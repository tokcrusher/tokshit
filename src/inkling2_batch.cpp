// Батч-путь Inkling2: B последовательных токенов за один проход по плотным
// весам. Слой-мажор: все токены батча проходят слот за слотом, поэтому вес
// Q/K/V/O/R/gate/sink и conv-проекций читается из DRAM один раз на B строк.
// Внимание, shortconv и MoE пока остаются последовательными по токенам:
// состояние слота (conv/KV) обновляется в порядке позиций, а union экспертов
// на B=3 всё равно почти не экономит чтения.
#include "tokshit/inkling2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

#if defined(__aarch64__) && defined(TOKSHIT_HAVE_NEON_DOTPROD)
#include <arm_neon.h>
#endif

#include "tokshit/batch.hpp"
#include "tokshit/dequant.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/pool.hpp"

namespace tokshit {
namespace {

inline float sigmoidf(float x) { return 1.0F / (1.0F + std::exp(-x)); }

// Optional batch-path breakdown (TOKSHIT_INK_BATCH_PROF=1).
double g_batch_dense_ms = 0.0;
double g_batch_moe_ms = 0.0;
double g_batch_attn_ms = 0.0;
double g_batch_conv_ms = 0.0;
double g_batch_head_ms = 0.0;

void add_ms(double& dst, std::chrono::steady_clock::time_point t0) {
  dst += std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
             .count();
}

// Q8_0 multi-activation, бит-в-бит как dot_rows_q8_0: тот же порядок блоков,
// тот же vdot по двум половинам и vaddvq_f32 в конце. dot_row_multi для
// Q8_0 зовёт dot_q8_0 с другим порядком аккумуляторов, и на 48 слоях это
// расходится до единиц в hidden.
#if defined(__aarch64__) && defined(TOKSHIT_HAVE_NEON_DOTPROD)
struct Q8MultiTask {
  const BatchSeg* seg = nullptr;
  std::size_t rows_per_task = 16;
};

inline float load_f16_bits(const uint8_t* p) {
  float16_t h;
  std::memcpy(&h, p, sizeof(h));
  return vgetq_lane_f32(vcvt_f32_f16(vdup_n_f16(h)), 0);
}

void q8_multi_rows(const BatchSeg& seg, std::size_t r0, std::size_t r1) {
  constexpr std::size_t kBs = 34;
  const std::size_t nb = seg.in_dim >> 5;
  const std::size_t b = seg.b;
  if (b == 0 || b > 16) return;
  for (std::size_t r = r0; r < r1; ++r) {
    const uint8_t* row = seg.bytes.data() + r * seg.rb;
    float32x4_t acc[16];
    for (std::size_t j = 0; j < b; ++j) acc[j] = vdupq_n_f32(0.0F);
    for (std::size_t blk = 0; blk < nb; ++blk) {
      const uint8_t* p = row + blk * kBs;
      const float d = load_f16_bits(p);
      const int8x16_t w0 =
          vld1q_s8(reinterpret_cast<const std::int8_t*>(p + 2));
      const int8x16_t w1 =
          vld1q_s8(reinterpret_cast<const std::int8_t*>(p + 18));
      for (std::size_t j = 0; j < b; ++j) {
        const int8_t* aq = seg.acts[j].q.data() + blk * 32;
        const int8x16_t a0 = vld1q_s8(aq);
        const int8x16_t a1 = vld1q_s8(aq + 16);
        int32x4_t q = vdotq_s32(vdupq_n_s32(0), w0, a0);
        q = vdotq_s32(q, w1, a1);
        acc[j] = vfmaq_f32(
            acc[j], vcvtq_f32_s32(q),
            vdupq_n_f32(d * seg.acts[j].scales[blk]));
      }
    }
    for (std::size_t j = 0; j < b; ++j) {
      seg.out[j * seg.rows + r] = vaddvq_f32(acc[j]);
    }
  }
}

void q8_multi_task(void* ctx, std::size_t task) {
  const auto& t = *static_cast<const Q8MultiTask*>(ctx);
  const std::size_t r0 = task * t.rows_per_task;
  if (r0 >= t.seg->rows) return;
  const std::size_t r1 =
      std::min(t.seg->rows, r0 + t.rows_per_task);
  q8_multi_rows(*t.seg, r0, r1);
}
#endif

// F32/BF16 matvec ровно как matvec_task: сырой float, без int8-кванта
// активации. Так считаются k_conv_proj/v_conv_proj у global-слоёв.
void matvec_f32_bf16(const InkMeta& m, const uint8_t* base, const float* x,
                     float* y) {
  for (std::size_t r = 0; r < m.rows; ++r) {
    const uint8_t* row = base + r * m.rb;
    float acc = 0.0F;
    if (m.quant == QuantType::F32) {
      const float* w = reinterpret_cast<const float*>(row);
      for (std::size_t i = 0; i < m.in_dim; ++i) acc += w[i] * x[i];
    } else {
      const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
      for (std::size_t i = 0; i < m.in_dim; ++i) {
        const uint32_t bits = static_cast<uint32_t>(w[i]) << 16;
        float f = 0.0F;
        std::memcpy(&f, &bits, sizeof(f));
        acc += f * x[i];
      }
    }
    y[r] = acc;
  }
}

void run_dense_bsegs(std::span<const BatchSeg> segs) {
  std::vector<BatchSeg> rest;
  rest.reserve(segs.size());
  for (const BatchSeg& s : segs) {
#if defined(__aarch64__) && defined(TOKSHIT_HAVE_NEON_DOTPROD)
    if (s.quant == QuantType::Q8_0 && s.in_dim % 32 == 0 &&
        kernels::have_dotprod() && s.b != 0 && s.b <= 16) {
      Q8MultiTask task{&s, 16};
      const std::size_t tasks = (s.rows + task.rows_per_task - 1) /
                                task.rows_per_task;
      pool::run_tasks(tasks, q8_multi_task, &task);
      continue;
    }
#endif
    rest.push_back(s);
  }
  if (!rest.empty()) {
    batchlib::run_bsegs(std::span<const BatchSeg>(rest.data(), rest.size()),
                        16);
  }
}

void dump_batch_vec(const char* name, const float* p, std::size_t n) {
  const char* dir = std::getenv("TOKSHIT_INK_BATCH_DUMP_DIR");
  if (dir == nullptr || *dir == '\0') return;
  const std::string path = std::string(dir) + "/" + name + ".bin";
  if (FILE* f = std::fopen(path.c_str(), "wb"); f != nullptr) {
    std::fwrite(p, sizeof(float), n, f);
    std::fclose(f);
  }
}

}  // namespace

void Inkling2::ensure_batch(std::size_t b) {
  const std::size_t h = hidden_;
  const std::size_t qdim = n_head_ * head_dim_;
  const std::size_t kvd = n_kv_ * head_dim_;
  const std::size_t dyn = l_cache_ * n_kv_;
  const std::size_t conv_n = (l_cache_ - 1) * sconv_channels_;
  const std::size_t ntap = tap_slots_.size();
  const std::size_t slots = slots_.size();

  // Сколько residual-блоков накопится к концу прохода. Точно как в
  // token_logits: прелюдия, core на каждый pass, эпилог.
  std::size_t need_blocks = 0;
  if (block_size_ != 0) {
    auto count = [&](std::size_t begin, std::size_t end) {
      for (std::size_t ip = begin; ip < end; ++ip) {
        if (ip % block_size_ == 0) ++need_blocks;
      }
    };
    count(0, core_start_);
    for (std::size_t p = 0; p < n_passes_; ++p) count(core_start_, core_end_);
    count(core_end_, n_phys_);
  }
  need_blocks = std::max<std::size_t>(need_blocks + 1, 2);

  if (b <= batch_cap_ && need_blocks <= max_blocks_) return;
  batch_cap_ = std::max(b, batch_cap_);
  max_blocks_ = std::max(need_blocks, max_blocks_);
  const std::size_t B = batch_cap_;
  const std::size_t MB = max_blocks_;

  xb_.assign(B * h, 0.0F);
  inp_sab_.assign(B * h, 0.0F);
  nrm3b_.assign(B * 3 * h, 0.0F);
  inp3b_.assign(B * 3 * h, 0.0F);
  mixb_.assign(B * 3 * h, 0.0F);
  mix2b_.assign(B * h, 0.0F);
  msg_b_.assign(B * 4 * h, 0.0F);
  ob_.assign(B * h, 0.0F);
  ffnb_.assign(B * h, 0.0F);
  ffn_in_b_.assign(B * h, 0.0F);
  prev_out_b_.assign(B * h, 0.0F);
  prelude_b_.assign(B * h, 0.0F);
  hb_.assign(B * h, 0.0F);

  qb_.assign(B * qdim, 0.0F);
  kb_.assign(B * kvd, 0.0F);
  vb_.assign(B * kvd, 0.0F);
  rb_.assign(B * n_relative_ * n_head_, 0.0F);
  gateb_.assign(B * qdim, 0.0F);
  attnb_.assign(B * qdim, 0.0F);
  sinkb_.assign(B * n_head_, 0.0F);
  dyn_kb_.assign(B * dyn, 0.0F);
  dyn_vb_.assign(B * dyn, 0.0F);

  block_bank_.assign(B * MB * h, 0.0F);
  bank_b_.assign(B * MB * h, 0.0F);
  prelude_blocks_b_.assign(B * MB * h, 0.0F);
  block_count_.assign(B, 0);
  bank_count_.assign(B, 0);
  prelude_count_.assign(B, 0);

  conv_before_.assign(slots * conv_n, 0.0F);
  conv_steps_.assign(B * slots * (sconv_channels_ + 2 * dyn), 0.0F);
  batch_taps_.assign(B * ntap * h, 0.0F);
  logits_b_.assign(B * vocab_padded_, 0.0F);
  acts_b_.resize(3 * B);
  state_ptrs_.assign(MB, nullptr);
}

void Inkling2::capture_batch_tap(std::size_t slot) {
  if (tap_slots_.empty() || last_batch_ == 0) return;
  const std::size_t h = hidden_;
  const std::size_t nt = tap_slots_.size();
  for (std::size_t k = 0; k < nt; ++k) {
    if (tap_slots_[k] != slot) continue;
    for (std::size_t j = 0; j < last_batch_; ++j) {
      std::memcpy(batch_taps_.data() + (j * nt + k) * h,
                  xb_.data() + j * h, h * sizeof(float));
    }
  }
}

void Inkling2::restore_batch(std::size_t accepted) {
  const std::size_t b = last_batch_;
  if (b == 0 || accepted > b) return;
  if (accepted == b) return;

  const std::size_t slots = slots_.size();
  const std::size_t kvd = n_kv_ * head_dim_;
  const std::size_t dyn = l_cache_ * n_kv_;
  const std::size_t conv_n = (l_cache_ - 1) * sconv_channels_;
  const std::size_t raw_n = sconv_channels_ + 2 * dyn;

  // Вернуть состояние перед батчем и прокрутить только принятые токены.
  // shortconv хранит последние L-1 входов, поэтому состояние после m
  // токенов однозначно восстанавливается из сохранённых входов.
  for (std::size_t s = 0; s < slots; ++s) {
    std::memcpy(slots_[s].conv.data(), conv_before_.data() + s * conv_n,
                conv_n * sizeof(float));
  }
  std::vector<float> tmp(sconv_channels_, 0.0F);
  for (std::size_t s = 0; s < slots; ++s) {
    if (accepted == 0) break;
    cur_slot_ = s;
    const InkLayer& L = phys(s);
    for (std::size_t j = 0; j < accepted; ++j) {
      const float* raw =
          conv_steps_.data() + (j * slots + s) * raw_n;
      std::memcpy(tmp.data(), raw, 2 * kvd * sizeof(float));
      shortconv(L, L.k_conv, raw + sconv_channels_, 0, kvd, tmp.data());
      shortconv(L, L.v_conv, raw + sconv_channels_ + dyn, kvd, kvd,
                tmp.data() + kvd);
      std::memcpy(tmp.data(), raw + 2 * kvd, 2 * hidden_ * sizeof(float));
      shortconv(L, L.post_conv, nullptr, 2 * kvd, hidden_, tmp.data());
      shortconv(L, L.ffn_conv, nullptr, 2 * kvd + hidden_, hidden_,
                tmp.data() + hidden_);
    }
    slots_[s].high = batch_pos0_ + accepted;
  }
  if (accepted == 0) {
    for (std::size_t s = 0; s < slots; ++s) slots_[s].high = batch_pos0_;
  }
  last_batch_ = accepted;
}

void Inkling2::run_app_batch(std::size_t slot, std::size_t ip, int pass,
                             const float* msg_b, std::size_t pos0,
                             std::size_t b) {
  const InkLayer& L = phys(slot);
  const bool swa = slot_swa_[slot] != 0;
  const std::size_t h = hidden_;
  const std::size_t hd = head_dim_;
  const std::size_t nh = n_head_;
  const std::size_t nkv = n_kv_;
  const std::size_t qdim = nh * hd;
  const std::size_t kvd = nkv * hd;
  const std::size_t dyn = l_cache_ * n_kv_;
  const std::size_t raw_n = sconv_channels_ + 2 * dyn;
  const bool has_msg = pass > 0 && msg_b != nullptr;
  cur_slot_ = slot;
  const bool dump0 = slot == 0 && ip == 0 && pass == 0 && pos0 == 0 && b == 1;

  const auto bytes_of = [&](const InkMeta& m) {
    const uint8_t* base = (m.hot ? hot_.data() : data_) + m.base;
    return std::span<const uint8_t>(
        base, static_cast<std::size_t>(m.rows) * m.rb);
  };
  const auto seg = [&](const InkMeta& m, const float* x,
                       const kernels::ActQ8* acts, float* y) {
    return BatchSeg{bytes_of(m), m.rb, m.quant, m.in_dim, x, acts, b, y,
                    m.rows};
  };

  for (std::size_t j = 0; j < b; ++j) {
    std::memcpy(inp_sab_.data() + j * h, xb_.data() + j * h,
                h * sizeof(float));
  }

  // ── входы внимания: depth-mix, control, нормы ──────────────────────────
  for (std::size_t j = 0; j < b; ++j) {
    const float* inputs[3] = {xb_.data() + j * h, xb_.data() + j * h,
                              xb_.data() + j * h};
    if (!L.attn_res_query.empty()) {
      const std::size_t nq = L.q_inp_norm.empty() ? 1 : 3;
      const std::size_t nb = block_count_[j];
      for (std::size_t s = 0; s < nb; ++s) {
        state_ptrs_[s] = block_bank_.data() + (j * max_blocks_ + s) * h;
      }
      depth_mix(std::span<const float* const>(state_ptrs_.data(), nb),
                xb_.data() + j * h, L.attn_res_query.data(), nq,
                mixb_.data() + j * 3 * h);
      for (std::size_t c = 0; c < 3; ++c) {
        inputs[c] = mixb_.data() + j * 3 * h + (nq == 1 ? 0 : c * h);
      }
    }
    for (std::size_t c = 0; c < 3; ++c) {
      std::memcpy(inp3b_.data() + (j * 3 + c) * h, inputs[c],
                  h * sizeof(float));
      if (has_msg) {
        control(inp3b_.data() + (j * 3 + c) * h,
                msg_b + (j * 4 + c) * h, L, static_cast<int>(c));
      }
    }

    if (!L.attn_norm.empty() && swa) {
      for (std::size_t c = 0; c < 3; ++c) {
        std::memcpy(nrm3b_.data() + (j * 3 + c) * h,
                    has_msg ? inp3b_.data() + (j * 3 + c) * h
                            : inp3b_.data() + j * 3 * h,
                    h * sizeof(float));
        rmsnorm(std::span<float>(nrm3b_.data() + (j * 3 + c) * h, h),
                std::span<const float>(L.attn_norm.data(), L.attn_norm.size()),
                eps_);
      }
    } else if (!L.attn_norm.empty()) {
      std::memcpy(nrm3b_.data() + j * 3 * h, inp3b_.data() + j * 3 * h,
                  h * sizeof(float));
      rmsnorm(std::span<float>(nrm3b_.data() + j * 3 * h, h),
              std::span<const float>(L.attn_norm.data(), L.attn_norm.size()),
              eps_);
      for (std::size_t c = 1; c < 3; ++c) {
        std::memcpy(nrm3b_.data() + (j * 3 + c) * h,
                    nrm3b_.data() + j * 3 * h, h * sizeof(float));
      }
    } else {
      for (std::size_t c = 0; c < 3; ++c) {
        std::memcpy(nrm3b_.data() + (j * 3 + c) * h,
                    inp3b_.data() + (j * 3 + c) * h, h * sizeof(float));
        const auto& w = c == 0 ? L.q_inp_norm
                               : (c == 1 ? L.k_inp_norm : L.v_inp_norm);
        rmsnorm(std::span<float>(nrm3b_.data() + (j * 3 + c) * h, h),
                std::span<const float>(w.data(), w.size()), eps_);
      }
    }
    for (std::size_t c = 0; c < 3; ++c) {
      kernels::quantize_into(
          std::span<const float>(nrm3b_.data() + (j * 3 + c) * h, h),
          acts_b_[c * b + j]);
    }
  }

  if (dump0) dump_batch_vec("xin", nrm3b_.data(), h);
  // ── плотные проекции: одна полоса весов на B токенов ───────────────────
  {
    // sink_proj идёт тем же путём, что matvec_multi: BF16-вес через
    // dot_fallback с квантованной в int8 активацией. Иначе sink уезжает на
    // ~0.5% и тянет за собой внимание.
    const BatchSeg segs[6] = {
        seg(L.q, inp3b_.data(), acts_b_.data() + 0 * b, qb_.data()),
        seg(L.k, inp3b_.data() + h, acts_b_.data() + 1 * b, kb_.data()),
        seg(L.v, inp3b_.data() + 2 * h, acts_b_.data() + 2 * b, vb_.data()),
        seg(L.rel, inp3b_.data(), acts_b_.data() + 0 * b, rb_.data()),
        seg(L.gate, inp3b_.data(), acts_b_.data() + 0 * b, gateb_.data()),
        seg(L.sink_proj, nrm3b_.data(), acts_b_.data() + 0 * b,
            sinkb_.data()),
    };
    const auto t_dense = std::chrono::steady_clock::now();
    run_dense_bsegs(std::span<const BatchSeg>(segs, 6));
    add_ms(g_batch_dense_ms, t_dense);
  }
  if (dump0) {
    dump_batch_vec("q_raw", qb_.data(), qdim);
    dump_batch_vec("k_raw", kb_.data(), kvd);
    dump_batch_vec("v_raw", vb_.data(), kvd);
    dump_batch_vec("r_raw", rb_.data(), n_relative_ * nh);
  }
  if (L.k_conv_proj.base) {
    const uint8_t* kb =
        (L.k_conv_proj.hot ? hot_.data() : data_) + L.k_conv_proj.base;
    const uint8_t* vb =
        (L.v_conv_proj.hot ? hot_.data() : data_) + L.v_conv_proj.base;
    const auto t_conv = std::chrono::steady_clock::now();
    for (std::size_t j = 0; j < b; ++j) {
      matvec_f32_bf16(L.k_conv_proj, kb,
                      nrm3b_.data() + (j * 3 + 1) * h,
                      dyn_kb_.data() + j * dyn);
      matvec_f32_bf16(L.v_conv_proj, vb,
                      nrm3b_.data() + (j * 3 + 2) * h,
                      dyn_vb_.data() + j * dyn);
    }
    add_ms(g_batch_conv_ms, t_conv);
  }

  // ── clip / norm / conv / KV-запись, по токенам в порядке позиций ───────
  std::vector<uint8_t> wrote_dyn(2, 0);
  wrote_dyn[0] = L.k_conv_proj.base ? 1 : 0;
  wrote_dyn[1] = L.v_conv_proj.base ? 1 : 0;
  for (std::size_t j = 0; j < b; ++j) {
    const std::size_t pos = pos0 + j;
    float* qv = qb_.data() + j * qdim;
    float* kvp = kb_.data() + j * kvd;
    float* vv = vb_.data() + j * kvd;
    float* rv = rb_.data() + j * n_relative_ * nh;

    // shortconv K/V: состояние слота обязано идти строго по позициям.
    // Вход сохраняем до conv — он же нужен для отката непринятого хвоста.
    const float* dk = wrote_dyn[0] ? dyn_kb_.data() + j * dyn : nullptr;
    const float* dv = wrote_dyn[1] ? dyn_vb_.data() + j * dyn : nullptr;
    float* raw = conv_steps_.data() + (j * slots_.size() + slot) * raw_n;
    std::memcpy(raw, kvp, kvd * sizeof(float));
    std::memcpy(raw + kvd, vv, kvd * sizeof(float));
    if (dk != nullptr) {
      std::memcpy(raw + sconv_channels_, dk, dyn * sizeof(float));
    }
    if (dv != nullptr) {
      std::memcpy(raw + sconv_channels_ + dyn, dv, dyn * sizeof(float));
    }
    if (dump0 && j == 0) {
      const float* st = slots_[slot].conv.data() + kvd * (l_cache_ - 1);
      dump_batch_vec("v_state_before", st, kvd * (l_cache_ - 1));
      dump_batch_vec("v_dyn_before", dv != nullptr ? dv : vv, dv ? dyn : 0);
    }
    shortconv(L, L.k_conv, dk, 0, kvd, kvp);
    shortconv(L, L.v_conv, dv, kvd, kvd, vv);
    if (dump0 && j == 0) dump_batch_vec("v_conv", vv, kvd);

    if (qk_clip_ > 0.0F) {
      for (std::size_t hh = 0; hh < nh; ++hh) {
        float* q = qv + hh * hd;
        float ss = 0.0F;
        for (std::size_t i = 0; i < hd; ++i) {
          q[i] *= L.q_clip[i + hd * hh];
          ss += q[i] * q[i];
        }
        const float f =
            std::min(1.0F, qk_clip_ / (std::sqrt(ss) + 1e-6F));
        for (std::size_t i = 0; i < hd; ++i) q[i] *= f;
      }
      for (std::size_t hh = 0; hh < nkv; ++hh) {
        float* k = kvp + hh * hd;
        float ss = 0.0F;
        for (std::size_t i = 0; i < hd; ++i) ss += k[i] * k[i];
        const float f =
            std::min(1.0F, qk_clip_ / (std::sqrt(ss) + 1e-6F));
        for (std::size_t i = 0; i < hd; ++i) k[i] *= f;
      }
    }
    for (std::size_t hh = 0; hh < nkv; ++hh) {
      rmsnorm(std::span<float>(vv + hh * hd, hd), {}, eps_);
    }
    if (rel_clip_ > 0.0F) {
      for (std::size_t hh = 0; hh < nh; ++hh) {
        float* r = rv + n_relative_ * hh;
        float ss = 0.0F;
        for (std::size_t i = 0; i < n_relative_; ++i) {
          r[i] *= L.rel_clip[i + n_relative_ * hh];
          ss += r[i] * r[i];
        }
        const float f =
            std::min(1.0F, rel_clip_ / (std::sqrt(ss) + 1e-6F));
        for (std::size_t i = 0; i < n_relative_; ++i) r[i] *= f;
      }
    }
    const float tau =
        (!swa && log_scale_floor_ != 0)
            ? 1.0F + log_scale_alpha_ *
                         std::log(std::max(
                             1.0F, static_cast<float>(pos + 1) /
                                       static_cast<float>(log_scale_floor_)))
            : 1.0F;
    if (!swa && tau != 1.0F) {
      for (std::size_t i = 0; i < qdim; ++i) qv[i] *= tau;
      for (std::size_t i = 0; i < n_relative_ * nh; ++i) rv[i] *= tau;
    }

    if (pos < kv_ctx_) {
      auto& st = slots_[slot];
      uint16_t* krow = st.k.data() + pos * kvd;
      uint16_t* vrow = st.v.data() + pos * kvd;
      for (std::size_t i = 0; i < kvd; ++i) {
        krow[i] = f32_to_f16(kvp[i]);
        vrow[i] = f32_to_f16(vv[i]);
      }
      if (pos + 1 > st.high) st.high = pos + 1;
    }
  }
  if (dump0) {
    dump_batch_vec("q", qb_.data(), qdim);
    dump_batch_vec("k", kb_.data(), kvd);
    dump_batch_vec("v", vb_.data(), kvd);
    dump_batch_vec("r", rb_.data(), n_relative_ * nh);
    dump_batch_vec("sink", sinkb_.data(), nh);
  }

  // ── внимание и гейт, по токенам ────────────────────────────────────────
  const auto t_attn = std::chrono::steady_clock::now();
  for (std::size_t j = 0; j < b; ++j) {
    const std::size_t pos = pos0 + j;
    std::memcpy(r_.data(), rb_.data() + j * n_relative_ * nh,
                n_relative_ * nh * sizeof(float));
    std::memcpy(sink_.data(), sinkb_.data() + j * nh,
                nh * sizeof(float));
    const float tau =
        (!swa && log_scale_floor_ != 0)
            ? 1.0F + log_scale_alpha_ *
                         std::log(std::max(
                             1.0F, static_cast<float>(pos + 1) /
                                       static_cast<float>(log_scale_floor_)))
            : 1.0F;
    attention(L, slot, pos, swa, tau, qb_.data() + j * qdim,
              kb_.data() + j * kvd, vb_.data() + j * kvd,
              attnb_.data() + j * qdim);
    for (std::size_t i = 0; i < qdim; ++i) {
      attnb_[j * qdim + i] *= sigmoidf(gateb_[j * qdim + i]);
    }
    kernels::quantize_into(
        std::span<const float>(attnb_.data() + j * qdim, qdim),
        acts_b_[j]);
  }
  add_ms(g_batch_attn_ms, t_attn);

  // ── O + post-conv + residual ───────────────────────────────────────────
  {
    const BatchSeg oseg =
        seg(L.o, attnb_.data(), acts_b_.data(), ob_.data());
    const auto t_dense = std::chrono::steady_clock::now();
    run_dense_bsegs(std::span<const BatchSeg>(&oseg, 1));
    add_ms(g_batch_dense_ms, t_dense);
  }
  const auto t_post = std::chrono::steady_clock::now();
  for (std::size_t j = 0; j < b; ++j) {
    float* o = ob_.data() + j * h;
    float* raw = conv_steps_.data() + (j * slots_.size() + slot) * raw_n;
    std::memcpy(raw + 2 * kvd, o, h * sizeof(float));
    shortconv(L, L.post_conv, nullptr, 2 * kvd, h, o);
  }
  add_ms(g_batch_conv_ms, t_post);
  if (dump0) dump_batch_vec("o_conv", ob_.data(), h);

  for (std::size_t j = 0; j < b; ++j) {
    const float* sa = inp_sab_.data() + j * h;
    const float* o = ob_.data() + j * h;
    if (ip % block_size_ == 0) {
      if (block_count_[j] < max_blocks_) {
        std::memcpy(block_bank_.data() + (j * max_blocks_ + block_count_[j]) * h,
                    sa, h * sizeof(float));
        ++block_count_[j];
      }
      std::memcpy(xb_.data() + j * h, o, h * sizeof(float));
    } else {
      for (std::size_t i = 0; i < h; ++i) {
        xb_[j * h + i] = sa[i] + o[i];
      }
    }
  }

  // ── FFN: depth-mix, control, MoE, post-conv ────────────────────────────
  static const bool noffn = std::getenv("TOKSHIT_INK_NOFFN") != nullptr;
  for (std::size_t j = 0; j < b; ++j) {
    std::memcpy(ffn_in_b_.data() + j * h, xb_.data() + j * h,
                h * sizeof(float));
    const std::size_t nb = block_count_[j];
    for (std::size_t s = 0; s < nb; ++s) {
      state_ptrs_[s] = block_bank_.data() + (j * max_blocks_ + s) * h;
    }
    depth_mix(std::span<const float* const>(state_ptrs_.data(), nb),
              xb_.data() + j * h, L.ffn_res_query.data(), 1,
              mix2b_.data() + j * h);
    if (has_msg) {
      control(mix2b_.data() + j * h, msg_b + (j * 4 + 3) * h, L, 3);
    }
    rmsnorm(std::span<float>(mix2b_.data() + j * h, h),
            std::span<const float>(L.ffn_norm.data(), L.ffn_norm.size()), eps_);
    float* ffn = ffnb_.data() + j * h;
    if (noffn) {
      std::fill(ffn, ffn + h, 0.0F);
    } else {
      const auto t_moe = std::chrono::steady_clock::now();
      moe_batch(L, mix2b_.data(), ffnb_.data(), b);
      add_ms(g_batch_moe_ms, t_moe);
    }
    if (dump0) {
      dump_batch_vec("moe_raw", ffn, h);
      dump_batch_vec("mix_ffn", mix2b_.data(), h);
      std::vector<float> dbg(64);
      for (std::size_t k = 0; k < n_used_; ++k) {
        dbg[k] = wts_[k];
        dbg[32 + k] = static_cast<float>(top_[k]);
      }
      dump_batch_vec("router", dbg.data(), dbg.size());
    }
    float* raw = conv_steps_.data() + (j * slots_.size() + slot) * raw_n;
    std::memcpy(raw + 2 * kvd + h, ffn, h * sizeof(float));
    if (!noffn) {
      const auto t_conv = std::chrono::steady_clock::now();
      shortconv(L, L.ffn_conv, nullptr, 2 * kvd + h, h, ffn);
      add_ms(g_batch_conv_ms, t_conv);
    }
    for (std::size_t i = 0; i < h; ++i) {
      xb_[j * h + i] = ffn_in_b_[j * h + i] + ffn[i];
    }
  }
  if (dump0) dump_batch_vec("x0", xb_.data(), h);
}

// ── moe_batch ───────────────────────────────────────────────────────────
// B токенов батча через MoE-слой: роутер — одна batched-полоса на все
// токены, затем union экспертов батча (gate_up + down) по эксперту с
// BatchSeg-ами: веса читаются один раз на эксперт, а обслуживают все его
// пары. Пер-токен топ-k, веса и накопление out[j] в том же порядке, что и
// у Inkling2::moe(), чтобы отличия были только из-за перегруппировки
// сложений (union vs цикл по k).
void Inkling2::moe_batch(const InkLayer& L, float* x, float* out,
                          std::size_t b) {
  const std::size_t h = hidden_;
  const std::size_t nff = n_ff_exp_;
  const std::size_t used =
      router_topk_ != 0 && router_topk_ < n_used_ ? router_topk_ : n_used_;
  std::fill(out, out + b * h, 0.0F);
  if (b == 0 || n_expert_ == 0 || used == 0) return;
  const std::size_t M = b * used;
  const auto bytes_of = [&](const InkMeta& m) {
    const uint8_t* base = (m.hot ? hot_.data() : data_) + m.base;
    return std::span<const uint8_t>(
        base, static_cast<std::size_t>(m.rows) * m.rb);
  };
  // срез одного эксперта: блок эксперта — `rows` строк по rb байт.
  const auto expert_bytes = [&](const InkMeta& m, std::size_t e) {
    const uint8_t* base = (m.hot ? hot_.data() : data_) + m.base +
                          e * static_cast<std::size_t>(m.rb) * m.rows;
    return std::span<const uint8_t>(
        base, static_cast<std::size_t>(m.rows) * m.rb);
  };
  constexpr std::size_t kNoPair = std::numeric_limits<std::size_t>::max();

  // ── роутер: gate_inp на B токенов через run_bsegs ────────────────────
  router_b_.resize(b * n_expert_);
  router_acts_.resize(b);
  for (std::size_t j = 0; j < b; ++j) {
    kernels::quantize_into(std::span<const float>(x + j * h, h),
                           router_acts_[j]);
  }
  {
    const BatchSeg seg{bytes_of(L.gate_inp), L.gate_inp.rb, L.gate_inp.quant,
                       L.gate_inp.in_dim, x, router_acts_.data(), b,
                       router_b_.data(), n_expert_};
    batchlib::run_bsegs(std::span<const BatchSeg>(&seg, 1), 16);
  }

  // ── пер-токен топ-k: тот же rule, что и Inkling2::moe() ───────────────
  top_b_.resize(M);
  wts_b_.resize(M);
  pair_at_b_.resize(M);
  const std::vector<float>& eb = L.exp_probs_b;
  for (std::size_t j = 0; j < b; ++j) {
    float* rr = router_b_.data() + j * n_expert_;
    for (std::size_t e = 0; e < n_expert_; ++e) {
      rr[e] = sigmoidf(rr[e]) + eb[e];
    }
    int* top = top_b_.data() + j * used;
    for (std::size_t k = 0; k < used; ++k) {
      int best = -1;
      float bv = -1e30F;
      for (std::size_t e = 0; e < n_expert_; ++e) {
        bool taken = false;
        for (std::size_t kk = 0; kk < k; ++kk) {
          if (top[kk] == static_cast<int>(e)) {
            taken = true;
            break;
          }
        }
        if (taken) continue;
        if (rr[e] > bv) {
          bv = rr[e];
          best = static_cast<int>(e);
        }
      }
      top[k] = best;
    }
    float* w = wts_b_.data() + j * used;
    float sum = 0.0F;
    for (std::size_t k = 0; k < used; ++k) {
      float p = 0.0F;
      if (top[k] >= 0) {
        const std::size_t e = static_cast<std::size_t>(top[k]);
        p = rr[e] - eb[e];
        if (p < 0.0F) p = 0.0F;
      }
      w[k] = p;
      sum += p;
    }
    const float inv = 1.0F / (sum + 1e-20F);
    for (std::size_t k = 0; k < used; ++k) {
      w[k] *= inv;
      if (expert_scale_ != 0.0F) w[k] *= expert_scale_;
    }
    if (b == 1) {  // для паритета с dump в run_app_batch
      for (std::size_t k = 0; k < used; ++k) {
        top_[k] = top[k];
        wts_[k] = w[k];
      }
    }
  }

  // ── union экспертов батча: seen/emap/utop, затем flat (j, вес) ──────
  seen_b_.resize(n_expert_);
  emap_b_.resize(n_expert_);
  std::fill(seen_b_.data(), seen_b_.data() + n_expert_, 0);
  std::fill(emap_b_.data(), emap_b_.data() + n_expert_, -1);
  utop_b_.clear();
  for (std::size_t j = 0; j < b; ++j) {
    const int* top = top_b_.data() + j * used;
    for (std::size_t k = 0; k < used; ++k) {
      const int e = top[k];
      if (e < 0) continue;
      if (seen_b_[e] == 0) {
        seen_b_[e] = 1;
        emap_b_[e] = static_cast<int>(utop_b_.size());
        utop_b_.push_back(static_cast<std::size_t>(e));
      }
    }
  }
  const std::size_t nu = utop_b_.size();
  moff_b_.resize(n_expert_);
  mcur_b_.resize(n_expert_);
  std::fill(moff_b_.data(), moff_b_.data() + nu, 0);
  std::size_t total = 0;
  for (std::size_t j = 0; j < b; ++j) {
    const int* top = top_b_.data() + j * used;
    for (std::size_t k = 0; k < used; ++k) {
      const int e = top[k];
      if (e < 0) continue;
      ++moff_b_[static_cast<std::size_t>(emap_b_[e])];
      ++total;
    }
  }
  {
    std::size_t cursor = 0;
    for (std::size_t u = 0; u < nu; ++u) {
      const std::size_t cnt = moff_b_[u];
      moff_b_[u] = cursor;
      mcur_b_[u] = cursor;
      cursor += cnt;
    }
  }

  // flat пары: gx_b_ — входы gate_up на пару, pair_at_b_ — пара → flat-индекс
  gx_b_.resize(total * h);
  moe_in_b_.resize(total);
  gu_b_.resize(total * 2 * nff);
  act_b_.resize(total * nff);
  moe_act_b_.resize(total);
  dn_b_.resize(total * h);
  for (std::size_t j = 0; j < b; ++j) {
    const int* top = top_b_.data() + j * used;
    for (std::size_t k = 0; k < used; ++k) {
      const int e = top[k];
      if (e < 0) {
        pair_at_b_[j * used + k] = kNoPair;
        continue;
      }
      const std::size_t u = static_cast<std::size_t>(emap_b_[e]);
      const std::size_t at = mcur_b_[u]++;
      pair_at_b_[j * used + k] = at;
      std::memcpy(gx_b_.data() + at * h, x + j * h, h * sizeof(float));
    }
  }
  for (std::size_t at = 0; at < total; ++at) {
    kernels::quantize_into(std::span<const float>(gx_b_.data() + at * h, h),
                           moe_in_b_[at]);
  }

  // ── per expert: gate_up (batched) → SiTU + quantize → down (batched) ─
  for (std::size_t u = 0; u < nu; ++u) {
    const std::size_t off = moff_b_[u];
    const std::size_t cnt =
        (u + 1 < nu ? moff_b_[u + 1] : total) - off;
    if (cnt == 0) continue;
    {
      const BatchSeg seg{expert_bytes(L.gate_up_exps, utop_b_[u]),
                         L.gate_up_exps.rb, L.gate_up_exps.quant,
                         L.gate_up_exps.in_dim, gx_b_.data() + off * h,
                         moe_in_b_.data() + off, cnt,
                         gu_b_.data() + off * 2 * nff, 2 * nff};
      batchlib::run_bsegs(std::span<const BatchSeg>(&seg, 1), 32);
    }
    const float gs = L.gate_up_scale.empty() ? 1.0F
                                             : L.gate_up_scale[utop_b_[u]];
    for (std::size_t mi = 0; mi < cnt; ++mi) {
      const float* gu = gu_b_.data() + (off + mi) * 2 * nff;
      float* dst = act_b_.data() + (off + mi) * nff;
      for (std::size_t i = 0; i < nff; ++i) {
        const float gi = gu[i] * gs;
        const float ui = gu[nff + i] * gs;
        float act = situ_beta_ * std::tanh(gi / situ_beta_) * sigmoidf(gi);
        act *= situ_linear_beta_ > 0.0F
                   ? situ_linear_beta_ * std::tanh(ui / situ_linear_beta_)
                   : ui;
        dst[i] = act;
      }
      kernels::quantize_into(std::span<const float>(dst, nff),
                             moe_act_b_[off + mi]);
    }
    {
      const BatchSeg seg{expert_bytes(L.down_exps, utop_b_[u]),
                         L.down_exps.rb, L.down_exps.quant,
                         L.down_exps.in_dim, act_b_.data() + off * nff,
                         moe_act_b_.data() + off, cnt, dn_b_.data() + off * h,
                         h};
      batchlib::run_bsegs(std::span<const BatchSeg>(&seg, 1), 32);
    }
  }

  // ── накопление out[j] += w_k · down[k] в порядке k, как в moe() ─────
  for (std::size_t j = 0; j < b; ++j) {
    float* dst = out + j * h;
    const int* top = top_b_.data() + j * used;
    for (std::size_t k = 0; k < used; ++k) {
      const std::size_t at = pair_at_b_[j * used + k];
      if (at == kNoPair) continue;
      const float w = wts_b_[j * used + k] *
                      (L.down_scale.empty() ? 1.0F : L.down_scale[top[k]]);
      const float* src = dn_b_.data() + at * h;
      for (std::size_t i = 0; i < h; ++i) dst[i] += w * src[i];
    }
  }
}

void Inkling2::token_logits_batch(std::span<const uint32_t> tokens,
                                  std::size_t pos, std::span<float> out) {
  const std::size_t b = tokens.size();
  const std::size_t vocab = vocab_unpadded_;
  if (b == 0) return;
  if (pos >= kv_ctx_ || pos + b > kv_ctx_ ||
      out.size() < b * vocab) {
    std::fill(out.begin(),
              out.begin() + static_cast<std::ptrdiff_t>(b * vocab), -1e30F);
    last_batch_ = 0;
    return;
  }
  for (uint32_t t : tokens) {
    if (t >= vocab) {
      std::fill(out.begin(),
                out.begin() + static_cast<std::ptrdiff_t>(b * vocab), -1e30F);
      last_batch_ = 0;
      return;
    }
  }

  if (pos == 0) reset();
  ensure_batch(b);
  last_batch_ = b;
  batch_pos0_ = pos;
  g_batch_dense_ms = 0.0;
  g_batch_moe_ms = 0.0;
  g_batch_attn_ms = 0.0;
  g_batch_conv_ms = 0.0;
  g_batch_head_ms = 0.0;
  const std::size_t h = hidden_;
  const std::size_t core = core_end_ - core_start_;
  const std::size_t slots = slots_.size();
  const std::size_t ntap = tap_slots_.size();
  const std::size_t conv_n = (l_cache_ - 1) * sconv_channels_;
  for (std::size_t s = 0; s < slots; ++s) {
    std::memcpy(conv_before_.data() + s * conv_n, slots_[s].conv.data(),
                conv_n * sizeof(float));
  }
  if (ntap != 0) {
    batch_taps_.assign(b * ntap * h, 0.0F);
  }

  for (std::size_t j = 0; j < b; ++j) {
    embed(tokens[j]);
    std::memcpy(xb_.data() + j * h, x_.data(), h * sizeof(float));
    block_count_[j] = 0;
    bank_count_[j] = 0;
    prelude_count_[j] = 0;
  }

  // ── прелюдия ───────────────────────────────────────────────────────────
  for (std::size_t slot = 0; slot < core_start_; ++slot) {
    capture_batch_tap(slot);
    run_app_batch(slot, slot, 0, nullptr, pos, b);
  }
  for (std::size_t j = 0; j < b; ++j) {
    std::memcpy(prelude_b_.data() + j * h, xb_.data() + j * h,
                h * sizeof(float));
    prelude_count_[j] = block_count_[j];
    std::memcpy(prelude_blocks_b_.data() + j * max_blocks_ * h,
                block_bank_.data() + j * max_blocks_ * h,
                max_blocks_ * h * sizeof(float));
  }

  // TOKSHIT_INK_PASSES повторяет расписание token_logits: меньше проходов —
  // меньше core-слотов, эпилог остаётся на своих физических слотах.
  std::size_t passes_run = n_passes_;
  if (const char* pv = std::getenv("TOKSHIT_INK_PASSES");
      pv != nullptr && *pv != '\0') {
    const unsigned long n = std::strtoul(pv, nullptr, 10);
    if (n >= 1 && n <= n_passes_) passes_run = static_cast<std::size_t>(n);
  }

  // ── core-луп ───────────────────────────────────────────────────────────
  for (std::size_t p = 0; p < passes_run; ++p) {
    if (p > 0) {
      for (std::size_t j = 0; j < b; ++j) {
        std::memcpy(xb_.data() + j * h, prelude_b_.data() + j * h,
                    h * sizeof(float));
        block_count_[j] = prelude_count_[j];
        std::memcpy(block_bank_.data() + j * max_blocks_ * h,
                    prelude_blocks_b_.data() + j * max_blocks_ * h,
                    max_blocks_ * h * sizeof(float));
      }
    }
    for (std::size_t ip = core_start_; ip < core_end_; ++ip) {
      const std::size_t slot = core_start_ + p * core + (ip - core_start_);
      const InkLayer& L = phys(slot);
      if (p > 0) {
        for (std::size_t j = 0; j < b; ++j) {
          const std::size_t nb = bank_count_[j];
          for (std::size_t s = 0; s < nb; ++s) {
            state_ptrs_[s] =
                bank_b_.data() + (j * max_blocks_ + s) * h;
          }
          for (int c = 0; c < 4; ++c) {
            depth_mix(
                std::span<const float* const>(state_ptrs_.data(), nb),
                prev_out_b_.data() + j * h, L.msg_query[c].data(), 1,
                msg_b_.data() + (j * 4 + c) * h);
          }
        }
      }
      capture_batch_tap(slot);
      run_app_batch(slot, ip, static_cast<int>(p),
                    p > 0 ? msg_b_.data() : nullptr, pos, b);
    }
    if (p + 1 < passes_run) {
      for (std::size_t j = 0; j < b; ++j) {
        const std::size_t first = prelude_count_[j] + 1;
        bank_count_[j] =
            block_count_[j] > first ? block_count_[j] - first : 0;
        for (std::size_t s = 0; s < bank_count_[j]; ++s) {
          std::memcpy(
              bank_b_.data() + (j * max_blocks_ + s) * h,
              block_bank_.data() + (j * max_blocks_ + first + s) * h,
              h * sizeof(float));
        }
        std::memcpy(prev_out_b_.data() + j * h, xb_.data() + j * h,
                    h * sizeof(float));
      }
    }
  }

  // ── эпилог ─────────────────────────────────────────────────────────────
  for (std::size_t ip = core_end_; ip < n_phys_; ++ip) {
    const std::size_t slot =
        ip - core_end_ + core_start_ + core * n_passes_;
    capture_batch_tap(slot);
    run_app_batch(slot, ip, 0, nullptr, pos, b);
  }

  // ── финальный depth-mix, out_norm, LM head ─────────────────────────────
  for (std::size_t j = 0; j < b; ++j) {
    const std::size_t nb = block_count_[j];
    for (std::size_t s = 0; s < nb; ++s) {
      state_ptrs_[s] = block_bank_.data() + (j * max_blocks_ + s) * h;
    }
    depth_mix(std::span<const float* const>(state_ptrs_.data(), nb),
              xb_.data() + j * h, out_res_query_.data(), 1,
              mix2b_.data() + j * h);
    rmsnorm(std::span<float>(mix2b_.data() + j * h, h),
            std::span<const float>(out_norm_.data(), out_norm_.size()), eps_);
    std::memcpy(xb_.data() + j * h, mix2b_.data() + j * h,
                h * sizeof(float));
  }
  capture_batch_tap(slot_count_);
  // output.weight — Q4_K: dot_rows_q4k и dot_row_multi(q4k) дают одинаковые
  // числа (в отличие от Q8_0), поэтому голову батчим и читаем 261 МБ один раз.
  const auto t_head = std::chrono::steady_clock::now();
  lm_head_batch(xb_.data(), b, h, logits_b_.data());
  add_ms(g_batch_head_ms, t_head);
  for (std::size_t j = 0; j < b; ++j) {
    float* row = logits_b_.data() + j * vocab_padded_;
    if (final_softcap_ > 0.0F) {
      for (std::size_t i = 0; i < vocab_padded_; ++i) {
        row[i] = final_softcap_ * std::tanh(row[i] / final_softcap_);
      }
    }
    std::copy(row, row + vocab,
              out.begin() + static_cast<std::ptrdiff_t>(j * vocab));
  }
  if (std::getenv("TOKSHIT_INK_BATCH_PROF") != nullptr) {
    std::fprintf(stderr,
                 "[batch prof] b=%zu dense=%.0fms moe=%.0fms attn=%.0fms "
                 "conv=%.0fms head=%.0fms\n",
                 b, g_batch_dense_ms, g_batch_moe_ms, g_batch_attn_ms,
                 g_batch_conv_ms, g_batch_head_ms);
  }
}

}  // namespace tokshit
