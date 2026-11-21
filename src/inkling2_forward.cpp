#include "tokshit/inkling2.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/pool.hpp"

namespace tokshit {
namespace {

struct MvTask {
  const uint8_t* data = nullptr;
  uint32_t rows = 0;
  uint32_t rb = 0;
  uint32_t in = 0;
  QuantType quant = QuantType::F32;
  const float* x = nullptr;
  float* y = nullptr;
  const kernels::ActQ8* act = nullptr;
  uint32_t rows_per_task = 32;
};

void matvec_task(void* ctx, std::size_t chunk) {
  const MvTask& t = *static_cast<const MvTask*>(ctx);
  const uint32_t r0 = static_cast<uint32_t>(chunk) * t.rows_per_task;
  if (r0 >= t.rows) return;
  const uint32_t n = std::min(t.rows_per_task, t.rows - r0);
  if (t.act != nullptr) {
    thread_local std::vector<float> scratch;
    kernels::dot_rows(
        t.quant, std::span<const uint8_t>(t.data + static_cast<std::size_t>(r0) * t.rb,
                                          static_cast<std::size_t>(n) * t.rb),
        t.rb, n, std::span<const float>(t.x, t.in), *t.act, t.in, t.y + r0,
        scratch);
    return;
  }
  for (uint32_t r = r0; r < r0 + n; ++r) {
    const uint8_t* row = t.data + static_cast<std::size_t>(r) * t.rb;
    float acc = 0.0F;
    if (t.quant == QuantType::F32) {
      const float* w = reinterpret_cast<const float*>(row);
      for (uint32_t i = 0; i < t.in; ++i) acc += w[i] * t.x[i];
    } else {
      const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
      for (uint32_t i = 0; i < t.in; ++i) {
        const uint32_t bits = static_cast<uint32_t>(w[i]) << 16;
        float f = 0.0F;
        std::memcpy(&f, &bits, sizeof(f));
        acc += f * t.x[i];
      }
    }
    t.y[r] = acc;
  }
}

struct ExpertJob {
  const std::uint8_t* data = nullptr;
  std::size_t stride = 0;
  std::uint32_t rows = 0;
  std::uint32_t rb = 0;
  std::uint32_t in = 0;
  std::uint32_t out_stride = 0;
  std::uint32_t chunk_rows = 64;
  std::uint32_t chunks = 0;
  tokshit::QuantType quant = tokshit::QuantType::F32;
  const tokshit::kernels::ActQ8* act = nullptr;
  const tokshit::kernels::ActQ8* act_block = nullptr;
  const int* ids = nullptr;
  const float* x = nullptr;
  std::uint32_t x_stride = 0;
  float* out = nullptr;
};

void expert_task(void* ctx, std::size_t t) {
  const ExpertJob& j = *static_cast<const ExpertJob*>(ctx);
  const std::size_t slot = t / j.chunks;
  const std::size_t e = j.ids != nullptr ? static_cast<std::size_t>(j.ids[slot]) : slot;
  const std::uint32_t r0 =
      static_cast<std::uint32_t>(t % j.chunks) * j.chunk_rows;
  if (r0 >= j.rows) return;
  const std::uint32_t n = std::min(j.chunk_rows, j.rows - r0);
  const std::uint8_t* base = j.data + e * j.stride;
  const float* xv = j.x + slot * j.x_stride;
  float* yv = j.out + slot * j.out_stride + r0;
  const tokshit::kernels::ActQ8& a =
      j.act_block != nullptr ? j.act_block[slot] : *j.act;
  thread_local std::vector<float> scratch;
  for (std::uint32_t r = 0; r < n; ++r) {
    yv[r] = tokshit::kernels::dot_row(
        j.quant, std::span<const std::uint8_t>(base + (r0 + r) * j.rb, j.rb),
        j.rb, std::span<const float>(xv, j.in), a, j.in, scratch);
  }
}

struct MultiJob {
  struct Seg {
    const std::uint8_t* data = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t rb = 0;
    std::uint32_t in = 0;
    std::uint32_t chunks = 0;
    tokshit::QuantType quant = tokshit::QuantType::F32;
    const tokshit::kernels::ActQ8* act = nullptr;
    const float* x = nullptr;
    float* y = nullptr;
  };
  std::vector<Seg> segs;
  std::vector<std::uint32_t> starts;
  std::uint32_t chunk_rows = 64;
};

void multi_task(void* ctx, std::size_t task) {
  const MultiJob& j = *static_cast<const MultiJob*>(ctx);
  std::size_t s = 0;
  while (s + 1 < j.starts.size() && task >= j.starts[s + 1]) ++s;
  const MultiJob::Seg& seg = j.segs[s];
  const std::uint32_t r0 =
      static_cast<std::uint32_t>(task - j.starts[s]) * j.chunk_rows;
  if (r0 >= seg.rows) return;
  const std::uint32_t n = std::min(j.chunk_rows, seg.rows - r0);
  thread_local std::vector<float> scratch;
  for (std::uint32_t r = 0; r < n; ++r) {
    seg.y[r0 + r] = tokshit::kernels::dot_row(
        seg.quant,
        std::span<const std::uint8_t>(seg.data + (r0 + r) * seg.rb, seg.rb),
        seg.rb, std::span<const float>(seg.x, seg.in), *seg.act, seg.in,
        scratch);
  }
}

inline float sigmoidf(float x) { return 1.0F / (1.0F + std::exp(-x)); }

void dump_vec(const char* name, const std::vector<float>& v) {
  const char* dir = std::getenv("TOKSHIT_INK_DUMP_DIR");
  if (dir == nullptr || *dir == '\0') return;
  const std::string path = std::string(dir) + "/" + name + ".bin";
  if (FILE* f = std::fopen(path.c_str(), "wb"); f != nullptr) {
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
  }
}

}  // namespace

void Inkling2::reset() {
  for (auto& st : slots_) {
    st.high = 0;
    std::fill(st.k.begin(), st.k.end(), 0);
    std::fill(st.v.begin(), st.v.end(), 0);
    std::fill(st.conv.begin(), st.conv.end(), 0.0F);
  }
}

void Inkling2::matvec(const InkMeta& m, const float* x, float* y) {
  matvec(m, x, y, nullptr);
}

void Inkling2::matvec(const InkMeta& m, const float* x, float* y,
                      float*) {
  const auto t0 = std::chrono::steady_clock::now();
  MvTask t;
  t.data = (m.hot ? hot_.data() : data_) + m.base;
  t.rows = m.rows;
  t.rb = m.rb;
  t.in = m.in_dim;
  t.quant = m.quant;
  t.x = x;
  t.y = y;
  kernels::ActQ8 act;
  if (m.quant != QuantType::F32 && m.quant != QuantType::Bf16) {
    kernels::quantize_into(std::span<const float>(x, m.in_dim), act);
    t.act = &act;
  }
  static const bool serial = std::getenv("TOKSHIT_INK_SERIAL") != nullptr;
  if (m.rows >= 256 && !serial) {
    t.rows_per_task = 64;
    const std::size_t tasks = (m.rows + t.rows_per_task - 1) / t.rows_per_task;
    pool::run_tasks(tasks, matvec_task, &t);
  } else {
    t.rows_per_task = m.rows;
    matvec_task(&t, 0);
  }
  if (std::getenv("TOKSHIT_INK_PROF") != nullptr) {
    prof_mv_ns_ += std::chrono::duration<double, std::nano>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
  }
}

void Inkling2::matvec_multi(std::span<const MatvecSeg> segs) {
  const auto t0 = std::chrono::steady_clock::now();
  MultiJob job;
  job.segs.reserve(segs.size());
  job.starts.reserve(segs.size() + 1);
  job.starts.push_back(0);
  std::uint32_t total = 0;
  for (const MatvecSeg& s : segs) {
    MultiJob::Seg seg;
    seg.data = s.meta->hot ? (hot_.data() + s.meta->base)
                           : (data_ + s.meta->base);
    seg.rows = s.meta->rows;
    seg.rb = s.meta->rb;
    seg.in = s.meta->in_dim;
    seg.chunks = (seg.rows + job.chunk_rows - 1) / job.chunk_rows;
    seg.quant = s.meta->quant;
    seg.act = s.act;
    seg.x = s.x;
    seg.y = s.y;
    total += seg.chunks;
    job.segs.push_back(seg);
    job.starts.push_back(total);
  }
  if (total == 0) return;
  pool::run_tasks(total, multi_task, &job);
  if (std::getenv("TOKSHIT_INK_PROF") != nullptr) {
    prof_mv_ns_ += std::chrono::duration<double, std::nano>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
  }
}

void Inkling2::rmsnorm(std::span<float> x, std::span<const float> w,
                       float eps) {
  float ss = 0.0F;
  for (float v : x) ss += v * v;
  const float scale = 1.0F / std::sqrt(ss / static_cast<float>(x.size()) + eps);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] *= scale;
    if (i < w.size()) x[i] *= w[i];
  }
}

void Inkling2::depth_mix(std::span<const float* const> states,
                         const float* cur, const float* query, std::size_t nq,
                         float* out) {
  const std::size_t n = states.size() + 1;
  float* norm = norm_states_.data();
  const std::size_t h = hidden_;

  std::memcpy(norm, cur, h * sizeof(float));
  for (std::size_t s = 0; s < states.size(); ++s) {
    std::memcpy(norm + (s + 1) * h, states[s], h * sizeof(float));
  }
  for (std::size_t s = 0; s < n; ++s) {
    float ss = 0.0F;
    float* v = norm + s * h;
    for (std::size_t i = 0; i < h; ++i) ss += v[i] * v[i];
    const float scale = 1.0F / std::sqrt(ss / static_cast<float>(h) + 1e-6F);
    for (std::size_t i = 0; i < h; ++i) v[i] *= scale;
  }

  const float inv = 1.0F / std::sqrt(static_cast<float>(h));
  for (std::size_t q = 0; q < nq; ++q) {
    const float* qv = query + q * h;
    float mx = 0.0F;
    for (std::size_t s = 0; s < n; ++s) {
      float dot = 0.0F;
      const float* v = norm + s * h;
      for (std::size_t i = 0; i < h; ++i) dot += v[i] * qv[i];
      dot *= inv;
      scores_[s] = dot;
      if (s == 0 || dot > mx) mx = dot;
    }
    float sum = 0.0F;
    for (std::size_t s = 0; s < n; ++s) {
      scores_[s] = std::exp(scores_[s] - mx);
      sum += scores_[s];
    }
    const float inv_sum = 1.0F / sum;
    float* o = out + q * h;
    std::fill(o, o + h, 0.0F);
    for (std::size_t s = 0; s < n; ++s) {
      const float p = scores_[s] * inv_sum;
      const float* v = norm + s * h;
      for (std::size_t i = 0; i < h; ++i) o[i] += p * v[i];
    }
  }
}

void Inkling2::shortconv(const InkLayer&, const InkMeta& w, const float* dyn,
                         std::size_t offset, std::size_t n_channels,
                         float* x) {
  // w: F32 [l_cache, n_channels], ne0 fastest
  const float* wf = reinterpret_cast<const float*>(data_ + w.base);
  const std::size_t L = l_cache_;
  const std::size_t n_heads = n_kv_;
  const std::size_t group = n_channels / n_heads;
  float xbuf[64];
  const std::size_t lag_n = L - 1;

  // состояние принадлежит слоту; сюда передаём текущий через thread-local
  // невозможно — используем член через параметр offset и текущий slot.
  // slot выбирается вызывающим: state лежит в slots_[cur_slot_].
  auto& st = slots_[cur_slot_].conv;
  for (std::size_t ch = 0; ch < n_channels; ++ch) {
    float* s = st.data() + offset * lag_n + ch * lag_n;
    for (std::size_t l = 0; l < lag_n; ++l) xbuf[l] = s[l];
    const float cur = x[ch];
    xbuf[lag_n] = cur;

    float acc = 0.0F;
    for (std::size_t l = 0; l < L; ++l) acc += wf[ch * L + l] * xbuf[l];
    if (dyn != nullptr) {
      const std::size_t head = ch / group;
      for (std::size_t lag = 0; lag < L; ++lag) {
        acc += dyn[lag + L * head] * xbuf[lag_n - lag];
      }
    }
    const float scale =
        1.0F / std::sqrt(static_cast<float>((dyn ? 2 : 1) * L + 1));
    x[ch] = (cur + acc) * scale;
    for (std::size_t l = 0; l + 1 < L; ++l) s[l] = xbuf[l + 1];
  }
}

void Inkling2::control(float* x, const float* msg, const InkLayer& L,
                       int ctrl) {
  const std::size_t h = hidden_;
  const bool dbg = std::getenv("TOKSHIT_INK_CTRL") != nullptr;
  std::memcpy(z_.data(), x, h * sizeof(float));
  std::memcpy(z_.data() + h, msg, h * sizeof(float));
  rmsnorm(std::span<float>(z_.data(), h), {}, 1e-6F);
  rmsnorm(std::span<float>(z_.data() + h, h), {}, 1e-6F);

  matvec(L.loop_down[ctrl], z_.data(), ctrl_z_.data());
  matvec(L.loop_gain[ctrl], ctrl_z_.data(), ctrl_g_.data());
  matvec(L.loop_forget[ctrl], ctrl_z_.data(), ctrl_f_.data());
  float xm = 0.0F, mm = 0.0F, gm = 0.0F, fm = 0.0F;
  if (dbg) {
    for (std::size_t i = 0; i < h; ++i) {
      xm = std::max(xm, std::fabs(x[i]));
      mm = std::max(mm, std::fabs(msg[i]));
      gm = std::max(gm, std::fabs(ctrl_g_[i]));
      fm = std::max(fm, std::fabs(ctrl_f_[i]));
    }
  }
  for (std::size_t i = 0; i < h; ++i) {
    const float g = std::tanh(ctrl_g_[i]);
    const float f = std::tanh(ctrl_f_[i]);
    x[i] = x[i] - f * x[i] + g * msg[i];
  }
  if (dbg) {
    float om = 0.0F;
    for (std::size_t i = 0; i < h; ++i) om = std::max(om, std::fabs(x[i]));
    std::fprintf(stderr,
                 "[ctrl] ctrl=%d |x|=%.2f |msg|=%.2f maxg=%.2f maxf=%.2f "
                 "|out|=%.2f\n",
                 ctrl, xm, mm, gm, fm, om);
  }
}

void Inkling2::attention(const InkLayer& L, std::size_t slot, std::size_t pos,
                         bool swa, float, const float* in_q,
                         const float* in_k, const float* in_v, float* out) {
  const std::size_t hd = head_dim_;
  const std::size_t nh = n_head_;
  const std::size_t nkv = n_kv_;
  const std::size_t group = nh / nkv;
  const std::size_t kvd = nkv * hd;
  auto& st = slots_[slot];

  // relative bias: table[e][head] = sum_i rel_bias[e][i] * R[i][head]
  const std::size_t nr = n_relative_;
  const std::size_t extent = relative_extent_;
  for (std::size_t e = 0; e <= extent; ++e) {
    const float* row = (e < extent) ? (L.rel_bias.data() + e * nr) : nullptr;
    for (std::size_t hh = 0; hh < nh; ++hh) {
      float acc = 0.0F;
      if (row != nullptr) {
        for (std::size_t i = 0; i < nr; ++i) {
          acc += row[i] * r_[i + nr * hh];
        }
      }
      table_[e * nh + hh] = acc;
    }
  }

  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  const bool mlx_attn = std::getenv("TOKSHIT_INK_LLAMA_ATTN") == nullptr;
  const std::size_t p0 = swa && pos + 1 > n_swa_ ? pos + 1 - n_swa_ : 0;

  for (std::size_t hh = 0; hh < nh; ++hh) {
    const float* q = in_q + hh * hd;
    const std::size_t kh = hh / group;
    float mx = mlx_attn ? sink_[hh] : 0.0F;
    std::size_t n = 0;
    for (std::size_t p = p0; p <= pos; ++p) {
      const std::size_t d = pos - p < extent ? pos - p : extent;
      float dot = 0.0F;
      const uint16_t* krow = st.k.data() + p * kvd + kh * hd;
      for (std::size_t i = 0; i < hd; ++i) {
        dot += q[i] * f16_to_f32(krow[i]);
      }
      const float s = mlx_attn
                          ? dot * scale + table_[d * nh + hh]
                          : (dot + table_[d * nh + hh] - sink_[hh]) * scale;
      scores_[n++] = s;
      if (s > mx) mx = s;
    }
    float denom = std::exp(mlx_attn ? sink_[hh] - mx : -mx);
    for (std::size_t i = 0; i < n; ++i) {
      scores_[i] = std::exp(scores_[i] - mx);
      denom += scores_[i];
    }
    const float inv = 1.0F / denom;
    float* o = out + hh * hd;
    std::fill(o, o + hd, 0.0F);
    for (std::size_t i = 0; i < n; ++i) {
      const float p = scores_[i] * inv;
      const uint16_t* vrow = st.v.data() + (p0 + i) * kvd + kh * hd;
      for (std::size_t d = 0; d < hd; ++d) o[d] += p * f16_to_f32(vrow[d]);
    }
  }
  (void)in_k;
  (void)in_v;
}

void Inkling2::moe(const InkLayer& L, const float* x, float* out) {
  const std::size_t h = hidden_;
  const std::size_t nff = n_ff_exp_;
  matvec(L.gate_inp, x, router_.data());

  // sigmoid + bias, top-k
  for (std::size_t e = 0; e < n_expert_; ++e) {
    router_[e] = sigmoidf(router_[e]) + L.exp_probs_b[e];
  }
  std::fill(top_.begin(), top_.end(), -1);
  for (std::size_t k = 0; k < n_used_; ++k) {
    int best = -1;
    float bv = -1e30F;
    for (std::size_t e = 0; e < n_expert_; ++e) {
      const bool used = std::find(top_.begin(), top_.begin() + k, static_cast<int>(e)) !=
                        top_.begin() + k;
      if (used) continue;
      if (router_[e] > bv) {
        bv = router_[e];
        best = static_cast<int>(e);
      }
    }
    top_[k] = best;
  }
  // веса из несмещённых sigmoid-вероятностей
  float sum = 0.0F;
  for (std::size_t k = 0; k < n_used_; ++k) {
    float p = 0.0F;
    if (top_[k] >= 0) {
      const std::size_t e = static_cast<std::size_t>(top_[k]);
      p = router_[e] - L.exp_probs_b[e];
      if (p < 0.0F) p = 0.0F;
    }
    wts_[k] = p;
    sum += p;
  }
  const float inv = 1.0F / (sum + 1e-20F);
  for (std::size_t k = 0; k < n_used_; ++k) {
    wts_[k] *= inv;
    if (expert_scale_ != 0.0F) wts_[k] *= expert_scale_;
  }

  std::fill(out, out + h, 0.0F);
  const std::size_t gu_size = static_cast<std::size_t>(L.gate_up_exps.rb) *
                              L.gate_up_exps.rows;
  const std::size_t dn_size =
      static_cast<std::size_t>(L.down_exps.rb) * L.down_exps.rows;
  kernels::quantize_into(std::span<const float>(x, h), act_q8_);

  {
    ExpertJob job;
    job.data = data_ + L.gate_up_exps.base;
    job.stride = gu_size;
    job.rows = L.gate_up_exps.rows;
    job.rb = L.gate_up_exps.rb;
    job.in = L.gate_up_exps.in_dim;
    job.out_stride = static_cast<std::uint32_t>(2 * nff);
    job.chunks = (job.rows + job.chunk_rows - 1) / job.chunk_rows;
    job.quant = L.gate_up_exps.quant;
    job.act = &act_q8_;
    job.ids = top_.data();
    job.x = x;
    job.x_stride = 0;
    job.out = gu_all_.data();
    const std::size_t tasks = static_cast<std::size_t>(n_used_) * job.chunks;
    pool::run_tasks(tasks, expert_task, &job);
  }

  for (std::size_t k = 0; k < n_used_; ++k) {
    const std::size_t e = static_cast<std::size_t>(top_[k]);
    const float* gu = gu_all_.data() + k * 2 * nff;
    const float gs = L.gate_up_scale.empty() ? 1.0F : L.gate_up_scale[e];
    float* dst = act_all_.data() + k * nff;
    for (std::size_t i = 0; i < nff; ++i) {
      const float gi = gu[i] * gs;
      const float ui = gu[nff + i] * gs;
      float act = situ_beta_ * std::tanh(gi / situ_beta_) * sigmoidf(gi);
      act *= situ_linear_beta_ > 0.0F
                 ? situ_linear_beta_ * std::tanh(ui / situ_linear_beta_)
                 : ui;
      dst[i] = act;
    }
    kernels::quantize_into(std::span<const float>(dst, nff), act_all_q8_[k]);
  }

  {
    ExpertJob job;
    job.data = data_ + L.down_exps.base;
    job.stride = dn_size;
    job.rows = L.down_exps.rows;
    job.rb = L.down_exps.rb;
    job.in = L.down_exps.in_dim;
    job.out_stride = static_cast<std::uint32_t>(h);
    job.chunks = (job.rows + job.chunk_rows - 1) / job.chunk_rows;
    job.quant = L.down_exps.quant;
    job.act_block = act_all_q8_.data();
    job.ids = top_.data();
    job.x = act_all_.data();
    job.x_stride = static_cast<std::uint32_t>(nff);
    job.out = down_all_.data();
    const std::size_t tasks = static_cast<std::size_t>(n_used_) * job.chunks;
    pool::run_tasks(tasks, expert_task, &job);
  }

  for (std::size_t k = 0; k < n_used_; ++k) {
    const std::size_t e = static_cast<std::size_t>(top_[k]);
    const float w = wts_[k] * (L.down_scale.empty() ? 1.0F : L.down_scale[e]);
    const float* src = down_all_.data() + k * h;
    for (std::size_t i = 0; i < h; ++i) out[i] += w * src[i];
  }
}

void Inkling2::embed(uint32_t token) {
  const std::size_t rb = tok_embd_.rb;
  const uint8_t* row = data_ + tok_embd_.base + rb * token;
  const auto deq = dequant_into(tok_embd_.quant,
                                std::span<const uint8_t>(row, rb), hidden_,
                                std::span<float>(x_.data(), hidden_));
  if (!deq.ok()) std::fill(x_.begin(), x_.end(), 0.0F);
  if (std::getenv("TOKSHIT_INK_NORMS") != nullptr) {
    float mx = 0.0F;
    for (float v : x_) mx = std::max(mx, std::fabs(v));
    std::fprintf(stderr, "[embd] token=%u max=%.3f\n", token, mx);
  }
  rmsnorm(std::span<float>(x_.data(), hidden_),
          std::span<const float>(tok_norm_.data(), tok_norm_.size()), eps_);
}

void Inkling2::token_logits(uint32_t token, std::size_t pos,
                            std::vector<float>& out) {
  if (pos == 0) reset();
  if (pos >= kv_ctx_ || token >= vocab_unpadded_) {
    std::fill(out.begin(), out.end(), -1e30F);
    return;
  }
  out.assign(vocab_unpadded_, 0.0F);
  cur_slot_ = 0;
  embed(token);
  if (pos == 0) dump_vec("embd", x_);

  const std::size_t h = hidden_;
  const std::size_t kvd = n_kv_ * head_dim_;
  const std::size_t core = core_end_ - core_start_;

  std::vector<std::vector<float>> blocks;
  std::vector<float> prelude;

  auto run_app = [&](std::size_t slot, std::size_t ip, int pass,
                     const float* msg_kv, const float* msg_res) {
    const InkLayer& L = phys(slot);
    const bool swa = slot_swa_[slot] != 0;
    cur_slot_ = slot;
    std::memcpy(inp_sa_.data(), x_.data(), h * sizeof(float));

    const bool has_msg = pass > 0 && msg_kv != nullptr && msg_res != nullptr;
    float* inputs[3] = {inp3_.data(), inp3_.data() + h, inp3_.data() + 2 * h};

    if (!L.attn_res_query.empty()) {
      const std::size_t nq = L.q_inp_norm.empty() ? 1 : 3;
      std::vector<const float*> st;
      st.reserve(blocks.size());
      for (const auto& b : blocks) st.push_back(b.data());
      depth_mix(std::span<const float* const>(st.data(), st.size()),
                x_.data(), L.attn_res_query.data(), nq, mix_.data());
      for (std::size_t j = 0; j < 3; ++j) {
        inputs[j] = mix_.data() + (nq == 1 ? 0 : j * h);
      }
    } else {
      for (int j = 0; j < 3; ++j) inputs[j] = x_.data();
    }

    // каждый вход — отдельный буфер: control() меняет его на месте, а для
    // SWA-слоёв все три указателя могут смотреть на один и тот же mixed[...,0]
    for (int j = 0; j < 3; ++j) {
      std::memcpy(inp3_.data() + j * h, inputs[j], h * sizeof(float));
      inputs[j] = inp3_.data() + j * h;
    }

    if (has_msg) {
      for (int j = 0; j < 3; ++j) control(inputs[j], msg_kv + j * h, L, j);
    }

    if (!L.attn_norm.empty() && swa) {
      for (int j = 0; j < 3; ++j) {
        std::memcpy(nrm3_.data() + j * h,
                    has_msg ? inputs[j] : inputs[0], h * sizeof(float));
        rmsnorm(std::span<float>(nrm3_.data() + j * h, h),
                std::span<const float>(L.attn_norm.data(),
                                       L.attn_norm.size()), eps_);
      }
    } else if (!L.attn_norm.empty()) {
      std::memcpy(nrm3_.data(), inputs[0], h * sizeof(float));
      rmsnorm(std::span<float>(nrm3_.data(), h),
              std::span<const float>(L.attn_norm.data(), L.attn_norm.size()),
              eps_);
      for (int j = 1; j < 3; ++j) std::memcpy(nrm3_.data() + j * h,
                                              nrm3_.data(), h * sizeof(float));
    } else {
      for (int j = 0; j < 3; ++j) {
        std::memcpy(nrm3_.data() + j * h, inputs[j], h * sizeof(float));
        const auto& w = j == 0 ? L.q_inp_norm : (j == 1 ? L.k_inp_norm : L.v_inp_norm);
        rmsnorm(std::span<float>(nrm3_.data() + j * h, h),
                std::span<const float>(w.data(), w.size()), eps_);
      }
    }

    if (std::getenv("TOKSHIT_INK_LOOPDBG") != nullptr && pass == 1 &&
        ip == 6 && pos == 8) {
      dump_vec("p1_l6_nrm", nrm3_);
    }
    const float* in0 = nrm3_.data();
    const float* in1 = nrm3_.data() + h;
    const float* in2 = nrm3_.data() + 2 * h;

    kernels::quantize_into(std::span<const float>(in0, h), act_in_[0]);
    kernels::quantize_into(std::span<const float>(in1, h), act_in_[1]);
    kernels::quantize_into(std::span<const float>(in2, h), act_in_[2]);
    {
      const MatvecSeg segs[] = {
          {&L.q, in0, q_.data(), &act_in_[0]},
          {&L.k, in1, k_.data(), &act_in_[1]},
          {&L.v, in2, v_.data(), &act_in_[2]},
          {&L.rel, in0, r_.data(), &act_in_[0]},
          {&L.gate, in0, gate_.data(), &act_in_[0]},
          {&L.sink_proj, in0, sink_.data(), &act_in_[0]},
      };
      matvec_multi(std::span<const MatvecSeg>(segs));
    }
    if (ip == 0 && (pos == 0 || pos == 8) && std::getenv("TOKSHIT_INK_DUMP_DIR") != nullptr) {
      dump_vec("xin", std::vector<float>(nrm3_.begin(), nrm3_.begin() + h));
      dump_vec("q_raw", q_);
      dump_vec("k_raw", k_);
      dump_vec("v_raw", v_);
      dump_vec("r_raw", r_);
    }

    float* dyn_k = nullptr;
    float* dyn_v = nullptr;
    if (!L.k_conv_proj.base) {
      dyn_k = nullptr;
    } else {
      matvec(L.k_conv_proj, in1, dyn_k_.data());
      dyn_k = dyn_k_.data();
    }
    if (!L.v_conv_proj.base) {
      dyn_v = nullptr;
    } else {
      matvec(L.v_conv_proj, in2, dyn_v_.data());
      dyn_v = dyn_v_.data();
    }
    shortconv(L, L.k_conv, dyn_k, 0, kvd, k_.data());
    shortconv(L, L.v_conv, dyn_v, kvd, kvd, v_.data());

    // QK-relative clip
    const std::size_t hd = head_dim_;
    const float qk_thr = qk_clip_;
    if (qk_thr > 0.0F) {
      for (std::size_t hh = 0; hh < n_head_; ++hh) {
        float* qv = q_.data() + hh * hd;
        float ss = 0.0F;
        for (std::size_t i = 0; i < hd; ++i) {
          qv[i] *= L.q_clip[i + hd * hh];
          ss += qv[i] * qv[i];
        }
        const float f = std::min(1.0F, qk_thr / (std::sqrt(ss) + 1e-6F));
        for (std::size_t i = 0; i < hd; ++i) qv[i] *= f;
      }
      for (std::size_t hh = 0; hh < n_kv_; ++hh) {
        float* kv = k_.data() + hh * hd;
        float ss = 0.0F;
        for (std::size_t i = 0; i < hd; ++i) ss += kv[i] * kv[i];
        const float f = std::min(1.0F, qk_thr / (std::sqrt(ss) + 1e-6F));
        for (std::size_t i = 0; i < hd; ++i) kv[i] *= f;
      }
    }
    for (std::size_t hh = 0; hh < n_kv_; ++hh) {
      rmsnorm(std::span<float>(v_.data() + hh * hd, hd), {}, eps_);
    }
    if (rel_clip_ > 0.0F) {
      for (std::size_t hh = 0; hh < n_head_; ++hh) {
        float* rv = r_.data() + n_relative_ * hh;
        float ss = 0.0F;
        for (std::size_t i = 0; i < n_relative_; ++i) {
          rv[i] *= L.rel_clip[i + n_relative_ * hh];
          ss += rv[i] * rv[i];
        }
        const float f = std::min(1.0F, rel_clip_ / (std::sqrt(ss) + 1e-6F));
        for (std::size_t i = 0; i < n_relative_; ++i) rv[i] *= f;
      }
    }

    const float tau =
        (!swa && log_scale_floor_ != 0)
            ? 1.0F + log_scale_alpha_ *
                         std::log(std::max(1.0F,
                                           static_cast<float>(pos + 1) /
                                               static_cast<float>(
                                                   log_scale_floor_)))
            : 1.0F;
    if (!swa && tau != 1.0F) {
      for (float& qv : q_) qv *= tau;
      for (float& rv : r_) rv *= tau;
    }

    if (ip == 0 && (pos == 0 || pos == 8) && std::getenv("TOKSHIT_INK_DUMP_DIR") != nullptr) {
      dump_vec("q", q_);
      dump_vec("k", k_);
      dump_vec("v", v_);
      dump_vec("r", r_);
      dump_vec("sink", sink_);
    }

    if (std::getenv("TOKSHIT_INK_LOOPDBG") != nullptr && pass == 1 &&
        ip == 6 && pos == 8) {
      const auto& st = slots_[slot];
      std::vector<float> ck(st.conv.begin(), st.conv.begin() + 3 * 1024);
      dump_vec("p1_ck", ck);
      std::vector<float> keys;
      keys.reserve((pos + 1) * 1024);
      for (std::size_t p = 0; p <= pos; ++p) {
        for (std::size_t i = 0; i < 1024; ++i) {
          keys.push_back(f16_to_f32(st.k[p * kvd + i]));
        }
      }
      dump_vec("p1_keys", keys);
      std::vector<float> qq(q_);
      dump_vec("p1_q", qq);
      std::vector<float> kk(k_);
      dump_vec("p1_k", kk);
    }

    // KV-запись слота
    if (pos < kv_ctx_) {
      auto& st = slots_[slot];
      uint16_t* krow = st.k.data() + pos * kvd;
      uint16_t* vrow = st.v.data() + pos * kvd;
      for (std::size_t i = 0; i < kvd; ++i) {
        krow[i] = f32_to_f16(k_[i]);
        vrow[i] = f32_to_f16(v_[i]);
      }
      if (pos + 1 > st.high) st.high = pos + 1;
    }

    attention(L, slot, pos, swa, tau, q_.data(), k_.data(), v_.data(),
              attn_.data());

    // gated attention output
    for (std::size_t i = 0; i < attn_.size(); ++i) {
      attn_[i] *= sigmoidf(gate_[i]);
    }
    matvec(L.o, attn_.data(), o_.data());
    shortconv(L, L.post_conv, nullptr, 2 * kvd, h, o_.data());
    if (ip == 0 && (pos == 0 || pos == 8)) dump_vec(pos == 0 ? "o_conv" : "o_conv8", o_);

    if (ip % block_size_ == 0) {
      blocks.emplace_back(inp_sa_);
      std::memcpy(x_.data(), o_.data(), h * sizeof(float));
    } else {
      for (std::size_t i = 0; i < h; ++i) x_[i] = inp_sa_[i] + o_[i];
    }

    // FFN
    float* ffn_in = ffn_in_.data();
    std::memcpy(ffn_in, x_.data(), h * sizeof(float));
    if (ip == 9 && pass == 0 && pos == 8 &&
        std::getenv("TOKSHIT_INK_DUMP_DIR") != nullptr) {
      dump_vec("ffn_in9", ffn_in_);
      for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
        dump_vec(("fblk9_" + std::to_string(bi)).c_str(), blocks[bi]);
      }
    }
    {
      std::vector<const float*> st;
      st.reserve(blocks.size());
      for (const auto& b : blocks) st.push_back(b.data());
      depth_mix(std::span<const float* const>(st.data(), st.size()),
                x_.data(), L.ffn_res_query.data(), 1, mix2_.data());
      if (has_msg) control(mix2_.data(), msg_res + 3 * h, L, 3);
    }
    rmsnorm(std::span<float>(mix2_.data(), h),
            std::span<const float>(L.ffn_norm.data(), L.ffn_norm.size()), eps_);
    static const bool noffn = std::getenv("TOKSHIT_INK_NOFFN") != nullptr;
    if (noffn) {
      std::fill(ffn_.begin(), ffn_.end(), 0.0F);
    } else {
      const auto tm0 = std::chrono::steady_clock::now();
      moe(L, mix2_.data(), ffn_.data());
      if (std::getenv("TOKSHIT_INK_PROF") != nullptr) {
        prof_moe_ns_ += std::chrono::duration<double, std::nano>(
                            std::chrono::steady_clock::now() - tm0)
                            .count();
      }
    }
    if ((ip == 0 && (pos == 0 || pos == 8)) ||
        (ip == 9 && pass == 0 && pos == 8)) {
      dump_vec(ip == 9 ? "moe9" : (pos == 0 ? "moe_raw" : "moe_raw8"), ffn_);
      dump_vec(ip == 9 ? "mix9" : (pos == 0 ? "mix_ffn" : "mix_ffn8"), mix2_);
      std::vector<float> dbg(64);
      for (std::size_t k = 0; k < n_used_; ++k) {
        dbg[k] = wts_[k];
        dbg[32 + k] = static_cast<float>(top_[k]);
      }
      dump_vec(ip == 9 ? "router9" : (pos == 0 ? "router" : "router8"), dbg);
    }
    if (!noffn) {
      shortconv(L, L.ffn_conv, nullptr, 2 * kvd + h, h, ffn_.data());
    }
    for (std::size_t i = 0; i < h; ++i) x_[i] = ffn_in[i] + ffn_[i];
    if (std::getenv("TOKSHIT_INK_NORMS") != nullptr) {
      float mx = 0.0F;
      for (float v : x_) mx = std::max(mx, std::fabs(v));
      std::fprintf(stderr, "[app] ip=%zu phys=%zu pass=%d max=%.3f\n", ip,
                   slot_phys_[slot], pass, mx);
    }
    if (ip == 0 && pass == 0 && (pos == 0 || pos == 8)) dump_vec(pos == 0 ? "x0" : "x8", x_);
    if (pos == 8 && std::getenv("TOKSHIT_INK_DUMP_ALL") != nullptr) {
      const std::string nm = "app" + std::to_string(ip) + "_p" + std::to_string(pass);
      dump_vec(nm.c_str(), x_);
    }
  };

  // прелюдия
  for (std::size_t slot = 0; slot < core_start_; ++slot) {
    run_app(slot, slot, 0, nullptr, nullptr);
  }
  prelude = x_;
  const auto prelude_blocks = blocks;

  std::vector<std::vector<float>> bank;
  std::vector<float> prev_out;
  for (std::size_t p = 0; p < n_passes_; ++p) {
    if (p > 0) {
      x_ = prelude;
      blocks = prelude_blocks;
    }
    for (std::size_t ip = core_start_; ip < core_end_; ++ip) {
      const std::size_t slot = core_start_ + p * core + (ip - core_start_);
      const InkLayer& L = phys(slot);
      const float* mkv = nullptr;
      const float* mres = nullptr;
      static const bool noloop = std::getenv("TOKSHIT_INK_NOLOOP") != nullptr;
      if (p > 0 && !bank.empty() && !noloop) {
        std::vector<const float*> st;
        st.reserve(bank.size());
        for (const auto& b : bank) st.push_back(b.data());
        for (int c = 0; c < 4; ++c) {
          depth_mix(std::span<const float* const>(st.data(), st.size()),
                    prev_out.data(), L.msg_query[c].data(), 1,
                    msg_.data() + c * h);
        }
        mkv = msg_.data();
        mres = msg_.data();
        if (std::getenv("TOKSHIT_INK_LOOPDBG") != nullptr && p == 1 &&
            ip == 6 && pos == 8) {
          if (!bank.empty()) dump_vec("bank0", bank[0]);
          dump_vec("prev_out", prev_out);
          std::fprintf(stderr, "[loop] bank=%zu prev_out=%.4f\n", bank.size(),
                       std::sqrt(prev_out.data()[0] * 0.0 +
                                 [&] {
                                   double s = 0;
                                   for (float v : prev_out) s += v * v;
                                   return static_cast<float>(s);
                                 }()));
          for (int c = 0; c < 4; ++c) {
            double s = 0;
            for (std::size_t i = 0; i < h; ++i) {
              s += msg_[c * h + i] * msg_[c * h + i];
            }
            std::fprintf(stderr,
                         "[loop] msg[%d] norm=%.4f v0=%.5f v1=%.5f\n", c,
                         std::sqrt(s), msg_[c * h], msg_[c * h + 1]);
          }
        }
      }
      run_app(slot, ip, static_cast<int>(p), mkv, mres);
    }
    if (p + 1 < n_passes_) {
      // банк: старт второго core-блока и выход прохода
      bank.clear();
      for (std::size_t i = prelude_blocks.size() + 1; i < blocks.size(); ++i) {
        bank.push_back(blocks[i]);
      }
      prev_out = x_;
    }
  }

  // эпилог
  for (std::size_t ip = core_end_; ip < n_phys_; ++ip) {
    const std::size_t slot = ip - core_end_ + core_start_ + core * n_passes_;
    run_app(slot, ip, 0, nullptr, nullptr);
  }

  // финальный mix и голова
  {
    std::vector<const float*> st;
    st.reserve(blocks.size());
    for (const auto& b : blocks) st.push_back(b.data());
    depth_mix(std::span<const float* const>(st.data(), st.size()), x_.data(),
              out_res_query_.data(), 1, mix2_.data());
    if (std::getenv("TOKSHIT_INK_OUT") != nullptr) {
      std::fprintf(stderr, "[out] blocks=%zu\n", blocks.size());
      // веса выходного depth-mix
      const std::size_t nstates = blocks.size() + 1;
      float norm[16] = {0};
      float sc[16] = {0};
      const float inv = 1.0F / std::sqrt(static_cast<float>(h));
      const float* sts[16] = {nullptr};
      sts[0] = x_.data();
      for (std::size_t i = 0; i < blocks.size(); ++i) sts[i + 1] = blocks[i].data();
      float mxw = 0.0F;
      for (std::size_t sidx = 0; sidx < nstates; ++sidx) {
        float ss = 0.0F;
        for (std::size_t i = 0; i < h; ++i) ss += sts[sidx][i] * sts[sidx][i];
        float rms = 1.0F / std::sqrt(ss / static_cast<float>(h) + 1e-6F);
        norm[sidx] = rms;
        float dot = 0.0F;
        for (std::size_t i = 0; i < h; ++i) {
          dot += sts[sidx][i] * rms * out_res_query_[i];
        }
        sc[sidx] = dot * inv;
        mxw = std::max(mxw, sc[sidx]);
      }
      float sum = 0.0F;
      for (std::size_t sidx = 0; sidx < nstates; ++sidx) {
        sc[sidx] = std::exp(sc[sidx] - mxw);
        sum += sc[sidx];
      }
      std::fprintf(stderr, "[out] mix_w:");
      for (std::size_t sidx = 0; sidx < nstates; ++sidx) {
        std::fprintf(stderr, " %.3f", sc[sidx] / sum);
      }
      std::fprintf(stderr, "\n");
      float mn = 1e30F, mx = -1e30F;
      for (float v : mix2_) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      }
      std::fprintf(stderr, "[out] mixed min=%.2f max=%.2f\n", mn, mx);
    }
  }
  rmsnorm(std::span<float>(mix2_.data(), h),
          std::span<const float>(out_norm_.data(), out_norm_.size()), eps_);
  if (pos == 8 && std::getenv("TOKSHIT_INK_DUMP_DIR") != nullptr) {
    dump_vec("out_hidden", mix2_);
    dump_vec("x_final", x_);
    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
      dump_vec(("oblk" + std::to_string(bi)).c_str(), blocks[bi]);
    }
  }
  matvec(out_, mix2_.data(), logits_.data());
  if (std::getenv("TOKSHIT_INK_OUT") != nullptr) {
    float hn = 0.0F;
    for (float v : mix2_) hn += v * v;
    std::fprintf(stderr,
                 "[out] hidden_norm=%.3f logit(think)=%.3f logit(is)=%.3f "
                 "logit(Paris)=%.3f\n",
                 std::sqrt(hn), logits_[200008], logits_[149884],
                 logits_[152717]);
  }
  if (std::getenv("TOKSHIT_INK_LOGITS") != nullptr) {
    float lo = 1e30F, hi = -1e30F;
    double sum = 0.0;
    for (std::size_t i = 0; i < vocab_padded_; ++i) {
      lo = std::min(lo, logits_[i]);
      hi = std::max(hi, logits_[i]);
      sum += std::fabs(logits_[i]);
    }
    std::fprintf(stderr, "[logits] token=%u pos=%zu pre min=%.3f max=%.3f mean|x|=%.3f\n", token, pos, lo, hi, sum / static_cast<double>(vocab_padded_));
  }
  if (final_softcap_ > 0.0F) {
    for (std::size_t i = 0; i < vocab_padded_; ++i) {
      logits_[i] =
          final_softcap_ * std::tanh(logits_[i] / final_softcap_);
    }
  }
  if (std::getenv("TOKSHIT_INK_PROF") != nullptr) {
    std::fprintf(stderr, "[prof] token=%u pos=%zu mv=%.0fms moe=%.0fms\n",
                 token, pos, prof_mv_ns_ / 1e6, prof_moe_ns_ / 1e6);
    prof_mv_ns_ = 0.0;
    prof_moe_ns_ = 0.0;
  }
  out.resize(vocab_unpadded_);
  std::copy(logits_.begin(), logits_.begin() + vocab_unpadded_, out.begin());
  if (std::getenv("TOKSHIT_INK_LOGITS") != nullptr) {
    float hi = out[0];
    for (float v : out) hi = std::max(hi, v);
    std::fprintf(stderr,
                 "[copy] pos=%zu out_max=%.3f src=%p dst=%p same=%d\n", pos, hi,
                 static_cast<const void*>(logits_.data()),
                 static_cast<const void*>(out.data()),
                 logits_.data() == out.data() ? 1 : 0);
  }
}

}  // namespace tokshit
