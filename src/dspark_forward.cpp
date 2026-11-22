#include "tokshit/dspark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include "tokshit/batch.hpp"
#include "tokshit/dequant.hpp"

namespace tokshit {
namespace {

inline float sigmoidf(float x) { return 1.0F / (1.0F + std::exp(-x)); }

inline float siluf(float x) { return x * sigmoidf(x); }

}  // namespace

void DSpark::rmsnorm(std::span<float> x, std::span<const float> w) const {
  float ss = 0.0F;
  for (float v : x) ss += v * v;
  const float scale =
      1.0F / std::sqrt(ss / static_cast<float>(x.size()) + eps_);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] *= scale;
    if (i < w.size()) x[i] *= w[i];
  }
}

void DSpark::rmsnorm(std::span<float> x) const {
  float ss = 0.0F;
  for (float v : x) ss += v * v;
  const float scale =
      1.0F / std::sqrt(ss / static_cast<float>(x.size()) + eps_);
  for (float& v : x) v *= scale;
}

void DSpark::matvec(const InkMeta& m, const float* x, float* y) {
  kernels::ActQ8 act;
  kernels::quantize_into(std::span<const float>(x, m.in_dim), act);
  const uint8_t* base = bytes(m);
  kernels::dot_rows(
      m.quant,
      std::span<const uint8_t>(base,
                               static_cast<std::size_t>(m.rows) * m.rb),
      m.rb, m.rows, std::span<const float>(x, m.in_dim), act, m.in_dim, y,
      scratch_);
}

void DSpark::matvec_f32(const std::vector<float>& w, std::size_t rows,
                        std::size_t cols, const float* x, float* y) const {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = w.data() + r * cols;
    float acc = 0.0F;
    for (std::size_t i = 0; i < cols; ++i) acc += row[i] * x[i];
    y[r] = acc;
  }
}

void DSpark::rope(std::span<float> x, std::size_t n_heads,
                  std::size_t pos) const {
  const std::size_t d = head_dim_;
  for (std::size_t h = 0; h < n_heads; ++h) {
    float* v = x.data() + h * d;
    for (std::size_t i = 0; i < d / 2; ++i) {
      const float freq =
          std::pow(rope_base_, -2.0F * static_cast<float>(i) /
                                   static_cast<float>(d));
      const float angle = static_cast<float>(pos) * freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const float x0 = v[2 * i];
      const float x1 = v[2 * i + 1];
      v[2 * i] = x0 * c - x1 * s;
      v[2 * i + 1] = x0 * s + x1 * c;
    }
  }
}

void DSpark::fuse(std::size_t row, std::span<const float> taps, float* out,
                  std::span<const float> norm) const {
  const float* wr =
      reinterpret_cast<const float*>(bytes(fuse_) + row * fuse_.rb);
  float w[4];
  float mx = wr[0];
  for (int i = 1; i < 4; ++i) mx = std::max(mx, wr[i]);
  float sum = 0.0F;
  for (int i = 0; i < 4; ++i) {
    w[i] = std::exp(wr[i] - mx);
    sum += w[i];
  }
  const float inv = 1.0F / sum;
  for (std::size_t i = 0; i < hidden_; ++i) out[i] = 0.0F;
  for (int k = 0; k < 4; ++k) {
    const float wk = w[k] * inv;
    const float* t = taps.data() + k * hidden_;
    for (std::size_t i = 0; i < hidden_; ++i) out[i] += wk * t[i];
  }
  rmsnorm(std::span<float>(out, hidden_), norm);
}

void DSpark::inject(std::size_t pos, std::span<const float> taps,
                    bool store_ctx) {
  if (pos >= kv_ctx_ || taps.size() < 4 * hidden_) return;
  const std::size_t kvd = n_kv_ * head_dim_;
  for (std::size_t l = 0; l < n_layer_; ++l) {
    Layer& L = layers_[l];
    fuse(l, taps, fuse_ctx_.data(), L.dflare_norm);
    matvec(L.k_ctx, fuse_ctx_.data(), kctx_.data());
    matvec(L.v_ctx, fuse_ctx_.data(), vctx_.data());
    for (std::size_t h = 0; h < n_kv_; ++h) {
      rmsnorm(std::span<float>(kctx_.data() + h * head_dim_, head_dim_),
              L.k_ctx_norm);
    }
    rope(kctx_, n_kv_, pos);
    uint16_t* kd = kv_[l].k.data() + pos * kvd;
    uint16_t* vd = kv_[l].v.data() + pos * kvd;
    for (std::size_t i = 0; i < kvd; ++i) {
      kd[i] = f32_to_f16(kctx_[i]);
      vd[i] = f32_to_f16(vctx_[i]);
    }
  }
  if (pos + 1 > injected_high_) injected_high_ = pos + 1;
  if (!store_ctx) return;
  fuse(3, taps, fuse_ctx_.data(), dspark_norm_);
  const std::size_t slot = pos % ctx_slots_;
  std::memcpy(ctx_ring_.data() + slot * hidden_, fuse_ctx_.data(),
              hidden_ * sizeof(float));
  ctx_pos_[slot] = static_cast<int64_t>(pos);
}

const float* DSpark::context(std::size_t pos) const {
  const std::size_t slot = pos % ctx_slots_;
  if (ctx_pos_[slot] != static_cast<int64_t>(pos)) return nullptr;
  return ctx_ring_.data() + slot * hidden_;
}

void DSpark::attention(std::size_t layer, std::size_t q_index,
                       std::size_t pos0, const float* q, float* out) {
  const LayerKv& kv = kv_[layer];
  const std::size_t hd = head_dim_;
  const std::size_t kvd = n_kv_ * hd;
  const std::size_t group = n_head_ / n_kv_;
  const std::size_t query_pos = pos0 + q_index;
  const std::size_t p0 =
      (swa_layer_[layer] && query_pos + 1 > swa_) ? pos0 + 1 - swa_ : 0;
  std::size_t p1 = pos0 + block_size_ - 1;
  if (p1 >= kv_ctx_) p1 = kv_ctx_ - 1;
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));

  for (std::size_t h = 0; h < n_head_; ++h) {
    const float* qh = q + h * hd;
    const std::size_t kh = h / group;
    float mx = -std::numeric_limits<float>::infinity();
    std::size_t n = 0;
    for (std::size_t p = p0; p <= p1; ++p) {
      if (p < pos0 && p >= injected_high_) continue;
      const uint16_t* krow = kv.k.data() + p * kvd + kh * hd;
      float dot = 0.0F;
      for (std::size_t i = 0; i < hd; ++i) {
        dot += qh[i] * f16_to_f32(krow[i]);
      }
      const float s = dot * scale;
      scores_[n++] = s;
      if (s > mx) mx = s;
    }
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
      scores_[i] = std::exp(scores_[i] - mx);
      sum += scores_[i];
    }
    const float inv = 1.0F / sum;
    float* o = out + h * hd;
    std::fill(o, o + hd, 0.0F);
    std::size_t at = 0;
    for (std::size_t p = p0; p <= p1; ++p) {
      if (p < pos0 && p >= injected_high_) continue;
      const float w = scores_[at++] * inv;
      const uint16_t* vrow = kv.v.data() + p * kvd + kh * hd;
      for (std::size_t i = 0; i < hd; ++i) {
        o[i] += w * f16_to_f32(vrow[i]);
      }
    }
  }
}

void DSpark::head_step(const float* hidden, const float* context, uint32_t prev,
                       std::vector<float>& state, std::vector<float>& logits,
                       bool have_base, const float* base) {
  const std::size_t rank = rank_;
  const std::size_t h = hidden_;
  const std::size_t joint_in = 2 * rank + 2 * h;
  const std::size_t joint_out = 3 * rank;

  const uint8_t* w1 = bytes(markov_w1_);
  const uint8_t* row = w1 + static_cast<std::size_t>(prev) * markov_w1_.rb;
  const auto deq = dequant_into(markov_w1_.quant,
                                std::span<const uint8_t>(row, markov_w1_.rb),
                                rank, std::span<float>(row_.data(), rank));
  if (!deq.ok()) std::fill(row_.begin(), row_.end(), 0.0F);

  std::memcpy(joint_buf_.data(), state.data(), rank * sizeof(float));
  std::memcpy(joint_buf_.data() + rank, row_.data(), rank * sizeof(float));
  std::memcpy(joint_buf_.data() + 2 * rank, hidden, h * sizeof(float));
  std::memcpy(joint_buf_.data() + 2 * rank + h, context, h * sizeof(float));

  matvec_f32(joint_f32_, joint_out, joint_in, joint_buf_.data(),
             joint_out_.data());
  for (std::size_t i = 0; i < joint_out; ++i) {
    joint_out_[i] += joint_bias_[i];
  }

  float* gate = joint_out_.data();
  float* cand = joint_out_.data() + rank;
  float* markov = joint_out_.data() + 2 * rank;
  for (std::size_t i = 0; i < rank; ++i) {
    const float g = sigmoidf(gate[i]);
    const float c = std::tanh(cand[i]);
    state[i] = c + g * (state[i] - c);
    markov[i] = std::tanh(markov[i]);
  }

  matvec_f32(hout_f32_, h, rank, markov, corr_.data());
  for (std::size_t i = 0; i < h; ++i) corr_[i] += hidden_out_bias_[i];

  // Пер-шаг bias: lm_head @ corr. base[j] = lm_head @ hidden_j уже посчитан
  // одним BatchSeg-проходом в draft(); тут — только сумма + argmax.
  bias_.resize(logits.size());
  target_->lm_head(corr_.data(), bias_.data());
  if (have_base) {
    for (std::size_t i = 0; i < vocab_padded_; ++i) {
      logits[i] = bias_[i] + base[i];
    }
  } else {
    std::copy(bias_.begin(), bias_.end(), logits.begin());
  }
  for (std::size_t i = 0; i < vocab_unpadded_; ++i) {
    logits[i] = softcap_ * std::tanh(logits[i] / softcap_);
  }
}

void DSpark::draft(uint32_t anchor, std::size_t pos0, std::size_t n,
                   std::vector<uint32_t>& out) {
  out.clear();
  n = std::min(n, max_draft());
  if (n == 0 || pos0 == 0 || pos0 >= kv_ctx_) return;
  const float* ctx = context(pos0 - 1);
  if (ctx == nullptr) return;

  const std::size_t B = block_size_;
  const std::size_t h = hidden_;
  const std::size_t qdim = n_head_ * head_dim_;
  const std::size_t kvd = n_kv_ * head_dim_;
  const std::size_t ffn = ffn_;

  const bool prof = std::getenv("TOKSHIT_DSPARK_PROF") != nullptr;
  const auto tp0 = std::chrono::steady_clock::now();

  target_->embed_normed(anchor, x_.data());
  for (std::size_t b = 1; b < B; ++b) {
    std::memcpy(x_.data() + b * h, mask_embd_.data(), h * sizeof(float));
  }

  for (std::size_t l = 0; l < n_layer_; ++l) {
    Layer& L = layers_[l];

    // attention input norm
    for (std::size_t b = 0; b < B; ++b) {
      std::memcpy(ffn_in_.data() + b * h, x_.data() + b * h,
                  h * sizeof(float));
      rmsnorm(std::span<float>(ffn_in_.data() + b * h, h), L.attn_norm);
      kernels::quantize_into(
          std::span<const float>(ffn_in_.data() + b * h, h), acts_[b]);
    }
    {
      const BatchSeg segs[3] = {
          {std::span<const uint8_t>(bytes(L.q),
                                    static_cast<std::size_t>(L.q.rows) * L.q.rb),
           L.q.rb, L.q.quant, L.q.in_dim, ffn_in_.data(), acts_.data(), B,
           q_.data(), qdim},
          {std::span<const uint8_t>(bytes(L.k),
                                    static_cast<std::size_t>(L.k.rows) * L.k.rb),
           L.k.rb, L.k.quant, L.k.in_dim, ffn_in_.data(), acts_.data(), B,
           k_.data(), kvd},
          {std::span<const uint8_t>(bytes(L.v),
                                    static_cast<std::size_t>(L.v.rows) * L.v.rb),
           L.v.rb, L.v.quant, L.v.in_dim, ffn_in_.data(), acts_.data(), B,
           v_.data(), kvd},
      };
      batchlib::run_bsegs(std::span<const BatchSeg>(segs, 3), 16);
    }

    for (std::size_t b = 0; b < B; ++b) {
      const std::size_t pos = pos0 + b;
      for (std::size_t hh = 0; hh < n_head_; ++hh) {
        rmsnorm(std::span<float>(q_.data() + b * qdim + hh * head_dim_,
                                 head_dim_),
                L.q_norm);
      }
      for (std::size_t hh = 0; hh < n_kv_; ++hh) {
        rmsnorm(std::span<float>(k_.data() + b * kvd + hh * head_dim_,
                                 head_dim_),
                L.k_norm);
      }
      rope(std::span<float>(q_.data() + b * qdim, qdim), n_head_, pos);
      rope(std::span<float>(k_.data() + b * kvd, kvd), n_kv_, pos);
      if (pos < kv_ctx_) {
        uint16_t* kd = kv_[l].k.data() + pos * kvd;
        uint16_t* vd = kv_[l].v.data() + pos * kvd;
        for (std::size_t i = 0; i < kvd; ++i) {
          kd[i] = f32_to_f16(k_[b * kvd + i]);
          vd[i] = f32_to_f16(v_[b * kvd + i]);
        }
      }
    }

    for (std::size_t b = 0; b < B; ++b) {
      attention(l, b, pos0, q_.data() + b * qdim, attn_.data() + b * qdim);
      kernels::quantize_into(
          std::span<const float>(attn_.data() + b * qdim, qdim), acts_[b]);
    }
    {
      const BatchSeg seg{
          std::span<const uint8_t>(bytes(L.o),
                                   static_cast<std::size_t>(L.o.rows) * L.o.rb),
          L.o.rb, L.o.quant, L.o.in_dim, attn_.data(), acts_.data(), B,
          out_.data(), h};
      batchlib::run_bsegs(std::span<const BatchSeg>(&seg, 1), 16);
    }
    for (std::size_t i = 0; i < B * h; ++i) x_[i] += out_[i];

    // FFN
    for (std::size_t b = 0; b < B; ++b) {
      std::memcpy(ffn_in_.data() + b * h, x_.data() + b * h,
                  h * sizeof(float));
      rmsnorm(std::span<float>(ffn_in_.data() + b * h, h), L.ffn_norm);
      kernels::quantize_into(
          std::span<const float>(ffn_in_.data() + b * h, h), acts_[b]);
    }
    {
      const BatchSeg segs[2] = {
          {std::span<const uint8_t>(
               bytes(L.gate), static_cast<std::size_t>(L.gate.rows) * L.gate.rb),
           L.gate.rb, L.gate.quant, L.gate.in_dim, ffn_in_.data(), acts_.data(),
           B, gate_.data(), ffn},
          {std::span<const uint8_t>(
               bytes(L.up), static_cast<std::size_t>(L.up.rows) * L.up.rb),
           L.up.rb, L.up.quant, L.up.in_dim, ffn_in_.data(), acts_.data(), B,
           up_.data(), ffn},
      };
      batchlib::run_bsegs(std::span<const BatchSeg>(segs, 2), 16);
    }
    for (std::size_t b = 0; b < B; ++b) {
      for (std::size_t i = 0; i < ffn; ++i) {
        act_[b * ffn + i] = siluf(gate_[b * ffn + i]) * up_[b * ffn + i];
      }
      kernels::quantize_into(
          std::span<const float>(act_.data() + b * ffn, ffn), acts_[b]);
    }
    {
      const BatchSeg seg{
          std::span<const uint8_t>(bytes(L.down),
                                   static_cast<std::size_t>(L.down.rows) *
                                       L.down.rb),
          L.down.rb, L.down.quant, L.down.in_dim, act_.data(), acts_.data(), B,
          out_.data(), h};
      batchlib::run_bsegs(std::span<const BatchSeg>(&seg, 1), 32);
    }
    for (std::size_t i = 0; i < B * h; ++i) x_[i] += out_[i];
  }

  for (std::size_t b = 0; b < B; ++b) {
    std::memcpy(out_.data() + b * h, x_.data() + b * h, h * sizeof(float));
    rmsnorm(std::span<float>(out_.data() + b * h, h), output_norm_);
  }

  const auto tp1 = std::chrono::steady_clock::now();
  target_->lm_head_batch(out_.data() + h, n, h, base_.data());
  const auto tp2 = std::chrono::steady_clock::now();

  std::fill(state_.begin(), state_.end(), 0.0F);
  uint32_t prev = anchor;
  for (std::size_t i = 0; i < n; ++i) {
    head_step(out_.data() + (i + 1) * h, ctx, prev, state_, logits_, true,
              base_.data() + i * vocab_padded_);
    std::size_t best = 0;
    for (std::size_t j = 1; j < vocab_unpadded_; ++j) {
      if (logits_[j] > logits_[best]) best = j;
    }
    out.push_back(static_cast<uint32_t>(best));
    prev = static_cast<uint32_t>(best);
  }
  if (prof) {
    const auto tp3 = std::chrono::steady_clock::now();
    const auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
                 "[dspark prof] block=%.0fms base=%.0fms head=%.0fms total=%.0fms\n",
                 ms(tp0, tp1), ms(tp1, tp2), ms(tp2, tp3), ms(tp0, tp3));
  }
}

}  // namespace tokshit
