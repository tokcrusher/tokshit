#include "tokshit/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <utility>
#include <vector>

#include "tokshit/metrics.hpp"

#include "model_internal.hpp"

namespace tokshit {

Model::Seg Model::make_seg(const Meta& meta, const float* x,
                           const kernels::ActQ8& act, float* y) const {
  return Seg{meta_bytes(meta), meta.rb, meta.quant, meta.in_dim, x, &act, y,
             meta.out_dim};
}

void Model::layer(std::size_t layer, Workspace& ws, KVCache& kv,
                  std::span<const std::pair<float, float>> rope_table) {
  // Пока считается attention текущего слоя, читатели тянут предсказанных
  // экспертов следующего. В idle-режиме смотрим только на один слой вперёд.
  if (prefetch_enabled() && io_ && layer + 1 >= n_dense_) {
    static const bool idle_pf = prefetch_idle();
    const std::size_t ahead_max = idle_pf ? 1 : 2;
    for (std::size_t ahead = 1; ahead <= ahead_max; ++ahead) {
      const std::size_t nl = layer + ahead;
      if (nl >= n_layers_) break;
      prefetch_experts(nl, freq_->last[nl]);
    }
  }

  layer_attention(layer, ws, kv, rope_table);

  const auto t_ffn = Clock::now();
  const Layer& lt = layers_[layer];
  std::copy(ws.x.begin(), ws.x.end(), ws.hf.begin());
  rmsnorm(ws.hf, lt.ffn_norm, eps_);
  if (layer < n_dense_) {
    layer_dense(layer, ws);
  } else {
    layer_moe(layer, ws);
  }
  tick(metrics().t_ffn, t_ffn);
}

void Model::layer_attention(
    std::size_t layer, Workspace& ws, KVCache& kv,
    std::span<const std::pair<float, float>> rope_table) {
  const auto t_attn = Clock::now();
  const Layer& lt = layers_[layer];
  const std::size_t kv_dim = n_kv_ * head_dim_;

  std::copy(ws.x.begin(), ws.x.end(), ws.hn.begin());
  rmsnorm(ws.hn, lt.attn_norm, eps_);

  std::fill(ws.q.begin(), ws.q.end(), 0.0F);
  std::fill(ws.k.begin(), ws.k.end(), 0.0F);
  std::fill(ws.v.begin(), ws.v.end(), 0.0F);
  const auto t_mm = Clock::now();
  {
    kernels::quantize_into(ws.hn, ws.act);
    const Seg segs[3] = {
        {meta_bytes(lt.attn_q), lt.attn_q.rb, lt.attn_q.quant, lt.attn_q.in_dim,
         ws.hn.data(), &ws.act, ws.q.data(), lt.attn_q.out_dim},
        {meta_bytes(lt.attn_k), lt.attn_k.rb, lt.attn_k.quant, lt.attn_k.in_dim,
         ws.hn.data(), &ws.act, ws.k.data(), lt.attn_k.out_dim},
        {meta_bytes(lt.attn_v), lt.attn_v.rb, lt.attn_v.quant, lt.attn_v.in_dim,
         ws.hn.data(), &ws.act, ws.v.data(), lt.attn_v.out_dim},
    };
    const uint64_t bytes =
        static_cast<uint64_t>(lt.attn_q.out_dim * lt.attn_q.rb +
                              lt.attn_k.out_dim * lt.attn_k.rb +
                              lt.attn_v.out_dim * lt.attn_v.rb);
    run_segs(std::span<const Seg>(segs, 3), 16);
    if (timing_enabled()) {
      metrics().t_mm.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                   t_mm)
                  .count()),
          std::memory_order_relaxed);
      metrics().c_mm_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
  }
  tick(metrics().t_attn_mm, t_mm);

  for (std::size_t hd = 0; hd < n_heads_; ++hd) {
    rmsnorm(std::span<float>(ws.q.data() + hd * head_dim_, head_dim_),
            lt.attn_q_norm, eps_);
  }
  for (std::size_t hd = 0; hd < n_kv_; ++hd) {
    rmsnorm(std::span<float>(ws.k.data() + hd * head_dim_, head_dim_),
            lt.attn_k_norm, eps_);
  }
  if (rope_on_[layer]) {
    for (std::size_t hd = 0; hd < n_heads_; ++hd) {
      rope_apply(std::span<float>(ws.q.data() + hd * head_dim_, head_dim_),
                 rope_table);
    }
    for (std::size_t hd = 0; hd < n_kv_; ++hd) {
      rope_apply(std::span<float>(ws.k.data() + hd * head_dim_, head_dim_),
                 rope_table);
    }
  }
  kv[layer].insert(kv[layer].end(), ws.k.begin(), ws.k.end());
  kv[layer].insert(kv[layer].end(), ws.v.begin(), ws.v.end());

  const auto t_core = Clock::now();
  const std::size_t n_hist = kv[layer].size() / (2 * kv_dim);
  const bool swa = swa_[layer];
  const std::size_t start = (swa && n_hist > window_) ? n_hist - window_ : 0;
  const std::size_t n_use = (swa && n_hist > window_) ? window_ : n_hist;

  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
  const std::size_t group = n_heads_ / n_kv_;
  std::fill(ws.attn.begin(), ws.attn.end(), 0.0F);
  for (std::size_t hd = 0; hd < n_heads_; ++hd) {
    const std::size_t kv_h = hd / group;
    const std::size_t qs = hd * head_dim_;
    ws.scores.clear();
    for (std::size_t p = start; p < start + n_use; ++p) {
      const std::size_t off = p * 2 * kv_dim + kv_h * head_dim_;
      float dot = 0.0F;
      for (std::size_t i = 0; i < head_dim_; ++i) {
        dot += ws.q[qs + i] * kv[layer][off + i];
      }
      ws.scores.push_back(dot * scale);
    }
    ws.scores.push_back(lt.attn_sinks[hd]);
    softmax(ws.scores);
    ws.scores.pop_back();
    for (std::size_t i = 0; i < head_dim_; ++i) {
      float acc = 0.0F;
      for (std::size_t p = 0; p < n_use; ++p) {
        const std::size_t off =
            (start + p) * 2 * kv_dim + kv_dim + kv_h * head_dim_;
        acc += ws.scores[p] * kv[layer][off + i];
      }
      ws.attn[qs + i] = acc;
    }
  }
  tick(metrics().t_attn_core, t_core);

  std::fill(ws.o.begin(), ws.o.end(), 0.0F);
  const auto t_o = Clock::now();
  kernels::quantize_into(ws.attn, ws.act);
  matvec_preact(lt.attn_o, ws.attn, ws.o, ws.act);
  tick(metrics().t_attn_mm, t_o);
  for (std::size_t i = 0; i < hidden_; ++i) ws.x[i] += ws.o[i];

  tick(metrics().t_attn, t_attn);
}

void Model::layer_dense(std::size_t layer, Workspace& ws) {
  const Layer& lt = layers_[layer];
  std::fill(ws.g.begin(), ws.g.end(), 0.0F);
  std::fill(ws.u.begin(), ws.u.end(), 0.0F);
  std::fill(ws.dd.begin(), ws.dd.end(), 0.0F);
  kernels::quantize_into(ws.hf, ws.act);
  {
    const auto t_gu = Clock::now();
    const Seg gu[2] = {
        make_seg(*lt.ffn_gate, ws.hf.data(), ws.act, ws.g.data()),
        make_seg(*lt.ffn_up, ws.hf.data(), ws.act, ws.u.data()),
    };
    run_segs(std::span<const Seg>(gu, 2), 32);
    if (timing_enabled()) {
      tick(metrics().t_mm, t_gu);
      metrics().c_mm_bytes.fetch_add(
          static_cast<uint64_t>(lt.ffn_gate->out_dim * lt.ffn_gate->rb +
                                lt.ffn_up->out_dim * lt.ffn_up->rb),
          std::memory_order_relaxed);
    }
  }
  swiglu(std::span<float>(ws.g.data(), dense_ffn_),
         std::span<const float>(ws.u.data(), dense_ffn_));
  kernels::quantize_into(std::span<float>(ws.g.data(), dense_ffn_), ws.act);
  matvec_preact(*lt.ffn_down, std::span<float>(ws.g.data(), dense_ffn_),
                std::span<float>(ws.dd.data(), hidden_), ws.act);
  for (std::size_t i = 0; i < hidden_; ++i) ws.x[i] += ws.dd[i];
}

}  // namespace tokshit
