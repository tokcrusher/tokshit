#include "tokshit/model.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "tokshit/metrics.hpp"
#include "tokshit/pool.hpp"

#include "model_internal.hpp"
#include "model_trace.hpp"

namespace tokshit {

// TOKSHIT_EXPERT_LOCALITY=1: accumulate the per-layer router top-k sets on
// every layer_moe() invocation and print pass-to-pass Jaccard overlap in
// expert_loc_report().  For the Model path an invocation == one pass over the
// layer; for Inkling2 the same signal would live in Inkling2::moe (see
// report note).
namespace {

struct LocAcc {
  std::mutex mu;
  // per layer: list of top-sets, one per invocation ("pass")
  std::vector<std::vector<std::set<std::size_t>>> passes;
};

LocAcc& loc_acc() {
  static LocAcc a;
  return a;
}

bool loc_enabled() {
  static const bool e = std::getenv("TOKSHIT_EXPERT_LOCALITY") != nullptr;
  return e;
}

double jaccard(const std::set<std::size_t>& a,
               const std::set<std::size_t>& b) {
  if (a.empty() && b.empty()) return 1.0;
  if (a.empty() || b.empty()) return 0.0;
  const std::size_t inter =
      static_cast<std::size_t>(std::count_if(a.begin(), a.end(), [&](auto e) {
        return b.count(e) != 0;
      }));
  return static_cast<double>(inter) /
         static_cast<double>(a.size() + b.size() - inter);
}

}  // namespace

void expert_loc_report() {
  LocAcc& a = loc_acc();
  std::lock_guard<std::mutex> lock(a.mu);
  double total = 0.0;
  std::size_t pairs = 0;
  for (std::size_t l = 0; l < a.passes.size(); ++l) {
    const auto& v = a.passes[l];
    if (v.size() < 2) continue;
    double j01 = 0.0, j12 = 0.0;
    const bool has01 = v.size() >= 2;
    const bool has12 = v.size() >= 3;
    if (has01) j01 = jaccard(v[0], v[1]);
    if (has12) j12 = jaccard(v[1], v[2]);
    const double mean = has01 ? j01 : 0.0;
    total += mean;
    ++pairs;
    std::fprintf(stderr, "[loc] layer %2zu passes=%zu j01=%.3f j12=%.3f mean=%.3f\n",
                 l, v.size(), has01 ? j01 : 0.0, has12 ? j12 : 0.0, mean);
  }
  if (pairs != 0) {
    std::fprintf(stderr, "[loc] mean overlap over %zu layers: %.3f\n", pairs,
                 total / static_cast<double>(pairs));
  }
}

}  // namespace tokshit

namespace tokshit {

void Model::layer_moe(std::size_t layer, Workspace& ws) {
  const Layer& lt = layers_[layer];
  const auto t_router = Clock::now();
  std::fill(ws.logits.begin(), ws.logits.end(), 0.0F);
  std::fill(ws.g.begin(), ws.g.end(), 0.0F);
  std::fill(ws.u.begin(), ws.u.end(), 0.0F);
  std::span<float> router(ws.logits.data(), n_experts_);
  kernels::quantize_into(ws.hf, ws.act);
  {
    // Router и shared gate/up читают один hf, поэтому идут одним барьером.
    const Seg pre[3] = {
        make_seg(*lt.gate_inp, ws.hf.data(), ws.act, router.data()),
        make_seg(*lt.gate_shexp, ws.hf.data(), ws.act, ws.g.data()),
        make_seg(*lt.up_shexp, ws.hf.data(), ws.act, ws.u.data()),
    };
    const auto t_pre = Clock::now();
    run_segs(std::span<const Seg>(pre, 3), 32);
    if (timing_enabled()) {
      tick(metrics().t_mm, t_pre);
      metrics().c_mm_bytes.fetch_add(
          static_cast<uint64_t>(lt.gate_inp->out_dim * lt.gate_inp->rb +
                                lt.gate_shexp->out_dim * lt.gate_shexp->rb +
                                lt.up_shexp->out_dim * lt.up_shexp->rb),
          std::memory_order_relaxed);
    }
  }

  const std::size_t k_used =
      router_topk_ != 0 && router_topk_ < n_used_ ? router_topk_ : n_used_;
  ws.idx.resize(n_experts_);
  for (std::size_t i = 0; i < n_experts_; ++i) ws.idx[i] = i;
  if (k_used < n_experts_) {
    std::nth_element(ws.idx.begin(),
                     ws.idx.begin() + static_cast<std::ptrdiff_t>(k_used),
                     ws.idx.end(), [&](std::size_t a, std::size_t b) {
                       return router[a] > router[b];
                     });
  }
  ws.top.assign(ws.idx.begin(),
                ws.idx.begin() + static_cast<std::ptrdiff_t>(k_used));

  ws.wts.resize(ws.top.size());
  for (std::size_t i = 0; i < ws.top.size(); ++i) {
    ws.wts[i] = 1.0F / (1.0F + std::exp(-router[ws.top[i]]));
  }
  float wsum = 0.0F;
  for (float w : ws.wts) wsum += w;
  for (float& w : ws.wts) w /= wsum;
  record_freq(layer, ws.top);
  trace_experts(layer, ws.top);
  if (loc_enabled()) {
    LocAcc& a = loc_acc();
    std::lock_guard<std::mutex> lock(a.mu);
    if (a.passes.size() <= layer) a.passes.resize(layer + 1);
    std::set<std::size_t> s(ws.top.begin(), ws.top.end());
    auto& v = a.passes[layer];
    if (v.size() < 64) v.push_back(std::move(s));
  }

  if (prefetch_enabled() && io_ != nullptr && resident_budget_ != 0) {
    for (std::size_t ahead = 1; ahead <= 2; ++ahead) {
      const std::size_t nl = layer + ahead;
      if (nl >= n_layers_) break;
      prefetch_experts(nl, ws.top);
    }
  }

  // Bulk-префилл выключен: он читает весь регион слоя ради нескольких
  // экспертов и на замерах проигрывает union-пути. Оставлен как ручной
  // эксперимент.
  static const bool bulk_mode = false;
  if (bulk_mode && prefilling_ && io_ && resident_budget_ != 0) {
    prefill_bulk_layer(layer);
  }
  tick(metrics().t_router, t_router);

  static const bool layer_dbg = std::getenv("TOKSHIT_LAYER_DEBUG") != nullptr;
  const auto t_expert = Clock::now();
  const uint64_t dbg_miss0 = layer_dbg ? metrics().c_expert_miss.load() : 0;
  const uint64_t dbg_hit0 = layer_dbg ? metrics().c_expert_hit.load() : 0;
  const uint64_t dbg_dem0 = layer_dbg ? metrics().c_dem_dup.load() : 0;
  const auto t_load = Clock::now();
  Clock::time_point dbg_load_end{}, dbg_gu_end{}, dbg_dw_end{}, dbg_d_end{};

  // В draft-режиме берём ровно выбранное число экспертов, не конфиг модели.
  const std::size_t n_used = k_used;
  const std::size_t ffn_dim = ffn_dim_;
  metrics().c_expert_union.fetch_add(n_used, std::memory_order_relaxed);
  metrics().c_expert_passes.fetch_add(1, std::memory_order_relaxed);
  const ExpertMeta em_g = *lt.gate_exps;
  const ExpertMeta em_u = *lt.up_exps;
  const ExpertMeta em_d = *lt.down_exps;
  static std::atomic<bool> expert_logged{false};
  if (!expert_logged.exchange(true)) {
    std::fprintf(
        stderr,
        "[expert] n_experts=%zu used=%zu ffn=%zu | gate in=%zu rb=%zu "
        "size=%zu q=%s | up in=%zu rb=%zu size=%zu q=%s | down in=%zu "
        "rb=%zu size=%zu q=%s\n",
        n_experts_, n_used, ffn_dim, em_g.in_dim, em_g.rb, em_g.size,
        std::string(quant_name(em_g.quant)).c_str(), em_u.in_dim, em_u.rb,
        em_u.size, std::string(quant_name(em_u.quant)).c_str(), em_d.in_dim,
        em_d.rb, em_d.size, std::string(quant_name(em_d.quant)).c_str());
  }

  ws.slices.resize(n_used * 3);
  ws.wg.resize(n_used);
  ws.wu.resize(n_used);
  ws.wd.resize(n_used);
  const std::size_t block = n_used * ffn_dim;
  std::fill(ws.gu.begin(), ws.gu.end(), 0.0F);
  kernels::quantize_into(ws.hf, ws.act);

  static const bool io_merge = io_merge_enabled();
  if (io_ != nullptr && resident_budget_ != 0) {
    const ExpertMeta* ems[3] = {&em_g, &em_u, &em_d};
    if (io_merge) {
      for (std::size_t pi = 0; pi < 3; ++pi) {
        std::vector<std::pair<uint64_t, uint64_t>> entries;
        entries.reserve(n_used);
        for (std::size_t j = 0; j < n_used; ++j) {
          const uint64_t key = (static_cast<uint64_t>(pi) << 32) |
                               static_cast<uint32_t>(ws.top[j]);
          if (io_->is_cached(layer, key)) {
            metrics().c_expert_hit.fetch_add(1, std::memory_order_relaxed);
          } else {
            metrics().c_expert_miss.fetch_add(1, std::memory_order_relaxed);
          }
          entries.emplace_back(key,
                               ems[pi]->base + ems[pi]->size * ws.top[j]);
        }
        io_->demand_merged(layer, entries, ems[pi]->size);
      }
    } else {
      for (std::size_t j = 0; j < n_used; ++j) {
        for (std::size_t pi = 0; pi < 2; ++pi) {
          const uint64_t key = (static_cast<uint64_t>(pi) << 32) |
                               static_cast<uint32_t>(ws.top[j]);
          const uint64_t off = ems[pi]->base + ems[pi]->size * ws.top[j];
          const bool cached = io_->demand_async(layer, key, off, ems[pi]->size);
          if (cached) {
            metrics().c_expert_hit.fetch_add(1, std::memory_order_relaxed);
          } else {
            metrics().c_expert_miss.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
      for (std::size_t j = 0; j < n_used; ++j) {
        const uint64_t key = (static_cast<uint64_t>(2) << 32) |
                             static_cast<uint32_t>(ws.top[j]);
        const uint64_t off = em_d.base + em_d.size * ws.top[j];
        const bool cached = io_->demand_async(layer, key, off, em_d.size);
        if (cached) {
          metrics().c_expert_hit.fetch_add(1, std::memory_order_relaxed);
        } else {
          metrics().c_expert_miss.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }

    // gate/up ждём все сразу: один батч параллелится лучше мелких запусков.
    for (std::size_t j = 0; j < n_used; ++j) {
      ws.wg[j] = take_expert(layer, 0, em_g, ws.top[j]);
      ws.wu[j] = take_expert(layer, 1, em_u, ws.top[j]);
    }
    tick(metrics().t_load, t_load);
    dbg_load_end = Clock::now();
  } else {
    LoadJob job;
    job.model = this;
    job.layer = layer;
    job.tops = ws.top.data();
    job.ems[0] = &em_g;
    job.ems[1] = &em_u;
    job.ems[2] = &em_d;
    job.n_used = n_used;
    job.out = ws.slices.data();
    pool::run_tasks(n_used * 3, &Model::load_task, &job);
    for (std::size_t j = 0; j < n_used; ++j) {
      ws.wg[j] = ws.slices[j];
      ws.wu[j] = ws.slices[n_used + j];
    }
    tick(metrics().t_load, t_load);
    dbg_load_end = Clock::now();
  }

  // Сегменты gate/up строятся одинаково для обоих путей загрузки.
  ws.segs.clear();
  for (std::size_t j = 0; j < n_used; ++j) {
    const std::size_t rb =
        std::min(em_g.rb, ws.wg[j].len / (ffn_dim == 0 ? 1 : ffn_dim));
    ws.segs.push_back(Seg{std::span<const uint8_t>(ws.wg[j].data, ws.wg[j].len),
                          rb, em_g.quant, em_g.in_dim, ws.hf.data(), &ws.act,
                          ws.gu.data() + j * ffn_dim, ffn_dim});
  }
  for (std::size_t j = 0; j < n_used; ++j) {
    const std::size_t rb =
        std::min(em_u.rb, ws.wu[j].len / (ffn_dim == 0 ? 1 : ffn_dim));
    ws.segs.push_back(Seg{std::span<const uint8_t>(ws.wu[j].data, ws.wu[j].len),
                          rb, em_u.quant, em_u.in_dim, ws.hf.data(), &ws.act,
                          ws.gu.data() + block + j * ffn_dim, ffn_dim});
  }
  {
    const auto t_mm = Clock::now();
    run_segs(ws.segs, 32);
    tick(metrics().t_expert_mm, t_mm);
    dbg_gu_end = Clock::now();
  }

  std::span<float> g_all(ws.gu.data(), block);
  std::span<float> u_all(ws.gu.data() + block, block);
  for (std::size_t j = 0; j < n_used; ++j) {
    const std::size_t s = j * ffn_dim;
    swiglu(g_all.subspan(s, ffn_dim), u_all.subspan(s, ffn_dim));
  }

  ws.expert_acts.resize(n_used);
  for (std::size_t j = 0; j < n_used; ++j) {
    kernels::quantize_into(g_all.subspan(j * ffn_dim, ffn_dim),
                           ws.expert_acts[j]);
  }
  std::fill(ws.dd.begin(), ws.dd.end(), 0.0F);
  if (io_ != nullptr && resident_budget_ != 0) {
    const auto t_dw = Clock::now();
    for (std::size_t j = 0; j < n_used; ++j) {
      ws.wd[j] = take_expert(layer, 2, em_d, ws.top[j]);
    }
    tick(metrics().t_down_wait, t_dw);
    dbg_dw_end = Clock::now();
  } else {
    for (std::size_t j = 0; j < n_used; ++j) {
      ws.wd[j] = ws.slices[2 * n_used + j];
    }
  }
  ws.down_segs.clear();
  for (std::size_t j = 0; j < n_used; ++j) {
    const std::size_t rb =
        std::min(em_d.rb, ws.wd[j].len / (hidden_ == 0 ? 1 : hidden_));
    ws.down_segs.push_back(
        Seg{std::span<const uint8_t>(ws.wd[j].data, ws.wd[j].len), rb,
            em_d.quant, em_d.in_dim, g_all.data() + j * ffn_dim,
            &ws.expert_acts[j], ws.dd.data() + j * hidden_, hidden_});
  }
  {
    const auto t_mm = Clock::now();
    run_segs(ws.down_segs, 32);
    tick(metrics().t_expert_mm, t_mm);
    dbg_d_end = Clock::now();
  }

  std::fill(ws.moe.begin(), ws.moe.end(), 0.0F);
  for (std::size_t j = 0; j < n_used; ++j) {
    const float w = ws.wts[j];
    const std::size_t base = j * hidden_;
    for (std::size_t i = 0; i < hidden_; ++i) ws.moe[i] += ws.dd[base + i] * w;
  }
  if (layer_dbg && !prefilling_) {
    const auto ms = [](Clock::time_point a, Clock::time_point b) {
      return a == Clock::time_point{}
                 ? 0.0
                 : std::chrono::duration<double>(b - a).count() * 1e3;
    };
    std::fprintf(
        stderr,
        "[layer %zu] load=%.2fms gu=%.2fms dw=%.2fms dmm=%.2fms "
        "miss=%llu hit=%llu dup=%llu q=%s\n",
        layer, ms(t_load, dbg_load_end), ms(dbg_load_end, dbg_gu_end),
        ms(dbg_gu_end, dbg_dw_end), ms(dbg_dw_end, dbg_d_end),
        static_cast<unsigned long long>(metrics().c_expert_miss.load() -
                                        dbg_miss0),
        static_cast<unsigned long long>(metrics().c_expert_hit.load() -
                                        dbg_hit0),
        static_cast<unsigned long long>(metrics().c_dem_dup.load() - dbg_dem0),
        std::string(quant_name(em_g.quant)).c_str());
  }
  tick(metrics().t_expert, t_expert);

  const auto t_shexp = Clock::now();
  // Shared expert: обычный плотный MLP, gate/up уже посчитаны с роутером.
  swiglu(std::span<float>(ws.g.data(), shared_dim_),
         std::span<const float>(ws.u.data(), shared_dim_));
  std::fill(ws.dd.begin(), ws.dd.end(), 0.0F);
  kernels::quantize_into(std::span<float>(ws.g.data(), shared_dim_), ws.act);
  matvec_preact(*lt.down_shexp, std::span<float>(ws.g.data(), shared_dim_),
                std::span<float>(ws.dd.data(), hidden_), ws.act);
  for (std::size_t i = 0; i < hidden_; ++i) ws.moe[i] += ws.dd[i];
  for (std::size_t i = 0; i < hidden_; ++i) ws.x[i] += ws.moe[i];
  tick(metrics().t_shexp, t_shexp);
}

}  // namespace tokshit
