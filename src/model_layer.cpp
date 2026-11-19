#include "tokshit/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "tokshit/metrics.hpp"
#include "tokshit/pool.hpp"

#include "model_internal.hpp"

namespace tokshit {

void Model::layer(std::size_t layer, Workspace& ws, KVCache& kv,
                  std::span<const std::pair<float, float>> rope_table) {
  // за один слой вперёд: пока считаем attention текущего слоя, читатели уже
  // тянут предсказанных экспертов следующего. два слоя вперёд тоже пробуем,
  // если они не попали в кэш и очередь не забита
  // TOKSHIT_PREFETCH_IDLE: только следующий слой — спекуляция живёт на
  // окнах простоя диска, читать на два слоя вперёд ей нечем
  // one layer ahead: while we do this layer's attention, the readers already
  // pull the next layer's predicted experts. two layers ahead too, if they
  // are not cached and the queue is not full. TOKSHIT_PREFETCH_IDLE: next
  // layer only — speculation lives on idle windows of the drive
  if (prefetch_enabled() && io_ && layer + 1 >= n_dense_) {
    static const bool idle_pf = prefetch_idle();
    const std::size_t ahead_max = idle_pf ? 1 : 2;
    for (std::size_t ahead = 1; ahead <= ahead_max; ++ahead) {
      const std::size_t nl = layer + ahead;
      if (nl >= n_layers_) break;
      prefetch_experts(nl, freq_->last[nl]);
    }
  }
  const auto t_attn = Clock::now();
  const Layer& lt = layers_[layer];
  const std::size_t kv_dim = n_kv_ * head_dim_;
  // один Seg на матрицу: fused-вызовы ниже собирают несколько независимых
  // matvec в один пул-барьер, чтобы не платить per-op overhead дважды
  // one Seg per matrix: fused calls below put independent matvecs behind a
  // single pool barrier instead of paying the per-op overhead twice
  auto make_seg = [this](const Meta& m, const float* x,
                         const kernels::ActQ8& act, float* y) {
    return Seg{meta_bytes(m), m.rb, m.quant, m.in_dim, x, &act, y, m.out_dim};
  };

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
  const std::size_t start =
      (swa && n_hist > window_) ? n_hist - window_ : 0;
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

  const auto t_ffn = Clock::now();
  // ffn: общая норма и дальше развилка плотный/moe
  // ffn: shared norm and then the dense/moe split
  std::copy(ws.x.begin(), ws.x.end(), ws.hf.begin());
  rmsnorm(ws.hf, lt.ffn_norm, eps_);

  if (layer < n_dense_) {
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
  } else {
    const auto t_router = Clock::now();
    // роутер: считаем логиты по экспертам, берём top-k, нормируем веса
    // router: score the experts, take top-k, normalize the weights
    std::fill(ws.logits.begin(), ws.logits.end(), 0.0F);
    std::fill(ws.g.begin(), ws.g.end(), 0.0F);
    std::fill(ws.u.begin(), ws.u.end(), 0.0F);
    std::span<float> router(ws.logits.data(), n_experts_);
    kernels::quantize_into(ws.hf, ws.act);
    {
      // router и gate/up shared expert читают один и тот же hf и ws.act:
      // три независимых matvec уходят в один барьер
      // router and the shared expert's gate/up read the same hf and ws.act:
      // three independent matvecs go behind one barrier
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
            static_cast<uint64_t>(
                lt.gate_inp->out_dim * lt.gate_inp->rb +
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

    // Предсказание следующего слоя по текущему роутингу. В top-1/top-2
    // задержка чтения — главная статья, а соседние слои часто выбирают тот же
    // эксперт; заказываем его до того, как дойдём до следующего слоя.
    // Predict the next layer from the current routing. In top-1/top-2 read
    // latency dominates, and neighbouring layers often pick the same expert;
    // order it before the next layer starts.
    if (prefetch_enabled() && io_ != nullptr && resident_budget_ != 0) {
      for (std::size_t ahead = 1; ahead <= 2; ++ahead) {
        const std::size_t nl = layer + ahead;
        if (nl >= n_layers_) break;
        prefetch_experts(nl, ws.top);
      }
    }

    // PREFILL BULK: sequential pread на весь слой, потом take_expert возьмёт hit
    static const bool bulk_mode = false;  // disabled: reads 252MB to use 8MB
    if (bulk_mode && prefilling_ && io_ && resident_budget_ != 0) {
      prefill_bulk_layer(layer);
    }
    tick(metrics().t_router, t_router);

    static const bool layer_dbg = std::getenv("TOKSHIT_LAYER_DEBUG") != nullptr;
    const auto t_expert = Clock::now();
    const uint64_t dbg_miss0 = layer_dbg ? metrics().c_expert_miss.load() : 0;
    const uint64_t dbg_hit0 = layer_dbg ? metrics().c_expert_hit.load() : 0;
    const uint64_t dbg_dem0 = layer_dbg ? metrics().c_dem_dup.load() : 0;
    // грузим выбранных экспертов (параллельно), потом gate/up одним батчем,
    // swiglu, и down. это самый дорогой кусок на реальной модели
    // load the chosen experts (parallel), then gate/up as one batch, swiglu,
    // then down. this is the expensive part on a real model
    const auto t_load = Clock::now();
    Clock::time_point dbg_load_end{}, dbg_gu_end{}, dbg_dw_end{}, dbg_d_end{};
    // в draft-режиме берём ровно выбранное количество, не конфиг модели:
    // ниже не хватает десятков мест, все живут от этой локальной
    // in draft mode use exactly what the router picked, not the model config:
    // everything downstream lives from this one local
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

    // заказываем сразу все чтения слоя. очередь спроса fifo, поэтому первые
    // матвеки не стоят за последними байтами
    // order all of the layer's reads at once. the demand queue is fifo, so
    // the first matmuls never wait behind the last bytes
    // TOKSHIT_IO_MERGE: соседи по proj лежат в файле подряд (эксперт e и
    // e+1 — смежные байты), выбранных экспертов сортируем по индексу и
    // цепочки уезжают одним pread каждый: меньше сисколлов, ближе к
    // sequential. попадания считаем той же бухгалтерией, что и раньше
    // TOKSHIT_IO_MERGE: neighbours within one proj are contiguous bytes
    // (expert e and e+1), so we sort the chosen experts by index and let
    // chains go out as one pread each: fewer syscalls, closer to sequential.
    // hits are accounted exactly like before
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

      // ждём gate/up всех экспертов. батч из 16 сегментов параллелится
      // лучше, чем 8 запусков по 2, и это перевешивает выигрыш от раннего
      // старта: замерено, батч быстрее
      // wait for every expert's gate/up. one 16-segment batch parallelises
      // better than 8 runs of 2, and that beats starting earlier: measured,
      // the batch wins
      for (std::size_t j = 0; j < n_used; ++j) {
        ws.wg[j] = take_expert(layer, 0, em_g, ws.top[j]);
        ws.wu[j] = take_expert(layer, 1, em_u, ws.top[j]);
      }
      tick(metrics().t_load, t_load);
      dbg_load_end = Clock::now();

      ws.segs.clear();
      for (std::size_t j = 0; j < n_used; ++j) {
        const std::size_t rb = std::min(
            em_g.rb, ws.wg[j].len / (ffn_dim == 0 ? 1 : ffn_dim));
        ws.segs.push_back(
            Seg{std::span<const uint8_t>(ws.wg[j].data, ws.wg[j].len), rb,
                em_g.quant, em_g.in_dim, ws.hf.data(), &ws.act,
                ws.gu.data() + j * ffn_dim, ffn_dim});
      }
      for (std::size_t j = 0; j < n_used; ++j) {
        const std::size_t rb = std::min(
            em_u.rb, ws.wu[j].len / (ffn_dim == 0 ? 1 : ffn_dim));
        ws.segs.push_back(
            Seg{std::span<const uint8_t>(ws.wu[j].data, ws.wu[j].len), rb,
                em_u.quant, em_u.in_dim, ws.hf.data(), &ws.act,
                ws.gu.data() + block + j * ffn_dim, ffn_dim});
      }
      const auto t_mm2 = Clock::now();
      run_segs(ws.segs, 32);
      tick(metrics().t_expert_mm, t_mm2);
      dbg_gu_end = Clock::now();
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

      ws.segs.clear();
      for (std::size_t j = 0; j < n_used; ++j) {
        const std::size_t rb = std::min(
            em_g.rb, ws.wg[j].len / (ffn_dim == 0 ? 1 : ffn_dim));
        ws.segs.push_back(
            Seg{std::span<const uint8_t>(ws.wg[j].data, ws.wg[j].len), rb,
                em_g.quant, em_g.in_dim, ws.hf.data(), &ws.act,
                ws.gu.data() + j * ffn_dim, ffn_dim});
      }
      for (std::size_t j = 0; j < n_used; ++j) {
        const std::size_t rb = std::min(
            em_u.rb, ws.wu[j].len / (ffn_dim == 0 ? 1 : ffn_dim));
        ws.segs.push_back(
            Seg{std::span<const uint8_t>(ws.wu[j].data, ws.wu[j].len), rb,
                em_u.quant, em_u.in_dim, ws.hf.data(), &ws.act,
                ws.gu.data() + block + j * ffn_dim, ffn_dim});
      }
      const auto t_mm2 = Clock::now();
      run_segs(ws.segs, 32);
      tick(metrics().t_expert_mm, t_mm2);
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
    // down ждём только сейчас: всё время пока считались gate/up и swiglu,
    // оно уже читалось в фоне
    // down is awaited only now: while gate/up and swiglu were running it was
    // already being read in the background
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
      const std::size_t rb = std::min(
          em_d.rb, ws.wd[j].len / (hidden_ == 0 ? 1 : hidden_));
      ws.down_segs.push_back(Seg{
          std::span<const uint8_t>(ws.wd[j].data, ws.wd[j].len), rb, em_d.quant,
          em_d.in_dim, g_all.data() + j * ffn_dim, &ws.expert_acts[j],
          ws.dd.data() + j * hidden_, hidden_});
    }
    {
      const auto t_mm3 = Clock::now();
      run_segs(ws.down_segs, 32);
      tick(metrics().t_expert_mm, t_mm3);
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
    // shared expert, обычный плотный mlp, добавляем к moe выходу.
    // gate/up уже посчитаны fused-вызовом вместе с router выше
    // shared expert, a plain dense mlp, added to the moe output.
    // gate/up were already computed in the fused router call above
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
  tick(metrics().t_ffn, t_ffn);
}

// выделяем все буферы один раз. дальше токен их только переиспользует.
// размеры берём из конфига модели
// allocate every buffer once. a token just reuses them after that. sizes come
}  // namespace tokshit
