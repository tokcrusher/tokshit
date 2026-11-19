// батч-путь модели: b токенов за один проход по весам. здесь живёт verify
// спекуляции и префилл длинных промптов. зеркалит Model::layer/token_logits,
// но каждая матрица весов читается один раз на b активаций
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "tokshit/attn_batch.hpp"
#include "tokshit/batch.hpp"
#include "tokshit/dequant.hpp"
#include "tokshit/kv.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/model.hpp"

namespace tokshit {

struct Model::BatchBuf {
  std::size_t b = 0;      // ёмкость строк / row capacity
  std::size_t maxm = 0;   // ёмкость пар эксперт-строка / expert-member pairs
  std::vector<float> x, hn, hf, o, moe, dd, g, u;
  std::vector<float> q, attn, k, v, router;
  std::vector<float> gu, uu, gx, ddm;
  std::vector<kernels::ActQ8> acts, acts2;
  std::vector<std::pair<float, float>> rope;
  std::vector<std::size_t> win;
  std::vector<std::vector<std::size_t>> top;
  std::vector<std::vector<float>> wts;
  std::vector<uint8_t> seen;
  std::vector<int> emap;
  std::vector<std::size_t> utop, moff, mcur, idx;
  // плоские пары эксперт-строка вместо vector<vector>: ноль аллокаций на слой
  std::vector<std::size_t> mem_j;
  std::vector<float> mem_w;
  std::vector<BatchSeg> segs;
  std::vector<WRef> wg, wu, wd;
};

// тут BatchBuf уже полный, поэтому особые члены модели можно сгенерировать
Model::Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
Model::~Model() = default;

void Model::ensure_batch_buf(std::size_t b) {
  if (!batch_buf_) batch_buf_ = std::make_unique<BatchBuf>();
  BatchBuf& bf = *batch_buf_;
  const std::size_t maxm = b * n_used_;
  if (bf.b >= b && bf.maxm >= maxm) return;
  bf.b = std::max(b, bf.b);
  bf.maxm = std::max(maxm, bf.maxm);
  const std::size_t B = bf.b, M = bf.maxm;
  bf.x.assign(B * hidden_, 0.0F);
  bf.hn.assign(B * hidden_, 0.0F);
  bf.hf.assign(B * hidden_, 0.0F);
  bf.o.assign(B * hidden_, 0.0F);
  bf.moe.assign(B * hidden_, 0.0F);
  bf.dd.assign(B * hidden_, 0.0F);
  const std::size_t ffn_cap = std::max(dense_ffn_, shared_dim_);
  bf.g.assign(B * ffn_cap, 0.0F);
  bf.u.assign(B * ffn_cap, 0.0F);
  bf.q.assign(B * n_heads_ * head_dim_, 0.0F);
  bf.attn.assign(B * n_heads_ * head_dim_, 0.0F);
  bf.k.assign(B * n_kv_ * head_dim_, 0.0F);
  bf.v.assign(B * n_kv_ * head_dim_, 0.0F);
  bf.router.assign(B * n_experts_, 0.0F);
  bf.gu.assign(M * ffn_dim_, 0.0F);
  bf.uu.assign(M * ffn_dim_, 0.0F);
  bf.gx.assign(M * hidden_, 0.0F);
  bf.ddm.assign(M * hidden_, 0.0F);
  bf.acts.resize(std::max(B, M));
  bf.acts2.resize(M);
  bf.rope.assign(B * inv_freq_.size(), {0.0F, 0.0F});
  bf.win.assign(B, 0);
  bf.top.assign(B, {});
  bf.wts.assign(B, {});
  bf.seen.assign(n_experts_, 0);
  bf.emap.assign(n_experts_, -1);
  bf.idx.resize(n_experts_);
  bf.utop.reserve(M);
  bf.moff.resize(M);
  bf.mcur.resize(M);
  bf.mem_j.resize(M);
  bf.mem_w.resize(M);
  bf.wg.resize(M);
  bf.wu.resize(M);
  bf.wd.resize(M);
}

// один слой на b строк. порядок строго как в Model::layer: attention, потом
// ffn. kv пишется по абсолютным позициям, строки внутри батча каузальны
void Model::layer_batch(std::size_t layer, Kv& kv, std::size_t pos,
                        std::size_t b,
                        std::span<const std::pair<float, float>> rope_table) {
  const Layer& lt = layers_[layer];
  BatchBuf& bf = *batch_buf_;
  // Батч-префилл: union экспертов слоя не влезает в штатный per-layer
  // бюджет, поэтому на время слоя расширяем его до общего и чистим кэш от
  // прошлых слоёв. Иначе блоки вытесняются до take_expert и читаются повторно.
  if (prefilling_ && layer >= n_dense_ && io_ != nullptr && resident_ != nullptr &&
      total_budget_ != 0) {
    resident_->clear();
    io_->set_budget(total_budget_);
  }
  const std::size_t hidden = hidden_;
  const std::size_t q_dim = n_heads_ * head_dim_;
  const std::size_t kv_dim = n_kv_ * head_dim_;
  const std::size_t R = inv_freq_.size();

  for (std::size_t j = 0; j < b; ++j) {
    std::copy(bf.x.begin() + j * hidden, bf.x.begin() + (j + 1) * hidden,
              bf.hn.begin() + j * hidden);
    rmsnorm(std::span<float>(bf.hn.data() + j * hidden, hidden), lt.attn_norm,
            eps_);
    kernels::quantize_into(
        std::span<const float>(bf.hn.data() + j * hidden, hidden), bf.acts[j]);
  }
  {
    const BatchSeg segs[3] = {
        {meta_bytes(lt.attn_q), lt.attn_q.rb, lt.attn_q.quant, lt.attn_q.in_dim,
         bf.hn.data(), bf.acts.data(), b, bf.q.data(), lt.attn_q.out_dim},
        {meta_bytes(lt.attn_k), lt.attn_k.rb, lt.attn_k.quant, lt.attn_k.in_dim,
         bf.hn.data(), bf.acts.data(), b, bf.k.data(), lt.attn_k.out_dim},
        {meta_bytes(lt.attn_v), lt.attn_v.rb, lt.attn_v.quant, lt.attn_v.in_dim,
         bf.hn.data(), bf.acts.data(), b, bf.v.data(), lt.attn_v.out_dim},
    };
    batchlib::run_bsegs(std::span<const BatchSeg>(segs, 3), 16);
  }
  for (std::size_t j = 0; j < b; ++j) {
    float* qr = bf.q.data() + j * q_dim;
    float* kr = bf.k.data() + j * kv_dim;
    for (std::size_t hd = 0; hd < n_heads_; ++hd) {
      rmsnorm(std::span<float>(qr + hd * head_dim_, head_dim_),
              lt.attn_q_norm, eps_);
    }
    for (std::size_t hd = 0; hd < n_kv_; ++hd) {
      rmsnorm(std::span<float>(kr + hd * head_dim_, head_dim_),
              lt.attn_k_norm, eps_);
    }
    if (rope_on_[layer]) {
      const auto rt = rope_table.subspan(j * R, R);
      for (std::size_t hd = 0; hd < n_heads_; ++hd) {
        rope_apply(std::span<float>(qr + hd * head_dim_, head_dim_), rt);
      }
      for (std::size_t hd = 0; hd < n_kv_; ++hd) {
        rope_apply(std::span<float>(kr + hd * head_dim_, head_dim_), rt);
      }
    }
  }

  kv.write(layer, pos, b, bf.k.data(), kv_dim, bf.v.data(), kv_dim);

  const bool swa = swa_[layer];
  for (std::size_t j = 0; j < b; ++j) {
    const std::size_t n_hist = pos + j + 1;
    bf.win[j] =
        (swa && n_hist > window_) ? n_hist - window_ : std::size_t{0};
  }
  attn::AttendArgs aa;
  aa.k = kv.k_slab(layer);
  aa.v = kv.v_slab(layer);
  aa.k_stride = kv.k_stride_bytes();
  aa.v_stride = kv.v_stride_bytes();
  aa.f16 = kv.f16();
  aa.q = bf.q.data();
  aa.sq = q_dim;
  aa.sinks = lt.attn_sinks.data();
  aa.out = bf.attn.data();
  aa.b = b;
  aa.n_heads = n_heads_;
  aa.n_kvh = n_kv_;
  aa.head_dim = head_dim_;
  aa.scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
  aa.first = pos;
  aa.cap = kv.cap();
  aa.win = bf.win.data();
  for (std::size_t j = 0; j < b; ++j) attn::attend_row(aa, j);

  for (std::size_t j = 0; j < b; ++j) {
    kernels::quantize_into(
        std::span<const float>(bf.attn.data() + j * q_dim, q_dim), bf.acts[j]);
  }
  {
    const BatchSeg seg{meta_bytes(lt.attn_o), lt.attn_o.rb, lt.attn_o.quant,
                       lt.attn_o.in_dim, bf.attn.data(), bf.acts.data(), b,
                       bf.o.data(), hidden};
    batchlib::run_bsegs(std::span<const BatchSeg>(&seg, 1), 16);
  }
  for (std::size_t j = 0; j < b; ++j) {
    for (std::size_t i = 0; i < hidden; ++i) {
      bf.x[j * hidden + i] += bf.o[j * hidden + i];
    }
  }

  for (std::size_t j = 0; j < b; ++j) {
    std::copy(bf.x.begin() + j * hidden, bf.x.begin() + (j + 1) * hidden,
              bf.hf.begin() + j * hidden);
    rmsnorm(std::span<float>(bf.hf.data() + j * hidden, hidden), lt.ffn_norm,
            eps_);
    kernels::quantize_into(
        std::span<const float>(bf.hf.data() + j * hidden, hidden), bf.acts[j]);
  }

  if (layer < n_dense_) {
    const std::size_t df = dense_ffn_;
    {
      const BatchSeg segs[2] = {
          {meta_bytes(*lt.ffn_gate), lt.ffn_gate->rb, lt.ffn_gate->quant,
           lt.ffn_gate->in_dim, bf.hf.data(), bf.acts.data(), b, bf.g.data(),
           df},
          {meta_bytes(*lt.ffn_up), lt.ffn_up->rb, lt.ffn_up->quant,
           lt.ffn_up->in_dim, bf.hf.data(), bf.acts.data(), b, bf.u.data(),
           df},
      };
      batchlib::run_bsegs(std::span<const BatchSeg>(segs, 2), 32);
    }
    for (std::size_t j = 0; j < b; ++j) {
      swiglu(std::span<float>(bf.g.data() + j * df, df),
             std::span<const float>(bf.u.data() + j * df, df));
      kernels::quantize_into(std::span<const float>(bf.g.data() + j * df, df),
                             bf.acts[j]);
    }
    {
      const BatchSeg seg{meta_bytes(*lt.ffn_down), lt.ffn_down->rb,
                         lt.ffn_down->quant, lt.ffn_down->in_dim,
                         bf.g.data(), bf.acts.data(), b, bf.dd.data(), hidden};
      batchlib::run_bsegs(std::span<const BatchSeg>(&seg, 1), 32);
    }
    for (std::size_t j = 0; j < b; ++j) {
      for (std::size_t i = 0; i < hidden; ++i) {
        bf.x[j * hidden + i] += bf.dd[j * hidden + i];
      }
    }
    return;
  }

  // MoE: роутер per row, затем union экспертов батча загружается один раз
  {
    const BatchSeg seg{meta_bytes(*lt.gate_inp), lt.gate_inp->rb,
                       lt.gate_inp->quant, lt.gate_inp->in_dim,
                       bf.hf.data(), bf.acts.data(), b, bf.router.data(),
                       n_experts_};
    batchlib::run_bsegs(std::span<const BatchSeg>(&seg, 1), 16);
  }
  const std::size_t k_used =
      router_topk_ != 0 && router_topk_ < n_used_ ? router_topk_ : n_used_;
  for (std::size_t j = 0; j < b; ++j) {
    const float* sc = bf.router.data() + j * n_experts_;
    std::iota(bf.idx.begin(), bf.idx.end(), std::size_t{0});
    if (k_used < n_experts_) {
      std::nth_element(bf.idx.begin(),
                       bf.idx.begin() + static_cast<std::ptrdiff_t>(k_used),
                       bf.idx.end(), [&](std::size_t a, std::size_t c) {
                         return sc[a] > sc[c];
                       });
    }
    bf.top[j].assign(bf.idx.begin(),
                     bf.idx.begin() + static_cast<std::ptrdiff_t>(k_used));
    bf.wts[j].resize(k_used);
    float wsum = 0.0F;
    for (std::size_t i = 0; i < k_used; ++i) {
      bf.wts[j][i] = 1.0F / (1.0F + std::exp(-sc[bf.top[j][i]]));
      wsum += bf.wts[j][i];
    }
    for (float& w : bf.wts[j]) w /= wsum;
  }
  for (std::size_t j = 0; j < b; ++j) {
    record_freq(layer, bf.top[j]);
  }

  std::fill(bf.seen.begin(), bf.seen.end(), uint8_t{0});
  bf.utop.clear();
  for (std::size_t j = 0; j < b; ++j) {
    for (std::size_t e : bf.top[j]) {
      if (bf.seen[e] == 0) {
        bf.seen[e] = 1;
        bf.emap[e] = static_cast<int>(bf.utop.size());
        bf.utop.push_back(e);
      }
    }
  }
  const std::size_t nu = bf.utop.size();
  metrics().c_expert_union.fetch_add(nu, std::memory_order_relaxed);
  metrics().c_expert_passes.fetch_add(1, std::memory_order_relaxed);

  // TOKSHIT_PREFILL_BULK: для длинного чанка union близок ко всем 256
  // экспертам, и случайные чтения по слою дешевле заменить одним
  // sequential-чтением региона. Мелкие чанки оставляем на demand-путь:
  // читать 252 МБ ради 40 экспертов бессмысленно.
  // TOKSHIT_PREFILL_BULK: for a long chunk the union is close to all 256
  if (prefilling_ && io_ != nullptr && resident_budget_ != 0 && b >= 16 &&
      nu >= 96) {
    static const bool bulk = [] {
      const char* v = std::getenv("TOKSHIT_PREFILL_BULK");
      return v != nullptr && *v != '\0' && v[0] != '0';
    }();
    if (bulk) {
      prefill_bulk_layer(layer,
                         std::span<const std::size_t>(bf.utop.data(), nu));
    }
  }

  // считаем размеры групп, потом раскладываем плоские пары (эксперт, строка)
  std::fill(bf.moff.begin(), bf.moff.begin() + static_cast<std::ptrdiff_t>(nu),
            std::size_t{0});
  std::size_t total = 0;
  for (std::size_t j = 0; j < b; ++j) {
    for (std::size_t ki = 0; ki < bf.top[j].size(); ++ki) {
      const std::size_t u = static_cast<std::size_t>(bf.emap[bf.top[j][ki]]);
      ++bf.moff[u];
      ++total;
    }
  }
  std::size_t cursor = 0;
  for (std::size_t u = 0; u < nu; ++u) {
    const std::size_t cnt = bf.moff[u];
    bf.moff[u] = cursor;
    bf.mcur[u] = cursor;
    cursor += cnt;
  }
  bf.mem_j.resize(total);
  bf.mem_w.resize(total);
  for (std::size_t j = 0; j < b; ++j) {
    for (std::size_t ki = 0; ki < bf.top[j].size(); ++ki) {
      const std::size_t u = static_cast<std::size_t>(bf.emap[bf.top[j][ki]]);
      const std::size_t at = bf.mcur[u]++;
      bf.mem_j[at] = j;
      bf.mem_w[at] = bf.wts[j][ki];
    }
  }

  const ExpertMeta em_g = *lt.gate_exps;
  const ExpertMeta em_u = *lt.up_exps;
  const ExpertMeta em_d = *lt.down_exps;
  bf.wg.resize(nu);
  bf.wu.resize(nu);
  bf.wd.resize(nu);

  if (io_ != nullptr && resident_budget_ != 0) {
    // hit/miss как в одиночном пути: demand_async вернул true (блок уже в
    // кэше или вне диапазона) — hit, иначе чтение реально заказано — miss.
    // дупликаат (чтение уже в полёте) тоже miss, чтобы совпадать с model.cpp
    const ExpertMeta* ems[2] = {&em_g, &em_u};
    for (std::size_t u = 0; u < nu; ++u) {
      const std::size_t e = bf.utop[u];
      for (std::size_t pi = 0; pi < 2; ++pi) {
        const uint64_t key = (uint64_t{pi} << 32) | e;
        const uint64_t off = ems[pi]->base + ems[pi]->size * e;
        if (io_->demand_async(layer, key, off, ems[pi]->size)) {
          metrics().c_expert_hit.fetch_add(1, std::memory_order_relaxed);
        } else {
          metrics().c_expert_miss.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
    for (std::size_t u = 0; u < nu; ++u) {
      const std::size_t e = bf.utop[u];
      const uint64_t key = (uint64_t{2} << 32) | e;
      const uint64_t off = em_d.base + em_d.size * e;
      if (io_->demand_async(layer, key, off, em_d.size)) {
        metrics().c_expert_hit.fetch_add(1, std::memory_order_relaxed);
      } else {
        metrics().c_expert_miss.fetch_add(1, std::memory_order_relaxed);
      }
    }
    for (std::size_t u = 0; u < nu; ++u) {
      bf.wg[u] = take_expert(layer, 0, em_g, bf.utop[u]);
      bf.wu[u] = take_expert(layer, 1, em_u, bf.utop[u]);
    }
  } else {
    for (std::size_t u = 0; u < nu; ++u) {
      bf.wg[u] = resident_expert(layer, 0, em_g, bf.utop[u]);
      bf.wu[u] = resident_expert(layer, 1, em_u, bf.utop[u]);
    }
  }

  for (std::size_t u = 0; u < nu; ++u) {
    const std::size_t off = bf.moff[u];
    const std::size_t cnt = (u + 1 < nu ? bf.moff[u + 1] : total) - off;
    for (std::size_t mi = 0; mi < cnt; ++mi) {
      const std::size_t j = bf.mem_j[off + mi];
      std::copy(bf.hf.begin() + j * hidden, bf.hf.begin() + (j + 1) * hidden,
                bf.gx.begin() + (off + mi) * hidden);
      kernels::quantize_into(
          std::span<const float>(bf.gx.data() + (off + mi) * hidden, hidden),
          bf.acts[off + mi]);
    }
  }
  {
    bf.segs.clear();
    bf.segs.reserve(2 * nu);
    const std::size_t ffn_dim = ffn_dim_;
    for (std::size_t u = 0; u < nu; ++u) {
      const std::size_t off = bf.moff[u];
      const std::size_t nb =
          (u + 1 < nu ? bf.moff[u + 1] : total) - off;
      const std::size_t rbg = std::min(em_g.rb, bf.wg[u].len / ffn_dim);
      const std::size_t rbu = std::min(em_u.rb, bf.wu[u].len / ffn_dim);
      bf.segs.push_back(BatchSeg{
          std::span<const uint8_t>(bf.wg[u].data, bf.wg[u].len), rbg,
          em_g.quant, em_g.in_dim, bf.gx.data() + off * hidden,
          bf.acts.data() + off, nb, bf.gu.data() + off * ffn_dim, ffn_dim});
      bf.segs.push_back(BatchSeg{
          std::span<const uint8_t>(bf.wu[u].data, bf.wu[u].len), rbu,
          em_u.quant, em_u.in_dim, bf.gx.data() + off * hidden,
          bf.acts.data() + off, nb, bf.uu.data() + off * ffn_dim, ffn_dim});
    }
    batchlib::run_bsegs(bf.segs, 32);
  }
  for (std::size_t m = 0; m < total; ++m) {
    swiglu(std::span<float>(bf.gu.data() + m * ffn_dim_, ffn_dim_),
           std::span<const float>(bf.uu.data() + m * ffn_dim_, ffn_dim_));
    kernels::quantize_into(
        std::span<const float>(bf.gu.data() + m * ffn_dim_, ffn_dim_),
        bf.acts2[m]);
  }
  if (io_ != nullptr && resident_budget_ != 0) {
    for (std::size_t u = 0; u < nu; ++u) {
      bf.wd[u] = take_expert(layer, 2, em_d, bf.utop[u]);
    }
  } else {
    for (std::size_t u = 0; u < nu; ++u) {
      bf.wd[u] = resident_expert(layer, 2, em_d, bf.utop[u]);
    }
  }
  {
    bf.segs.clear();
    bf.segs.reserve(nu);
    for (std::size_t u = 0; u < nu; ++u) {
      const std::size_t off = bf.moff[u];
      const std::size_t nb =
          (u + 1 < nu ? bf.moff[u + 1] : total) - off;
      const std::size_t rbd = std::min(em_d.rb, bf.wd[u].len / hidden);
      bf.segs.push_back(BatchSeg{
          std::span<const uint8_t>(bf.wd[u].data, bf.wd[u].len), rbd,
          em_d.quant, em_d.in_dim, bf.gu.data() + off * ffn_dim_,
          bf.acts2.data() + off, nb, bf.ddm.data() + off * hidden, hidden});
    }
    batchlib::run_bsegs(bf.segs, 32);
  }

  std::fill(bf.moe.begin(), bf.moe.end(), 0.0F);
  for (std::size_t u = 0; u < nu; ++u) {
    const std::size_t off = bf.moff[u];
    const std::size_t cnt = (u + 1 < nu ? bf.moff[u + 1] : total) - off;
    for (std::size_t mi = 0; mi < cnt; ++mi) {
      const std::size_t j = bf.mem_j[off + mi];
      const float w = bf.mem_w[off + mi];
      const float* src = bf.ddm.data() + (off + mi) * hidden;
      float* dst = bf.moe.data() + j * hidden;
      for (std::size_t i = 0; i < hidden; ++i) dst[i] += w * src[i];
    }
  }

  // shared expert, плотный mlp как в одиночном пути
  for (std::size_t j = 0; j < b; ++j) {
    kernels::quantize_into(
        std::span<const float>(bf.hf.data() + j * hidden, hidden), bf.acts[j]);
  }
  {
    const std::size_t sd = shared_dim_;
    const BatchSeg segs[2] = {
        {meta_bytes(*lt.gate_shexp), lt.gate_shexp->rb, lt.gate_shexp->quant,
         lt.gate_shexp->in_dim, bf.hf.data(), bf.acts.data(), b, bf.g.data(),
         sd},
        {meta_bytes(*lt.up_shexp), lt.up_shexp->rb, lt.up_shexp->quant,
         lt.up_shexp->in_dim, bf.hf.data(), bf.acts.data(), b, bf.u.data(),
         sd},
    };
    batchlib::run_bsegs(std::span<const BatchSeg>(segs, 2), 32);
  }
  for (std::size_t j = 0; j < b; ++j) {
    swiglu(std::span<float>(bf.g.data() + j * shared_dim_, shared_dim_),
           std::span<const float>(bf.u.data() + j * shared_dim_, shared_dim_));
    kernels::quantize_into(
        std::span<const float>(bf.g.data() + j * shared_dim_, shared_dim_),
        bf.acts[j]);
  }
  {
    const BatchSeg seg{
        meta_bytes(*lt.down_shexp), lt.down_shexp->rb, lt.down_shexp->quant,
        lt.down_shexp->in_dim, bf.g.data(), bf.acts.data(), b, bf.dd.data(),
        hidden};
    batchlib::run_bsegs(std::span<const BatchSeg>(&seg, 1), 32);
  }
  for (std::size_t j = 0; j < b; ++j) {
    for (std::size_t i = 0; i < hidden; ++i) {
      bf.x[j * hidden + i] += bf.moe[j * hidden + i] + bf.dd[j * hidden + i];
    }
  }
}

void Model::token_logits_batch(std::span<const uint32_t> tokens, Kv& kv,
                               std::size_t pos, std::span<float> outs) {
  const std::size_t b = tokens.size();
  if (b == 0) return;
  ensure_batch_buf(b);
  BatchBuf& bf = *batch_buf_;
  const std::size_t hidden = hidden_;
  const std::size_t R = inv_freq_.size();

  const auto embd_bytes = meta_bytes(embd_);
  for (std::size_t j = 0; j < b; ++j) {
    auto dst = std::span<float>(bf.x.data() + j * hidden, hidden);
    // невалидный id — нулевая строка, как при провале деквантования
    if (static_cast<std::size_t>(tokens[j]) >= vocab_) {
      std::fill(dst.begin(), dst.end(), 0.0F);
      continue;
    }
    const std::size_t off = embd_.rb * static_cast<std::size_t>(tokens[j]);
    const auto row = embd_bytes.subspan(off, embd_.rb);
    const auto deq = dequant_into(embd_.quant, row, hidden, dst);
    if (!deq.ok()) std::fill(dst.begin(), dst.end(), 0.0F);
  }
  for (std::size_t j = 0; j < b; ++j) {
    for (std::size_t i = 0; i < R; ++i) {
      const float angle = static_cast<float>(pos + j) * inv_freq_[i];
      bf.rope[j * R + i] = {std::sin(angle), std::cos(angle)};
    }
  }

  std::size_t n_run = n_layers_;
  if (const char* v = std::getenv("GGUF_NLAYERS"); v != nullptr && *v != '\0') {
    n_run = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
    n_run = std::min(n_run, n_layers_);
  }
  const std::span<const std::pair<float, float>> rope_all(bf.rope.data(),
                                                          b * R);
  for (std::size_t layer = 0; layer < n_run; ++layer) {
    layer_batch(layer, kv, pos, b, rope_all);
  }
  // вернуть штатный бюджет после батч-префилла, иначе декод будет держать
  // слишком много блоков на слой и упрётся в общий AHB
  if (prefilling_ && io_ != nullptr) io_->set_budget(resident_budget_);

  for (std::size_t j = 0; j < b; ++j) {
    auto row = std::span<float>(bf.x.data() + j * hidden, hidden);
    rmsnorm(row, out_norm_, eps_);
    kernels::quantize_into(row, bf.acts[j]);
  }
  const BatchSeg head{meta_bytes(out_), out_.rb, out_.quant, out_.in_dim,
                      bf.x.data(), bf.acts.data(), b, outs.data(), vocab_};
  batchlib::run_bsegs(std::span<const BatchSeg>(&head, 1), 16);
}

}  // namespace tokshit
