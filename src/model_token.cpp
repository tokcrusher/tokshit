#include "tokshit/model.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/pool.hpp"

#include "model_internal.hpp"

namespace tokshit {

void Model::row_task(void* ctx, std::size_t task) {
  const RowJob& job = *static_cast<const RowJob*>(ctx);
  std::size_t s = 0;
  while (s + 1 < job.task_starts_len && task >= job.task_starts[s + 1]) ++s;
  const std::size_t local = task - job.task_starts[s];
  const Seg& seg = job.segs[s];
  const std::size_t r0 = local * job.rows_per_task;
  const std::size_t r1 = std::min(r0 + job.rows_per_task, seg.rows);
  thread_local std::vector<float> scratch;
  const std::size_t nrows = r1 - r0;
  if (nrows == 0) return;
  const auto block = seg.bytes.subspan(r0 * seg.rb, nrows * seg.rb);
  kernels::dot_rows(
      seg.quant, block, seg.rb, nrows,
      std::span<const float>(seg.x, seg.in_dim), *seg.act, seg.in_dim,
      seg.out + r0, scratch);
}

// грузим один эксперт одной проекции через кэш. задачи намеренно плоские:
// индекс i кодирует (какой эксперт, какая проекция)
void Model::load_task(void* ctx, std::size_t i) {
  LoadJob& job = *static_cast<LoadJob*>(ctx);
  const std::size_t e = i % job.n_used;
  const auto proj = static_cast<uint8_t>(i / job.n_used);
  job.out[i] = job.model->resident_expert(job.layer, proj, *job.ems[proj],
                                          job.tops[e]);
}

// главный загрузчик. читает конфиг, строит Layer с Meta смещениями, валидирует
// паттерны и создаёт workspace. тяжёлых копий нет (кроме hot и norm векторов)

std::span<const uint8_t> Model::meta_bytes(const Meta& meta) const {
  if (meta.hot) {
    const uint8_t* p = hot_.data_at(meta.base);
    if (p != nullptr) {
      return std::span<const uint8_t>(p, hot_.size() - meta.base);
    }
  }
  const auto bytes = mm_.bytes();
  return std::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(bytes.data()) + meta.base,
      bytes.size() - meta.base);
}

// удобная версия: квантуем активацию во временный буфер и считаем. нужна
// только холодным вызовам, горячий путь юзает версию с переданным act
void Model::matvec(const Meta& meta, std::span<const float> x,
                   std::span<float> y) const {
  kernels::ActQ8 act;
  matvec(meta, x, y, act);
}

// квантуем x в act и считаем. активация остаётся снаружи чтобы можно было
// переиспользовать её для нескольких матриц с тем же входом
void Model::matvec(const Meta& meta, std::span<const float> x,
                   std::span<float> y, kernels::ActQ8& act) const {
  kernels::quantize_into(x, act);
  matvec_preact(meta, x, y, act);
}

// версия с уже готовой активацией. именно её зовёт горячий путь, тут нет ни
// одной аллокации
void Model::matvec_preact(const Meta& meta, std::span<const float> x,
                          std::span<float> y,
                          const kernels::ActQ8& act) const {
  const auto bytes = meta_bytes(meta);
  const auto t0 = Clock::now();
  const Seg seg{bytes, meta.rb, meta.quant, meta.in_dim, x.data(), &act,
                y.data(), meta.out_dim};
  run_segs(std::span<const Seg>(&seg, 1), 16);
  if (timing_enabled()) {
    metrics().t_mm.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                 t0)
                .count()),
        std::memory_order_relaxed);
    metrics().c_mm_bytes.fetch_add(
        static_cast<uint64_t>(meta.out_dim * meta.rb),
        std::memory_order_relaxed);
  }
}

// раскладываем все сегменты в задачи по строкам и кидаем в пул. вызывающий
// поток тоже работает, так что не простаивает
void Model::run_segs(std::span<const Seg> segs,
                     std::size_t rows_per_task) const {
  thread_local std::vector<std::size_t> task_starts;
  task_starts.clear();
  task_starts.reserve(segs.size() + 1);
  task_starts.push_back(0);
  std::size_t total = 0;
  for (const Seg& seg : segs) {
    total += (seg.rows + rows_per_task - 1) / rows_per_task;
    task_starts.push_back(total);
  }
  if (total == 0) return;

  const RowJob job{segs.data(), segs.size(), task_starts.data(),
                   task_starts.size(), rows_per_task};
  static const bool dbg = std::getenv("TOKSHIT_SEG_DEBUG") != nullptr;
  const auto pf0 = dbg ? page_faults() : std::pair<uint64_t, uint64_t>{0, 0};
  const auto t0 = Clock::now();
  pool::run_tasks(total, &Model::row_task,
                  const_cast<RowJob*>(&job));

  if (dbg) {
    static std::atomic<uint64_t> n{0};
    static std::atomic<uint64_t> total_bytes{0};
    static std::atomic<uint64_t> total_ns{0};
    const uint64_t k = n.fetch_add(1, std::memory_order_relaxed);
    std::size_t rows = 0;
    std::size_t bytes = 0;
    for (const Seg& seg : segs) {
      rows += seg.rows;
      bytes += seg.rows * seg.rb;
    }
    const double dt =
        std::chrono::duration<double>(Clock::now() - t0).count();
    const uint64_t dt_ns = static_cast<uint64_t>(dt * 1e9);
    const uint64_t acc_b = total_bytes.fetch_add(bytes) + bytes;
    const uint64_t acc_ns = total_ns.fetch_add(dt_ns) + dt_ns;
    const QuantType q0 = segs.empty() ? QuantType::F32 : segs[0].quant;
    const std::size_t rb0 = segs.empty() ? 0 : segs[0].rb;
    const std::size_t in0 = segs.empty() ? 0 : segs[0].in_dim;
    static const bool dbg_all =
        std::getenv("TOKSHIT_SEG_DEBUG_ALL") != nullptr;
    if (k < 12 || k % 500 == 0 ||
        (dbg_all && k < 400 && q0 != QuantType::Q8_0)) {
      const auto pf1 = page_faults();
      std::fprintf(stderr,
                   "[seg %llu] tasks=%zu rows=%zu segs=%zu q=%.*s rb=%zu "
                   "in=%zu %.1f MB %.0f us %.2f GB/s avg=%.2f GB/s active=%d "
                   "minflt=%llu majflt=%llu\n",
                   static_cast<unsigned long long>(k), total, rows, segs.size(),
                   static_cast<int>(quant_name(q0).size()),
                   quant_name(q0).data(), rb0, in0,
                   static_cast<double>(bytes) / 1e6, dt * 1e6,
                   static_cast<double>(bytes) / dt / 1e9,
                   acc_ns == 0 ? 0.0
                               : static_cast<double>(acc_b) / acc_ns,
                   pool::is_active() ? 1 : 0,
                   static_cast<unsigned long long>(pf1.first - pf0.first),
                   static_cast<unsigned long long>(pf1.second - pf0.second));
    }
  }
}

// достаём эксперта. бюджет 0 = отдаём span прямо в mmap. иначе через
// фоновых читателей: они и кэш проверят, и подхватят уже начатое чтение.
// ключ это (proj << 32) | expert

void Model::prepare_workspace() {
  ws_ = std::make_unique<Workspace>();
  Workspace& w = *ws_;
  const std::size_t q_dim = n_heads_ * head_dim_;
  const std::size_t kv_dim = n_kv_ * head_dim_;
  const std::size_t ffn_cap = std::max({dense_ffn_, shared_dim_, std::size_t{1}});
  const std::size_t expert_elem =
      std::max(n_used_ * hidden_, std::max(hidden_, std::size_t{1}));

  w.x.assign(hidden_, 0.0F);
  w.hn.assign(hidden_, 0.0F);
  w.hf.assign(hidden_, 0.0F);
  w.o.assign(hidden_, 0.0F);
  w.dd.assign(expert_elem, 0.0F);
  w.moe.assign(hidden_, 0.0F);
  w.q.assign(q_dim, 0.0F);
  w.k.assign(kv_dim, 0.0F);
  w.v.assign(kv_dim, 0.0F);
  w.attn.assign(q_dim, 0.0F);
  w.g.assign(ffn_cap, 0.0F);
  w.u.assign(ffn_cap, 0.0F);
  w.gu.assign(2 * std::max(n_used_ * ffn_dim_, std::size_t{1}), 0.0F);
  w.scores.clear();
  w.scores.reserve(std::max(window_, std::size_t{4096}));
  w.logits.assign(vocab_, 0.0F);
  w.wts.assign(n_used_, 0.0F);
  w.idx.resize(n_experts_);
  w.top.assign(n_used_, 0);
  w.slices.resize(3 * n_used_);
  w.wg.resize(n_used_);
  w.wu.resize(n_used_);
  w.wd.resize(n_used_);
  w.segs.reserve(2 * n_used_);
  w.down_segs.reserve(n_used_);
  w.expert_acts.resize(n_used_);
  w.rope_table.resize(inv_freq_.size());
}

// главный вход на токен, без аллокаций (при условии что out уже нужного
// размера). делает эмбеддинг, все слои, финальную норму и голову
void Model::token_logits(uint32_t token, KVCache& kv, std::size_t pos,
                         std::vector<float>& out) {
  Workspace& ws = *ws_;
  // prefill: только первые вызовы (до начала генерации). main.cpp
  // выставит prefilling_ = false перед генерацией
  const auto t_embd = Clock::now();
  const auto embd_bytes = meta_bytes(embd_);
  // id вне словаря (битый сэмплер или промпт) даёт нулевые логиты вместо
  // выхода за буфер весов.
  if (static_cast<std::size_t>(token) >= vocab_) {
    std::fill(out.begin(), out.end(), 0.0F);
    return;
  }
  const std::size_t off = embd_.rb * static_cast<std::size_t>(token);
  const auto row = embd_bytes.subspan(off, embd_.rb);
  const auto deq = dequant_into(embd_.quant, row, hidden_, ws.x);
  if (!deq.ok()) {
    std::fill(out.begin(), out.end(), 0.0F);
    return;
  }
  tick(metrics().t_embd, t_embd);

  for (std::size_t i = 0; i < inv_freq_.size(); ++i) {
    const float angle = static_cast<float>(pos) * inv_freq_[i];
    ws.rope_table[i] = {std::sin(angle), std::cos(angle)};
  }

  std::size_t n_run = n_layers_;
  if (const char* v = std::getenv("GGUF_NLAYERS"); v != nullptr && *v != '\0') {
    n_run = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
    n_run = std::min(n_run, n_layers_);
  }
  if (std::getenv("TOKSHIT_DEBUG_NRUN") != nullptr) {
    std::fprintf(stderr, "[nrun] %zu/%zu\n", n_run, n_layers_);
  }

  // новый токен = новые предсказания. старое из очереди выбрасываем, а
  // предсказанное прошлым токеном ставим в очередь. только первый moe слой:
  // читать весь токен вперёд бессмысленно, диск и так в полку, а промахи
  // предсказания тратят полосу
  if (prefetch_enabled() && io_ && resident_budget_ != 0) {
    io_->new_epoch();
    static const bool idle_pf = prefetch_idle();
    if (n_dense_ < n_run) {
      prefetch_experts(n_dense_, freq_->last[n_dense_]);
      // в idle-режиме второй слой вперёд не читаем: спекуляция только на
      // окнах простоя, а их хватает на один слой
      if (!idle_pf && n_dense_ + 1 < n_run) {
        prefetch_experts(n_dense_ + 1, freq_->last[n_dense_ + 1]);
      }
    }
  }
  for (std::size_t layer = 0; layer < n_run; ++layer) {
    this->layer(layer, ws, kv, ws.rope_table);
  }

  const auto t_head = Clock::now();
  rmsnorm(ws.x, out_norm_, eps_);
  out.resize(vocab_);
  std::fill(out.begin(), out.end(), 0.0F);
  kernels::quantize_into(ws.x, ws.act);
  matvec_preact(out_, ws.x, out, ws.act);
  tick(metrics().t_head, t_head);
}

// удобная обёртка для холодных вызовов, возвращает vector по значению
std::vector<float> Model::token_logits(uint32_t token, KVCache& kv,
                                        std::size_t pos) {
  std::vector<float> out(vocab_, 0.0F);
  token_logits(token, kv, pos, out);
  return out;
}

}  // namespace tokshit
