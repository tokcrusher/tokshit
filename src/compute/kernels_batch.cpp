#include "tokshit/batch.hpp"

#include <algorithm>
#include <vector>

#include "tokshit/pool.hpp"

namespace tokshit::batchlib {

namespace {

// задача: чанк строк весов одного сегмента. внутри — b активаций по этой
// строке. строку весов читаем через существующие ядра (на aarch64 это
// dotprod-варианты), повторные проходы по b ловят её из L1
struct BJob {
  const BatchSeg* segs = nullptr;
  std::size_t nsegs = 0;
  const std::size_t* starts = nullptr;  // nsegs+1
  std::size_t rows_per_task = 8;
};

thread_local std::vector<float> scratch;
thread_local std::vector<float> out_scratch;

void btask(void* ctx, std::size_t task) {
  auto* job = static_cast<BJob*>(ctx);
  std::size_t si = 0;
  while (job->starts[si + 1] <= task) ++si;
  const BatchSeg& seg = job->segs[si];
  const std::size_t local = task - job->starts[si];
  const std::size_t r0 = local * job->rows_per_task;
  const std::size_t r1 = std::min(seg.rows, r0 + job->rows_per_task);
  if (seg.b == 1) {
    for (std::size_t r = r0; r < r1; ++r) {
      const auto row = seg.bytes.subspan(r * seg.rb, seg.rb);
      seg.out[r] = kernels::dot_row(
          seg.quant, row, seg.rb, std::span<const float>(seg.x, seg.in_dim),
          seg.acts[0], seg.in_dim, scratch);
    }
    return;
  }
  out_scratch.resize(seg.b);
  for (std::size_t r = r0; r < r1; ++r) {
    const auto row = seg.bytes.subspan(r * seg.rb, seg.rb);
    kernels::dot_row_multi(seg.quant, row, seg.rb, seg.acts, seg.b, seg.in_dim,
                           out_scratch.data(), scratch);
    for (std::size_t j = 0; j < seg.b; ++j) {
      seg.out[j * seg.rows + r] = out_scratch[j];
    }
  }
}

}  // namespace

void run_bsegs(std::span<const BatchSeg> segs, std::size_t rows_per_task) {
  thread_local std::vector<std::size_t> starts;
  starts.clear();
  starts.reserve(segs.size() + 1);
  starts.push_back(0);
  std::size_t total = 0;
  for (const BatchSeg& s : segs) {
    total += (s.rows + rows_per_task - 1) / rows_per_task;
    starts.push_back(total);
  }
  if (total == 0) return;
  BJob job{segs.data(), segs.size(), starts.data(), rows_per_task};
  pool::run_tasks(total, &btask, &job);
}

}  // namespace tokshit::batchlib
