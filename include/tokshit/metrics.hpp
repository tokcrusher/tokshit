#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

namespace tokshit {

// Profiling counters. Names and output format match the Rust reference so logs
// from both implementations can be compared directly.
struct Metrics {
  std::atomic<uint64_t> t_attn{0};
  std::atomic<uint64_t> t_mm{0};
  std::atomic<uint64_t> c_mm_bytes{0};
  std::atomic<uint64_t> t_attn_mm{0};
  std::atomic<uint64_t> t_attn_core{0};
  std::atomic<uint64_t> t_ffn{0};
  std::atomic<uint64_t> t_head{0};
  std::atomic<uint64_t> t_embd{0};
  std::atomic<uint64_t> t_router{0};
  std::atomic<uint64_t> t_expert{0};
  std::atomic<uint64_t> t_shexp{0};
  std::atomic<uint64_t> t_load{0};
  std::atomic<uint64_t> t_down_wait{0};
  std::atomic<uint64_t> t_expert_mm{0};
  std::atomic<uint64_t> c_inline{0};
  std::atomic<uint64_t> c_wait_null{0};
  std::atomic<uint64_t> c_wait_ok{0};
  std::atomic<uint64_t> c_dem_queued{0};
  std::atomic<uint64_t> c_dem_cached{0};
  std::atomic<uint64_t> c_dem_dup{0};
  std::atomic<uint64_t> c_dem_range{0};
  // TOKSHIT_IO_MERGE: сколько слоёв-задач слилось в один pread и сколько
  // отдельных чтений они заменили
  // TOKSHIT_IO_MERGE: how many jobs were merged into one pread and how many
  // individual reads they replaced
  std::atomic<uint64_t> c_dem_merged_jobs{0};
  std::atomic<uint64_t> c_dem_merged_slices{0};
  // TOKSHIT_PREFETCH_IDLE: спек-заказы, отброшенные из-за занятого диска
  // TOKSHIT_PREFETCH_IDLE: spec orders dropped because the drive was busy
  std::atomic<uint64_t> c_spec_idle_drop{0};
  std::atomic<uint64_t> c_read_fail{0};
  std::atomic<uint64_t> c_expert_hit{0};
  std::atomic<uint64_t> c_expert_miss{0};
  // сколько уникальных экспертов в среднем тянет один проход moe-слоя
  // average number of unique experts a single moe layer pass pulls
  std::atomic<uint64_t> c_expert_union{0};
  std::atomic<uint64_t> c_expert_passes{0};
};

[[nodiscard]] Metrics& metrics() noexcept;

// обнуляет таймеры и счётчики. нужен профилировщику, чтобы отделить
// генерацию от prefill
// zeroes timers and counters. the profiler uses it to separate decode from
// prefill
void metrics_reset() noexcept;

[[nodiscard]] bool timing_enabled() noexcept;
[[nodiscard]] bool prefetch_enabled() noexcept;
// TOKSHIT_PREFETCH_IDLE: спекулятивные заказы ставить в очередь только когда
// дисковый путь полностью свободен (demand пуст, воркеры не читают)
// TOKSHIT_PREFETCH_IDLE: queue speculative orders only when the drive path is
// completely idle (demand empty, no worker reading)
[[nodiscard]] bool prefetch_idle() noexcept;
// TOKSHIT_IO_MERGE: соседние срезы экспертов одного proj одним pread
// TOKSHIT_IO_MERGE: adjacent expert slices of one proj in a single pread
[[nodiscard]] bool io_merge_enabled() noexcept;

using Clock = std::chrono::steady_clock;

inline void tick(std::atomic<uint64_t>& counter, Clock::time_point start) {
  if (timing_enabled()) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now() - start)
                        .count();
    counter.fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
  }
}

[[nodiscard]] uint64_t io_read_bytes() noexcept;
[[nodiscard]] std::pair<uint64_t, uint64_t> page_faults() noexcept;
[[nodiscard]] uint64_t rss_mb() noexcept;
[[nodiscard]] std::vector<uint32_t> cpu_freqs() noexcept;

}  // namespace tokshit
