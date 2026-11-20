#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <span>

namespace tokshit {

// Debug-трасса роутинга: TOKSHIT_EXPERT_TRACE=<path> пишет по строке на
// каждый проход слоя: "layer e0 e1 ... e5". В обычном запуске выключена.
inline void trace_experts(std::size_t layer,
                          std::span<const std::size_t> top) {
  static const char* path = std::getenv("TOKSHIT_EXPERT_TRACE");
  if (path == nullptr || *path == '\0') return;
  static std::mutex mu;
  static FILE* f = std::fopen(path, "w");
  if (f == nullptr) return;
  std::lock_guard<std::mutex> lock(mu);
  std::fprintf(f, "%zu", layer);
  for (std::size_t e : top) std::fprintf(f, " %zu", e);
  std::fputc('\n', f);
  std::fflush(f);
}

}  // namespace tokshit
