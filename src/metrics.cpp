#include "tokshit/metrics.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace tokshit {
namespace {

bool env_present(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr;
}

}  // namespace

Metrics& metrics() noexcept {
  static Metrics instance;
  return instance;
}

bool timing_enabled() noexcept {
  static const bool enabled = env_present("GGUF_TIME");
  return enabled;
}

bool prefetch_enabled() noexcept {
    static const bool e = [] {
        const char* v = std::getenv("TOKSHIT_PREFETCH");
        return v != nullptr && *v != '\0' && v[0] != '0';
    }();
    return e;
}

bool prefetch_idle() noexcept {
    static const bool e = [] {
        const char* v = std::getenv("TOKSHIT_PREFETCH_IDLE");
        return v != nullptr && *v != '\0' && v[0] != '0';
    }();
    return e;
}

bool io_merge_enabled() noexcept {
    static const bool e = [] {
        const char* v = std::getenv("TOKSHIT_IO_MERGE");
        return v != nullptr && *v != '\0' && v[0] != '0';
    }();
    return e;
}

uint64_t io_read_bytes() noexcept {
  std::ifstream in("/proc/self/io");
  if (!in) return 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("read_bytes:", 0) == 0) {
      std::istringstream ss(line.substr(11));
      uint64_t value = 0;
      if (ss >> value) return value;
    }
  }
  return 0;
}

std::pair<uint64_t, uint64_t> page_faults() noexcept {
  std::ifstream in("/proc/self/stat");
  if (!in) return {0, 0};
  std::string content((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
  const auto close = content.rfind(')');
  if (close == std::string::npos) return {0, 0};
  std::istringstream ss(content.substr(close + 1));
  std::string field;
  uint64_t values[16] = {};
  size_t i = 0;
  while (i < 16 && (ss >> field)) {
    values[i++] = static_cast<uint64_t>(std::strtoull(field.c_str(), nullptr, 10));
  }
  // Tail starts at field 3 (state); minflt is field 10 -> index 7, majflt 12 ->
  // index 9.
  return {values[7], values[9]};
}

uint64_t rss_mb() noexcept {
  std::ifstream in("/proc/self/status");
  if (!in) return 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::istringstream ss(line.substr(6));
      uint64_t kb = 0;
      if (ss >> kb) return kb / 1024;
    }
  }
  return 0;
}

std::vector<uint32_t> cpu_freqs() noexcept {
  std::vector<uint32_t> out;
  out.reserve(8);
  for (uint32_t i = 0; i < 8; ++i) {
    std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(i) +
                     "/cpufreq/scaling_cur_freq");
    uint32_t khz = 0;
    if (in && (in >> khz)) {
      out.push_back(khz / 1000);
    } else {
      out.push_back(0);
    }
  }
  return out;
}

void metrics_reset() noexcept {
  Metrics& m = metrics();
  m.t_attn = 0; m.t_mm = 0; m.c_mm_bytes = 0; m.c_dot_fallback = 0;
  for (auto& q : m.c_fallback_q) q.store(0, std::memory_order_relaxed);
  m.t_attn_mm = 0; m.t_attn_core = 0; m.t_ffn = 0;
  m.t_head = 0; m.t_embd = 0; m.t_router = 0;
  m.t_expert = 0; m.t_shexp = 0; m.t_load = 0;
  m.t_down_wait = 0; m.t_expert_mm = 0;
  m.c_inline = 0; m.c_wait_null = 0; m.c_wait_ok = 0;
  m.c_dem_queued = 0; m.c_dem_cached = 0; m.c_dem_dup = 0;
  m.c_dem_range = 0; m.c_dem_merged_jobs = 0; m.c_dem_merged_slices = 0;
  m.c_spec_idle_drop = 0; m.c_read_fail = 0;
  m.c_expert_hit = 0; m.c_expert_miss = 0;
  m.c_expert_union = 0; m.c_expert_passes = 0;
}

}  // namespace tokshit
