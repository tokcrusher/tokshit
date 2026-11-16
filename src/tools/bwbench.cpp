// Memory bandwidth on a real model file.
// usage: bwbench <path> [region_mb] [iters]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "tokshit/gguf.hpp"
#include "tokshit/metrics.hpp"

namespace {

uint64_t sum_words(std::span<const uint8_t> s) {
  // 8 bytes per load: keep the ALU out of the way.
  uint64_t acc = 0;
  const std::size_t n = s.size() / 8;
  for (std::size_t i = 0; i < n; ++i) {
    uint64_t word = 0;
    std::memcpy(&word, s.data() + i * 8, sizeof(word));
    acc += word;
  }
  return acc;
}

}  // namespace

int main(int argc, char** argv) {
  std::string path = argc > 1
                         ? argv[1]
                         : "/data/data/com.termux/files/home/models/"
                           "flint-UD-Q3_K_XL.gguf";
  std::size_t region_mb = 2048;
  if (argc > 2) region_mb = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
  std::size_t iters = 3;
  if (argc > 3) iters = static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10));

  auto opened = tokshit::Gguf::open(path);
  if (!opened.ok()) {
    std::fprintf(stderr, "gguf: %s: %s\n",
                 tokshit::to_string(opened.error().code).data(),
                 opened.error().message.c_str());
    return 1;
  }
  tokshit::Gguf gguf = std::move(opened.value());
  auto mapped = gguf.mmap_file();
  if (!mapped.ok()) {
    std::fprintf(stderr, "mmap: %s\n", mapped.error().message.c_str());
    return 1;
  }
  tokshit::Mmap mm = std::move(mapped.value());
  const auto bytes = mm.bytes();
  const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
  const std::size_t start = static_cast<std::size_t>(gguf.data_offset());
  const std::size_t len =
      std::min(region_mb * 1024 * 1024, bytes.size() - start);
  const std::span<const uint8_t> region(data + start, len);
  const std::size_t threads =
      std::max<std::size_t>(1, std::thread::hardware_concurrency());

  // Warm: pull pages into the page cache.
  const auto t0 = tokshit::Clock::now();
  const uint64_t warm = sum_words(region);
  const double warm_dt =
      std::chrono::duration<double>(tokshit::Clock::now() - t0).count();
  std::fprintf(stderr,
               "warm: %.2f GB/s (%.0f ms for %zu MB, %zu threads, chk=%llu)\n",
               static_cast<double>(len) / warm_dt / 1e9, warm_dt * 1e3,
               len / (1024 * 1024), threads,
               static_cast<unsigned long long>(warm));

  const std::size_t chunk = len / threads;
  for (std::size_t it = 0; it < iters; ++it) {
    const auto ti = tokshit::Clock::now();
    std::vector<uint64_t> partial(threads, 0);
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (std::size_t tno = 0; tno < threads; ++tno) {
      const std::size_t begin = tno * chunk;
      const std::size_t end = tno + 1 == threads ? len : (tno + 1) * chunk;
      workers.emplace_back([&region, &partial, tno, begin, end]() {
        partial[tno] = sum_words(region.subspan(begin, end - begin));
      });
    }
    uint64_t acc = 0;
    for (std::thread& worker : workers) worker.join();
    for (uint64_t value : partial) acc += value;
    const double dt =
        std::chrono::duration<double>(tokshit::Clock::now() - ti).count();
    std::fprintf(stderr, "iter %zu: %.2f GB/s (%.1f ms, checksum %llu)\n", it,
                 static_cast<double>(len) / dt / 1e9, dt * 1e3,
                 static_cast<unsigned long long>(acc));
  }
  return 0;
}
