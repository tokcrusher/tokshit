// ahbtest: проверка пула expert-кэша.
//
//   ./ahbtest [--mb N] [--blocks N]
//
// Меряет две вещи: что выделенные блоки действительно лежат в AHB (а не в
// malloc-fallback) и какая у них скорость последовательного чтения. Для
// сравнения печатается полоса обычного malloc-региона.

#include "tokshit/ahb.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

double read_gbps(std::vector<uint8_t*> const& blocks, std::size_t bytes,
                 int iters) {
  // прогреваем, чтобы не мерить page faults на первом проходе
  volatile uint64_t sink = 0;
  for (auto* p : blocks) {
    const auto* q = reinterpret_cast<const uint64_t*>(p);
    uint64_t acc = 0;
    for (std::size_t k = 0; k < bytes / 8; ++k) acc += q[k];
    sink = sink + acc;
  }
  const auto t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    for (auto* p : blocks) {
      const auto* q = reinterpret_cast<const uint64_t*>(p);
      uint64_t acc = 0;
      for (std::size_t k = 0; k < bytes / 8; ++k) acc += q[k];
      asm volatile("" :: "r"(acc) : "memory");
    }
  }
  const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
  const double total = static_cast<double>(blocks.size()) * bytes * iters;
  (void)sink;
  return dt > 0 ? total / dt / 1e9 : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace tokshit::ahb;
  std::size_t mb = 1024;
  std::size_t blocks = 256;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--mb") == 0) {
      mb = std::strtoull(argv[i + 1], nullptr, 10);
    } else if (std::strcmp(argv[i], "--blocks") == 0) {
      blocks = std::strtoull(argv[i + 1], nullptr, 10);
    }
  }

  Config cfg;
  cfg.slot_sizes = {328 * 1024, 576 * 1024};
  cfg.slot_bytes = 576 * 1024;
  cfg.slab_bytes = 128 * 1024 * 1024;
  cfg.total_bytes = mb * 1024 * 1024;
  Pool::instance().init(cfg);

  std::printf("[test] active=%d classes=%zu slots=%zu free=%zu used=%zu MB\n",
              Pool::instance().ahb_active(), Pool::instance().class_count(),
              Pool::instance().slots_total(), Pool::instance().slots_free(),
              Pool::instance().bytes_used() / (1024 * 1024));

  std::vector<std::unique_ptr<uint8_t[]>> malloc_blocks;
  std::vector<uint8_t*> ahb_blocks;
  std::vector<std::size_t> sizes;
  std::size_t live = 0;
  for (std::size_t i = 0; i < blocks; ++i) {
    const std::size_t size = (i % 2 == 0) ? 328 * 1024 : 576 * 1024;
    uint8_t* p = Pool::instance().allocate(size);
    if (p == nullptr) {
      std::printf("[test] allocate failed at i=%zu\n", i);
      break;
    }
    std::memset(p, static_cast<int>(i & 0xFF), size);
    ahb_blocks.push_back(p);
    sizes.push_back(size);
    if (Pool::instance().is_ahb(p)) ++live;
  }
  std::printf("[test] allocated %zu blocks, %zu really on AHB\n",
              ahb_blocks.size(), live);

  for (std::size_t i = 0; i < 64; ++i) {
    auto p = std::make_unique<uint8_t[]>(328 * 1024);
    std::memset(p.get(), 1, 328 * 1024);
    malloc_blocks.push_back(std::move(p));
  }

  std::vector<uint8_t*> small;
  for (std::size_t i = 0; i < ahb_blocks.size(); ++i) {
    if (sizes[i] == 328 * 1024) small.push_back(ahb_blocks[i]);
  }
  if (!small.empty()) {
    std::printf("[test] AHB   328K read: %.2f GB/s\n",
                read_gbps(small, 328 * 1024, 20));
  }
  std::vector<uint8_t*> plain;
  for (auto& p : malloc_blocks) plain.push_back(p.get());
  if (!plain.empty()) {
    std::printf("[test] malloc 328K read: %.2f GB/s\n",
                read_gbps(plain, 328 * 1024, 20));
  }

  for (std::size_t i = 0; i < ahb_blocks.size(); ++i) {
    Pool::instance().deallocate(ahb_blocks[i], sizes[i]);
  }
  std::printf("[test] after dealloc: free=%zu used=%zu MB\n",
              Pool::instance().slots_free(),
              Pool::instance().bytes_used() / (1024 * 1024));
  return 0;
}
