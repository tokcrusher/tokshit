// Simulates the decode gather pattern: for a synthetic token, compute the
// router logits on layer-0 expert regions (same top-8 rule as the model)
// and then, for `slots` slots (48 by default), read the selected experts'
// regions at random offsets within each expert, without computing. Reports
// delivered BW (bytes / time).
// usage: gather_bwbench <gguf> [iters=50] [slots=48]

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "tokshit/gguf.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/mmap.hpp"
#include "tokshit/quant.hpp"

namespace {

using tokshit::Clock;
using tokshit::QuantType;

struct Region {
  const uint8_t* data = nullptr;
  std::size_t rb = 0;
  std::size_t nrows = 0;
};

[[nodiscard]] std::size_t find_tensor(
    const std::vector<tokshit::GgufTensor>& ts, const std::string& contains) {
  for (std::size_t i = 0; i < ts.size(); ++i)
    if (ts[i].name.find(contains) != std::string::npos) return i;
  return ts.size();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <gguf> [iters=50] [slots=48]\n", argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  const int iters = argc > 2 ? std::atoi(argv[2]) : 50;
  const int slots = argc > 3 ? std::atoi(argv[3]) : 48;
  if (iters <= 0 || slots <= 0) return 2;

  auto opened = tokshit::Gguf::open(path);
  if (!opened.ok()) {
    std::fprintf(stderr, "gguf: %s\n", opened.error().message.c_str());
    return 1;
  }
  tokshit::Gguf gguf = std::move(opened.value());
  auto mapped = gguf.mmap_file();
  if (!mapped.ok()) {
    std::fprintf(stderr, "mmap: %s\n", mapped.error().message.c_str());
    return 1;
  }
  tokshit::Mmap mm = std::move(mapped.value());
  const uint8_t* base = reinterpret_cast<const uint8_t*>(mm.bytes().data());
  const std::size_t base_off = static_cast<std::size_t>(gguf.data_offset());
  const auto& tensors = gguf.tensors();

  // ---- router: blk.0.ffn_gate_inp + all expert rows
  const std::size_t ri = find_tensor(tensors, "ffn_gate_inp");
  if (ri == tensors.size()) {
    std::fprintf(stderr, "router tensor not found\n");
    return 1;
  }
  const tokshit::GgufTensor& router = tensors[ri];
  const std::size_t rin = static_cast<std::size_t>(router.dims[0]);
  std::size_t n_experts = 1;
  for (std::size_t i = 1; i < router.dims.size(); ++i)
    n_experts *= static_cast<std::size_t>(router.dims[i]);
  const std::size_t rrb = static_cast<std::size_t>(
      tokshit::row_bytes(router.quant, router.dims[0]).value());
  const uint8_t* rdata = base + base_off + router.offset;

  std::vector<float> x(rin);
  for (std::size_t i = 0; i < rin; ++i)
    x[i] = (static_cast<float>((i * 37 + 11) % 101) - 50.0F) / 50.0F;
  const auto act = tokshit::kernels::quantize(std::span<const float>(x));
  thread_local std::vector<float> scratch;
  std::vector<float> logits(n_experts, 0.0F);
  tokshit::kernels::dot_rows(router.quant,
                             std::span<const uint8_t>(rdata, rrb * n_experts),
                             rrb, n_experts, std::span<const float>(x), act,
                             rin, logits.data(), scratch);

  std::vector<std::size_t> top8(8);
  std::iota(top8.begin(), top8.end(), 0);
  std::partial_sort(top8.begin(), top8.begin() + 8, top8.end(),
                    [&](std::size_t a, std::size_t b) {
                      return logits[a] > logits[b];
                    });
  top8.resize(8);

  // ---- regions: gate/up/down exps for layers 0..3
  std::vector<Region> regions;
  for (std::size_t layer = 0; layer < 4; ++layer) {
    for (const char* proj : {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
                             "ffn_gate_up_exps"}) {
      const std::string nm = "blk." + std::to_string(layer) + "." + proj +
                             ".weight";
      const std::size_t ti = find_tensor(tensors, nm);
      if (ti == tensors.size()) continue;
      const tokshit::GgufTensor& t = tensors[ti];
      Region r;
      r.rb = static_cast<std::size_t>(tokshit::row_bytes(t.quant, t.dims[0]).value());
      r.nrows = 1;
      for (std::size_t i = 1; i < t.dims.size(); ++i)
        r.nrows *= static_cast<std::size_t>(t.dims[i]);
      r.data = base + base_off + t.offset;
      regions.push_back(r);
    }
  }
  if (regions.empty()) {
    std::fprintf(stderr, "no expert regions found\n");
    return 1;
  }

  // prefault every region page (steady-state: faults excluded from timing)
  for (const Region& r : regions) {
    const std::size_t region_len = r.rb * r.nrows / n_experts;
    for (std::size_t p = 0; p + 4096 <= region_len; p += 4096) {
      volatile const uint8_t* page = r.data + p;
      (void)*page;
    }
  }
  // ---- simulate slot reads: read a chunk at a random page-aligned offset
  // within a randomly selected expert, without computing
  std::mt19937_64 rng(0xBADC0FFEEULL);
  std::uint64_t sink = 0;
  std::uint64_t total_bytes = 0;
  const std::size_t kChunk = 8192;
  const auto t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    for (int s = 0; s < slots; ++s) {
      const Region& r = regions[static_cast<std::size_t>(rng() % regions.size())];
      const std::size_t e = top8[static_cast<std::size_t>(rng() % top8.size())];
      const std::size_t region_len = r.rb * r.nrows / n_experts;
      const std::size_t off =
          static_cast<std::size_t>(rng() % (region_len / 4096 + 1)) * 4096;
      const std::size_t chunk = std::min(kChunk, region_len - off);
      const uint8_t* p = r.data + e * region_len + off;
      std::uint64_t acc = 0;
      for (std::size_t i = 0; i < chunk; ++i) acc ^= p[i];
      sink ^= acc;
      total_bytes += chunk;
    }
  }
  const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
  const double gb = static_cast<double>(total_bytes) / dt / 1e9;
  std::printf("top8:");
  for (std::size_t e : top8) std::printf(" %zu", e);
  std::printf("\nregions=%zu slots=%d iters=%d\n", regions.size(), slots,
              iters);
  std::printf("delivered: %.2f GB/s (%llu bytes in %.4fs, sink=%llu)\n", gb,
              static_cast<unsigned long long>(total_bytes), dt,
              static_cast<unsigned long long>(sink));
  return 0;
}
