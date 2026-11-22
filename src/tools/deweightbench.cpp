// Microbenchmark of per-expert-region GEMV cost: pure dequant time via
// dequant_into, plus dot_row_multi / dot_rows time on an 8-row sample of the
// same region.
// usage: deweightbench <gguf> [iters] [tensor...]
//
// Tensors named on the command line come from the model when present;
// otherwise a synthetic region (8 rows, real-width, random bytes) is used.
// With no names, one expert-shaped region per quant is benchmarked: real
// tensor when the model has one of that quant, synthetic otherwise.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/gguf.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/mmap.hpp"
#include "tokshit/quant.hpp"

namespace {

using tokshit::Clock;
using tokshit::QuantType;

struct Region {
  std::string name;
  const uint8_t* data = nullptr;
  std::size_t rb = 0;
  std::size_t in_dim = 0;
  std::size_t rows = 0;
  QuantType quant = QuantType::F32;
  bool synthetic = false;
};

std::vector<uint8_t> make_random_bytes(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<uint8_t> v(n);
  for (std::size_t i = 0; i < n; i += 8) {
    const std::uint64_t x = rng();
    for (std::size_t k = 0; k < 8 && i + k < n; ++k)
      v[i + k] = static_cast<uint8_t>(x >> (8 * k));
  }
  return v;
}

double us_per_call(const std::function<void()>& fn, int iters) {
  const auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) fn();
  const auto t1 = Clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <gguf> [iters] [tensor...]\n", argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  const int iters = argc > 2 ? std::atoi(argv[2]) : 5;
  std::vector<std::string> names;
  for (int i = 3; i < argc; ++i) names.emplace_back(argv[i]);

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

  std::vector<Region> regions;
  if (!names.empty()) {
    for (const auto& name : names) {
      auto t = gguf.tensor(name);
      if (!t.ok()) {
        std::fprintf(stderr, "skip %s: %s\n", name.c_str(),
                     t.error().message.c_str());
        continue;
      }
      const tokshit::GgufTensor& tt = *t.value();
      Region r;
      r.name = name;
      r.quant = tt.quant;
      r.in_dim = static_cast<std::size_t>(tt.dims.empty() ? 0 : tt.dims[0]);
      r.rows = 1;
      for (std::size_t i = 1; i < tt.dims.size(); ++i)
        r.rows *= static_cast<std::size_t>(tt.dims[i]);
      auto rb = tokshit::row_bytes(tt.quant, tt.dims[0]);
      if (!rb.ok()) {
        std::fprintf(stderr, "skip %s: bad rb\n", name.c_str());
        continue;
      }
      r.rb = static_cast<std::size_t>(rb.value());
      r.data = base + base_off + tt.offset;
      regions.push_back(r);
    }
  } else {
    // benchmark each listed expert quant: real tensor when the model has one
    // of that quant, synthetic region otherwise
    const QuantType want[] = {
        QuantType::Iq2S,   QuantType::Iq2Xxs, QuantType::Iq1S,
        QuantType::Q2_0,   QuantType::Nvfp4,  QuantType::Mxfp4,
        QuantType::Q4K,    QuantType::Q6K,    QuantType::Q4_0,
        QuantType::Iq3S,   QuantType::Q8_0};
    for (QuantType q : want) {
      bool found = false;
      for (const auto* tn : {"blk.0.ffn_gate_exps.weight",
                             "blk.0.ffn_up_exps.weight",
                             "blk.0.ffn_down_exps.weight",
                             "blk.0.ffn_gate_up_exps.weight",
                             "blk.0.ffn_gate.weight"}) {
        auto t = gguf.tensor(tn);
        if (t.ok() && t.value()->quant == q) {
          const tokshit::GgufTensor& tt = *t.value();
          Region r;
          r.name = tt.name;
          r.quant = tt.quant;
          r.in_dim = static_cast<std::size_t>(tt.dims[0]);
          r.rows = tt.dims.size() > 1 ? static_cast<std::size_t>(tt.dims[1]) : 1;
          r.rb = static_cast<std::size_t>(
              tokshit::row_bytes(q, tt.dims[0]).value());
          r.data = base + base_off + tt.offset;
          found = true;
          regions.push_back(r);
          break;
        }
      }
      if (!found) {
        const auto spec = tokshit::block_spec(q);
        if (!spec.has_value()) continue;
        Region r;
        r.name = std::string(tokshit::quant_name(q));
        r.quant = q;
        r.in_dim = 2048;
        r.rows = 8;
        r.rb = 2048 / spec->weights * spec->bytes;
        static std::vector<std::vector<uint8_t>> keep_alive;
        keep_alive.push_back(make_random_bytes(r.rb * 8, static_cast<std::uint64_t>(q)));
        r.data = keep_alive.back().data();
        r.synthetic = true;
        regions.push_back(r);
      }
    }
  }

  std::vector<float> x;
  tokshit::kernels::ActQ8 act;
  for (const Region& r : regions) {
    if (r.in_dim == 0) continue;
    x.assign(r.in_dim, 0.0F);
    for (std::size_t i = 0; i < r.in_dim; ++i)
      x[i] = (static_cast<float>((i * 37 + 11) % 101) - 50.0F) / 50.0F;
    act = tokshit::kernels::quantize(std::span<const float>(x));
    break;
  }

  std::printf("%-16s %8s %8s %8s | %12s %11s | %13s %13s\n", "region", "quant",
              "in_dim", "rows", "dequant us/row", "MB/s", "dot_multi ns",
              "dot_rows ns");

  for (const Region& r : regions) {
    const std::size_t n = r.in_dim;
    if (n == 0) continue;
    // ---- dequant per row of in_dim weights
    std::vector<uint8_t> sample_row =
        r.synthetic ? make_random_bytes(r.rb, 0x5eed) : make_random_bytes(r.rb, 0);
    if (!r.synthetic && r.data != nullptr) {
      std::copy_n(r.data, r.rb, sample_row.begin());
    }
    std::vector<float> dst(n);
    const auto t_deq_us = us_per_call([&] {
      (void)tokshit::dequant_into(r.quant,
                                  std::span<const uint8_t>(sample_row), n, dst);
    }, std::max(1, iters));
    const double deq_mbps =
        n * 4.0 / (t_deq_us > 1e-9 ? t_deq_us : 1.0) * 1e-3;

    // ---- 8-row sample of the same quant
    std::vector<uint8_t> sample(r.rb * 8, 0);
    for (std::size_t j = 0; j < 8 && !r.synthetic; ++j)
      std::copy_n(r.data + j * r.rb, r.rb, sample.data() + j * r.rb);
    if (r.synthetic) {
      const uint8_t* sp = sample_row.data();
      for (std::size_t j = 0; j < 8; ++j)
        std::copy_n(sp, r.rb, sample.data() + j * r.rb);
    }
    const std::span<const uint8_t> rows_span(sample);
    std::vector<tokshit::kernels::ActQ8> acts(8, act);
    std::vector<float> outm(8 * 64, 0.0F);

    const double t_multi_ns =
        us_per_call([&] {
          thread_local std::vector<float> scratch;
          tokshit::kernels::dot_row_multi(r.quant, rows_span, r.rb, acts.data(),
                                          8, n, outm.data(), scratch);
        }, std::max(1, iters)) * 1000.0;
    const double t_rows_ns =
        us_per_call([&] {
          thread_local std::vector<float> scratch;
          tokshit::kernels::dot_rows(r.quant, rows_span, r.rb, 8,
                                     std::span<const float>(x), act, n,
                                     outm.data(), scratch);
        }, std::max(1, iters)) * 1000.0;

    std::printf("%-16s %8s %8zu %8zu | %12.1f %11.1f | %13.1f %13.1f\n",
                r.name.c_str(), std::string(tokshit::quant_name(r.quant)).c_str(),
                n, r.rows, t_deq_us, deq_mbps, t_multi_ns, t_rows_ns);
  }
  return 0;
}
