// Check: dot_row (raw quantized bytes) vs scalar dot over dequant_into.
// usage: vcheck <gguf> <tensor>...

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/gguf.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/quant.hpp"

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : std::string();
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
  const uint8_t* data = reinterpret_cast<const uint8_t*>(bytes.data());
  const std::size_t base_off = static_cast<std::size_t>(gguf.data_offset());

  for (int a = 2; a < argc; ++a) {
    const std::string name = argv[a];
    auto tensor = gguf.tensor(name);
    if (!tensor.ok()) {
      std::fprintf(stderr, "skip %s: %s\n", name.c_str(),
                   tensor.error().message.c_str());
      continue;
    }
    const tokshit::GgufTensor& t = *tensor.value();
    const std::size_t in_dim = static_cast<std::size_t>(t.dims.empty() ? 0 : t.dims[0]);
    auto rb_result = tokshit::row_bytes(t.quant, in_dim);
    if (!rb_result.ok()) {
      std::fprintf(stderr, "skip %s: %s\n", name.c_str(),
                   rb_result.error().message.c_str());
      continue;
    }
    const std::size_t rb = static_cast<std::size_t>(rb_result.value());
    const std::size_t base = base_off + static_cast<std::size_t>(t.offset);

    std::vector<float> x(in_dim);
    for (std::size_t i = 0; i < in_dim; ++i) {
      x[i] = (static_cast<float>((i * 37 + 11) % 101) - 50.0F) / 50.0F;
    }
    const auto act = tokshit::kernels::quantize(x);
    std::vector<float> scratch;
    std::size_t bad = 0;
    float worst = 0.0F;

    std::size_t nrows = 1;
    for (std::size_t i = 1; i < t.dims.size(); ++i) {
      nrows *= static_cast<std::size_t>(t.dims[i]);
    }
    const std::size_t rows = std::min<std::size_t>(nrows, 8);
    for (std::size_t r = 0; r < rows; ++r) {
      const auto row =
          std::span<const uint8_t>(data + base + r * rb, rb);
      const float kernel = tokshit::kernels::dot_row(
          t.quant, row, rb, x, act, in_dim, scratch);
      std::vector<float> w(in_dim);
      auto deq = tokshit::dequant_into(t.quant, row, in_dim, w);
      if (!deq.ok()) {
        std::fprintf(stderr, "skip %s: %s\n", name.c_str(),
                     deq.error().message.c_str());
        break;
      }
      float scalar = 0.0F;
      for (std::size_t i = 0; i < in_dim; ++i) scalar += w[i] * x[i];
      const float denom = std::max({std::fabs(kernel), std::fabs(scalar), 1e-6F});
      const float rel = std::fabs(kernel - scalar) / denom;
      if (rel > 1e-3F) ++bad;
      worst = std::max(worst, rel);
      if (r < 3) {
        std::printf("%s row%zu kernel=%.6f dequant=%.6f rel=%.4f\n", name.c_str(),
                    r, static_cast<double>(kernel), static_cast<double>(scalar),
                    static_cast<double>(rel));
      }
    }
    std::printf("%s %s rows_checked=%zu bad=%zu worst_rel=%.4f\n", name.c_str(),
                std::string(tokshit::quant_name(t.quant)).c_str(), rows, bad,
                static_cast<double>(worst));
  }
  return 0;
}
