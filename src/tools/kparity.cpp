// debug: compare dot_rows 4-row kernels against per-row dot_row
// usage: kparity

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/quant.hpp"

using tokshit::QuantType;

namespace {
uint32_t lcg(uint64_t& state) {
  state = state * 6364136223846793005ull + 1ull;
  return static_cast<uint32_t>(state >> 33);
}

double check(QuantType quant, size_t bs, size_t in_dim, int rows) {
  uint64_t state = 0x123456789ABCDEF0ull;
  std::vector<float> x(in_dim);
  for (float& v : x) v = (static_cast<float>(lcg(state)) / UINT32_MAX - 0.5F) * 4.0F;
  const auto act = tokshit::kernels::quantize(x);
  const auto rb = tokshit::row_bytes(quant, in_dim).value();
  std::vector<uint8_t> blob(rows * rb);
  for (uint8_t& b : blob) b = static_cast<uint8_t>(lcg(state));
  // varying per-block d
  size_t blk = 0;
  for (size_t off = 0; off + bs <= blob.size(); off += bs, ++blk) {
    const uint16_t bits = tokshit::f32_to_f16(1.0F + 0.25F * (blk % 5));
    blob[off] = static_cast<uint8_t>(bits & 0xFFU);
    blob[off + 1] = static_cast<uint8_t>(bits >> 8);
    if (quant == QuantType::Q4K) { blob[off + 2] = 0; blob[off + 3] = 0x3C; }
    if (quant == QuantType::Mxfp4) { blob[off] = 0; }  // e8m0 at scale
  }
  std::vector<float> scratch;
  std::vector<float> per_row(rows), bulk(rows), truth(rows);
  for (int r = 0; r < rows; ++r) {
    per_row[r] = tokshit::kernels::dot_row(
        quant, std::span<const uint8_t>(blob.data() + r * rb, rb), rb, x, act,
        in_dim, scratch);
    // ground truth: dequant + scalar dot on the same activation
    auto w = tokshit::dequant(quant, std::span<const uint8_t>(blob.data() + r * rb, rb),
                              static_cast<uint64_t>(in_dim));
    float s = 0.0F;
    for (size_t i = 0; i < in_dim; ++i)
      s += w.value()[i] * (static_cast<float>(act.q[i]) * act.scales[i / 32]);
    truth[r] = s;
  }
  tokshit::kernels::dot_rows(quant, blob, rb, rows, x, act, in_dim,
                             bulk.data(), scratch);
  double worst = 0, worst_truth = 0;
  for (int r = 0; r < rows; ++r) {
    const double denom = std::max(static_cast<double>(std::fabs(per_row[r])), 1.0);
    const double d = std::fabs(per_row[r] - bulk[r]) / denom;
    worst = std::max(worst, d);
    const double dt = std::fabs(truth[r] - bulk[r]) / std::max(static_cast<double>(std::fabs(truth[r])), 1.0);
    worst_truth = std::max(worst_truth, dt);
  }
  std::printf("   per_row[0]=%.4f bulk[0]=%.4f truth[0]=%.4f\n", per_row[0],
              bulk[0], truth[0]);
  return std::max(worst, worst_truth);
}
}  // namespace

int main() {
  for (auto [q, bs, id] : {
           std::tuple{QuantType::Iq2S, 82, 512},
           std::tuple{QuantType::Iq2Xxs, 66, 512},
           std::tuple{QuantType::Iq2Xs, 74, 512},
           std::tuple{QuantType::Iq1S, 50, 512},
           std::tuple{QuantType::Mxfp4, 17, 512},
           std::tuple{QuantType::Nvfp4, 36, 512},
           std::tuple{QuantType::Q2_0, 18, 512},
           std::tuple{QuantType::Q6K, 210, 512},
       }) {
    std::printf("%s bs=%zu worst=%.4f\n", tokshit::quant_name(q).data(), bs,
                check(q, bs, id, 9));
  }
  return 0;
}
