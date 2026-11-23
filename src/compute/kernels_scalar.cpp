#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/quant.hpp"
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tokshit::kernels {
namespace {

[[nodiscard]] size_t expected_row_bytes(QuantType quant,
                                        size_t in_dim) noexcept {
  const auto spec = block_spec(quant);
  if (!spec.has_value() || spec->weights == 0) return 0;
  return (in_dim / spec->weights) * spec->bytes;
}

}  // namespace

ActQ8 quantize(std::span<const float> x) {
  ActQ8 act;
  quantize_into(x, act);
  return act;
}

// Симметричный int8 квант блоками по 32: x ~= q * scale, q = max/127.
void quantize_into(std::span<const float> x, ActQ8& act) {
  act.len = x.size();
  act.q.resize(act.len);
  std::fill(act.q.begin(), act.q.end(), 0);
  const size_t nb = (act.len + 31) / 32;
  act.scales.resize(nb);
  std::fill(act.scales.begin(), act.scales.end(), 0.0F);

  for (size_t b = 0; b < nb; ++b) {
    const size_t base = b * 32;
    const size_t n = std::min<size_t>(32, act.len - base);
    const float* xb = x.data() + base;
    float amax = 0.0F;
    for (size_t j = 0; j < n; ++j) {
      const float a = std::fabs(xb[j]);
      if (a > amax) amax = a;
    }
    if (amax == 0.0F || !std::isfinite(amax)) continue;

    const float inv = 127.0F / amax;
    act.scales[b] = amax / 127.0F;
    int8_t* qb = act.q.data() + base;
    size_t j = 0;
#if defined(__aarch64__)
            {
                const float32x4_t iv  = vdupq_n_f32(inv);
                const float32x4_t vlo = vdupq_n_f32(-127.0F);
                const float32x4_t vhi = vdupq_n_f32(127.0F);
                for (; j + 16 <= n; j += 16) {
                    const int32x4_t q0 = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vrndaq_f32(vmulq_f32(vld1q_f32(xb + j),      iv)), vlo), vhi));
                    const int32x4_t q1 = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vrndaq_f32(vmulq_f32(vld1q_f32(xb + j + 4),  iv)), vlo), vhi));
                    const int32x4_t q2 = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vrndaq_f32(vmulq_f32(vld1q_f32(xb + j + 8),  iv)), vlo), vhi));
                    const int32x4_t q3 = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vrndaq_f32(vmulq_f32(vld1q_f32(xb + j + 12), iv)), vlo), vhi));
                    const int16x8_t h0 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
                    const int16x8_t h1 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
                    vst1q_s8(qb + j, vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
                }
            }
#endif
            for (; j < n; ++j) {
                const float v = std::round(xb[j] * inv);
                qb[j] = static_cast<int8_t>(std::clamp(v, -127.0F, 127.0F));
            }
  }
}

// резервный путь: сначала разложить строку в float, потом скалярно умножить.
// медленно, но нужно для тестов и для платформ без neon
float dot_fallback(QuantType quant, std::span<const uint8_t> row, size_t rb,
                   std::span<const float> x, const ActQ8& act, size_t in_dim,
                   std::vector<float>& scratch) {
  (void)x;
  metrics().c_dot_fallback.fetch_add(1, std::memory_order_relaxed);
  const int qi = static_cast<int>(quant);
  if (qi >= 0 && qi < 64) {
    metrics().c_fallback_q[qi].fetch_add(1, std::memory_order_relaxed);
  }
  const size_t expected = expected_row_bytes(quant, in_dim);
  const size_t needed = expected == 0 ? rb : std::min(rb, expected);

  scratch.resize(in_dim);

  if (row.size() >= needed) {
    const auto result = dequant_into(quant, row.first(needed), in_dim, scratch);
    if (!result.ok()) return 0.0F;
  } else {
    std::vector<uint8_t> pad(needed, 0);
    const size_t copy = std::min(row.size(), needed);
    std::copy_n(row.data(), copy, pad.data());
    const auto result = dequant_into(quant, pad, in_dim, scratch);
    if (!result.ok()) return 0.0F;
  }

  float sum = 0.0F;
  for (size_t i = 0; i < in_dim; ++i) {
    const float xq =
        static_cast<float>(act.q[i]) * act.scales[i / 32];
    sum += scratch[i] * xq;
  }
  return sum;
}

}  // namespace tokshit::kernels
