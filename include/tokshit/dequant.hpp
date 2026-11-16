#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "tokshit/quant.hpp"
#include "tokshit/result.hpp"

namespace tokshit {

// IEEE half -> float, matching the reference fast path plus a slow fallback
// for subnormals/NaN/Inf.
[[nodiscard]] float f16_to_f32(uint16_t bits) noexcept;

// float -> IEEE half, round-to-nearest-even.
[[nodiscard]] uint16_t f32_to_f16(float value) noexcept;

// Dequantize `numel` weights into `out` (which must have at least `numel`
// elements). `src` must contain the full quantized block range.
[[nodiscard]] Result<void> dequant_into(QuantType quant,
                                        std::span<const uint8_t> src,
                                        uint64_t numel,
                                        std::span<float> out);

// Allocating convenience wrapper used by cold paths (embedding rows, vcheck).
[[nodiscard]] Result<std::vector<float>> dequant(QuantType quant,
                                                 std::span<const uint8_t> src,
                                                 uint64_t numel);

// Quantize `numel` floats to Q8_0 (block of 32: f16 scale, 32 int8) into `dst`.
// `numel` must be a multiple of 32 and `dst` must hold numel/32*34 bytes.
// Used once at load time to requantize dense weights that have no NEON dot
// kernel (Q5K/Q6K/Bf16) into the format the hot path knows how to run.
[[nodiscard]] Result<void> quantize_q8_0_into(std::span<const float> x,
                                              std::span<uint8_t> dst);

// Quantize `numel` floats to Q4_0 (block of 32: f16 scale, 16 packed nibbles)
// into `dst`. `numel` must be a multiple of 32 and `dst` must hold
// numel/32*18 bytes. Used only for the opt-in Q4 hot-weight turbo mode.
[[nodiscard]] Result<void> quantize_q4_0_into(std::span<const float> x,
                                              std::span<uint8_t> dst);

}  // namespace tokshit
