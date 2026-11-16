#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "tokshit/error.hpp"
#include "tokshit/result.hpp"

namespace tokshit {

// Geometry of one quantization block: how many weights it covers and how many
// bytes it occupies.
struct BlockSpec {
  uint32_t weights = 0;
  uint32_t bytes = 0;

  constexpr BlockSpec() = default;
  constexpr BlockSpec(uint32_t w, uint32_t b) : weights(w), bytes(b) {}

  friend constexpr bool operator==(BlockSpec, BlockSpec) = default;
};

// GGML-compatible quant codes (0..30) plus the tokshit group=64 v2 codes
// (100..102). The numeric values are part of the GGUF file format and must not
// be renumbered.
enum class QuantType : uint16_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2K = 10,
  Q3K = 11,
  Q4K = 12,
  Q5K = 13,
  Q6K = 14,
  Q8K = 15,
  Iq2Xxs = 16,
  Iq2Xs = 17,
  Iq3Xxs = 18,
  Iq1S = 19,
  Iq4Nl = 20,
  Iq3S = 21,
  Iq2S = 22,
  Iq4Xs = 23,
  Iq1M = 29,
  Bf16 = 30,
  TokshitQ4K64 = 100,
  TokshitQ5K64 = 101,
  TokshitQ6K64 = 102,
};

[[nodiscard]] constexpr uint16_t quant_code(QuantType quant) noexcept {
  return static_cast<uint16_t>(quant);
}

// Debug/statistics name, matching Rust's `{:?}` output.
[[nodiscard]] constexpr std::string_view quant_name(QuantType quant) noexcept {
  switch (quant) {
    case QuantType::F32:
      return "F32";
    case QuantType::F16:
      return "F16";
    case QuantType::Q4_0:
      return "Q4_0";
    case QuantType::Q4_1:
      return "Q4_1";
    case QuantType::Q5_0:
      return "Q5_0";
    case QuantType::Q5_1:
      return "Q5_1";
    case QuantType::Q8_0:
      return "Q8_0";
    case QuantType::Q8_1:
      return "Q8_1";
    case QuantType::Q2K:
      return "Q2K";
    case QuantType::Q3K:
      return "Q3K";
    case QuantType::Q4K:
      return "Q4K";
    case QuantType::Q5K:
      return "Q5K";
    case QuantType::Q6K:
      return "Q6K";
    case QuantType::Q8K:
      return "Q8K";
    case QuantType::Iq2Xxs:
      return "Iq2Xxs";
    case QuantType::Iq2Xs:
      return "Iq2Xs";
    case QuantType::Iq3Xxs:
      return "Iq3Xxs";
    case QuantType::Iq1S:
      return "Iq1S";
    case QuantType::Iq4Nl:
      return "Iq4Nl";
    case QuantType::Iq3S:
      return "Iq3S";
    case QuantType::Iq2S:
      return "Iq2S";
    case QuantType::Iq4Xs:
      return "Iq4Xs";
    case QuantType::Iq1M:
      return "Iq1M";
    case QuantType::Bf16:
      return "Bf16";
    case QuantType::TokshitQ4K64:
      return "TokshitQ4K64";
    case QuantType::TokshitQ5K64:
      return "TokshitQ5K64";
    case QuantType::TokshitQ6K64:
      return "TokshitQ6K64";
  }
  return "Unknown";
}

// Returns `nullopt` for codes this build does not know. The caller turns that
// into `ErrCode::UnsupportedQuant` with the original code in the message.
[[nodiscard]] constexpr std::optional<QuantType> quant_from_code(
    uint16_t code) noexcept {
  switch (code) {
    case 0:
      return QuantType::F32;
    case 1:
      return QuantType::F16;
    case 2:
      return QuantType::Q4_0;
    case 3:
      return QuantType::Q4_1;
    case 6:
      return QuantType::Q5_0;
    case 7:
      return QuantType::Q5_1;
    case 8:
      return QuantType::Q8_0;
    case 9:
      return QuantType::Q8_1;
    case 10:
      return QuantType::Q2K;
    case 11:
      return QuantType::Q3K;
    case 12:
      return QuantType::Q4K;
    case 13:
      return QuantType::Q5K;
    case 14:
      return QuantType::Q6K;
    case 15:
      return QuantType::Q8K;
    case 16:
      return QuantType::Iq2Xxs;
    case 17:
      return QuantType::Iq2Xs;
    case 18:
      return QuantType::Iq3Xxs;
    case 19:
      return QuantType::Iq1S;
    case 20:
      return QuantType::Iq4Nl;
    case 21:
      return QuantType::Iq3S;
    case 22:
      return QuantType::Iq2S;
    case 23:
      return QuantType::Iq4Xs;
    case 29:
      return QuantType::Iq1M;
    case 30:
      return QuantType::Bf16;
    case 100:
      return QuantType::TokshitQ4K64;
    case 101:
      return QuantType::TokshitQ5K64;
    case 102:
      return QuantType::TokshitQ6K64;
    default:
      return std::nullopt;
  }
}

[[nodiscard]] constexpr std::optional<BlockSpec> block_spec(
    QuantType quant) noexcept {
  switch (quant) {
    case QuantType::F32:
      return BlockSpec{1, 4};
    case QuantType::F16:
    case QuantType::Bf16:
      return BlockSpec{1, 2};
    case QuantType::Q4_0:
    case QuantType::Iq4Nl:
      return BlockSpec{32, 18};
    case QuantType::Q4_1:
      return BlockSpec{32, 20};
    case QuantType::Q5_0:
      return BlockSpec{32, 22};
    case QuantType::Q5_1:
      return BlockSpec{32, 24};
    case QuantType::Q8_0:
      return BlockSpec{32, 34};
    case QuantType::Q8_1:
      return BlockSpec{32, 36};
    case QuantType::Q2K:
      return BlockSpec{256, 84};
    case QuantType::Q3K:
      return BlockSpec{256, 110};
    case QuantType::Q4K:
      return BlockSpec{256, 144};
    case QuantType::Q5K:
      return BlockSpec{256, 176};
    case QuantType::Q6K:
      return BlockSpec{256, 210};
    case QuantType::Q8K:
      return BlockSpec{256, 292};
    case QuantType::Iq2Xxs:
      return BlockSpec{256, 66};
    case QuantType::Iq2Xs:
      return BlockSpec{256, 74};
    case QuantType::Iq3Xxs:
      return BlockSpec{256, 98};
    case QuantType::Iq1S:
      return BlockSpec{256, 50};
    case QuantType::Iq3S:
      return BlockSpec{256, 110};
    case QuantType::Iq2S:
      return BlockSpec{256, 82};
    case QuantType::Iq4Xs:
      return BlockSpec{256, 136};
    case QuantType::Iq1M:
      return BlockSpec{256, 56};
    case QuantType::TokshitQ4K64:
      return BlockSpec{64, 36};
    case QuantType::TokshitQ5K64:
      return BlockSpec{64, 44};
    case QuantType::TokshitQ6K64:
      return BlockSpec{64, 52};
  }
  return std::nullopt;
}

[[nodiscard]] constexpr bool is_quantized(QuantType quant) noexcept {
  return quant != QuantType::F32 && quant != QuantType::F16 &&
         quant != QuantType::Bf16;
}

[[nodiscard]] constexpr bool is_tokshit_v2(QuantType quant) noexcept {
  return quant == QuantType::TokshitQ4K64 ||
         quant == QuantType::TokshitQ5K64 ||
         quant == QuantType::TokshitQ6K64;
}

struct QuantLayout {
  uint64_t blocks = 0;
  uint64_t byte_len = 0;
  BlockSpec spec{};

  friend constexpr bool operator==(const QuantLayout&,
                                   const QuantLayout&) = default;
};

// Number of blocks and bytes for `element_count` weights.
[[nodiscard]] Result<QuantLayout> layout_for(QuantType quant,
                                             uint64_t element_count);

// Same, for one row of `row_len` weights.
[[nodiscard]] Result<QuantLayout> row_layout(QuantType quant, uint32_t row_len);

// Validates that `byte_len` matches the expected layout.
[[nodiscard]] Result<QuantLayout> tensor_layout(QuantType quant,
                                                uint64_t element_count,
                                                uint64_t byte_len);

// `row_bytes(quant, n) = n / spec.weights * spec.bytes`, with validation.
[[nodiscard]] Result<uint64_t> row_bytes(QuantType quant, uint64_t n);

}  // namespace tokshit
