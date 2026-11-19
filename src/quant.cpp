#include "tokshit/quant.hpp"

#include <limits>
#include <string>

namespace tokshit {
namespace {

[[nodiscard]] std::string unsupported_quant_message(uint16_t code) {
  return "unsupported quant code " + std::to_string(code);
}

}  // namespace

// раскладка кванта для n элементов: сколько байт и сколько блоков
Result<QuantLayout> layout_for(QuantType quant, uint64_t element_count) {
  const auto spec = block_spec(quant);
  if (!spec.has_value()) {
    return make_error(ErrCode::UnsupportedQuant, "quant",
                      unsupported_quant_message(quant_code(quant)));
  }
  if (spec->bytes == 0) {
    return make_error(ErrCode::InvalidValue, "quant.block_bytes",
                      "block byte size must be non-zero");
  }

  const uint64_t block_weights = spec->weights;
  if (element_count % block_weights != 0) {
    return make_error(
        ErrCode::QuantBlockMismatch, "element_count",
        "element count " + std::to_string(element_count) +
            " is not a multiple of block weights " +
            std::to_string(block_weights));
  }

  const uint64_t blocks = element_count / block_weights;
  if (blocks > std::numeric_limits<uint64_t>::max() / spec->bytes) {
    return make_error(ErrCode::Overflow, "quant.byte_len",
                      "quantized byte length overflows uint64");
  }

  return QuantLayout{blocks, blocks * spec->bytes, *spec};
}

Result<QuantLayout> row_layout(QuantType quant, uint32_t row_len) {
  return layout_for(quant, row_len);
}

Result<QuantLayout> tensor_layout(QuantType quant, uint64_t element_count,
                                  uint64_t byte_len) {
  TOKSHIT_TRY(layout, layout_for(quant, element_count));
  if (layout.byte_len != byte_len) {
    return make_error(ErrCode::TensorSizeMismatch, "tensor.byte_len",
                      "expected " + std::to_string(layout.byte_len) +
                          " bytes, found " + std::to_string(byte_len));
  }
  return layout;
}

Result<uint64_t> row_bytes(QuantType quant, uint64_t n) {
  TOKSHIT_TRY(layout, layout_for(quant, n));
  return layout.byte_len;
}

}  // namespace tokshit
