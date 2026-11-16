#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace tokshit {

// Stable error codes. Messages are free-form and only live on the cold path.
enum class ErrCode : uint8_t {
  Truncated,
  BadMagic,
  UnsupportedVersion,
  UnsupportedQuant,
  QuantBlockMismatch,
  TensorSizeMismatch,
  Io,
  MissingTensor,
  InvalidField,
  InvalidValue,
  Overflow,
  OutOfBounds,
};

[[nodiscard]] constexpr std::string_view to_string(ErrCode code) noexcept {
  switch (code) {
    case ErrCode::Truncated:
      return "Truncated";
    case ErrCode::BadMagic:
      return "BadMagic";
    case ErrCode::UnsupportedVersion:
      return "UnsupportedVersion";
    case ErrCode::UnsupportedQuant:
      return "UnsupportedQuant";
    case ErrCode::QuantBlockMismatch:
      return "QuantBlockMismatch";
    case ErrCode::TensorSizeMismatch:
      return "TensorSizeMismatch";
    case ErrCode::Io:
      return "Io";
    case ErrCode::MissingTensor:
      return "MissingTensor";
    case ErrCode::InvalidField:
      return "InvalidField";
    case ErrCode::InvalidValue:
      return "InvalidValue";
    case ErrCode::Overflow:
      return "Overflow";
    case ErrCode::OutOfBounds:
      return "OutOfBounds";
  }
  return "Unknown";
}

struct Error {
  ErrCode code = ErrCode::InvalidValue;
  std::string field;
  std::string message;

  Error() = default;

  Error(ErrCode c, std::string f = {}, std::string m = {})
      : code(c), field(std::move(f)), message(std::move(m)) {}
};

[[nodiscard]] inline bool operator==(const Error& a, const Error& b) {
  return a.code == b.code && a.field == b.field && a.message == b.message;
}

[[nodiscard]] inline bool operator!=(const Error& a, const Error& b) {
  return !(a == b);
}

[[nodiscard]] inline Error make_error(ErrCode code, std::string field = {},
                                      std::string message = {}) {
  return Error{code, std::move(field), std::move(message)};
}

}  // namespace tokshit
