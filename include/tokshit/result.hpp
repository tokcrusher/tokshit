#pragma once

#include <utility>
#include <variant>

#include "tokshit/error.hpp"

namespace tokshit {

// аналог rust Result<T, Error>, без исключений на горячем пути
//
// `Result<T>` специально маленький: внутри либо значение либо `Error`.
// парсер/загрузчик могут строить длинные тексты ошибок, а вычислительный путь
// ошибок вообще не создаёт
// rust-like Result<T, Error> without exceptions on the hot path.
// Result<T> is intentionally small, it owns either a value or an Error.
// loader/parser may build long strings, the compute path never errors
template <class T>
class [[nodiscard]] Result {
 public:
  using value_type = T;

  Result(T value) : v_(std::move(value)) {}
  Result(Error error) : v_(std::move(error)) {}

  [[nodiscard]] bool ok() const noexcept { return v_.index() == 0; }
  explicit operator bool() const noexcept { return ok(); }

  T& value() & { return std::get<0>(v_); }
  const T& value() const& { return std::get<0>(v_); }
  T&& value() && { return std::get<0>(std::move(v_)); }

  Error& error() & { return std::get<1>(v_); }
  const Error& error() const& { return std::get<1>(v_); }
  Error&& error() && { return std::get<1>(std::move(v_)); }

 private:
  std::variant<T, Error> v_;
};

// `Result<void>` для операций где нужно только успех/провал
// `Result<void>` for operations that only need success/failure
template <>
class [[nodiscard]] Result<void> {
 public:
  Result() noexcept = default;
  Result(Error error) : error_(std::move(error)), ok_(false) {}

  [[nodiscard]] bool ok() const noexcept { return ok_; }
  explicit operator bool() const noexcept { return ok_; }

  Error& error() & { return error_; }
  const Error& error() const& { return error_; }

 private:
  Error error_{};
  bool ok_ = true;
};

}  // namespace tokshit

// выкинуть ошибку из функции возвращающей `tokshit::Result<...>`
//
//   TOKSHIT_TRY(layout, layout_for(quant, n));
//
// объявляет `layout` при успехе и возвращает `error()` иначе. имя временной
// переменной берётся из `__LINE__`, так что несколько штук в одном скопе можно
// propagate an error out of a fn returning Result<...>. declares var on
// success, returns error() otherwise. name comes from __LINE__, so several in
// one scope are fine
#define TOKSHIT_DETAIL_CONCAT_IMPL(a, b) a##b
#define TOKSHIT_DETAIL_CONCAT(a, b) TOKSHIT_DETAIL_CONCAT_IMPL(a, b)

#define TOKSHIT_TRY(var, expr)                                          \
  auto&& TOKSHIT_DETAIL_CONCAT(_tokshit_tmp_, __LINE__) = (expr);       \
  if (!TOKSHIT_DETAIL_CONCAT(_tokshit_tmp_, __LINE__).ok())             \
    return TOKSHIT_DETAIL_CONCAT(_tokshit_tmp_, __LINE__).error();      \
  auto var = std::move(TOKSHIT_DETAIL_CONCAT(_tokshit_tmp_, __LINE__).value())

// то же самое но значение на успехе выбрасываем. функция может возвращать
// любой `tokshit::Result<...>`
// same but discards the value on success, enclosing fn can return any Result
#define TOKSHIT_TRY_VOID(expr)                                          \
  do {                                                                  \
    auto&& _tokshit_void_tmp_ = (expr);                                 \
    if (!_tokshit_void_tmp_.ok()) return _tokshit_void_tmp_.error();    \
  } while (0)
