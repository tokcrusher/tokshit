#pragma once

// runtime-бюджет thinking-токенов: считаем токены внутри <think/> и после
// исчерпания бюджета принудительно закрываем блок

#include <cstdint>

#include "tokshit/special_tokens.hpp"

namespace tokshit {

class ReasoningBudget {
 public:
  explicit ReasoningBudget(int budget = -1) noexcept;

  // сбросить состояние, бюджет сохраняется
  void reset() noexcept;

  // учесть сгенерированный токен; открывающий токен не считается
  void observe(std::uint32_t token) noexcept;

  // вернуть kThinkClose, если бюджет исчерпан, иначе sampled
  [[nodiscard]] std::uint32_t constrain(std::uint32_t sampled) const noexcept;

  [[nodiscard]] bool in_think() const noexcept { return in_think_; }
  [[nodiscard]] std::uint32_t used() const noexcept { return used_; }
  [[nodiscard]] int budget() const noexcept { return budget_; }

 private:
  int budget_ = -1;
  bool in_think_ = false;
  std::uint32_t used_ = 0;
};

}  // namespace tokshit
