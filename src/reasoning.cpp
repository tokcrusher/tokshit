#include "tokshit/reasoning.hpp"

namespace tokshit {

ReasoningBudget::ReasoningBudget(int budget) noexcept
    // отрицательные значения кроме -1 трактуем как «без лимита»
    : budget_(budget < 0 ? -1 : budget) {}

void ReasoningBudget::reset() noexcept {
  in_think_ = false;
  used_ = 0;
}

void ReasoningBudget::observe(std::uint32_t token) noexcept {
  if (token == kThinkOpen) {
    if (!in_think_) {
      // открывающий токен запускает блок, но в бюджет не входит
      in_think_ = true;
      used_ = 0;
    } else {
      // вложенный open — обычный токен внутри блока
      ++used_;
    }
    return;
  }
  if (token == kThinkClose) {
    // счётчик не трогаем: used() показывает, сколько токенов было в
    // последнем блоке; новый kThinkOpen сбросит его
    in_think_ = false;
    return;
  }
  if (in_think_) ++used_;
}

std::uint32_t ReasoningBudget::constrain(std::uint32_t sampled) const noexcept {
  if (in_think_ && budget_ >= 0 &&
      used_ >= static_cast<std::uint32_t>(budget_)) {
    return kThinkClose;
  }
  return sampled;
}

}  // namespace tokshit
