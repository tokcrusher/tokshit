#pragma once

#include <cstdint>

namespace tokshit {

// Спецтокены Flint. Значения фиксированы моделью и совпадают в chat,
// speculative и reasoning-путях.
inline constexpr std::uint32_t kUser = 150003;
inline constexpr std::uint32_t kAssistant = 150004;
inline constexpr std::uint32_t kThinkOpen = 150006;
inline constexpr std::uint32_t kThinkClose = 150007;

}  // namespace tokshit
