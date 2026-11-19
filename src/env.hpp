#pragma once

#include <cstdint>
#include <cstdlib>

namespace tokshit {

// Флаг из окружения: true для любого непустого значения, кроме "0".
inline bool env_flag(const char* name) noexcept {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' && v[0] != '0';
}

// Число из окружения: пусто или мусор — fallback.
inline uint64_t env_u64(const char* name, uint64_t fallback) noexcept {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  return std::strtoull(v, nullptr, 10);
}

inline std::size_t env_usize(const char* name, std::size_t fallback) noexcept {
  return static_cast<std::size_t>(env_u64(name, fallback));
}

}  // namespace tokshit
