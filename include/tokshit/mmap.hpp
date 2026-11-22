#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "tokshit/result.hpp"

namespace tokshit {

// raii-обёртка над read-only маппингом файла. madvise/mlock это best-effort:
// android/termux могут их отклонить, а рантайм должен продолжать работать
class Mmap {
 public:
  Mmap() = default;
  ~Mmap();

  Mmap(const Mmap&) = delete;
  Mmap& operator=(const Mmap&) = delete;
  Mmap(Mmap&& other) noexcept;
  Mmap& operator=(Mmap&& other) noexcept;

  [[nodiscard]] static Result<Mmap> map_file(int fd, uint64_t len);

  [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
  [[nodiscard]] bool empty() const noexcept { return len_ == 0; }
  [[nodiscard]] size_t size() const noexcept { return len_; }

  void advise_willneed() const noexcept;
  void advise_range(size_t off, size_t len) const noexcept;
  void advise_cold(size_t off, size_t len) const noexcept;
  [[nodiscard]] bool lock_range(size_t off, size_t len) const noexcept;
  // MADV_HUGEPAGE: резерв поддержки 2MB-страниц для gather-доступа к
  // экспертам; ошибку не фиксируем, обязательное ограничение.
  [[nodiscard]] bool advise_hugepage() const noexcept;

 private:
  Mmap(std::byte* ptr, size_t len) noexcept : ptr_(ptr), len_(len) {}

  void reset() noexcept;

  std::byte* ptr_ = nullptr;
  size_t len_ = 0;
};

}  // namespace tokshit
