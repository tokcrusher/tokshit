#include "tokshit/mmap.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#ifndef MADV_COLD
#define MADV_COLD 20
#endif
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 36
#endif

namespace tokshit {
namespace {

[[nodiscard]] Error errno_error(ErrCode code, std::string field) {
  return make_error(code, std::move(field), std::strerror(errno));
}

[[nodiscard]] bool valid_range(size_t off, size_t len, size_t total) noexcept {
  return off <= total && len <= total - off;
}

}  // namespace

Mmap::~Mmap() { reset(); }

Mmap::Mmap(Mmap&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)),
      len_(std::exchange(other.len_, 0)) {}

Mmap& Mmap::operator=(Mmap&& other) noexcept {
  if (this != &other) {
    reset();
    ptr_ = std::exchange(other.ptr_, nullptr);
    len_ = std::exchange(other.len_, 0);
  }
  return *this;
}

void Mmap::reset() noexcept {
  if (ptr_ != nullptr) {
    ::munmap(ptr_, len_);
    ptr_ = nullptr;
    len_ = 0;
  }
}

// мапим файл целиком, read-only. ошибки маппинга возвращаем как обычно
Result<Mmap> Mmap::map_file(int fd, uint64_t len) {
  if (len == 0) {
    return make_error(ErrCode::InvalidValue, "mmap.len",
                      "cannot map an empty file");
  }
  if (len > std::numeric_limits<size_t>::max()) {
    return make_error(ErrCode::Overflow, "mmap.len",
                      "file is larger than address space");
  }
  if (fd < 0) {
    return make_error(ErrCode::InvalidValue, "mmap.fd", "invalid file descriptor");
  }

  void* ptr = ::mmap(nullptr, static_cast<size_t>(len), PROT_READ, MAP_PRIVATE,
                     fd, 0);
  if (ptr == MAP_FAILED) {
    return errno_error(ErrCode::Io, "mmap");
  }
  return Mmap(static_cast<std::byte*>(ptr), static_cast<size_t>(len));
}

std::span<const std::byte> Mmap::bytes() const noexcept {
  return {ptr_, len_};
}

void Mmap::advise_willneed() const noexcept {
  if (ptr_ == nullptr || len_ == 0) return;
  ::madvise(ptr_, len_, MADV_WILLNEED);
}

void Mmap::advise_range(size_t off, size_t len) const noexcept {
  if (ptr_ == nullptr || len == 0 || !valid_range(off, len, len_)) return;
  ::madvise(ptr_ + off, len, MADV_WILLNEED);
}

void Mmap::advise_cold(size_t off, size_t len) const noexcept {
  if (ptr_ == nullptr || len == 0 || !valid_range(off, len, len_)) return;
  ::madvise(ptr_ + off, len, MADV_COLD);
}

bool Mmap::lock_range(size_t off, size_t len) const noexcept {
  if (ptr_ == nullptr || len == 0 || !valid_range(off, len, len_)) return false;
  return ::mlock(ptr_ + off, len) == 0;
}

bool Mmap::advise_hugepage() const noexcept {
  if (ptr_ == nullptr || len_ == 0) return false;
  return ::madvise(ptr_, len_, MADV_HUGEPAGE) == 0;
}

}  // namespace tokshit
