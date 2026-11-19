#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "tokshit/mmap.hpp"
#include "tokshit/quant.hpp"
#include "tokshit/result.hpp"

namespace tokshit {

// One GGUF metadata value. Mirrors the GGUF `Value` type. Arrays are shared to
// keep `Value` copyable while still allowing recursion.
class Value {
 public:
  enum class Type : uint8_t {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
  };

  using Array = std::vector<Value>;

  Value() = default;
  explicit Value(uint8_t v) : v_(v) {}
  explicit Value(int8_t v) : v_(v) {}
  explicit Value(uint16_t v) : v_(v) {}
  explicit Value(int16_t v) : v_(v) {}
  explicit Value(uint32_t v) : v_(v) {}
  explicit Value(int32_t v) : v_(v) {}
  explicit Value(float v) : v_(v) {}
  explicit Value(bool v) : v_(v) {}
  explicit Value(std::string v) : v_(std::move(v)) {}
  explicit Value(std::shared_ptr<const Array> v) : v_(std::move(v)) {}
  explicit Value(uint64_t v) : v_(v) {}
  explicit Value(int64_t v) : v_(v) {}
  explicit Value(double v) : v_(v) {}

  [[nodiscard]] Type type() const noexcept {
    return static_cast<Type>(v_.index());
  }

  [[nodiscard]] std::optional<uint64_t> as_u64() const noexcept;
  [[nodiscard]] std::optional<float> as_f32() const noexcept;
  [[nodiscard]] std::optional<double> as_f64() const noexcept;
  [[nodiscard]] std::optional<bool> as_bool() const noexcept;
  [[nodiscard]] std::optional<std::string_view> as_str() const noexcept;
  [[nodiscard]] const Array* as_array() const noexcept;

 private:
  using Storage =
      std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
                   float, bool, std::string, std::shared_ptr<const Array>,
                   uint64_t, int64_t, double>;
  Storage v_{};
};

// Description of one tensor from the GGUF header.
struct GgufTensor {
  std::string name;
  // GGUF order: dims[0] (ne0) is the contiguous axis (matvec input), rows
  // follow ne1, then ne2 (experts).
  std::vector<uint64_t> dims;
  QuantType quant = QuantType::F32;
  uint64_t offset = 0;
  uint64_t byte_len = 0;
};

struct TransparentStringHash {
  using is_transparent = void;
  [[nodiscard]] size_t operator()(std::string_view s) const noexcept {
    return std::hash<std::string_view>{}(s);
  }
  [[nodiscard]] size_t operator()(const std::string& s) const noexcept {
    return std::hash<std::string_view>{}(s);
  }
  [[nodiscard]] size_t operator()(const char* s) const noexcept {
    return std::hash<std::string_view>{}(s);
  }
};

struct TransparentStringEq {
  using is_transparent = void;
  [[nodiscard]] bool operator()(std::string_view a,
                                std::string_view b) const noexcept {
    return a == b;
  }
};

using MetadataMap =
    std::unordered_map<std::string, Value, TransparentStringHash,
                       TransparentStringEq>;

// сырое чтение по абсолютному смещению. без Gguf, чтобы фоновые читатели
// экспертов не держали ссылку на объект, который может переехать при move
[[nodiscard]] Result<void> pread_at(int fd, uint64_t offset,
                                    std::span<uint8_t> dst);

// POSIX_FADV_DONTNEED на диапазон, best effort
void fadvise_dontneed(int fd, uint64_t offset, uint64_t len) noexcept;

// GGUF v2/v3 reader. Owns a read-only file descriptor; tensor data is fetched
// on demand with `pread` (zero-copy `mmap` is available separately).
class Gguf {
 public:
  Gguf() = default;
  ~Gguf();

  Gguf(const Gguf&) = delete;
  Gguf& operator=(const Gguf&) = delete;
  Gguf(Gguf&& other) noexcept;
  Gguf& operator=(Gguf&& other) noexcept;

  [[nodiscard]] static Result<Gguf> open(const std::string& path);

  [[nodiscard]] int fd() const noexcept { return fd_; }
  [[nodiscard]] uint64_t file_len() const noexcept { return file_len_; }
  [[nodiscard]] uint64_t data_offset() const noexcept { return data_offset_; }
  [[nodiscard]] const std::vector<GgufTensor>& tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] const MetadataMap& metadata() const noexcept {
    return metadata_;
  }

  // Map the whole file read-only. The mapping stays valid while this `Gguf`
  // is open (the underlying fd may be closed after mapping on POSIX).
  [[nodiscard]] Result<Mmap> mmap_file() const;

  [[nodiscard]] Result<const GgufTensor*> tensor(
      std::string_view name) const;

  [[nodiscard]] Result<std::vector<uint8_t>> read_tensor(
      std::string_view name) const;

  [[nodiscard]] Result<void> read_into(std::string_view name,
                                       uint64_t byte_offset,
                                       std::span<uint8_t> buf) const;

  // сырое чтение по абсолютному смещению. кэш экспертов читает прямо в свой
  // буфер, а не через page fault на mmap
  [[nodiscard]] Result<void> read_at(uint64_t offset,
                                     std::span<uint8_t> dst) const;

  // роняем page cache для диапазона (POSIX_FADV_DONTNEED). нужно чтобы поток
  // экспертов не вытеснял неэкспертные веса
  void fadvise_dontneed(uint64_t offset, uint64_t len) const noexcept;

  [[nodiscard]] Result<std::vector<uint8_t>> read_tensor_slice(
      std::string_view name, uint64_t byte_offset, uint64_t byte_len) const;

  [[nodiscard]] Result<const Value*> get(std::string_view key) const;
  [[nodiscard]] Result<uint64_t> u64_kv(std::string_view key) const;
  [[nodiscard]] Result<float> f32_kv(std::string_view key) const;
  [[nodiscard]] Result<std::vector<std::string>> strings(
      std::string_view key) const;
  [[nodiscard]] Result<std::vector<uint32_t>> u32s(std::string_view key) const;

 private:
  void reset() noexcept;

  int fd_ = -1;
  uint64_t file_len_ = 0;
  uint64_t data_offset_ = 0;
  std::vector<GgufTensor> tensors_;
  MetadataMap metadata_;
  std::unordered_map<std::string, size_t, TransparentStringHash,
                     TransparentStringEq>
      index_;
};

}  // namespace tokshit
