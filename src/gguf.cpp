#include "tokshit/gguf.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace tokshit {
namespace {

constexpr std::string_view kGgufMagic = "GGUF";
constexpr uint64_t kMaxItems = 10'000'000;
constexpr uint64_t kMaxString = 64ull * 1024 * 1024;

// типы значений метадаты gguf (ggml)
// gguf metadata value types (ggml)
constexpr uint32_t kTypeUint8 = 0;
constexpr uint32_t kTypeInt8 = 1;
constexpr uint32_t kTypeUint16 = 2;
constexpr uint32_t kTypeInt16 = 3;
constexpr uint32_t kTypeUint32 = 4;
constexpr uint32_t kTypeInt32 = 5;
constexpr uint32_t kTypeFloat32 = 6;
constexpr uint32_t kTypeBool = 7;
constexpr uint32_t kTypeString = 8;
constexpr uint32_t kTypeArray = 9;
constexpr uint32_t kTypeUint64 = 10;
constexpr uint32_t kTypeInt64 = 11;
constexpr uint32_t kTypeFloat64 = 12;

[[nodiscard]] Error errno_error(ErrCode code, std::string field) {
  return make_error(code, std::move(field), std::strerror(errno));
}

[[nodiscard]] Error truncated_error(std::string field = "gguf.header") {
  return make_error(ErrCode::Truncated, std::move(field),
                    "gguf header truncated");
}

[[nodiscard]] bool valid_utf8(const uint8_t* s, size_t n) noexcept {
  size_t i = 0;
  while (i < n) {
    const uint8_t lead = s[i];
    if (lead < 0x80) {
      ++i;
      continue;
    }

    size_t extra = 0;
    uint32_t cp = 0;
    if ((lead & 0xE0) == 0xC0) {
      extra = 1;
      cp = lead & 0x1Fu;
    } else if ((lead & 0xF0) == 0xE0) {
      extra = 2;
      cp = lead & 0x0Fu;
    } else if ((lead & 0xF8) == 0xF0) {
      extra = 3;
      cp = lead & 0x07u;
    } else {
      return false;
    }

    if (extra >= n - i) return false;
    for (size_t k = 1; k <= extra; ++k) {
      const uint8_t cont = s[i + k];
      if ((cont & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (cont & 0x3Fu);
    }

    constexpr uint32_t kMinCp[4] = {0, 0x80, 0x800, 0x10000};
    if (cp < kMinCp[extra] || cp > 0x10FFFF) return false;
    if (cp >= 0xD800 && cp <= 0xDFFF) return false;
    i += extra + 1;
  }
  return true;
}

[[nodiscard]] uint16_t load_le16(const uint8_t* p) noexcept {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                               (static_cast<uint16_t>(p[1]) << 8));
}

[[nodiscard]] uint32_t load_le32(const uint8_t* p) noexcept {
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

[[nodiscard]] uint64_t load_le64(const uint8_t* p) noexcept {
  return static_cast<uint64_t>(load_le32(p)) |
         (static_cast<uint64_t>(load_le32(p + 4)) << 32);
}

class Reader {
 public:
  Reader(const uint8_t* data, size_t len) noexcept : data_(data), len_(len) {}

  [[nodiscard]] size_t pos() const noexcept { return pos_; }

  [[nodiscard]] bool take(size_t n, const uint8_t** out) noexcept {
    if (n > len_ - pos_) return false;
    *out = data_ + pos_;
    pos_ += n;
    return true;
  }

  [[nodiscard]] bool u32(uint32_t& out) noexcept {
    const uint8_t* p = nullptr;
    if (!take(4, &p)) return false;
    out = load_le32(p);
    return true;
  }

  [[nodiscard]] bool u64(uint64_t& out) noexcept {
    const uint8_t* p = nullptr;
    if (!take(8, &p)) return false;
    out = load_le64(p);
    return true;
  }

  [[nodiscard]] Result<std::string> string() {
    uint64_t n = 0;
    if (!u64(n)) return truncated_error();
    if (n > kMaxString) {
      return make_error(ErrCode::InvalidValue, "gguf.string",
                        "gguf string too long");
    }
    const uint8_t* p = nullptr;
    if (!take(static_cast<size_t>(n), &p)) return truncated_error();
    if (!valid_utf8(p, static_cast<size_t>(n))) {
      return make_error(ErrCode::InvalidValue, "gguf.string",
                        "gguf string not utf-8");
    }
    return std::string(reinterpret_cast<const char*>(p),
                       static_cast<size_t>(n));
  }

  [[nodiscard]] Result<Value> value(uint32_t vtype) {
    const uint8_t* p = nullptr;
    switch (vtype) {
      case kTypeUint8:
        if (!take(1, &p)) return truncated_error();
        return Value{static_cast<uint8_t>(p[0])};
      case kTypeInt8:
        if (!take(1, &p)) return truncated_error();
        return Value{static_cast<int8_t>(p[0])};
      case kTypeUint16:
        if (!take(2, &p)) return truncated_error();
        return Value{load_le16(p)};
      case kTypeInt16:
        if (!take(2, &p)) return truncated_error();
        return Value{static_cast<int16_t>(load_le16(p))};
      case kTypeUint32: {
        uint32_t v = 0;
        if (!u32(v)) return truncated_error();
        return Value{v};
      }
      case kTypeInt32: {
        uint32_t v = 0;
        if (!u32(v)) return truncated_error();
        return Value{static_cast<int32_t>(v)};
      }
      case kTypeFloat32: {
        uint32_t bits = 0;
        if (!u32(bits)) return truncated_error();
        return Value{std::bit_cast<float>(bits)};
      }
      case kTypeBool:
        if (!take(1, &p)) return truncated_error();
        return Value{p[0] != 0};
      case kTypeString: {
        TOKSHIT_TRY(text, string());
        return Value{std::move(text)};
      }
      case kTypeArray: {
        uint32_t elem_type = 0;
        if (!u32(elem_type)) return truncated_error();
        uint64_t n = 0;
        if (!u64(n)) return truncated_error();
        if (n > kMaxItems) {
          return make_error(ErrCode::InvalidValue, "gguf.array",
                            "gguf array too long");
        }
        auto items = std::make_shared<Value::Array>();
        items->reserve(static_cast<size_t>(std::min<uint64_t>(n, 1u << 20)));
        for (uint64_t i = 0; i < n; ++i) {
          TOKSHIT_TRY(item, value(elem_type));
          items->push_back(std::move(item));
        }
        return Value{std::move(items)};
      }
      case kTypeUint64: {
        uint64_t v = 0;
        if (!u64(v)) return truncated_error();
        return Value{v};
      }
      case kTypeInt64: {
        uint64_t v = 0;
        if (!u64(v)) return truncated_error();
        return Value{static_cast<int64_t>(v)};
      }
      case kTypeFloat64: {
        uint64_t bits = 0;
        if (!u64(bits)) return truncated_error();
        return Value{std::bit_cast<double>(bits)};
      }
      default:
        return make_error(ErrCode::InvalidField, "gguf.value_type",
                          "unknown gguf value type " +
                              std::to_string(vtype));
    }
  }

 private:
  const uint8_t* data_ = nullptr;
  size_t len_ = 0;
  size_t pos_ = 0;
};

struct ParsedHeader {
  MetadataMap metadata;
  std::vector<GgufTensor> tensors;
  uint64_t data_offset = 0;
};

[[nodiscard]] Result<ParsedHeader> parse_header(const uint8_t* buf, size_t len,
                                                uint64_t file_len) {
  Reader reader(buf, len);

  const uint8_t* magic = nullptr;
  if (!reader.take(4, &magic)) return truncated_error();
  if (std::memcmp(magic, kGgufMagic.data(), 4) != 0) {
    return make_error(ErrCode::BadMagic, "gguf.magic", "not a GGUF file");
  }

  uint32_t version = 0;
  if (!reader.u32(version)) return truncated_error();
  if (version < 2 || version > 3) {
    return make_error(ErrCode::UnsupportedVersion, "gguf.version",
                      "unsupported GGUF version " + std::to_string(version));
  }

  uint64_t tensor_count = 0;
  uint64_t kv_count = 0;
  if (!reader.u64(tensor_count)) return truncated_error();
  if (!reader.u64(kv_count)) return truncated_error();
  if (tensor_count > kMaxItems || kv_count > kMaxItems) {
    return make_error(ErrCode::InvalidValue, "gguf.count",
                      "implausible gguf counts");
  }

  ParsedHeader out;
  out.metadata.reserve(
      static_cast<size_t>(std::min<uint64_t>(kv_count, 1u << 20)));

  for (uint64_t i = 0; i < kv_count; ++i) {
    TOKSHIT_TRY(key, reader.string());
    uint32_t vtype = 0;
    if (!reader.u32(vtype)) return truncated_error();
    TOKSHIT_TRY(value, reader.value(vtype));
    out.metadata.insert_or_assign(std::move(key), std::move(value));
  }

  out.tensors.reserve(
      static_cast<size_t>(std::min<uint64_t>(tensor_count, 1u << 20)));

  for (uint64_t i = 0; i < tensor_count; ++i) {
    TOKSHIT_TRY(name, reader.string());

    uint32_t n_dims = 0;
    if (!reader.u32(n_dims)) return truncated_error();
    if (n_dims < 1 || n_dims > 4) {
      return make_error(ErrCode::InvalidValue, "gguf.tensor.dims",
                        "tensor `" + name + "` has " +
                            std::to_string(n_dims) + " dimensions");
    }

    GgufTensor tensor;
    tensor.name = std::move(name);
    tensor.dims.reserve(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      uint64_t dim = 0;
      if (!reader.u64(dim)) return truncated_error();
      tensor.dims.push_back(dim);
    }

    uint32_t type_code = 0;
    if (!reader.u32(type_code)) return truncated_error();
    uint64_t offset = 0;
    if (!reader.u64(offset)) return truncated_error();

    const auto quant = quant_from_code(static_cast<uint16_t>(type_code));
    if (!quant.has_value()) {
      return make_error(ErrCode::UnsupportedQuant, "gguf.tensor.quant",
                        "tensor `" + tensor.name +
                            "`: unsupported quant code " +
                            std::to_string(type_code));
    }
    const auto spec = block_spec(*quant);
    if (!spec.has_value()) {
      return make_error(ErrCode::UnsupportedQuant, "gguf.tensor.quant",
                        "tensor `" + tensor.name +
                            "`: unsupported quant code " +
                            std::to_string(type_code));
    }

    uint64_t numel = 1;
    for (const uint64_t dim : tensor.dims) {
      if (dim != 0 && numel > std::numeric_limits<uint64_t>::max() / dim) {
        return make_error(ErrCode::Overflow, "gguf.tensor.numel",
                          "tensor `" + tensor.name +
                              "`: element count overflows uint64");
      }
      numel *= dim;
    }
    if (numel % spec->weights != 0) {
      return make_error(ErrCode::QuantBlockMismatch, "gguf.tensor.numel",
                        "tensor `" + tensor.name + "`: numel " +
                            std::to_string(numel) +
                            " not multiple of " +
                            std::to_string(spec->weights));
    }
    const uint64_t blocks = numel / spec->weights;
    if (blocks > std::numeric_limits<uint64_t>::max() / spec->bytes) {
      return make_error(ErrCode::Overflow, "gguf.tensor.byte_len",
                        "tensor `" + tensor.name +
                            "`: byte length overflows uint64");
    }

    tensor.quant = *quant;
    tensor.offset = offset;
    tensor.byte_len = blocks * spec->bytes;
    out.tensors.push_back(std::move(tensor));
  }

  uint64_t alignment = 32;
  const auto alignment_it = out.metadata.find("general.alignment");
  if (alignment_it != out.metadata.end()) {
    const auto value = alignment_it->second.as_u64();
    if (value.has_value()) alignment = *value;
  }
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return make_error(ErrCode::InvalidValue, "gguf.alignment",
                      "invalid gguf alignment");
  }

  const uint64_t header_len = static_cast<uint64_t>(reader.pos());
  if (header_len > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
    return make_error(ErrCode::Overflow, "gguf.data_offset",
                      "data offset overflows uint64");
  }
  out.data_offset = ((header_len + alignment - 1) / alignment) * alignment;
  if (out.data_offset > file_len) {
    return make_error(ErrCode::OutOfBounds, "gguf.data_offset",
                      "data offset beyond file");
  }

  const uint64_t data_len = file_len - out.data_offset;
  for (const GgufTensor& tensor : out.tensors) {
    if (tensor.offset > data_len || tensor.byte_len > data_len - tensor.offset) {
      return make_error(ErrCode::OutOfBounds, "gguf.tensor.offset",
                        "tensor `" + tensor.name + "` is beyond file bounds");
    }
  }

  return out;
}

[[nodiscard]] Result<void> read_exact_at(int fd, uint8_t* dst, size_t len,
                                         uint64_t offset) {
  size_t done = 0;
  while (done < len) {
    const size_t want = len - done;
    const uint64_t at = offset + static_cast<uint64_t>(done);
    const ssize_t n = ::pread(fd, dst + done, want, static_cast<off_t>(at));
    if (n < 0) {
      if (errno == EINTR) continue;
      return errno_error(ErrCode::Io, "gguf.pread");
    }
    if (n == 0) {
      return make_error(ErrCode::Truncated, "gguf.pread",
                        "short read at offset " + std::to_string(at));
    }
    done += static_cast<size_t>(n);
  }
  return {};
}

}  // namespace

Result<void> pread_at(int fd, uint64_t offset, std::span<uint8_t> dst) {
  if (fd < 0) {
    return make_error(ErrCode::InvalidValue, "gguf.fd",
                      "cannot read from a closed file");
  }
  return read_exact_at(fd, dst.data(), dst.size(), offset);
}

void fadvise_dontneed(int fd, uint64_t offset, uint64_t len) noexcept {
  if (fd < 0 || len == 0) return;
  (void)::posix_fadvise(fd, static_cast<off_t>(offset),
                        static_cast<off_t>(len), POSIX_FADV_DONTNEED);
}

std::optional<uint64_t> Value::as_u64() const noexcept {
  switch (type()) {
    case Type::U8:
      return std::get<uint8_t>(v_);
    case Type::I8: {
      const auto v = std::get<int8_t>(v_);
      if (v < 0) return std::nullopt;
      return static_cast<uint64_t>(v);
    }
    case Type::U16:
      return std::get<uint16_t>(v_);
    case Type::I16: {
      const auto v = std::get<int16_t>(v_);
      if (v < 0) return std::nullopt;
      return static_cast<uint64_t>(v);
    }
    case Type::U32:
      return std::get<uint32_t>(v_);
    case Type::I32: {
      const auto v = std::get<int32_t>(v_);
      if (v < 0) return std::nullopt;
      return static_cast<uint64_t>(v);
    }
    case Type::U64:
      return std::get<uint64_t>(v_);
    case Type::I64: {
      const auto v = std::get<int64_t>(v_);
      if (v < 0) return std::nullopt;
      return static_cast<uint64_t>(v);
    }
    default:
      return std::nullopt;
  }
}

std::optional<float> Value::as_f32() const noexcept {
  switch (type()) {
    case Type::F32:
      return std::get<float>(v_);
    case Type::F64:
      return static_cast<float>(std::get<double>(v_));
    case Type::U8:
      return static_cast<float>(std::get<uint8_t>(v_));
    case Type::I8:
      return static_cast<float>(std::get<int8_t>(v_));
    case Type::U16:
      return static_cast<float>(std::get<uint16_t>(v_));
    case Type::I16:
      return static_cast<float>(std::get<int16_t>(v_));
    case Type::U32:
      return static_cast<float>(std::get<uint32_t>(v_));
    case Type::I32:
      return static_cast<float>(std::get<int32_t>(v_));
    case Type::U64:
      return static_cast<float>(std::get<uint64_t>(v_));
    case Type::I64:
      return static_cast<float>(std::get<int64_t>(v_));
    default:
      return std::nullopt;
  }
}

std::optional<bool> Value::as_bool() const noexcept {
  if (type() == Type::Bool) return std::get<bool>(v_);
  const auto v = as_u64();
  if (v.has_value()) return *v != 0;
  return std::nullopt;
}

std::optional<double> Value::as_f64() const noexcept {
  switch (type()) {
    case Type::F64:
      return std::get<double>(v_);
    case Type::F32:
      return static_cast<double>(std::get<float>(v_));
    default: {
      const auto v = as_u64();
      if (v.has_value()) return static_cast<double>(*v);
      return std::nullopt;
    }
  }
}

std::optional<std::string_view> Value::as_str() const noexcept {
  if (type() != Type::String) return std::nullopt;
  return std::get<std::string>(v_);
}

const Value::Array* Value::as_array() const noexcept {
  if (type() != Type::Array) return nullptr;
  return std::get<std::shared_ptr<const Array>>(v_).get();
}

Gguf::~Gguf() { reset(); }

void Gguf::reset() noexcept {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

Gguf::Gguf(Gguf&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      file_len_(std::exchange(other.file_len_, 0)),
      data_offset_(std::exchange(other.data_offset_, 0)),
      tensors_(std::move(other.tensors_)),
      metadata_(std::move(other.metadata_)),
      index_(std::move(other.index_)) {}

Gguf& Gguf::operator=(Gguf&& other) noexcept {
  if (this != &other) {
    reset();
    fd_ = std::exchange(other.fd_, -1);
    file_len_ = std::exchange(other.file_len_, 0);
    data_offset_ = std::exchange(other.data_offset_, 0);
    tensors_ = std::move(other.tensors_);
    metadata_ = std::move(other.metadata_);
    index_ = std::move(other.index_);
  }
  return *this;
}

// открываем файл, читаем заголовок. если распарсить не вышло из-за обрезки то
// увеличиваем буфер и дочитываем только хвост (а не с нуля)
// open the file and parse the header. if it was truncated, grow the buffer
// and read only the new tail (not from zero again)
Result<Gguf> Gguf::open(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return errno_error(ErrCode::Io, "gguf.open");

  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    const Error err = errno_error(ErrCode::Io, "gguf.fstat");
    ::close(fd);
    return err;
  }
  if (st.st_size < 0) {
    ::close(fd);
    return make_error(ErrCode::InvalidValue, "gguf.file_len",
                      "negative file size");
  }
  const uint64_t file_len = static_cast<uint64_t>(st.st_size);

  uint64_t cap = 1ull << 20;
  std::vector<uint8_t> buf;
  for (;;) {
    const uint64_t want64 = std::min<uint64_t>(cap, file_len);
    const size_t want = static_cast<size_t>(want64);
    // Grow incrementally: a Truncated parse only needs the bytes we have not
    // read yet, so we never re-read the tokenizer/header prefix from offset 0.
    const size_t have = buf.size();
    if (want > have) {
      buf.resize(want);
      const auto read = read_exact_at(fd, buf.data() + have, want - have, have);
      if (!read.ok()) {
        const Error err = read.error();
        ::close(fd);
        return err;
      }
    }

    auto parsed = parse_header(buf.data(), buf.size(), file_len);
    if (parsed.ok()) {
      Gguf gguf;
      gguf.fd_ = fd;
      gguf.file_len_ = file_len;
      gguf.data_offset_ = parsed.value().data_offset;
      gguf.tensors_ = std::move(parsed.value().tensors);
      gguf.metadata_ = std::move(parsed.value().metadata);
      gguf.index_.reserve(gguf.tensors_.size());
      for (size_t i = 0; i < gguf.tensors_.size(); ++i) {
        gguf.index_.emplace(gguf.tensors_[i].name, i);
      }
      return gguf;
    }

    if (parsed.error().code == ErrCode::Truncated && cap < file_len) {
      if (cap > (1ull << 30) / 4) {
        const Error err = parsed.error();
        ::close(fd);
        return err;
      }
      cap *= 4;
      continue;
    }

    const Error err = parsed.error();
    ::close(fd);
    return err;
  }
}

Result<void> Gguf::read_at(uint64_t offset, std::span<uint8_t> dst) const {
  if (fd_ < 0) {
    return make_error(ErrCode::InvalidValue, "gguf.fd",
                      "cannot read from a closed file");
  }
  if (offset > file_len_ || dst.size() > file_len_ - offset) {
    return make_error(ErrCode::OutOfBounds, "gguf.read_at",
                      "read beyond the file");
  }
  return read_exact_at(fd_, dst.data(), dst.size(), offset);
}

void Gguf::fadvise_dontneed(uint64_t offset, uint64_t len) const noexcept {
  if (fd_ < 0 || len == 0) return;
  (void)::posix_fadvise(fd_, static_cast<off_t>(offset),
                        static_cast<off_t>(len), POSIX_FADV_DONTNEED);
}

Result<Mmap> Gguf::mmap_file() const {
  if (fd_ < 0) {
    return make_error(ErrCode::InvalidValue, "gguf.fd",
                      "cannot mmap a closed file");
  }
  return Mmap::map_file(fd_, file_len_);
}

Result<const GgufTensor*> Gguf::tensor(std::string_view name) const {
  const auto it = index_.find(name);
  if (it == index_.end()) {
    return make_error(ErrCode::MissingTensor, "gguf.tensor",
                      "no tensor `" + std::string(name) + "`");
  }
  return &tensors_[it->second];
}

Result<std::vector<uint8_t>> Gguf::read_tensor(std::string_view name) const {
  TOKSHIT_TRY(tensor_ptr, tensor(name));
  const GgufTensor& t = *tensor_ptr;
  if (t.byte_len > std::numeric_limits<size_t>::max()) {
    return make_error(ErrCode::Overflow, "gguf.tensor.byte_len",
                      "tensor `" + t.name + "` is larger than address space");
  }
  std::vector<uint8_t> buf(static_cast<size_t>(t.byte_len));
  const auto read =
      read_exact_at(fd_, buf.data(), buf.size(), data_offset_ + t.offset);
  if (!read.ok()) return read.error();
  return buf;
}

Result<void> Gguf::read_into(std::string_view name, uint64_t byte_offset,
                             std::span<uint8_t> buf) const {
  TOKSHIT_TRY(tensor_ptr, tensor(name));
  const GgufTensor& t = *tensor_ptr;
  if (byte_offset > t.byte_len || buf.size() > t.byte_len - byte_offset) {
    return make_error(ErrCode::OutOfBounds, "gguf.tensor.slice",
                      "slice beyond `" + t.name + "`");
  }
  return read_exact_at(fd_, buf.data(), buf.size(),
                       data_offset_ + t.offset + byte_offset);
}

Result<std::vector<uint8_t>> Gguf::read_tensor_slice(
    std::string_view name, uint64_t byte_offset, uint64_t byte_len) const {
  TOKSHIT_TRY(tensor_ptr, tensor(name));
  const GgufTensor& t = *tensor_ptr;
  if (byte_offset > t.byte_len || byte_len > t.byte_len - byte_offset) {
    return make_error(ErrCode::OutOfBounds, "gguf.tensor.slice",
                      "slice beyond `" + t.name + "`");
  }
  if (byte_len > std::numeric_limits<size_t>::max()) {
    return make_error(ErrCode::Overflow, "gguf.tensor.slice",
                      "slice is larger than address space");
  }
  std::vector<uint8_t> buf(static_cast<size_t>(byte_len));
  const auto read = read_exact_at(fd_, buf.data(), buf.size(),
                                  data_offset_ + t.offset + byte_offset);
  if (!read.ok()) return read.error();
  return buf;
}

Result<const Value*> Gguf::get(std::string_view key) const {
  const auto it = metadata_.find(key);
  if (it == metadata_.end()) {
    return make_error(ErrCode::MissingTensor, "gguf.metadata",
                      "no kv `" + std::string(key) + "`");
  }
  return &it->second;
}

Result<uint64_t> Gguf::u64_kv(std::string_view key) const {
  TOKSHIT_TRY(value, get(key));
  const auto v = value->as_u64();
  if (!v.has_value()) {
    return make_error(ErrCode::InvalidField, "gguf.metadata",
                      "kv `" + std::string(key) + "` not int");
  }
  return *v;
}

Result<float> Gguf::f32_kv(std::string_view key) const {
  TOKSHIT_TRY(value, get(key));
  const auto v = value->as_f32();
  if (!v.has_value()) {
    return make_error(ErrCode::InvalidField, "gguf.metadata",
                      "kv `" + std::string(key) + "` not float");
  }
  return *v;
}

Result<std::vector<std::string>> Gguf::strings(std::string_view key) const {
  TOKSHIT_TRY(value, get(key));
  const Value::Array* arr = value->as_array();
  if (arr == nullptr) {
    return make_error(ErrCode::InvalidField, "gguf.metadata",
                      "kv `" + std::string(key) + "` not array");
  }
  std::vector<std::string> out;
  out.reserve(arr->size());
  for (const Value& item : *arr) {
    const auto text = item.as_str();
    if (!text.has_value()) {
      return make_error(ErrCode::InvalidField, "gguf.metadata",
                        "kv `" + std::string(key) + "`: not strings");
    }
    out.emplace_back(*text);
  }
  return out;
}

Result<std::vector<uint32_t>> Gguf::u32s(std::string_view key) const {
  TOKSHIT_TRY(value, get(key));
  const Value::Array* arr = value->as_array();
  if (arr == nullptr) {
    return make_error(ErrCode::InvalidField, "gguf.metadata",
                      "kv `" + std::string(key) + "` not array");
  }
  std::vector<uint32_t> out;
  out.reserve(arr->size());
  for (const Value& item : *arr) {
    const auto v = item.as_u64();
    if (!v.has_value() || *v > std::numeric_limits<uint32_t>::max()) {
      return make_error(ErrCode::InvalidField, "gguf.metadata",
                        "kv `" + std::string(key) + "`: not u32");
    }
    out.push_back(static_cast<uint32_t>(*v));
  }
  return out;
}

}  // namespace tokshit
