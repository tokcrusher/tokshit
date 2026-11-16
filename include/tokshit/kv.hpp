#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tokshit {

// полноценный kv-кеш на заранее отведённый контур. геометрия флинта: 49
// слоёв, 4 kv-головы × 128, kv_dim = 512. f16 даёт 2 байта на элемент —
// 2к контекста ≈ 49 × 2048 × 512 × 4 байта ≈ 195 мб, f32 вдвое дороже.
// главная собственность: позиции абсолютные, строки перезаписываемые —
// черновики спекуляции пишутся поверх тех же слотов, truncate до принятого
// префикса откатывает кэш без пересылок
// full kv cache on a fixed preallocated budget. flint geometry: 49 layers,
// 4 kv heads × 128, kv_dim = 512. f16 is 2 bytes/elem so 2k context is
// ~49 × 2048 × 512 × 4 bytes ≈ 195 mb, f32 is double. the important
// property: absolute positions, rewritable rows — speculative drafts are
// written over the same slots, truncate-to-prefix rolls the cache back
// without any data movement
struct KvCap {
  std::size_t layers = 0;
  std::size_t kv_dim = 0;  // n_kv_heads * head_dim
  std::size_t cap = 0;     // токенов на слой / positions per layer
  bool f16 = true;         // f16 слябы, иначе plain f32
};

class Kv {
 public:
  Kv() = default;
  // аллоцирует слябы k и v всех слоёв сразу. страницы anon пока не тронуты —
  // физические выделятся по мере записи, это энергетически дешевле, чем
  // пересчитывать memset по всему контуру
  // allocates k and v slabs for all layers at once. anon pages stay
  // untouched until written, so physical pages appear as rows land
  static Kv alloc(const KvCap& cap);

  // пишет b токенов с абсолютной позиции pos. k/v полустроками: строка j
  // начинается с k[j * k_stride], kv_dim элементов, stride в float'ах
  // writes b token rows at absolute position pos. k/v row-major: row j
  // starts at k[j * k_stride], kv_dim long, stride counted in floats
  void write(std::size_t layer, std::size_t pos, std::size_t b,
             const float* k, std::size_t k_stride, const float* v,
             std::size_t v_stride);
  void write_row(std::size_t layer, std::size_t pos, const float* k,
                 const float* v);

  // сколько позиций в слое: high-water = максимум pos+1 когда-либо
  // записанных, truncate снижает
  // per-layer count: high-water = largest pos+1 ever written, truncate lowers
  [[nodiscard]] std::size_t tokens(std::size_t layer) const;

  // откат: позиции >= n больше не существуют. чуть позже тот же диапазон
  // перепишут другие токены — это штатный сценарий
  // rollback: positions >= n are gone. the same range gets rewritten by other
  // tokens later, that is the normal path
  void truncate(std::size_t layer, std::size_t n);
  void truncate_all(std::size_t n);

  [[nodiscard]] std::size_t cap() const noexcept;
  [[nodiscard]] bool full(std::size_t layer, std::size_t next_pos) const;
  [[nodiscard]] bool f16() const noexcept;
  [[nodiscard]] std::size_t layers() const noexcept;
  [[nodiscard]] std::size_t kv_dim() const noexcept;

  // сырой сляб для attention core: cap × stride байт на слой, строка токена
  // p лежит на slab[p * k_stride_bytes()]
  // raw slab for the attention core: cap × stride bytes per layer, token p
  // sits at slab[p * k_stride_bytes()]
  [[nodiscard]] const uint8_t* k_slab(std::size_t layer) const noexcept;
  [[nodiscard]] const uint8_t* v_slab(std::size_t layer) const noexcept;
  // байт на токен: f16 = kv_dim × 2, f32 = kv_dim × 4
  // bytes per token: f16 = kv_dim × 2, f32 = kv_dim × 4
  [[nodiscard]] std::size_t k_stride_bytes() const noexcept;
  [[nodiscard]] std::size_t v_stride_bytes() const noexcept;

  // суммарно выделено под слябы, для отчётирования
  // total slab bytes, for reporting
  [[nodiscard]] std::size_t bytes() const noexcept;

 private:
  KvCap cap_;
  // слябы сплошные по слоям: слой l начинается с l * cap * stride
  // slabs are contiguous across layers: layer l starts at l * cap * stride
  std::unique_ptr<uint8_t[]> k_;
  std::unique_ptr<uint8_t[]> v_;
  std::vector<std::size_t> high_;
  bool use_f16_ = true;
};

}  // namespace tokshit
