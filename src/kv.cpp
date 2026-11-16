#include "tokshit/kv.hpp"

#include <algorithm>
#include <cstring>

#include "tokshit/dequant.hpp"

namespace tokshit {

// полный kv-кеш: два сплошных сляба (k и v), слой l начинается с
// l * cap * stride. f16 вдвое дешевле f32 по полосе внимания — на 2К это
// ~195 мб вместо ~390, и attention читает вдвое меньше из RAM
// full kv cache: two flat slabs (k and v), layer l starts at
// l * cap * stride. f16 halves the attention bandwidth vs f32 — ~195 mb
// instead of ~390 at 2k, and attention reads half as much from RAM
Kv Kv::alloc(const KvCap& cap) {
  Kv kv;
  kv.cap_ = cap;
  kv.use_f16_ = cap.f16;
  const std::size_t stride = cap.kv_dim * (cap.f16 ? 2 : 4);
  const std::size_t per_layer = cap.cap * stride;
  kv.k_ = std::make_unique<uint8_t[]>(cap.layers * per_layer);
  kv.v_ = std::make_unique<uint8_t[]>(cap.layers * per_layer);
  kv.high_.assign(cap.layers, 0);
  return kv;
}

namespace {

// упаковка float -> fp16 пачкой. скалярно, но с расчётом на автовектор
// pack float -> fp16 in bulk. scalar, written to auto-vectorize
inline void store_f16(uint8_t* dst, const float* src, std::size_t n) {
  auto* out = reinterpret_cast<uint16_t*>(dst);
  for (std::size_t i = 0; i < n; ++i) out[i] = f32_to_f16(src[i]);
}

inline void store_f32(uint8_t* dst, const float* src, std::size_t n) {
  std::memcpy(dst, src, n * sizeof(float));
}

}  // namespace

void Kv::write(std::size_t layer, std::size_t pos, std::size_t b,
               const float* k, std::size_t k_stride, const float* v,
               std::size_t v_stride) {
  if (b == 0 || layer >= cap_.layers) return;
  const std::size_t per_layer = cap_.cap * k_stride_bytes();
  uint8_t* kd = k_.get() + layer * per_layer;
  uint8_t* vd = v_.get() + layer * per_layer;
  const std::size_t writable =
      pos < cap_.cap ? std::min(b, cap_.cap - pos) : 0;
  for (std::size_t j = 0; j < writable; ++j) {
    const std::size_t p = pos + j;
    uint8_t* krow = kd + p * k_stride_bytes();
    uint8_t* vrow = vd + p * v_stride_bytes();
    if (use_f16_) {
      store_f16(krow, k + j * k_stride, cap_.kv_dim);
      store_f16(vrow, v + j * v_stride, cap_.kv_dim);
    } else {
      store_f32(krow, k + j * k_stride, cap_.kv_dim);
      store_f32(vrow, v + j * v_stride, cap_.kv_dim);
    }
  }
  // high-water только по реально записанным строкам: раньше hi = pos + b
  // заявлял позиции за контуром, tokens() сообщал о несуществующих строках,
  // и attention читал мусор вместо k/v. при writable == 0 (pos за контуром)
  // high-water не двигается вовсе
  // high-water over actually written rows only: hi = pos + b used to claim
  // positions past the slab, tokens() reported rows that were never written,
  // and attention read garbage instead of k/v. with writable == 0 (pos past
  // the slab) the high-water does not move at all
  if (writable != 0) {
    const std::size_t hi = pos + writable;
    if (hi > high_[layer]) high_[layer] = hi;
  }
}

void Kv::write_row(std::size_t layer, std::size_t pos, const float* k,
                   const float* v) {
  write(layer, pos, 1, k, cap_.kv_dim, v, cap_.kv_dim);
}

std::size_t Kv::tokens(std::size_t layer) const {
  return layer < high_.size() ? high_[layer] : 0;
}

void Kv::truncate(std::size_t layer, std::size_t n) {
  if (layer < high_.size() && n < high_[layer]) high_[layer] = n;
}

void Kv::truncate_all(std::size_t n) {
  for (auto& h : high_) h = std::min(h, n);
}

std::size_t Kv::cap() const noexcept { return cap_.cap; }
bool Kv::f16() const noexcept { return use_f16_; }
std::size_t Kv::layers() const noexcept { return cap_.layers; }
std::size_t Kv::kv_dim() const noexcept { return cap_.kv_dim; }
bool Kv::full(std::size_t layer, std::size_t next_pos) const {
  return next_pos >= cap_.cap || layer >= cap_.layers;
}

std::size_t Kv::k_stride_bytes() const noexcept {
  return cap_.kv_dim * (use_f16_ ? 2 : 4);
}
std::size_t Kv::v_stride_bytes() const noexcept { return k_stride_bytes(); }

const uint8_t* Kv::k_slab(std::size_t layer) const noexcept {
  return k_.get() + layer * cap_.cap * k_stride_bytes();
}
const uint8_t* Kv::v_slab(std::size_t layer) const noexcept {
  return v_.get() + layer * cap_.cap * v_stride_bytes();
}

std::size_t Kv::bytes() const noexcept {
  return 2 * cap_.layers * cap_.cap * k_stride_bytes();
}

}  // namespace tokshit
