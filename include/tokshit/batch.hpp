#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "tokshit/kernels.hpp"
#include "tokshit/quant.hpp"

namespace tokshit {

// батч-матвек: одна полоска весов применяется к нескольким активациям.
// в этом весь смысл verify-батча спекуляции — веса читаются один раз, а
// обслуживают b токенов, DRAM-трафик на выход делится на b
struct BatchSeg {
  std::span<const uint8_t> bytes;
  std::size_t rb = 0;
  QuantType quant = QuantType::F32;
  std::size_t in_dim = 0;
  // активации: b строк по in_dim; acts[j] — ActQ8 строки j, уже квантизованная
  const float* x = nullptr;
  const kernels::ActQ8* acts = nullptr;
  std::size_t b = 0;
  // выход: b × rows, row-major: out[j * rows + r]
  float* out = nullptr;
  std::size_t rows = 0;
};

namespace batchlib {
// раскладывает сегменты в задачи «строки весов × активации» и кидает в пул.
// вызывающий поток тоже работает, как в run_segs
void run_bsegs(std::span<const BatchSeg> segs, std::size_t rows_per_task);
}  // namespace batchlib

}  // namespace tokshit
