// i8mm (SMMLA) ядра для matvec (aarch64 + FEAT_I8MM + FEAT_DotProd).
//
// файл компилится с -march=armv8.2-a+dotprod+i8mm, так что smmla не протекает
// в базовый бинарь. включается через TOKSHIT_I8MM=1 при наличии фичи на cpu
// (kernels::have_i8mm); по умолчанию работает dotprod-путь.
//
// семантика lane SMMLA (проверено пробником на целевом cpu):
//
// для чистого dot произведения полезны только диагональные lane 0 и 3
// (lane 1/2 — кросс-члены a[k]b[8+k]); то есть SMMLA даёт те же 16 полезных
// MAC на оп, что SDOT, без перепаковки весов под пару строк. ядро ниже —
// natural-раскладка, 2 SMMLA на строку на блок 32 весов, и замер решает,
// выигрывает ли оно у vdotq-ядра (та же длина зависимости, латентность
// SMMLA выше, но меньше операций на блок шкалы).
//
//
//
//

#include "kernels_internal.hpp"

#if defined(__aarch64__) && defined(TOKSHIT_HAVE_NEON_I8MM)

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tokshit/dequant.hpp"

namespace tokshit::kernels::neon {
namespace {

// f16 -> float, как в dotprod-ядрах
// f16 -> float, same as the dotprod kernels
[[nodiscard]] inline float load_f16(const std::uint8_t* p) noexcept {
  float16_t h;
  std::memcpy(&h, p, sizeof(h));
  return vgetq_lane_f32(vcvt_f32_f16(vdup_n_f16(h)), 0);
}

[[nodiscard]] inline int8x16_t load_w(const std::uint8_t* p) noexcept {
  return vld1q_s8(reinterpret_cast<const std::int8_t*>(p));
}

}  // namespace

// q8_0, по две строки за проход с независимыми цепочками SMMLA.
void dot_rows_q8_0_i8mm(const std::uint8_t* rows, std::size_t rb,
                        std::size_t nrows, const ActQ8& act,
                        std::size_t in_dim, float* out) noexcept {
  constexpr std::size_t kBs = 34;
  const std::size_t nb = in_dim >> 5;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  std::size_t r = 0;
  for (; r + 2 <= nrows; r += 2) {
    float32x4_t facc0 = vdupq_n_f32(0.0F);
    float32x4_t facc1 = vdupq_n_f32(0.0F);
    for (std::size_t b = 0; b < nb; ++b) {
      const int8x16_t a0 = vld1q_s8(ap + b * 32);
      const int8x16_t a1 = vld1q_s8(ap + b * 32 + 16);

      const std::uint8_t* p0 = rows + r * rb + b * kBs;
      const std::uint8_t* p1 = rows + (r + 1) * rb + b * kBs;
      // шкала веса d_b — своя у каждого блока 34B, выносить её из блока
      // нельзя: произведение d_b * s_b живёт только внутри блока
      const float32x4_t sc0 = vdupq_n_f32(load_f16(p0) * sp[b]);
      const float32x4_t sc1 = vdupq_n_f32(load_f16(p1) * sp[b]);

      // диагональные lane 0 и 3 несут полный dot блока, кросс-члены в
      // lane 1/2 копятся в отдельных lane фасc и не читаются
      int32x4_t acc0 = vmmlaq_s32(vdupq_n_s32(0), a0, load_w(p0 + 2));
      acc0 = vmmlaq_s32(acc0, a1, load_w(p0 + 18));
      facc0 = vfmaq_f32(facc0, vcvtq_f32_s32(acc0), sc0);

      int32x4_t acc1 = vmmlaq_s32(vdupq_n_s32(0), a0, load_w(p1 + 2));
      acc1 = vmmlaq_s32(acc1, a1, load_w(p1 + 18));
      facc1 = vfmaq_f32(facc1, vcvtq_f32_s32(acc1), sc1);
    }

    out[r] = vgetq_lane_f32(facc0, 0) + vgetq_lane_f32(facc0, 3);
    out[r + 1] = vgetq_lane_f32(facc1, 0) + vgetq_lane_f32(facc1, 3);
  }
  for (; r < nrows; ++r) {
    out[r] = dot_q8_0(rows + r * rb, act, in_dim);
  }
}

}  // namespace tokshit::kernels::neon

#endif  // __aarch64__ && TOKSHIT_HAVE_NEON_I8MM
