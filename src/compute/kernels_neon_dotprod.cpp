// neon ядра для скалярного произведения (aarch64 + FEAT_DotProd)
//
// этот файл компилится с -march=armv8.2-a+dotprod, так что vdotq_s32 не
// протекают в базовый бинарь. каждое ядро сначала проверяется против
// скалярного эталона в tests/test_kernels.cpp, и только потом попадает в
// горячий путь
//

#include "kernels_internal.hpp"

#if defined(__aarch64__) && defined(TOKSHIT_HAVE_NEON_DOTPROD)

#include <arm_neon.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../dequant_tables.hpp"
#include "tokshit/dequant.hpp"
#include "tokshit/gguf_tables.hpp"

namespace tokshit::kernels::neon {
namespace {

constexpr std::array<std::uint64_t, 256> build_mask8() {
  std::array<std::uint64_t, 256> out{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint64_t v = 0;
    for (std::uint32_t k = 0; k < 8; ++k) {
      if (((i >> k) & 1U) != 0U) {
        v |= static_cast<std::uint64_t>(0xFFU) << (8 * k);
      }
    }
    out[i] = v;
  }
  return out;
}

constexpr std::array<std::uint64_t, 256> kMask8 = build_mask8();
// grid entry pair loaded straight into neon lanes: no GPR shift/or, no stack
[[nodiscard]] inline int8x8_t grid2_iq3s(std::uint32_t i1, std::uint32_t i2) noexcept {
    uint32x2_t v = vld1_lane_u32(&tables::iq3s_grid[i1], vdup_n_u32(0), 0);
    v = vld1_lane_u32(&tables::iq3s_grid[i2], v, 1);
    return vreinterpret_s8_u32(v);
}
[[nodiscard]] inline int8x8_t grid2_iq3xxs(std::uint32_t i1, std::uint32_t i2) noexcept {
    uint32x2_t v = vld1_lane_u32(&tables::iq3xxs_grid[i1], vdup_n_u32(0), 0);
    v = vld1_lane_u32(&tables::iq3xxs_grid[i2], v, 1);
    return vreinterpret_s8_u32(v);
}

[[nodiscard]] inline std::uint16_t load_u16(const std::uint8_t* p) noexcept {
  return static_cast<std::uint16_t>(
      static_cast<std::uint16_t>(p[0]) |
      static_cast<std::uint16_t>(static_cast<std::uint16_t>(p[1]) << 8));
}

[[nodiscard]] inline float load_f16(const std::uint8_t* p) noexcept {
  // аппаратная конвертация half->float: софтовая f16_to_f32 в горячем цикле
  // стоила примерно столько же, сколько сам vdotq, потому что вызывалась на
  // каждый блок из 32 весов
  // hardware half->float: the software f16_to_f32 cost about as much as the
  float16_t h;
  std::memcpy(&h, p, sizeof(h));
  return vgetq_lane_f32(vcvt_f32_f16(vdup_n_f16(h)), 0);
}

[[nodiscard]] inline std::uint32_t load_u32(const std::uint8_t* p) noexcept {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

// e8m0 -> float, уже поделённый на два: таблица e2m1 в dequant удвоена.
// формула ровно как ggml_e8m0_to_fp32_half
constexpr std::array<float, 256> build_e8m0_half() {
  std::array<float, 256> out{};
  for (std::uint32_t x = 0; x < 256; ++x) {
    const std::uint32_t bits =
        x < 2 ? (0x00200000U << x) : ((x - 1) << 23);
    out[x] = std::bit_cast<float>(bits);
  }
  return out;
}

// ue4m3 (4-бит экспонента со сдвигом 7, 3-бит мантисса) -> float, тоже
// пополам. 0x7f — NaN этого формата, dequant отдаёт для него 0
constexpr std::array<float, 256> build_ue4m3_half() {
  std::array<float, 256> out{};
  for (std::uint32_t x = 0; x < 256; ++x) {
    const std::uint32_t e = (x >> 3) & 0x0FU;
    const std::uint32_t m = x & 7U;
    if (x == 0x7FU) {
      out[x] = 0.0F;
      continue;
    }
    if (e == 0) {
      out[x] = static_cast<float>(m) * (1.0F / 1024.0F);
      continue;
    }
    float v = 1.0F + static_cast<float>(m) * 0.125F;
    for (std::uint32_t i = 7; i < e; ++i) v *= 2.0F;
    for (std::uint32_t i = e; i < 7; ++i) v *= 0.5F;
    out[x] = v * 0.5F;
  }
  return out;
}

const std::array<float, 256> kE8m0Half = build_e8m0_half();
const std::array<float, 256> kUe4m3Half = build_ue4m3_half();

// q2_0: 16 байт = 64 веса, по 4 кода на байт. tbl раскладывает 4 байта в 16
// значений, signed shift {0,-2,-4,-6} вытаскивает код, маска и -1 дают
// значение -1..2 (как q - 1 в dequant)
const std::array<std::uint8_t, 16> kQ2IdxLo = {
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
const std::array<std::uint8_t, 16> kQ2IdxHi = {
    4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};
const std::array<std::int8_t, 16> kQ2Shifts = {
    0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};

[[nodiscard]] inline int8x8_t grid8(const std::uint64_t& entry) noexcept {
  return vld1_s8(reinterpret_cast<const std::int8_t*>(&entry));
}

}  // namespace

// q8_0: блок 32 веса, f16 scale + 32 int8. активация уже int8, так что просто
// vdotq и в конце умножаем на scale весов и scale активации
float dot_q8_0(const std::uint8_t* row, const ActQ8& act,
               std::size_t in_dim) noexcept {
  const std::size_t nb = in_dim >> 5;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  float32x4_t a0 = vdupq_n_f32(0.0F);
  float32x4_t a1 = vdupq_n_f32(0.0F);
  float32x4_t a2 = vdupq_n_f32(0.0F);
  float32x4_t a3 = vdupq_n_f32(0.0F);

  std::size_t b = 0;
  while (b + 4 <= nb) {
    const std::size_t base = b * 34;
    const float d0 = load_f16(row + base);
    const float d1 = load_f16(row + base + 34);
    const float d2 = load_f16(row + base + 68);
    const float d3 = load_f16(row + base + 102);

    const int8x16_t w00 =
        vld1q_s8(reinterpret_cast<const std::int8_t*>(row + base + 2));
    const int8x16_t w01 =
        vld1q_s8(reinterpret_cast<const std::int8_t*>(row + base + 18));
    const int8x16_t w10 =
        vld1q_s8(reinterpret_cast<const std::int8_t*>(row + base + 36));
    const int8x16_t w11 =
        vld1q_s8(reinterpret_cast<const std::int8_t*>(row + base + 52));
    const int8x16_t w20 =
        vld1q_s8(reinterpret_cast<const std::int8_t*>(row + base + 70));
    const int8x16_t w21 =
        vld1q_s8(reinterpret_cast<const std::int8_t*>(row + base + 86));
    const int8x16_t w30 =
        vld1q_s8(reinterpret_cast<const std::int8_t*>(row + base + 104));
    const int8x16_t w31 =
        vld1q_s8(reinterpret_cast<const std::int8_t*>(row + base + 120));

    int32x4_t s0 = vdotq_s32(vdupq_n_s32(0), w00, vld1q_s8(ap + b * 32));
    s0 = vdotq_s32(s0, w01, vld1q_s8(ap + b * 32 + 16));
    int32x4_t s1 = vdotq_s32(vdupq_n_s32(0), w10, vld1q_s8(ap + b * 32 + 32));
    s1 = vdotq_s32(s1, w11, vld1q_s8(ap + b * 32 + 48));
    int32x4_t s2 = vdotq_s32(vdupq_n_s32(0), w20, vld1q_s8(ap + b * 32 + 64));
    s2 = vdotq_s32(s2, w21, vld1q_s8(ap + b * 32 + 80));
    int32x4_t s3 = vdotq_s32(vdupq_n_s32(0), w30, vld1q_s8(ap + b * 32 + 96));
    s3 = vdotq_s32(s3, w31, vld1q_s8(ap + b * 32 + 112));

    a0 = vfmaq_f32(a0, vcvtq_f32_s32(s0), vdupq_n_f32(d0 * sp[b]));
    a1 = vfmaq_f32(a1, vcvtq_f32_s32(s1), vdupq_n_f32(d1 * sp[b + 1]));
    a2 = vfmaq_f32(a2, vcvtq_f32_s32(s2), vdupq_n_f32(d2 * sp[b + 2]));
    a3 = vfmaq_f32(a3, vcvtq_f32_s32(s3), vdupq_n_f32(d3 * sp[b + 3]));
    b += 4;
  }
  while (b < nb) {
    const std::size_t base = b * 34;
    const float d = load_f16(row + base);
    const int8x16_t w0 =
        vld1q_s8(reinterpret_cast<const std::int8_t*>(row + base + 2));
    const int8x16_t w1 =
        vld1q_s8(reinterpret_cast<const std::int8_t*>(row + base + 18));
    int32x4_t s = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(ap + b * 32));
    s = vdotq_s32(s, w1, vld1q_s8(ap + b * 32 + 16));
    a0 = vfmaq_f32(a0, vcvtq_f32_s32(s), vdupq_n_f32(d * sp[b]));
    ++b;
  }

  return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}

// q4_0: 16 байт на 32 веса, половинки разворачиваем и поднимаем до int8
float dot_q4_0(const std::uint8_t* row, const ActQ8& act,
               std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 18;
  const std::size_t nb = in_dim >> 5;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  float32x4_t a0 = vdupq_n_f32(0.0F);
  float32x4_t a1 = vdupq_n_f32(0.0F);
  float32x4_t a2 = vdupq_n_f32(0.0F);
  float32x4_t a3 = vdupq_n_f32(0.0F);

  const int8x16_t eight = vdupq_n_s8(8);
  const uint8x16_t nib = vdupq_n_u8(0x0FU);

  std::size_t b = 0;
  while (b + 4 <= nb) {
    const std::size_t o = b * kBs;
    int32x4_t sv[4];
    float dd[4];
    for (std::size_t k = 0; k < 4; ++k) {
      const std::size_t p = o + k * kBs;
      dd[k] = load_f16(row + p);
      const uint8x16_t q = vld1q_u8(row + p + 2);
      const int8x16_t lo =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, nib)), eight);
      const int8x16_t hi =
          vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), eight);
      const std::size_t a = (b + k) * 32;
      int32x4_t s = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(ap + a));
      s = vdotq_s32(s, hi, vld1q_s8(ap + a + 16));
      sv[k] = s;
    }
    a0 = vfmaq_f32(a0, vcvtq_f32_s32(sv[0]), vdupq_n_f32(dd[0] * sp[b]));
    a1 = vfmaq_f32(a1, vcvtq_f32_s32(sv[1]), vdupq_n_f32(dd[1] * sp[b + 1]));
    a2 = vfmaq_f32(a2, vcvtq_f32_s32(sv[2]), vdupq_n_f32(dd[2] * sp[b + 2]));
    a3 = vfmaq_f32(a3, vcvtq_f32_s32(sv[3]), vdupq_n_f32(dd[3] * sp[b + 3]));
    b += 4;
  }
  while (b < nb) {
    const std::size_t o = b * kBs;
    const float d = load_f16(row + o);
    const uint8x16_t q = vld1q_u8(row + o + 2);
    const int8x16_t lo =
        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, nib)), eight);
    const int8x16_t hi =
        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), eight);
    const std::size_t a = b * 32;
    int32x4_t s = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(ap + a));
    s = vdotq_s32(s, hi, vld1q_s8(ap + a + 16));
    a0 = vfmaq_f32(a0, vcvtq_f32_s32(s), vdupq_n_f32(d * sp[b]));
    ++b;
  }

  return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}

// iq4_xs: кодбук + знаки, вес сразу в векторном регистре (без wbuf), dot,
// потом две шкалы. чтобы независимые idp и fmla перестали стоять в очереди,
// используем две цепочки аккумуляторов
float dot_iq4xs(const std::uint8_t* row, const ActQ8& act,
                std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 136;
  const std::size_t nb = in_dim >> 8;
  const int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  const int8x16_t tab = vld1q_s8(tables::kvalues_iq4nl.data());
  const uint8x16_t lowmask = vdupq_n_u8(0x0FU);

  float32x4_t accv_even = vdupq_n_f32(0.0F);
  float32x4_t accv_odd = vdupq_n_f32(0.0F);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t base = b * kBs;
    const float d = load_f16(row + base);
    const std::uint32_t sh = load_u16(row + base + 2);
    const std::uint8_t* sl = row + base + 4;
    const std::uint8_t* qs = row + base + 8;
    for (std::size_t ib = 0; ib < 8; ++ib) {
      const std::uint32_t ls =
          (static_cast<std::uint32_t>(
               (static_cast<std::uint32_t>(sl[ib / 2]) >>
                (4 * (ib % 2))) &
               0x0FU)) |
          ((static_cast<std::uint32_t>(sh >> (2 * ib)) & 3U) << 4);
      const float dl = d * static_cast<float>(static_cast<int32_t>(ls) - 32);

      const uint8x16_t qv = vld1q_u8(qs + ib * 16);
      const uint8x16_t lo = vandq_u8(qv, lowmask);
      const uint8x16_t hi = vshrq_n_u8(qv, 4);
      const int8x16_t wlo = vqtbl1q_s8(tab, lo);
      const int8x16_t whi = vqtbl1q_s8(tab, hi);

      const std::size_t a = b * 256 + ib * 32;
      int32x4_t s = vdotq_s32(vdupq_n_s32(0), wlo, vld1q_s8(ap + a));
      s = vdotq_s32(s, whi, vld1q_s8(ap + a + 16));
      const float32x4_t sc = vdupq_n_f32(dl * sp[a >> 5]);
      // две цепочки аккумуляторов, чередуем по ib: одна цепочка это 16 vfma
      // подряд на одном регистре, чистая латентность
      if (((b + ib) & 1) != 0) {
        accv_odd = vfmaq_f32(accv_odd, vcvtq_f32_s32(s), sc);
      } else {
        accv_even = vfmaq_f32(accv_even, vcvtq_f32_s32(s), sc);
      }
    }
  }
  return vaddvq_f32(vaddq_f32(accv_even, accv_odd));
}

// iq2_s: 2 бита на вес. маски/сдвиги как в llama.cpp, потом vdotq. тут
// 4 независимые цепочки аккумуляторов, индексы решётки без сдвигового
// хоп через int64 и шкала, сплавленная в один scalar mul
float dot_iq2s(const std::uint8_t* row, const ActQ8& act,
               std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 82;
  const std::size_t nb = in_dim >> 8;
  const int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  float32x4_t acc2 = vdupq_n_f32(0.0F);
  float32x4_t acc3 = vdupq_n_f32(0.0F);

  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t base = b * kBs;
    const float d = load_f16(row + base);
    const std::uint8_t* qs = row + base + 2;
    const std::uint8_t* qh = row + base + 66;
    const std::uint8_t* scales = row + base + 74;
    const std::uint8_t* signs = qs + 32;
    // d * 0.25 выносим на блок: сабблок платит один scalar mul
    const float d4 = d * 0.25F;

    for (std::size_t ib = 0; ib < 8; ++ib) {
      // старший байт индекса решётки из nibble-байта qh[ib]:
      // idx = qsb[l] | (((qh >> 2l) & 3) << 8) == старый (qh << (8-2l)) & 0x300
      const std::uint8_t qhb = qh[ib];
      const std::uint8_t* qsb = qs + ib * 4;
      const std::uint8_t* sgb = signs + ib * 4;
      const std::uint32_t i0 = static_cast<std::uint32_t>(qsb[0]) |
                               ((static_cast<std::uint32_t>(qhb) & 0x03U) << 8);
      const std::uint32_t i1 = static_cast<std::uint32_t>(qsb[1]) |
                               ((static_cast<std::uint32_t>(qhb) & 0x0CU) << 6);
      const std::uint32_t i2 = static_cast<std::uint32_t>(qsb[2]) |
                               ((static_cast<std::uint32_t>(qhb) & 0x30U) << 4);
      const std::uint32_t i3 = static_cast<std::uint32_t>(qsb[3]) |
                               ((static_cast<std::uint32_t>(qhb) & 0xC0U) << 2);
      // решётку и знак тащим прямо в неоновые регистры, минуя int64 hop
      const int8x8_t g0 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &tables::iq2s_grid[i0]));
      const int8x8_t g1 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &tables::iq2s_grid[i1]));
      const int8x8_t g2 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &tables::iq2s_grid[i2]));
      const int8x8_t g3 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &tables::iq2s_grid[i3]));
      const int8x8_t m0 = vld1_s8(
          reinterpret_cast<const std::int8_t*>(&kMask8[sgb[0]]));
      const int8x8_t m1 = vld1_s8(
          reinterpret_cast<const std::int8_t*>(&kMask8[sgb[1]]));
      const int8x8_t m2 = vld1_s8(
          reinterpret_cast<const std::int8_t*>(&kMask8[sgb[2]]));
      const int8x8_t m3 = vld1_s8(
          reinterpret_cast<const std::int8_t*>(&kMask8[sgb[3]]));
      const int8x16_t w01 = vcombine_s8(vsub_s8(veor_s8(g0, m0), m0),
                                        vsub_s8(veor_s8(g1, m1), m1));
      const int8x16_t w23 = vcombine_s8(vsub_s8(veor_s8(g2, m2), m2),
                                        vsub_s8(veor_s8(g3, m3), m3));

      const std::size_t a = b * 256 + ib * 32;
      const float s01 =
          d4 * (0.5F + static_cast<float>(scales[ib] & 0x0FU)) * sp[a >> 5];
      const float s23 =
          d4 * (0.5F + static_cast<float>((scales[ib] >> 4) & 0x0FU)) *
          sp[a >> 5];
      const float32x4_t f01 = vdupq_n_f32(s01);
      const float32x4_t f23 = vdupq_n_f32(s23);
      const int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), w01, vld1q_s8(ap + a));
      const int32x4_t p1 =
          vdotq_s32(vdupq_n_s32(0), w23, vld1q_s8(ap + a + 16));
      // чередование ib/чёт: одна цепочка живёт ~25 ops, латентность закрыта
      if ((ib & 1) == 0) {
        acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(p0), f01);
        acc2 = vfmaq_f32(acc2, vcvtq_f32_s32(p1), f23);
      } else {
        acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(p0), f01);
        acc3 = vfmaq_f32(acc3, vcvtq_f32_s32(p1), f23);
      }
    }
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

// iq2_s, четыре строки: решётка/знаки/шкалы так же как в dot_iq2s, входные
// активации (a0/a1) загружаются один раз на сабблок и делятся между строками —
// независимые пары vdot+fmla закрывают латентность vdot
void dot_rows_iq2s(const std::uint8_t* rows, std::size_t rb, std::size_t nrows,
                   const ActQ8& act, std::size_t in_dim, float* out) noexcept {
  constexpr std::size_t kBs = 82;
  const std::size_t nb = in_dim >> 8;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  std::size_t r = 0;
  for (; r + 4 <= nrows; r += 4) {
    const std::uint8_t* p0 = rows + (r + 0) * rb;
    const std::uint8_t* p1 = rows + (r + 1) * rb;
    const std::uint8_t* p2 = rows + (r + 2) * rb;
    const std::uint8_t* p3 = rows + (r + 3) * rb;
    float32x4_t acc0 = vdupq_n_f32(0.0F);
    float32x4_t acc1 = vdupq_n_f32(0.0F);
    float32x4_t acc2 = vdupq_n_f32(0.0F);
    float32x4_t acc3 = vdupq_n_f32(0.0F);

    // одна строка: решётка/знаки/шкалы как в dot_iq2s
    auto row_dot = [&](const std::uint8_t* p, float32x4_t& acc, std::size_t b,
                       std::size_t ib, const int8x16_t& a0,
                       const int8x16_t& a1) {
      const std::size_t base = b * kBs;
      const std::size_t a = b * 256 + ib * 32;
      const float d4 = load_f16(p + base) * 0.25F;
      const std::uint8_t qhb = p[base + 66 + ib];
      const std::uint8_t* qsb = p + base + 2 + ib * 4;
      const std::uint8_t* sgb = p + base + 34 + ib * 4;
      const float spa = sp[a >> 5];
      const std::uint32_t i00 =
          static_cast<std::uint32_t>(qsb[0]) |
          ((static_cast<std::uint32_t>(qhb) & 0x03U) << 8);
      const std::uint32_t i01 =
          static_cast<std::uint32_t>(qsb[1]) |
          ((static_cast<std::uint32_t>(qhb) & 0x0CU) << 6);
      const std::uint32_t i02 =
          static_cast<std::uint32_t>(qsb[2]) |
          ((static_cast<std::uint32_t>(qhb) & 0x30U) << 4);
      const std::uint32_t i03 =
          static_cast<std::uint32_t>(qsb[3]) |
          ((static_cast<std::uint32_t>(qhb) & 0xC0U) << 2);
      const int8x8_t g00 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &tables::iq2s_grid[i00]));
      const int8x8_t g01 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &tables::iq2s_grid[i01]));
      const int8x8_t g02 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &tables::iq2s_grid[i02]));
      const int8x8_t g03 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &tables::iq2s_grid[i03]));
      const int8x8_t m00 =
          vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb[0]]));
      const int8x8_t m01 =
          vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb[1]]));
      const int8x8_t m02 =
          vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb[2]]));
      const int8x8_t m03 =
          vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb[3]]));
      const int8x16_t w01 =
          vcombine_s8(vsub_s8(veor_s8(g00, m00), m00),
                      vsub_s8(veor_s8(g01, m01), m01));
      const int8x16_t w23 =
          vcombine_s8(vsub_s8(veor_s8(g02, m02), m02),
                      vsub_s8(veor_s8(g03, m03), m03));
      const float s01 =
          d4 * (0.5F + static_cast<float>(p[base + 74 + ib] & 0x0Fu)) * spa;
      const float s23 =
          d4 * (0.5F + static_cast<float>((p[base + 74 + ib] >> 4) & 0x0Fu)) *
          spa;
      const int32x4_t q0 = vdotq_s32(vdupq_n_s32(0), w01, a0);
      acc = vfmaq_f32(acc, vcvtq_f32_s32(q0), vdupq_n_f32(s01));
      const int32x4_t q1 = vdotq_s32(vdupq_n_s32(0), w23, a1);
      acc = vfmaq_f32(acc, vcvtq_f32_s32(q1), vdupq_n_f32(s23));
    };

    for (std::size_t b = 0; b < nb; ++b) {
      for (std::size_t ib = 0; ib < 8; ++ib) {
        const std::size_t a = b * 256 + ib * 32;
        const int8x16_t a0 = vld1q_s8(ap + a);
        const int8x16_t a1 = vld1q_s8(ap + a + 16);
        row_dot(p0, acc0, b, ib, a0, a1);
        row_dot(p1, acc1, b, ib, a0, a1);
        row_dot(p2, acc2, b, ib, a0, a1);
        row_dot(p3, acc3, b, ib, a0, a1);
      }
    }
    out[r + 0] = vaddvq_f32(acc0);
    out[r + 1] = vaddvq_f32(acc1);
    out[r + 2] = vaddvq_f32(acc2);
    out[r + 3] = vaddvq_f32(acc3);
  }
  for (; r < nrows; ++r) {
    out[r] = dot_iq2s(rows + r * rb, act, in_dim);
  }
}

// iq3_s: без промежуточного wbuf на стеке, считаем сразу по распакованным int8.
float dot_iq3s(const std::uint8_t* row, const ActQ8& act,
               std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 110;
  const std::size_t nb = in_dim >> 8;
  const int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  float32x4_t accv = vdupq_n_f32(0.0F);
    float32x4_t accv2 = vdupq_n_f32(0.0F);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t base = b * kBs;
    const float d = load_f16(row + base);
    const std::uint8_t* qs = row + base + 2;
    const std::uint8_t* qh = row + base + 66;
    const std::uint8_t* signs = row + base + 74;
    const std::uint8_t* scales = row + base + 106;
    for (std::size_t ib = 0; ib < 4; ++ib) {
      const std::uint8_t sc = scales[ib];
      const float db1 =
          d * (1.0F + 2.0F * static_cast<float>(sc & 0x0FU));
      const float db2 =
          d * (1.0F + 2.0F * static_cast<float>(sc >> 4));
      for (std::size_t half = 0; half < 2; ++half) {
        const float db = half == 0 ? db1 : db2;
        const std::uint32_t qhb = qh[ib * 2 + half];
        const std::uint8_t* qsb = qs + ib * 16 + half * 8;
        const std::uint8_t* sgb = signs + ib * 8 + half * 4;

        const std::uint32_t i00 = static_cast<std::uint32_t>(qsb[0]) | ((qhb << 8) & 256U);
                const std::uint32_t i01 = static_cast<std::uint32_t>(qsb[1]) | ((qhb << 7) & 256U);
                const std::uint32_t i10 = static_cast<std::uint32_t>(qsb[2]) | ((qhb << 6) & 256U);
                const std::uint32_t i11 = static_cast<std::uint32_t>(qsb[3]) | ((qhb << 5) & 256U);
                const std::uint32_t i20 = static_cast<std::uint32_t>(qsb[4]) | ((qhb << 4) & 256U);
                const std::uint32_t i21 = static_cast<std::uint32_t>(qsb[5]) | ((qhb << 3) & 256U);
                const std::uint32_t i30 = static_cast<std::uint32_t>(qsb[6]) | ((qhb << 2) & 256U);
                const std::uint32_t i31 = static_cast<std::uint32_t>(qsb[7]) | ((qhb << 1) & 256U);
                const int8x8_t g0 = grid2_iq3s(i00, i01);
                const int8x8_t g1 = grid2_iq3s(i10, i11);
                const int8x8_t g2 = grid2_iq3s(i20, i21);
                const int8x8_t g3 = grid2_iq3s(i30, i31);
                const int8x8_t m0 = vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb[0]]));
                const int8x8_t m1 = vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb[1]]));
                const int8x8_t m2 = vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb[2]]));
                const int8x8_t m3 = vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb[3]]));
        const int8x16_t w01 = vcombine_s8(vsub_s8(veor_s8(g0, m0), m0),
                                          vsub_s8(veor_s8(g1, m1), m1));
        const int8x16_t w23 = vcombine_s8(vsub_s8(veor_s8(g2, m2), m2),
                                          vsub_s8(veor_s8(g3, m3), m3));

        const std::size_t a = b * 256 + ib * 64 + half * 32;
        int32x4_t s = vdotq_s32(vdupq_n_s32(0), w01, vld1q_s8(ap + a));
        s = vdotq_s32(s, w23, vld1q_s8(ap + a + 16));
        const float32x4_t scale = vdupq_n_f32(db * sp[a >> 5]);
        if (half == 0) { accv = vfmaq_f32(accv, vcvtq_f32_s32(s), scale); }
                else { accv2 = vfmaq_f32(accv2, vcvtq_f32_s32(s), scale); }
      }
    }
  }
  return vaddvq_f32(vaddq_f32(accv, accv2));
}

// iq3_xxs: как iq3_s, другая раскладка qh/scales, тоже без wbuf.
float dot_iq3xxs(const std::uint8_t* row, const ActQ8& act,
                 std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 98;
  const std::size_t nb = in_dim >> 8;
  const int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  float32x4_t accv = vdupq_n_f32(0.0F);
    float32x4_t accv2 = vdupq_n_f32(0.0F);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t base = b * kBs;
    const float d = load_f16(row + base);
    const std::uint8_t* qs = row + base + 2;
    const std::uint8_t* ss = row + base + 66;
    for (std::size_t ib = 0; ib < 8; ++ib) {
      const std::uint32_t aux = load_u32(ss + ib * 4);
      const float db =
          d * (0.5F + static_cast<float>((aux >> 28) & 0x0FU)) * 0.5F;

      const std::uint8_t* qsb = qs + ib * 8;
                const int8x8_t g0 = grid2_iq3xxs(qsb[0], qsb[1]);
                const int8x8_t g1 = grid2_iq3xxs(qsb[2], qsb[3]);
                const int8x8_t g2 = grid2_iq3xxs(qsb[4], qsb[5]);
                const int8x8_t g3 = grid2_iq3xxs(qsb[6], qsb[7]);
                const int8x8_t m0 = vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[tables::ksigns_iq2xs[(aux >> 0) & 127U]]));
                const int8x8_t m1 = vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[tables::ksigns_iq2xs[(aux >> 7) & 127U]]));
                const int8x8_t m2 = vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[tables::ksigns_iq2xs[(aux >> 14) & 127U]]));
                const int8x8_t m3 = vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[tables::ksigns_iq2xs[(aux >> 21) & 127U]]));
      const int8x16_t w01 = vcombine_s8(vsub_s8(veor_s8(g0, m0), m0),
                                        vsub_s8(veor_s8(g1, m1), m1));
      const int8x16_t w23 = vcombine_s8(vsub_s8(veor_s8(g2, m2), m2),
                                        vsub_s8(veor_s8(g3, m3), m3));

      const std::size_t a = b * 256 + ib * 32;
      int32x4_t s = vdotq_s32(vdupq_n_s32(0), w01, vld1q_s8(ap + a));
      s = vdotq_s32(s, w23, vld1q_s8(ap + a + 16));
      const float32x4_t scale = vdupq_n_f32(db * sp[a >> 5]);
      if ((ib & 1) == 0) { accv = vfmaq_f32(accv, vcvtq_f32_s32(s), scale); }
                else { accv2 = vfmaq_f32(accv2, vcvtq_f32_s32(s), scale); }
    }
  }
  return vaddvq_f32(vaddq_f32(accv, accv2));
}

// Four rows at once against one activation. The activation block is loaded
// once, and four independent vdot/fmla chains keep the pipeline busy. This is
// the matvec shape the model actually uses: many rows, one activation.
void dot_rows_q8_0(const std::uint8_t* rows, std::size_t rb, std::size_t nrows,
                   const ActQ8& act, std::size_t in_dim, float* out) noexcept {
  constexpr std::size_t kBs = 34;
  const std::size_t nb = in_dim >> 5;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  std::size_t r = 0;
  for (; r + 4 <= nrows; r += 4) {
    float32x4_t acc0 = vdupq_n_f32(0.0F);
    float32x4_t acc1 = vdupq_n_f32(0.0F);
    float32x4_t acc2 = vdupq_n_f32(0.0F);
    float32x4_t acc3 = vdupq_n_f32(0.0F);
    for (std::size_t b = 0; b < nb; ++b) {
      const int8x16_t a0 = vld1q_s8(ap + b * 32);
      const int8x16_t a1 = vld1q_s8(ap + b * 32 + 16);
      const float s = sp[b];

      const std::uint8_t* p0 = rows + (r + 0) * rb + b * kBs;
      const std::uint8_t* p1 = rows + (r + 1) * rb + b * kBs;
      const std::uint8_t* p2 = rows + (r + 2) * rb + b * kBs;
      const std::uint8_t* p3 = rows + (r + 3) * rb + b * kBs;

      int32x4_t q0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(reinterpret_cast<const std::int8_t*>(p0 + 2)), a0);
      int32x4_t q1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(reinterpret_cast<const std::int8_t*>(p1 + 2)), a0);
      int32x4_t q2 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(reinterpret_cast<const std::int8_t*>(p2 + 2)), a0);
      int32x4_t q3 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(reinterpret_cast<const std::int8_t*>(p3 + 2)), a0);
      q0 = vdotq_s32(q0, vld1q_s8(reinterpret_cast<const std::int8_t*>(p0 + 18)), a1);
      q1 = vdotq_s32(q1, vld1q_s8(reinterpret_cast<const std::int8_t*>(p1 + 18)), a1);
      q2 = vdotq_s32(q2, vld1q_s8(reinterpret_cast<const std::int8_t*>(p2 + 18)), a1);
      q3 = vdotq_s32(q3, vld1q_s8(reinterpret_cast<const std::int8_t*>(p3 + 18)), a1);

      acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(q0),
                       vdupq_n_f32(load_f16(p0) * s));
      acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(q1),
                       vdupq_n_f32(load_f16(p1) * s));
      acc2 = vfmaq_f32(acc2, vcvtq_f32_s32(q2),
                       vdupq_n_f32(load_f16(p2) * s));
      acc3 = vfmaq_f32(acc3, vcvtq_f32_s32(q3),
                       vdupq_n_f32(load_f16(p3) * s));
    }
    out[r + 0] = vaddvq_f32(acc0);
    out[r + 1] = vaddvq_f32(acc1);
    out[r + 2] = vaddvq_f32(acc2);
    out[r + 3] = vaddvq_f32(acc3);
  }
  for (; r < nrows; ++r) {
    out[r] = dot_q8_0(rows + r * rb, act, in_dim);
  }
}

void dot_rows_q4_0(const std::uint8_t* rows, std::size_t rb, std::size_t nrows,
                   const ActQ8& act, std::size_t in_dim, float* out) noexcept {
  constexpr std::size_t kBs = 18;
  const std::size_t nb = in_dim >> 5;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const int8x16_t eight = vdupq_n_s8(8);
  const uint8x16_t nib = vdupq_n_u8(0x0FU);

  std::size_t r = 0;
  for (; r + 4 <= nrows; r += 4) {
    float32x4_t acc0 = vdupq_n_f32(0.0F);
    float32x4_t acc1 = vdupq_n_f32(0.0F);
    float32x4_t acc2 = vdupq_n_f32(0.0F);
    float32x4_t acc3 = vdupq_n_f32(0.0F);
    for (std::size_t b = 0; b < nb; ++b) {
      const int8x16_t a0 = vld1q_s8(ap + b * 32);
      const int8x16_t a1 = vld1q_s8(ap + b * 32 + 16);
      const float s = sp[b];

      const std::uint8_t* p0 = rows + (r + 0) * rb + b * kBs;
      const std::uint8_t* p1 = rows + (r + 1) * rb + b * kBs;
      const std::uint8_t* p2 = rows + (r + 2) * rb + b * kBs;
      const std::uint8_t* p3 = rows + (r + 3) * rb + b * kBs;

      const uint8x16_t q0 = vld1q_u8(p0 + 2);
      const uint8x16_t q1 = vld1q_u8(p1 + 2);
      const uint8x16_t q2 = vld1q_u8(p2 + 2);
      const uint8x16_t q3 = vld1q_u8(p3 + 2);

      const int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q0, nib)), eight);
      const int8x16_t l1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q1, nib)), eight);
      const int8x16_t l2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q2, nib)), eight);
      const int8x16_t l3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3, nib)), eight);
      const int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), eight);
      const int8x16_t h1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)), eight);
      const int8x16_t h2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q2, 4)), eight);
      const int8x16_t h3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q3, 4)), eight);

      int32x4_t s0 = vdotq_s32(vdupq_n_s32(0), l0, a0);
      int32x4_t s1 = vdotq_s32(vdupq_n_s32(0), l1, a0);
      int32x4_t s2 = vdotq_s32(vdupq_n_s32(0), l2, a0);
      int32x4_t s3 = vdotq_s32(vdupq_n_s32(0), l3, a0);
      s0 = vdotq_s32(s0, h0, a1);
      s1 = vdotq_s32(s1, h1, a1);
      s2 = vdotq_s32(s2, h2, a1);
      s3 = vdotq_s32(s3, h3, a1);

      acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(s0),
                       vdupq_n_f32(load_f16(p0) * s));
      acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(s1),
                       vdupq_n_f32(load_f16(p1) * s));
      acc2 = vfmaq_f32(acc2, vcvtq_f32_s32(s2),
                       vdupq_n_f32(load_f16(p2) * s));
      acc3 = vfmaq_f32(acc3, vcvtq_f32_s32(s3),
                       vdupq_n_f32(load_f16(p3) * s));
    }
    out[r + 0] = vaddvq_f32(acc0);
    out[r + 1] = vaddvq_f32(acc1);
    out[r + 2] = vaddvq_f32(acc2);
    out[r + 3] = vaddvq_f32(acc3);
  }
  for (; r < nrows; ++r) {
    out[r] = dot_q4_0(rows + r * rb, act, in_dim);
  }
}

// ---- q4_k: блок 256 весов, 8 сабблоков по 32, шкала+min на сабблок (6 бит,
// упакованы в 12 байт). значение = nibble 0..15, оффсцент уже в min.
// dot с int8-активацией считается точно: w = d*sc*v - dmin*m, поэтому
// sum(w*a) = d*sc*sum(v*q) - dmin*m*sum(q), где сумма q на 32-блоке
// предвычислена один раз на вызов

void get_scale_min_k4(size_t j, const uint8_t* scales, uint8_t& d,
                      uint8_t& m) noexcept {
  if (j < 4) {
    d = static_cast<uint8_t>(scales[j] & 63u);
    m = static_cast<uint8_t>(scales[j + 4] & 63u);
  } else {
    d = static_cast<uint8_t>(static_cast<uint32_t>(scales[j + 4] & 0x0Fu) |
                             (static_cast<uint32_t>(scales[j - 4] >> 6) << 4));
    m = static_cast<uint8_t>(static_cast<uint32_t>(scales[j + 4] >> 4) |
                             (static_cast<uint32_t>(scales[j] >> 6) << 4));
  }
}

namespace {

// сумма int8-активации на каждый блок 32. вызывается раз на dot_rows/dot_row
const std::int32_t* act_block_sums(const ActQ8& act, std::size_t nsb,
                                   std::vector<std::int32_t>& buf) {
  buf.assign(nsb, 0);
  const int8x16_t ones = vreinterpretq_s8_u8(vdupq_n_u8(1));
  const std::int8_t* ap = act.q.data();
  for (std::size_t b = 0; b < nsb; ++b) {
    const int32x4_t s0 =
        vdotq_s32(vdupq_n_s32(0), ones, vld1q_s8(ap + b * 32));
    const int32x4_t s1 =
        vdotq_s32(vdupq_n_s32(0), ones, vld1q_s8(ap + b * 32 + 16));
    buf[b] = vaddvq_s32(vaddq_s32(s0, s1));
  }
  return buf.data();
}

// один 32-весовой сабблок q4_k: nibbles chunk (lo или hi половина) против
// 32 значений активации. возвращает int32 4-lane частичную сумму (полный dot
// сабблока после vaddvq)
[[nodiscard]] inline int32x4_t q4k_sub_dot(const std::uint8_t* chunk,
                                           bool hi, const std::int8_t* ap,
                                           std::size_t a,
                                           const uint8x16_t& nib) noexcept {
  const uint8x16_t c0 = vld1q_u8(chunk);
  const uint8x16_t c1 = vld1q_u8(chunk + 16);
  const int8x16_t w0 =
      hi ? vreinterpretq_s8_u8(vshrq_n_u8(c0, 4))
         : vreinterpretq_s8_u8(vandq_u8(c0, nib));
  const int8x16_t w1 =
      hi ? vreinterpretq_s8_u8(vshrq_n_u8(c1, 4))
         : vreinterpretq_s8_u8(vandq_u8(c1, nib));
  int32x4_t s = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(ap + a));
  s = vdotq_s32(s, w1, vld1q_s8(ap + a + 16));
  return s;
}

// один 32-весовой сабблок q5_k: nibble + старший 5-й бит из qh (по одному
// биту на вес, бит (2j+half) в байте qh[l]). как и q4_k, dot точен:
// w = d*sc*v - dmin*m, поэтому sum(w*a) = d*sc*sum(v*a) - dmin*m*sum(a)
[[nodiscard]] inline int32x4_t q5k_sub_dot(const std::uint8_t* chunk,
                                           const std::uint8_t* qh,
                                           std::size_t sh, bool hi,
                                           const std::int8_t* ap,
                                           std::size_t a,
                                           const uint8x16_t& nib) noexcept {
  const uint8x16_t c0 = vld1q_u8(chunk);
  const uint8x16_t c1 = vld1q_u8(chunk + 16);
  const uint8x16_t b0 =
      vandq_u8(vshlq_u8(vld1q_u8(qh), vdupq_n_s8(-static_cast<int>(sh))),
               vdupq_n_u8(0x01));
  const uint8x16_t b1 =
      vandq_u8(vshlq_u8(vld1q_u8(qh + 16), vdupq_n_s8(-static_cast<int>(sh))),
               vdupq_n_u8(0x01));
  const uint8x16_t n0 =
      hi ? vshrq_n_u8(c0, 4) : vandq_u8(c0, nib);
  const uint8x16_t n1 =
      hi ? vshrq_n_u8(c1, 4) : vandq_u8(c1, nib);
  const int8x16_t w0 =
      vreinterpretq_s8_u8(vorrq_u8(n0, vshlq_n_u8(b0, 4)));
  const int8x16_t w1 =
      vreinterpretq_s8_u8(vorrq_u8(n1, vshlq_n_u8(b1, 4)));
  int32x4_t s = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(ap + a));
  s = vdotq_s32(s, w1, vld1q_s8(ap + a + 16));
  return s;
}

}  // namespace

float dot_q4k(const std::uint8_t* row, const ActQ8& act,
              std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 144;
  const std::size_t nb = in_dim >> 8;
  const std::size_t nsb = in_dim >> 5;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const uint8x16_t nib = vdupq_n_u8(0x0FU);

  thread_local std::vector<std::int32_t> asum_buf;
  const std::int32_t* asum = act_block_sums(act, nsb, asum_buf);

  float32x4_t facc = vdupq_n_f32(0.0F);
  float minacc = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t base = b * kBs;
    const float d = load_f16(row + base);
    const float dmin = load_f16(row + base + 2);
    const std::uint8_t* scales = row + base + 4;
    const std::uint8_t* qs = row + base + 16;
    for (std::size_t j = 0; j < 4; ++j) {
      const std::uint8_t* chunk = qs + j * 32;
      const std::size_t a = b * 256 + j * 64;
      for (std::size_t half = 0; half < 2; ++half) {
        const std::size_t sub = b * 8 + 2 * j + half;
        std::uint8_t sc = 0;
        std::uint8_t m = 0;
        get_scale_min_k4(2 * j + half, scales, sc, m);
        const int32x4_t s =
            q4k_sub_dot(chunk, half != 0, ap, a + half * 32, nib);
        const float scf = d * static_cast<float>(sc) * sp[a >> 5];
        facc = vfmaq_f32(facc, vcvtq_f32_s32(s), vdupq_n_f32(scf));
        minacc += static_cast<float>(asum[sub]) *
                  (dmin * static_cast<float>(m) * sp[a >> 5]);
      }
    }
  }
  return vaddvq_f32(facc) - minacc;
}

// ---- q5_k: блок 256 весов, 176 байт: d, dmin, 12 байт 6-битных scale/min,
// 32 байта qh (старший 5-й бит), 128 байт nibble-кодов. значение =
// d*sc*(nib | bit<<4) - dmin*m. точный dot через vdot + sum(q) на 32-блоке
float dot_q5k(const std::uint8_t* row, const ActQ8& act,
              std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 176;
  const std::size_t nb = in_dim >> 8;
  const std::size_t nsb = in_dim >> 5;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const uint8x16_t nib = vdupq_n_u8(0x0FU);

  thread_local std::vector<std::int32_t> asum_buf;
  const std::int32_t* asum = act_block_sums(act, nsb, asum_buf);

  float32x4_t facc = vdupq_n_f32(0.0F);
  float minacc = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t base = b * kBs;
    const float d = load_f16(row + base);
    const float dmin = load_f16(row + base + 2);
    const std::uint8_t* scales = row + base + 4;
    const std::uint8_t* qh = row + base + 16;
    const std::uint8_t* qs = row + base + 48;
    for (std::size_t j = 0; j < 4; ++j) {
      const std::uint8_t* chunk = qs + j * 32;
      const std::size_t a = b * 256 + j * 64;
      for (std::size_t half = 0; half < 2; ++half) {
        const std::size_t sub = b * 8 + 2 * j + half;
        std::uint8_t sc = 0;
        std::uint8_t m = 0;
        get_scale_min_k4(2 * j + half, scales, sc, m);
        const int32x4_t s = q5k_sub_dot(chunk, qh, 2 * j + half, half != 0,
                                        ap, a + half * 32, nib);
        const float scf = d * static_cast<float>(sc) * sp[a >> 5];
        facc = vfmaq_f32(facc, vcvtq_f32_s32(s), vdupq_n_f32(scf));
        minacc += static_cast<float>(asum[sub]) *
                  (dmin * static_cast<float>(m) * sp[a >> 5]);
      }
    }
  }
  return vaddvq_f32(facc) - minacc;
}

// ---- bf16: bf16 = старшие 16 бит float. конверсия сдвигом, блок активации
// 32: sum(w * q) * scale_b. шкала весов своя на элемент, поэтому применяем
// масштаб активации один раз на 32-блок
float dot_bf16(const std::uint8_t* row, const ActQ8& act,
               std::size_t in_dim) noexcept {
  const std::size_t nb = in_dim >> 5;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const std::uint16_t* wp = reinterpret_cast<const std::uint16_t*>(row);

  float32x4_t facc = vdupq_n_f32(0.0F);
  for (std::size_t b = 0; b < nb; ++b) {
    const uint16x8_t h0 = vld1q_u16(wp + b * 32);
    const uint16x8_t h1 = vld1q_u16(wp + b * 32 + 8);
    const uint16x8_t h2 = vld1q_u16(wp + b * 32 + 16);
    const uint16x8_t h3 = vld1q_u16(wp + b * 32 + 24);
    const float32x4_t w0 = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_low_u16(h0)), 16));
    const float32x4_t w1 = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_high_u16(h0)), 16));
    const float32x4_t w2 = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_low_u16(h1)), 16));
    const float32x4_t w3 = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_high_u16(h1)), 16));
    const float32x4_t w4 = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_low_u16(h2)), 16));
    const float32x4_t w5 = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_high_u16(h2)), 16));
    const float32x4_t w6 = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_low_u16(h3)), 16));
    const float32x4_t w7 = vreinterpretq_f32_u32(
        vshlq_n_u32(vmovl_u16(vget_high_u16(h3)), 16));

    const int8x16_t q0 = vld1q_s8(ap + b * 32);
    const int8x16_t q1 = vld1q_s8(ap + b * 32 + 16);
    const int16x8_t s00 = vmovl_s8(vget_low_s8(q0));
    const int16x8_t s01 = vmovl_s8(vget_high_s8(q0));
    const int16x8_t s10 = vmovl_s8(vget_low_s8(q1));
    const int16x8_t s11 = vmovl_s8(vget_high_s8(q1));
    const float32x4_t x0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s00)));
    const float32x4_t x1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s00)));
    const float32x4_t x2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s01)));
    const float32x4_t x3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s01)));
    const float32x4_t x4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s10)));
    const float32x4_t x5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s10)));
    const float32x4_t x6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s11)));
    const float32x4_t x7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s11)));

    float32x4_t acc = vmulq_f32(w0, x0);
    acc = vfmaq_f32(acc, w1, x1);
    acc = vfmaq_f32(acc, w2, x2);
    acc = vfmaq_f32(acc, w3, x3);
    acc = vfmaq_f32(acc, w4, x4);
    acc = vfmaq_f32(acc, w5, x5);
    acc = vfmaq_f32(acc, w6, x6);
    acc = vfmaq_f32(acc, w7, x7);
    facc = vfmaq_f32(facc, acc, vdupq_n_f32(sp[b]));
  }
  return vaddvq_f32(facc);
}

void dot_rows_q4k(const std::uint8_t* rows, std::size_t rb, std::size_t nrows,
                  const ActQ8& act, std::size_t in_dim, float* out) noexcept {
  constexpr std::size_t kBs = 144;
  const std::size_t nb = in_dim >> 8;
  const std::size_t nsb = in_dim >> 5;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const uint8x16_t nib = vdupq_n_u8(0x0FU);

  thread_local std::vector<std::int32_t> asum_buf;
  const std::int32_t* asum = act_block_sums(act, nsb, asum_buf);

  std::size_t r = 0;
  for (; r + 4 <= nrows; r += 4) {
    float32x4_t f0 = vdupq_n_f32(0.0F);
    float32x4_t f1 = vdupq_n_f32(0.0F);
    float32x4_t f2 = vdupq_n_f32(0.0F);
    float32x4_t f3 = vdupq_n_f32(0.0F);
    float mn0 = 0.0F;
    float mn1 = 0.0F;
    float mn2 = 0.0F;
    float mn3 = 0.0F;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::size_t base = b * kBs;
      const std::uint8_t* p0 = rows + (r + 0) * rb + base;
      const std::uint8_t* p1 = rows + (r + 1) * rb + base;
      const std::uint8_t* p2 = rows + (r + 2) * rb + base;
      const std::uint8_t* p3 = rows + (r + 3) * rb + base;
      // d/dmin не зависят от сабблока — поднимаем на блок
      const float d0 = load_f16(p0);
      const float d1 = load_f16(p1);
      const float d2 = load_f16(p2);
      const float d3 = load_f16(p3);
      const float dm0 = load_f16(p0 + 2);
      const float dm1 = load_f16(p1 + 2);
      const float dm2 = load_f16(p2 + 2);
      const float dm3 = load_f16(p3 + 2);
      const std::uint8_t* q0 = p0 + 16;
      const std::uint8_t* q1 = p1 + 16;
      const std::uint8_t* q2 = p2 + 16;
      const std::uint8_t* q3 = p3 + 16;
      for (std::size_t j = 0; j < 4; ++j) {
        const std::uint8_t* c0 = q0 + j * 32;
        const std::uint8_t* c1 = q1 + j * 32;
        const std::uint8_t* c2 = q2 + j * 32;
        const std::uint8_t* c3 = q3 + j * 32;
        const std::size_t a = b * 256 + j * 64;
        const float spa = sp[a >> 5];
        for (std::size_t half = 0; half < 2; ++half) {
          const std::size_t sub = b * 8 + 2 * j + half;
          std::uint8_t sc0 = 0;
          std::uint8_t m0 = 0;
          std::uint8_t sc1 = 0;
          std::uint8_t m1 = 0;
          std::uint8_t sc2 = 0;
          std::uint8_t m2 = 0;
          std::uint8_t sc3 = 0;
          std::uint8_t m3 = 0;
          get_scale_min_k4(2 * j + half, p0 + 4, sc0, m0);
          get_scale_min_k4(2 * j + half, p1 + 4, sc1, m1);
          get_scale_min_k4(2 * j + half, p2 + 4, sc2, m2);
          get_scale_min_k4(2 * j + half, p3 + 4, sc3, m3);
          const int32x4_t s0 =
              q4k_sub_dot(c0, half != 0, ap, a + half * 32, nib);
          const int32x4_t s1 =
              q4k_sub_dot(c1, half != 0, ap, a + half * 32, nib);
          const int32x4_t s2 =
              q4k_sub_dot(c2, half != 0, ap, a + half * 32, nib);
          const int32x4_t s3 =
              q4k_sub_dot(c3, half != 0, ap, a + half * 32, nib);
          f0 = vfmaq_f32(f0, vcvtq_f32_s32(s0),
                         vdupq_n_f32(d0 * static_cast<float>(sc0) * spa));
          f1 = vfmaq_f32(f1, vcvtq_f32_s32(s1),
                         vdupq_n_f32(d1 * static_cast<float>(sc1) * spa));
          f2 = vfmaq_f32(f2, vcvtq_f32_s32(s2),
                         vdupq_n_f32(d2 * static_cast<float>(sc2) * spa));
          f3 = vfmaq_f32(f3, vcvtq_f32_s32(s3),
                         vdupq_n_f32(d3 * static_cast<float>(sc3) * spa));
          mn0 += static_cast<float>(asum[sub]) *
                 (dm0 * static_cast<float>(m0) * spa);
          mn1 += static_cast<float>(asum[sub]) *
                 (dm1 * static_cast<float>(m1) * spa);
          mn2 += static_cast<float>(asum[sub]) *
                 (dm2 * static_cast<float>(m2) * spa);
          mn3 += static_cast<float>(asum[sub]) *
                 (dm3 * static_cast<float>(m3) * spa);
        }
      }
    }
    out[r + 0] = vaddvq_f32(f0) - mn0;
    out[r + 1] = vaddvq_f32(f1) - mn1;
    out[r + 2] = vaddvq_f32(f2) - mn2;
    out[r + 3] = vaddvq_f32(f3) - mn3;
  }
  for (; r < nrows; ++r) {
    out[r] = dot_q4k(rows + r * rb, act, in_dim);
  }
}

// ---- двухстрочные IQ-ядра: пара строк (r, r+1) за проход. деквант каждой
// строки остаётся своим (решётки/знаки разные), но загрузка активации и
// независимые цепочки деквант+dot идут параллельно, поэтому латентность
// закрывается вдвое плотнее. включается TOKSHIT_IQ2ROW=1
// TOKSHIT_IQ2ROW=1

// iq4_xs, две строки
void dot_rows_iq4xs(const std::uint8_t* rows, std::size_t rb,
                    std::size_t nrows, const ActQ8& act, std::size_t in_dim,
                    float* out) noexcept {
  constexpr std::size_t kBs = 136;
  const std::size_t nb = in_dim >> 8;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  const int8x16_t tab = vld1q_s8(tables::kvalues_iq4nl.data());
  const uint8x16_t lowmask = vdupq_n_u8(0x0FU);

  std::size_t r = 0;
  for (; r + 2 <= nrows; r += 2) {
    float32x4_t e0 = vdupq_n_f32(0.0F);
    float32x4_t o0 = vdupq_n_f32(0.0F);
    float32x4_t e1 = vdupq_n_f32(0.0F);
    float32x4_t o1 = vdupq_n_f32(0.0F);
    for (std::size_t b = 0; b < nb; ++b) {
      const std::size_t base0 = r * rb + b * kBs;
      const std::size_t base1 = (r + 1) * rb + b * kBs;
      const float d0 = load_f16(rows + base0);
      const float d1 = load_f16(rows + base1);
      const std::uint32_t sh0 = load_u16(rows + base0 + 2);
      const std::uint32_t sh1 = load_u16(rows + base1 + 2);
      const std::uint8_t* sl0 = rows + base0 + 4;
      const std::uint8_t* sl1 = rows + base1 + 4;
      const std::uint8_t* qs0 = rows + base0 + 8;
      const std::uint8_t* qs1 = rows + base1 + 8;
      for (std::size_t ib = 0; ib < 8; ++ib) {
        const std::uint32_t ls0 =
            (static_cast<std::uint32_t>(
                 (static_cast<std::uint32_t>(sl0[ib / 2]) >> (4 * (ib % 2))) &
                 0x0FU)) |
            ((static_cast<std::uint32_t>(sh0 >> (2 * ib)) & 3U) << 4);
        const std::uint32_t ls1 =
            (static_cast<std::uint32_t>(
                 (static_cast<std::uint32_t>(sl1[ib / 2]) >> (4 * (ib % 2))) &
                 0x0FU)) |
            ((static_cast<std::uint32_t>(sh1 >> (2 * ib)) & 3U) << 4);
        const float dl0 = d0 * static_cast<float>(static_cast<int32_t>(ls0) - 32);
        const float dl1 = d1 * static_cast<float>(static_cast<int32_t>(ls1) - 32);

        const std::size_t a = b * 256 + ib * 32;
        const int8x16_t a0 = vld1q_s8(ap + a);
        const int8x16_t a1 = vld1q_s8(ap + a + 16);

        const uint8x16_t qv0 = vld1q_u8(qs0 + ib * 16);
        const int8x16_t wlo0 = vqtbl1q_s8(tab, vandq_u8(qv0, lowmask));
        const int8x16_t whi0 = vqtbl1q_s8(tab, vshrq_n_u8(qv0, 4));
        const uint8x16_t qv1 = vld1q_u8(qs1 + ib * 16);
        const int8x16_t wlo1 = vqtbl1q_s8(tab, vandq_u8(qv1, lowmask));
        const int8x16_t whi1 = vqtbl1q_s8(tab, vshrq_n_u8(qv1, 4));

        int32x4_t s0 = vdotq_s32(vdupq_n_s32(0), wlo0, a0);
        s0 = vdotq_s32(s0, whi0, a1);
        int32x4_t s1 = vdotq_s32(vdupq_n_s32(0), wlo1, a0);
        s1 = vdotq_s32(s1, whi1, a1);

        const float32x4_t sc0 = vdupq_n_f32(dl0 * sp[a >> 5]);
        const float32x4_t sc1 = vdupq_n_f32(dl1 * sp[a >> 5]);
        if (((b + ib) & 1) != 0) {
          o0 = vfmaq_f32(o0, vcvtq_f32_s32(s0), sc0);
          o1 = vfmaq_f32(o1, vcvtq_f32_s32(s1), sc1);
        } else {
          e0 = vfmaq_f32(e0, vcvtq_f32_s32(s0), sc0);
          e1 = vfmaq_f32(e1, vcvtq_f32_s32(s1), sc1);
        }
      }
    }
    out[r] = vaddvq_f32(vaddq_f32(e0, o0));
    out[r + 1] = vaddvq_f32(vaddq_f32(e1, o1));
  }
  for (; r < nrows; ++r) {
    out[r] = dot_iq4xs(rows + r * rb, act, in_dim);
  }
}

// iq3_s, две строки
void dot_rows_iq3s(const std::uint8_t* rows, std::size_t rb, std::size_t nrows,
                   const ActQ8& act, std::size_t in_dim, float* out) noexcept {
  constexpr std::size_t kBs = 110;
  const std::size_t nb = in_dim >> 8;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  std::size_t r = 0;
  for (; r + 2 <= nrows; r += 2) {
    float32x4_t f0 = vdupq_n_f32(0.0F);
    float32x4_t f0b = vdupq_n_f32(0.0F);
    float32x4_t f1 = vdupq_n_f32(0.0F);
    float32x4_t f1b = vdupq_n_f32(0.0F);
    for (std::size_t b = 0; b < nb; ++b) {
      const std::size_t base0 = r * rb + b * kBs;
      const std::size_t base1 = (r + 1) * rb + b * kBs;
      const float d0 = load_f16(rows + base0);
      const float d1 = load_f16(rows + base1);
      const std::uint8_t* qs0 = rows + base0 + 2;
      const std::uint8_t* qs1 = rows + base1 + 2;
      const std::uint8_t* qh0 = rows + base0 + 66;
      const std::uint8_t* qh1 = rows + base1 + 66;
      const std::uint8_t* sg0 = rows + base0 + 74;
      const std::uint8_t* sg1 = rows + base1 + 74;
      const std::uint8_t* scales0 = rows + base0 + 106;
      const std::uint8_t* scales1 = rows + base1 + 106;
      for (std::size_t ib = 0; ib < 4; ++ib) {
        const std::uint8_t sc0 = scales0[ib];
        const std::uint8_t sc1 = scales1[ib];
        const float db0 =
            d0 * (1.0F + 2.0F * static_cast<float>(sc0 & 0x0FU));
        const float db0b =
            d0 * (1.0F + 2.0F * static_cast<float>(sc0 >> 4));
        const float db1 =
            d1 * (1.0F + 2.0F * static_cast<float>(sc1 & 0x0FU));
        const float db1b =
            d1 * (1.0F + 2.0F * static_cast<float>(sc1 >> 4));
        for (std::size_t half = 0; half < 2; ++half) {
          const std::uint32_t qhb0 = qh0[ib * 2 + half];
          const std::uint32_t qhb1 = qh1[ib * 2 + half];
          const std::uint8_t* qsb0 = qs0 + ib * 16 + half * 8;
          const std::uint8_t* qsb1 = qs1 + ib * 16 + half * 8;
          const std::uint8_t* sgb0 = sg0 + ib * 8 + half * 4;
          const std::uint8_t* sgb1 = sg1 + ib * 8 + half * 4;

          const std::uint32_t i00 =
              static_cast<std::uint32_t>(qsb0[0]) | ((qhb0 << 8) & 256U);
          const std::uint32_t i01 =
              static_cast<std::uint32_t>(qsb0[1]) | ((qhb0 << 7) & 256U);
          const std::uint32_t i10 =
              static_cast<std::uint32_t>(qsb0[2]) | ((qhb0 << 6) & 256U);
          const std::uint32_t i11 =
              static_cast<std::uint32_t>(qsb0[3]) | ((qhb0 << 5) & 256U);
          const std::uint32_t i20 =
              static_cast<std::uint32_t>(qsb0[4]) | ((qhb0 << 4) & 256U);
          const std::uint32_t i21 =
              static_cast<std::uint32_t>(qsb0[5]) | ((qhb0 << 3) & 256U);
          const std::uint32_t i30 =
              static_cast<std::uint32_t>(qsb0[6]) | ((qhb0 << 2) & 256U);
          const std::uint32_t i31 =
              static_cast<std::uint32_t>(qsb0[7]) | ((qhb0 << 1) & 256U);
          const std::uint32_t j00 =
              static_cast<std::uint32_t>(qsb1[0]) | ((qhb1 << 8) & 256U);
          const std::uint32_t j01 =
              static_cast<std::uint32_t>(qsb1[1]) | ((qhb1 << 7) & 256U);
          const std::uint32_t j10 =
              static_cast<std::uint32_t>(qsb1[2]) | ((qhb1 << 6) & 256U);
          const std::uint32_t j11 =
              static_cast<std::uint32_t>(qsb1[3]) | ((qhb1 << 5) & 256U);
          const std::uint32_t j20 =
              static_cast<std::uint32_t>(qsb1[4]) | ((qhb1 << 4) & 256U);
          const std::uint32_t j21 =
              static_cast<std::uint32_t>(qsb1[5]) | ((qhb1 << 3) & 256U);
          const std::uint32_t j30 =
              static_cast<std::uint32_t>(qsb1[6]) | ((qhb1 << 2) & 256U);
          const std::uint32_t j31 =
              static_cast<std::uint32_t>(qsb1[7]) | ((qhb1 << 1) & 256U);

          const int8x8_t g00 = grid2_iq3s(i00, i01);
          const int8x8_t g01 = grid2_iq3s(i10, i11);
          const int8x8_t g02 = grid2_iq3s(i20, i21);
          const int8x8_t g03 = grid2_iq3s(i30, i31);
          const int8x8_t g10 = grid2_iq3s(j00, j01);
          const int8x8_t g11 = grid2_iq3s(j10, j11);
          const int8x8_t g12 = grid2_iq3s(j20, j21);
          const int8x8_t g13 = grid2_iq3s(j30, j31);
          const int8x8_t m00 =
              vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb0[0]]));
          const int8x8_t m01 =
              vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb0[1]]));
          const int8x8_t m02 =
              vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb0[2]]));
          const int8x8_t m03 =
              vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb0[3]]));
          const int8x8_t m10 =
              vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb1[0]]));
          const int8x8_t m11 =
              vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb1[1]]));
          const int8x8_t m12 =
              vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb1[2]]));
          const int8x8_t m13 =
              vld1_s8(reinterpret_cast<const std::int8_t*>(&kMask8[sgb1[3]]));

          const std::size_t a = b * 256 + ib * 64 + half * 32;
          const int8x16_t w00 = vcombine_s8(
              vsub_s8(veor_s8(g00, m00), m00), vsub_s8(veor_s8(g01, m01), m01));
          const int8x16_t w01 = vcombine_s8(
              vsub_s8(veor_s8(g02, m02), m02), vsub_s8(veor_s8(g03, m03), m03));
          const int8x16_t w10 = vcombine_s8(
              vsub_s8(veor_s8(g10, m10), m10), vsub_s8(veor_s8(g11, m11), m11));
          const int8x16_t w11 = vcombine_s8(
              vsub_s8(veor_s8(g12, m12), m12), vsub_s8(veor_s8(g13, m13), m13));

          int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), w00, vld1q_s8(ap + a));
          p0 = vdotq_s32(p0, w01, vld1q_s8(ap + a + 16));
          int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), w10, vld1q_s8(ap + a));
          p1 = vdotq_s32(p1, w11, vld1q_s8(ap + a + 16));

          const float32x4_t sc0 =
              vdupq_n_f32((half == 0 ? db0 : db0b) * sp[a >> 5]);
          const float32x4_t sc1 =
              vdupq_n_f32((half == 0 ? db1 : db1b) * sp[a >> 5]);
          if (half == 0) {
            f0 = vfmaq_f32(f0, vcvtq_f32_s32(p0), sc0);
            f1 = vfmaq_f32(f1, vcvtq_f32_s32(p1), sc1);
          } else {
            f0b = vfmaq_f32(f0b, vcvtq_f32_s32(p0), sc0);
            f1b = vfmaq_f32(f1b, vcvtq_f32_s32(p1), sc1);
          }
        }
      }
    }
    out[r] = vaddvq_f32(vaddq_f32(f0, f0b));
    out[r + 1] = vaddvq_f32(vaddq_f32(f1, f1b));
  }
  for (; r < nrows; ++r) {
    out[r] = dot_iq3s(rows + r * rb, act, in_dim);
  }
}

// iq3_xxs, две строки
void dot_rows_iq3xxs(const std::uint8_t* rows, std::size_t rb,
                     std::size_t nrows, const ActQ8& act, std::size_t in_dim,
                     float* out) noexcept {
  constexpr std::size_t kBs = 98;
  const std::size_t nb = in_dim >> 8;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  std::size_t r = 0;
  for (; r + 2 <= nrows; r += 2) {
    float32x4_t f0 = vdupq_n_f32(0.0F);
    float32x4_t f0b = vdupq_n_f32(0.0F);
    float32x4_t f1 = vdupq_n_f32(0.0F);
    float32x4_t f1b = vdupq_n_f32(0.0F);
    for (std::size_t b = 0; b < nb; ++b) {
      const std::size_t base0 = r * rb + b * kBs;
      const std::size_t base1 = (r + 1) * rb + b * kBs;
      const float d0 = load_f16(rows + base0);
      const float d1 = load_f16(rows + base1);
      const std::uint8_t* qs0 = rows + base0 + 2;
      const std::uint8_t* qs1 = rows + base1 + 2;
      const std::uint8_t* ss0 = rows + base0 + 66;
      const std::uint8_t* ss1 = rows + base1 + 66;
      for (std::size_t ib = 0; ib < 8; ++ib) {
        const std::uint32_t aux0 = load_u32(ss0 + ib * 4);
        const std::uint32_t aux1 = load_u32(ss1 + ib * 4);
        const float db0 =
            d0 * (0.5F + static_cast<float>((aux0 >> 28) & 0x0FU)) * 0.5F;
        const float db1 =
            d1 * (0.5F + static_cast<float>((aux1 >> 28) & 0x0FU)) * 0.5F;

        const std::uint8_t* qsb0 = qs0 + ib * 8;
        const std::uint8_t* qsb1 = qs1 + ib * 8;
        const int8x8_t g00 = grid2_iq3xxs(qsb0[0], qsb0[1]);
        const int8x8_t g01 = grid2_iq3xxs(qsb0[2], qsb0[3]);
        const int8x8_t g02 = grid2_iq3xxs(qsb0[4], qsb0[5]);
        const int8x8_t g03 = grid2_iq3xxs(qsb0[6], qsb0[7]);
        const int8x8_t g10 = grid2_iq3xxs(qsb1[0], qsb1[1]);
        const int8x8_t g11 = grid2_iq3xxs(qsb1[2], qsb1[3]);
        const int8x8_t g12 = grid2_iq3xxs(qsb1[4], qsb1[5]);
        const int8x8_t g13 = grid2_iq3xxs(qsb1[6], qsb1[7]);
        const int8x8_t m00 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &kMask8[tables::ksigns_iq2xs[(aux0 >> 0) & 127U]]));
        const int8x8_t m01 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &kMask8[tables::ksigns_iq2xs[(aux0 >> 7) & 127U]]));
        const int8x8_t m02 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &kMask8[tables::ksigns_iq2xs[(aux0 >> 14) & 127U]]));
        const int8x8_t m03 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &kMask8[tables::ksigns_iq2xs[(aux0 >> 21) & 127U]]));
        const int8x8_t m10 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &kMask8[tables::ksigns_iq2xs[(aux1 >> 0) & 127U]]));
        const int8x8_t m11 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &kMask8[tables::ksigns_iq2xs[(aux1 >> 7) & 127U]]));
        const int8x8_t m12 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &kMask8[tables::ksigns_iq2xs[(aux1 >> 14) & 127U]]));
        const int8x8_t m13 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &kMask8[tables::ksigns_iq2xs[(aux1 >> 21) & 127U]]));

        const std::size_t a = b * 256 + ib * 32;
        const int8x16_t w00 = vcombine_s8(
            vsub_s8(veor_s8(g00, m00), m00), vsub_s8(veor_s8(g01, m01), m01));
        const int8x16_t w01 = vcombine_s8(
            vsub_s8(veor_s8(g02, m02), m02), vsub_s8(veor_s8(g03, m03), m03));
        const int8x16_t w10 = vcombine_s8(
            vsub_s8(veor_s8(g10, m10), m10), vsub_s8(veor_s8(g11, m11), m11));
        const int8x16_t w11 = vcombine_s8(
            vsub_s8(veor_s8(g12, m12), m12), vsub_s8(veor_s8(g13, m13), m13));

        int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), w00, vld1q_s8(ap + a));
        p0 = vdotq_s32(p0, w01, vld1q_s8(ap + a + 16));
        int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), w10, vld1q_s8(ap + a));
        p1 = vdotq_s32(p1, w11, vld1q_s8(ap + a + 16));

        const float32x4_t sc0 = vdupq_n_f32(db0 * sp[a >> 5]);
        const float32x4_t sc1 = vdupq_n_f32(db1 * sp[a >> 5]);
        if ((ib & 1) == 0) {
          f0 = vfmaq_f32(f0, vcvtq_f32_s32(p0), sc0);
          f1 = vfmaq_f32(f1, vcvtq_f32_s32(p1), sc1);
        } else {
          f0b = vfmaq_f32(f0b, vcvtq_f32_s32(p0), sc0);
          f1b = vfmaq_f32(f1b, vcvtq_f32_s32(p1), sc1);
        }
      }
    }
    out[r] = vaddvq_f32(vaddq_f32(f0, f0b));
    out[r + 1] = vaddvq_f32(vaddq_f32(f1, f1b));
  }
  for (; r < nrows; ++r) {
    out[r] = dot_iq3xxs(rows + r * rb, act, in_dim);
  }
}


// ---- q2_0: блок 64, 2 бита, значения -1,0,1,2; шкала f16 на 64.
// активация квантована по 32, поэтому блок веса делится на два сабблока.
float dot_q2_0(const std::uint8_t* row, const ActQ8& act,
               std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 18;
  constexpr std::size_t kQk = 64;
  const std::size_t nb = in_dim / kQk;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  float32x4_t acc = vdupq_n_f32(0.0F);
  const uint8x16_t mask2 = vdupq_n_u8(0x03U);
  const int8x16_t one = vdupq_n_s8(1);
  for (std::size_t b = 0; b < nb; ++b) {
    const float d = load_f16(row + b * kBs);
    const uint8x16_t q = vld1q_u8(row + b * kBs + 2);
    const uint8x16_t a = vandq_u8(q, mask2);
    const uint8x16_t c =
        vandq_u8(vshrq_n_u8(q, 2), mask2);
    const uint8x16_t e =
        vandq_u8(vshrq_n_u8(q, 4), mask2);
    const uint8x16_t f = vshrq_n_u8(q, 6);
    const uint8x16_t ab_lo = vzip1q_u8(a, c);
    const uint8x16_t ab_hi = vzip2q_u8(a, c);
    const uint8x16_t ef_lo = vzip1q_u8(e, f);
    const uint8x16_t ef_hi = vzip2q_u8(e, f);
    // второй уровень склейки — по 16-битным парам: (a,c) с (e,f), иначе
    // получается a,e,c,f вместо a,c,e,f
    const uint16x8_t ab16 = vreinterpretq_u16_u8(ab_lo);
    const uint16x8_t ef16 = vreinterpretq_u16_u8(ef_lo);
    const uint16x8_t ab16h = vreinterpretq_u16_u8(ab_hi);
    const uint16x8_t ef16h = vreinterpretq_u16_u8(ef_hi);
    const int8x16_t v0 =
        vsubq_s8(vreinterpretq_s8_u8(vreinterpretq_u8_u16(
                     vzip1q_u16(ab16, ef16))), one);
    const int8x16_t v1 =
        vsubq_s8(vreinterpretq_s8_u8(vreinterpretq_u8_u16(
                     vzip2q_u16(ab16, ef16))), one);
    const int8x16_t v2 =
        vsubq_s8(vreinterpretq_s8_u8(vreinterpretq_u8_u16(
                     vzip1q_u16(ab16h, ef16h))), one);
    const int8x16_t v3 =
        vsubq_s8(vreinterpretq_s8_u8(vreinterpretq_u8_u16(
                     vzip2q_u16(ab16h, ef16h))), one);
    int32x4_t s0 = vdotq_s32(vdupq_n_s32(0), v0, vld1q_s8(ap + b * 64));
    s0 = vdotq_s32(s0, v1, vld1q_s8(ap + b * 64 + 16));
    int32x4_t s1 = vdotq_s32(vdupq_n_s32(0), v2, vld1q_s8(ap + b * 64 + 32));
    s1 = vdotq_s32(s1, v3, vld1q_s8(ap + b * 64 + 48));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(s0),
                    vdupq_n_f32(d * sp[2 * b]));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(s1),
                    vdupq_n_f32(d * sp[2 * b + 1]));
  }
  return vaddvq_f32(acc);
}

void dot_rows_q2_0(const std::uint8_t* rows, std::size_t rb,
                   std::size_t nrows, const ActQ8& act,
                   std::size_t in_dim, float* out) noexcept {
  constexpr std::size_t kBs = 18;
  constexpr std::size_t kQk = 64;
  const std::size_t nb = in_dim / kQk;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const uint8x16_t mask2 = vdupq_n_u8(0x03U);
  const int8x16_t one = vdupq_n_s8(1);
  std::size_t r = 0;
  for (; r + 4 <= nrows; r += 4) {
    const std::uint8_t* r0 = rows + (r + 0) * rb;
    const std::uint8_t* r1 = rows + (r + 1) * rb;
    const std::uint8_t* r2 = rows + (r + 2) * rb;
    const std::uint8_t* r3 = rows + (r + 3) * rb;
    float32x4_t a0 = vdupq_n_f32(0.0F);
    float32x4_t a1 = vdupq_n_f32(0.0F);
    float32x4_t a2 = vdupq_n_f32(0.0F);
    float32x4_t a3 = vdupq_n_f32(0.0F);
    for (std::size_t b = 0; b < nb; ++b) {
      const int8x16_t x0 = vld1q_s8(ap + b * 64);
      const int8x16_t x1 = vld1q_s8(ap + b * 64 + 16);
      const int8x16_t x2 = vld1q_s8(ap + b * 64 + 32);
      const int8x16_t x3 = vld1q_s8(ap + b * 64 + 48);
      const float s0 = sp[2 * b];
      const float s1 = sp[2 * b + 1];
      auto acc_row = [&](const std::uint8_t* row, float32x4_t& acc) {
        const float d = load_f16(row + b * kBs);
        const uint8x16_t q = vld1q_u8(row + b * kBs + 2);
        const uint8x16_t a = vandq_u8(q, mask2);
        const uint8x16_t c = vandq_u8(vshrq_n_u8(q, 2), mask2);
        const uint8x16_t e = vandq_u8(vshrq_n_u8(q, 4), mask2);
        const uint8x16_t f = vshrq_n_u8(q, 6);
        const uint8x16_t ab_lo = vzip1q_u8(a, c);
        const uint8x16_t ab_hi = vzip2q_u8(a, c);
        const uint8x16_t ef_lo = vzip1q_u8(e, f);
        const uint8x16_t ef_hi = vzip2q_u8(e, f);
        const uint16x8_t ab16 = vreinterpretq_u16_u8(ab_lo);
        const uint16x8_t ef16 = vreinterpretq_u16_u8(ef_lo);
        const uint16x8_t ab16h = vreinterpretq_u16_u8(ab_hi);
        const uint16x8_t ef16h = vreinterpretq_u16_u8(ef_hi);
        const int8x16_t v0 =
            vsubq_s8(vreinterpretq_s8_u8(vreinterpretq_u8_u16(
                         vzip1q_u16(ab16, ef16))), one);
        const int8x16_t v1 =
            vsubq_s8(vreinterpretq_s8_u8(vreinterpretq_u8_u16(
                         vzip2q_u16(ab16, ef16))), one);
        const int8x16_t v2 =
            vsubq_s8(vreinterpretq_s8_u8(vreinterpretq_u8_u16(
                         vzip1q_u16(ab16h, ef16h))), one);
        const int8x16_t v3 =
            vsubq_s8(vreinterpretq_s8_u8(vreinterpretq_u8_u16(
                         vzip2q_u16(ab16h, ef16h))), one);
        int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), v0, x0);
        p0 = vdotq_s32(p0, v1, x1);
        int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), v2, x2);
        p1 = vdotq_s32(p1, v3, x3);
        acc = vfmaq_f32(acc, vcvtq_f32_s32(p0),
                        vdupq_n_f32(d * s0));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(p1),
                        vdupq_n_f32(d * s1));
      };
      acc_row(r0, a0);
      acc_row(r1, a1);
      acc_row(r2, a2);
      acc_row(r3, a3);
    }
    out[r + 0] = vaddvq_f32(a0);
    out[r + 1] = vaddvq_f32(a1);
    out[r + 2] = vaddvq_f32(a2);
    out[r + 3] = vaddvq_f32(a3);
  }
  for (; r < nrows; ++r) {
    out[r] = dot_q2_0(rows + r * rb, act, in_dim);
  }
}

[[nodiscard]] inline float e8m0_half(std::uint8_t x) noexcept {
  const std::uint32_t bits =
      x < 2 ? (0x00200000u << x) : (static_cast<std::uint32_t>(x - 1) << 23);
  return std::bit_cast<float>(bits);
}

[[nodiscard]] inline float ue4m3_half(std::uint8_t x) noexcept {
  if (x == 0 || x == 0x7F) return 0.0F;
  const int exp = (x >> 3) & 0xF;
  const int man = x & 0x7;
  const float raw =
      exp == 0 ? std::ldexp(static_cast<float>(man), -9)
               : std::ldexp(1.0F + static_cast<float>(man) / 8.0F, exp - 7);
  return raw * 0.5F;
}

// ---- mxfp4: блок 32, e8m0 + 16 байт e2m1 (кодбук удвоен).
float dot_mxfp4(const std::uint8_t* row, const ActQ8& act,
                std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 17;
  const std::size_t nb = in_dim / 32;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const int8x16_t tbl = vld1q_s8(
      reinterpret_cast<const std::int8_t*>(dequant_tables::kvalues_fp4.data()));
  const uint8x16_t mask4 = vdupq_n_u8(0x0FU);
  float32x4_t acc = vdupq_n_f32(0.0F);
  for (std::size_t b = 0; b < nb; ++b) {
    const float d = e8m0_half(row[b * kBs]);
    const uint8x16_t q = vld1q_u8(row + b * kBs + 1);
    const int8x16_t wlo = vqtbl1q_s8(tbl, vandq_u8(q, mask4));
    const int8x16_t whi = vqtbl1q_s8(tbl, vshrq_n_u8(q, 4));
    int32x4_t s = vdotq_s32(vdupq_n_s32(0), wlo, vld1q_s8(ap + b * 32));
    s = vdotq_s32(s, whi, vld1q_s8(ap + b * 32 + 16));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(s), vdupq_n_f32(d * sp[b]));
  }
  return vaddvq_f32(acc);
}

// ---- nvfp4: блок 64, 4 ue4m3 сабшкалы на 16 весов, кодбук e2m1.
float dot_nvfp4(const std::uint8_t* row, const ActQ8& act,
                std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 36;
  const std::size_t nb = in_dim / 64;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const int8x16_t tbl = vld1q_s8(
      reinterpret_cast<const std::int8_t*>(dequant_tables::kvalues_fp4.data()));
  const uint8x16_t mask4 = vdupq_n_u8(0x0FU);
  float32x4_t acc = vdupq_n_f32(0.0F);
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t s = 0; s < 4; ++s) {
      const std::size_t base = b * kBs + 4 + s * 8;
      const float d = ue4m3_half(row[b * kBs + s]);
      const uint8x8_t q = vld1_u8(row + base);
      const int8x8_t wlo =
          vqtbl1_s8(tbl, vand_u8(q, vget_low_u8(mask4)));
      const int8x8_t whi = vqtbl1_s8(tbl, vshr_n_u8(q, 4));
      const int8x16_t w = vcombine_s8(wlo, whi);
      const std::size_t a = b * 64 + s * 16;
      const int32x4_t t = vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(ap + a));
      acc = vfmaq_f32(acc, vcvtq_f32_s32(t),
                      vdupq_n_f32(d * sp[a / 32]));
    }
  }
  return vaddvq_f32(acc);
}


void dot_rows_nvfp4(const std::uint8_t* rows, std::size_t rb,
                    std::size_t nrows, const ActQ8& act,
                    std::size_t in_dim, float* out) noexcept {
  constexpr std::size_t kBs = 36;
  const std::size_t nb = in_dim / 64;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const int8x16_t tbl = vld1q_s8(
      reinterpret_cast<const std::int8_t*>(dequant_tables::kvalues_fp4.data()));
  const uint8x16_t mask4 = vdupq_n_u8(0x0FU);

  // ue4m3 scale lookup: the scalar ue4m3_half() calls ldexp/bit twiddling and
  // shows up hot (4 scales per 64 weights per row).
  static const std::array<float, 256> ue4m3_lut = [] {
    std::array<float, 256> t{};
    for (std::size_t i = 0; i < 256; ++i) t[i] = ue4m3_half(i);
    return t;
  }();

  // Permute the activation once for the whole call. The 4-bit code layout
  // puts low nibbles of a 16-byte chunk at weights [0..7] and [16..23], high
  // nibbles at [8..15] and [24..31]; the permuted buffer lets one vdotq cover
  // two subblocks with per-lane scales.
  thread_local std::vector<std::int8_t> perm;
  const std::size_t need = nb * 64;
  if (perm.size() < need) perm.resize(need);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::int8_t* a = ap + b * 64;
    const int8x16_t x0 = vld1q_s8(a);
    const int8x16_t x1 = vld1q_s8(a + 16);
    const int8x16_t x2 = vld1q_s8(a + 32);
    const int8x16_t x3 = vld1q_s8(a + 48);
    vst1q_s8(perm.data() + b * 64,
             vcombine_s8(vget_low_s8(x0), vget_low_s8(x1)));
    vst1q_s8(perm.data() + b * 64 + 16,
             vcombine_s8(vget_high_s8(x0), vget_high_s8(x1)));
    vst1q_s8(perm.data() + b * 64 + 32,
             vcombine_s8(vget_low_s8(x2), vget_low_s8(x3)));
    vst1q_s8(perm.data() + b * 64 + 48,
             vcombine_s8(vget_high_s8(x2), vget_high_s8(x3)));
  }
  const std::int8_t* pp = perm.data();

  std::size_t r = 0;
  for (; r + 4 <= nrows; r += 4) {
    const std::uint8_t* r0 = rows + (r + 0) * rb;
    const std::uint8_t* r1 = rows + (r + 1) * rb;
    const std::uint8_t* r2 = rows + (r + 2) * rb;
    const std::uint8_t* r3 = rows + (r + 3) * rb;
    float32x4_t a0 = vdupq_n_f32(0.0F);
    float32x4_t a1 = vdupq_n_f32(0.0F);
    float32x4_t a2 = vdupq_n_f32(0.0F);
    float32x4_t a3 = vdupq_n_f32(0.0F);
    for (std::size_t b = 0; b < nb; ++b) {
      const int8x16_t x0 = vld1q_s8(pp + b * 64);
      const int8x16_t x1 = vld1q_s8(pp + b * 64 + 16);
      const int8x16_t x2 = vld1q_s8(pp + b * 64 + 32);
      const int8x16_t x3 = vld1q_s8(pp + b * 64 + 48);
      const float asp0 = sp[2 * b];
      const float asp1 = sp[2 * b + 1];
      auto acc_row = [&](const std::uint8_t* row, float32x4_t& acc) {
        const std::uint8_t* p = row + b * kBs;
        const float d0 = ue4m3_lut[p[0]];
        const float d1 = ue4m3_lut[p[1]];
        const float d2 = ue4m3_lut[p[2]];
        const float d3 = ue4m3_lut[p[3]];
        const uint8x16_t q0 = vld1q_u8(p + 4);
        const uint8x16_t q1 = vld1q_u8(p + 20);
        const int8x16_t lo0 = vqtbl1q_s8(tbl, vandq_u8(q0, mask4));
        const int8x16_t hi0 = vqtbl1q_s8(tbl, vshrq_n_u8(q0, 4));
        const int8x16_t lo1 = vqtbl1q_s8(tbl, vandq_u8(q1, mask4));
        const int8x16_t hi1 = vqtbl1q_s8(tbl, vshrq_n_u8(q1, 4));
        const float32x4_t sc01 =
            vmulq_n_f32(vcombine_f32(vdup_n_f32(d0), vdup_n_f32(d1)), asp0);
        const float32x4_t sc23 =
            vmulq_n_f32(vcombine_f32(vdup_n_f32(d2), vdup_n_f32(d3)), asp1);
        int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), lo0, x0);
        p0 = vdotq_s32(p0, hi0, x1);
        int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), lo1, x2);
        p1 = vdotq_s32(p1, hi1, x3);
        acc = vfmaq_f32(acc, vcvtq_f32_s32(p0), sc01);
        acc = vfmaq_f32(acc, vcvtq_f32_s32(p1), sc23);
      };
      acc_row(r0, a0);
      acc_row(r1, a1);
      acc_row(r2, a2);
      acc_row(r3, a3);
    }
    out[r + 0] = vaddvq_f32(a0);
    out[r + 1] = vaddvq_f32(a1);
    out[r + 2] = vaddvq_f32(a2);
    out[r + 3] = vaddvq_f32(a3);
  }
  for (; r < nrows; ++r) {
    out[r] = dot_nvfp4(rows + r * rb, act, in_dim);
  }
}

// ---- iq1_s: 256 весов, 8 кодов решётки по 8 значений + дельта 0.125.
// в отличие от q8_K-версии, активация квантована по 32, поэтому каждый
// 32-сабблок считает свой bsum и свою шкалу активации.
float dot_iq1s(const std::uint8_t* row, const ActQ8& act,
               std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 50;
  constexpr float kDelta = 0.125F;
  const std::size_t nb = in_dim / 256;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  float total = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t base = b * kBs;
    const float d = load_f16(row + base);
    const std::uint8_t* qs = row + base + 2;
    const std::uint16_t* qh =
        reinterpret_cast<const std::uint16_t*>(row + base + 34);
    float acc = 0.0F;
    for (std::size_t j = 0; j < 8; ++j) {
      const std::uint32_t q = qh[j];
      const std::uint32_t idx0 =
          static_cast<std::uint32_t>(qs[4 * j + 0]) | ((q << 8) & 0x700U);
      const std::uint32_t idx1 =
          static_cast<std::uint32_t>(qs[4 * j + 1]) | ((q << 5) & 0x700U);
      const std::uint32_t idx2 =
          static_cast<std::uint32_t>(qs[4 * j + 2]) | ((q << 2) & 0x700U);
      const std::uint32_t idx3 =
          static_cast<std::uint32_t>(qs[4 * j + 3]) | ((q >> 1) & 0x700U);
      const int8x8_t g0 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &dequant_tables::iq1s_grid[idx0]));
      const int8x8_t g1 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &dequant_tables::iq1s_grid[idx1]));
      const int8x8_t g2 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &dequant_tables::iq1s_grid[idx2]));
      const int8x8_t g3 = vld1_s8(reinterpret_cast<const std::int8_t*>(
          &dequant_tables::iq1s_grid[idx3]));
      const std::size_t a = b * 256 + j * 32;
      int32x4_t p = vdotq_s32(vdupq_n_s32(0), vcombine_s8(g0, g1),
                              vld1q_s8(ap + a));
      p = vdotq_s32(p, vcombine_s8(g2, g3), vld1q_s8(ap + a + 16));
      const int psum = vaddvq_s32(p);
      const int bsum = vaddlvq_s8(vld1q_s8(ap + a)) +
                       vaddlvq_s8(vld1q_s8(ap + a + 16));
      const float ls = 2.0F * static_cast<float>((q >> 12) & 7U) + 1.0F;
      const float sign = (q & 0x8000U) != 0 ? -1.0F : 1.0F;
      acc += (static_cast<float>(psum) +
              kDelta * sign * static_cast<float>(bsum)) *
             ls * sp[a >> 5];
    }
    total += d * acc;
  }
  return total;
}


// ---- iq2_xxs: 256 весов, 32 кода по 8 значений, знаки — чётные 7-битные
// маски. шкала 0.5+3bit на 32-сабблок, активация — своя на 32.
float dot_iq2xxs(const std::uint8_t* row, const ActQ8& act,
                 std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 66;
  const std::size_t nb = in_dim / 256;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  float total = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t base = b * kBs;
    const float d = load_f16(row + base);
    const std::uint8_t* qp = row + base + 2;
    float acc = 0.0F;
    for (std::size_t k = 0; k < 4; ++k) {
      std::uint32_t aux[4];
      std::memcpy(aux, qp + k * 16, 16);
      const std::uint8_t* aux8 = reinterpret_cast<const std::uint8_t*>(aux);
      const std::uint32_t idx[8] = {aux8[0], aux8[1], aux8[2], aux8[3],
                                    aux8[8], aux8[9], aux8[10], aux8[11]};
      const std::uint32_t sgn[8] = {(aux[1] >> 0) & 127U, (aux[1] >> 7) & 127U,
                                    (aux[1] >> 14) & 127U, (aux[1] >> 21) & 127U,
                                    (aux[3] >> 0) & 127U, (aux[3] >> 7) & 127U,
                                    (aux[3] >> 14) & 127U, (aux[3] >> 21) & 127U};
      const float sub1 = 0.5F + static_cast<float>(aux[1] >> 28);
      const float sub2 = 0.5F + static_cast<float>(aux[3] >> 28);
      for (std::size_t c = 0; c < 8; ++c) {
        const int8x8_t g = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq2xxs_grid[idx[c]]));
        const int8x8_t m = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &kMask8[tables::ksigns_iq2xs[sgn[c]]]));
        const int8x8_t w = vsub_s8(veor_s8(g, m), m);
        const std::size_t a = b * 256 + k * 64 + c * 8;
        const int32x2_t p =
            vdot_s32(vdup_n_s32(0), w, vld1_s8(ap + a));
        acc += static_cast<float>(vget_lane_s32(p, 0) + vget_lane_s32(p, 1)) *
               (c < 4 ? sub1 : sub2) * sp[a >> 5];
      }
    }
    total += d * acc;
  }
  return 0.25F * total;
}

// ---- iq2_xs: 9-битный код решётки + 7-битные знаки в uint16; 4 шкалы
// (2*4bit+1) на 64 веса.
float dot_iq2xs(const std::uint8_t* row, const ActQ8& act,
                std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 74;
  const std::size_t nb = in_dim / 256;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  float total = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t base = b * kBs;
    const float d = load_f16(row + base);
    const std::uint16_t* qs =
        reinterpret_cast<const std::uint16_t*>(row + base + 2);
    const std::uint8_t* scales = row + base + 66;
    float acc = 0.0F;
    for (std::size_t k = 0; k < 4; ++k) {
      for (std::size_t c = 0; c < 8; ++c) {
        const std::uint16_t code = qs[k * 8 + c];
        const int8x8_t g = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq2xs_grid[code & 511U]));
        const int8x8_t m = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &kMask8[tables::ksigns_iq2xs[code >> 9]]));
        const int8x8_t w = vsub_s8(veor_s8(g, m), m);
        const std::size_t a = b * 256 + k * 64 + c * 8;
        const int32x2_t p =
            vdot_s32(vdup_n_s32(0), w, vld1_s8(ap + a));
        const std::size_t si = 4 * k + c / 2;
        const std::uint8_t sc = scales[si / 2];
        const float ws = 2.0F * static_cast<float>(
                                    (sc >> (4 * (si % 2))) & 0x0F) +
                         1.0F;
        acc += static_cast<float>(vget_lane_s32(p, 0) + vget_lane_s32(p, 1)) *
               ws * sp[a >> 5];
      }
    }
    total += d * acc;
  }
  return 0.125F * total;
}


// ---- iq2_xs, две строки: пара 9-битных кодов (общая 4-битная шкала на 16
// весов) склеивается в int8x16 и идёт одним vdotq вместо двух vdot_s32.
// построчный dot_iq2xs делает на код два скалярных load-а, сдвиг шкалы и
// vdot_s32 — здесь всё это вдвое реже.
void dot_rows_iq2xs(const std::uint8_t* rows, std::size_t rb,
                    std::size_t nrows, const ActQ8& act, std::size_t in_dim,
                    float* out) noexcept {
  constexpr std::size_t kBs = 74;
  const std::size_t nb = in_dim / 256;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  std::size_t r = 0;
  for (; r + 2 <= nrows; r += 2) {
    const std::uint8_t* p0 = rows + (r + 0) * rb;
    const std::uint8_t* p1 = rows + (r + 1) * rb;
    float o0 = 0.0F, o1 = 0.0F;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::size_t base = b * kBs;
      const float d0 = load_f16(p0 + base) * 0.125F;
      const float d1 = load_f16(p1 + base) * 0.125F;
      const std::uint8_t* sc0 = p0 + base + 66;
      const std::uint8_t* sc1 = p1 + base + 66;
      float32x4_t f0 = vdupq_n_f32(0.0F);
      float32x4_t f1 = vdupq_n_f32(0.0F);
      for (std::size_t k = 0; k < 4; ++k) {
        const std::size_t a0 = b * 256 + k * 64;
        for (std::size_t pair = 0; pair < 4; ++pair) {
          const std::size_t a = a0 + pair * 16;
          const int8x16_t xa = vld1q_s8(ap + a);
          const std::size_t si = 4 * k + pair;
          const float spv = sp[a >> 5];
          auto row16 = [&](const std::uint8_t* p, const std::uint8_t* scl,
                           float d8, float32x4_t& acc) {
            const std::uint8_t* qb =
                p + base + 2 + (k * 8 + pair * 2) * 2;
            std::uint32_t qw = 0;
            std::memcpy(&qw, qb, 4);
            const std::size_t i0 = qw & 511U;
            const std::size_t i1 = (qw >> 16) & 511U;
            const std::size_t sg0 = (qw >> 9) & 127U;
            const std::size_t sg1 = (qw >> 25) & 127U;
            const int8x8_t m0 = vld1_s8(
                reinterpret_cast<const std::int8_t*>(
                    &kMask8[tables::ksigns_iq2xs[sg0]]));
            const int8x8_t m1 = vld1_s8(
                reinterpret_cast<const std::int8_t*>(
                    &kMask8[tables::ksigns_iq2xs[sg1]]));
            const int8x8_t g0 = vld1_s8(
                reinterpret_cast<const std::int8_t*>(
                    &dequant_tables::iq2xs_grid[i0]));
            const int8x8_t g1 = vld1_s8(
                reinterpret_cast<const std::int8_t*>(
                    &dequant_tables::iq2xs_grid[i1]));
            const int8x16_t w = vcombine_s8(vsub_s8(veor_s8(g0, m0), m0),
                                            vsub_s8(veor_s8(g1, m1), m1));
            const std::uint8_t sc = scl[si / 2];
            const float ws =
                2.0F * static_cast<float>((sc >> (4 * (si % 2))) & 0x0FU) +
                1.0F;
            const int32x4_t pp = vdotq_s32(vdupq_n_s32(0), w, xa);
            acc = vfmaq_f32(acc, vcvtq_f32_s32(pp),
                            vdupq_n_f32(d8 * ws * spv));
          };
          row16(p0, sc0, d0, f0);
          row16(p1, sc1, d1, f1);
        }
      }
      o0 += vaddvq_f32(f0);
      o1 += vaddvq_f32(f1);
    }
    out[r + 0] = o0;
    out[r + 1] = o1;
  }
  for (; r < nrows; ++r) {
    out[r] = dot_iq2xs(rows + r * rb, act, in_dim);
  }
}

// ---- iq2_xxs, две строки: кодовые пары соединены в int8x16, один vdotq
// на 16 весов; активация читается один раз и делится между двумя строками.
// 2-row выбран вместо 4-row: на 4 строках компилятор выливает аккумуляторы
// на стек и ядро выходит вдвое медленнее построчного fallback-а
void dot_rows_iq2xxs(const std::uint8_t* rows, std::size_t rb,
                     std::size_t nrows, const ActQ8& act, std::size_t in_dim,
                     float* out) noexcept {
  constexpr std::size_t kBs = 66;
  const std::size_t nb = in_dim / 256;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  std::size_t r = 0;
  for (; r + 2 <= nrows; r += 2) {
    const std::uint8_t* p0 = rows + (r + 0) * rb;
    const std::uint8_t* p1 = rows + (r + 1) * rb;
    float o0 = 0.0F, o1 = 0.0F;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::size_t base = b * kBs;
      const float d0 = load_f16(p0 + base);
      const float d1 = load_f16(p1 + base);
      float32x4_t f0 = vdupq_n_f32(0.0F);
      float32x4_t f1 = vdupq_n_f32(0.0F);
      for (std::size_t k = 0; k < 4; ++k) {
        std::uint32_t aa0[4];
        std::uint32_t aa1[4];
        std::memcpy(aa0, p0 + base + 2 + k * 16, 16);
        std::memcpy(aa1, p1 + base + 2 + k * 16, 16);
        const std::size_t a0 = b * 256 + k * 64;
        for (std::size_t half = 0; half < 2; ++half) {
          const std::size_t a1 = a0 + half * 32;
          const float spv = sp[a1 >> 5];
          const float s0 =
              (0.5F + static_cast<float>(aa0[1 + 2 * half] >> 28)) * spv;
          const float s1 =
              (0.5F + static_cast<float>(aa1[1 + 2 * half] >> 28)) * spv;
          for (std::size_t pair = 0; pair < 2; ++pair) {
            const std::size_t c = half * 4 + pair * 2;
            const std::int8_t* x = ap + a1 + pair * 16;
            const int8x16_t xa = vld1q_s8(x);
            auto row16 = [&](const std::uint32_t* aq, float32x4_t& acc,
                             float scale) {
              const std::uint8_t* a8 =
                  reinterpret_cast<const std::uint8_t*>(aq);
              const std::size_t i0 = a8[c < 4 ? c : c + 4];
              const std::size_t i1 = a8[c < 4 ? c + 1 : c + 5];
              const std::size_t sh = 7 * (c % 4);
              const std::uint32_t sw = aq[c < 4 ? 1 : 3];
              const std::size_t sg0 = (sw >> sh) & 127U;
              const std::size_t sg1 = (sw >> (sh + 7)) & 127U;
              const int8x8_t m0 = vld1_s8(
                  reinterpret_cast<const std::int8_t*>(
                      &kMask8[tables::ksigns_iq2xs[sg0]]));
              const int8x8_t m1 = vld1_s8(
                  reinterpret_cast<const std::int8_t*>(
                      &kMask8[tables::ksigns_iq2xs[sg1]]));
              const int8x8_t g0 = vld1_s8(
                  reinterpret_cast<const std::int8_t*>(
                      &dequant_tables::iq2xxs_grid[i0]));
              const int8x8_t g1 = vld1_s8(
                  reinterpret_cast<const std::int8_t*>(
                      &dequant_tables::iq2xxs_grid[i1]));
              const int8x16_t w = vcombine_s8(vsub_s8(veor_s8(g0, m0), m0),
                                              vsub_s8(veor_s8(g1, m1), m1));
              const int32x4_t pp = vdotq_s32(vdupq_n_s32(0), w, xa);
              acc = vfmaq_f32(acc, vcvtq_f32_s32(pp),
                              vdupq_n_f32(scale));
            };
            row16(aa0, f0, s0);
            row16(aa1, f1, s1);
          }
        }
      }
      o0 += 0.25F * d0 * vaddvq_f32(f0);
      o1 += 0.25F * d1 * vaddvq_f32(f1);
    }
    out[r + 0] = o0;
    out[r + 1] = o1;
  }
  for (; r < nrows; ++r) {
    out[r] = dot_iq2xxs(rows + r * rb, act, in_dim);
  }
}

// ---- q6_k: 256 весов, 6 бит; шкала int8 на 16 весов, d на блок.
// раскладка: две половины по 128, в каждой q1/q2/q3/q4 по 32 значения.
float dot_q6k(const std::uint8_t* row, const ActQ8& act,
              std::size_t in_dim) noexcept {
  constexpr std::size_t kBs = 210;
  const std::size_t nb = in_dim / 256;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  float total = 0.0F;
  const int8x16_t k32 = vdupq_n_s8(32);
  const uint8x16_t low4 = vdupq_n_u8(0x0F);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t base = b * kBs;
    const float d = load_f16(row + base + 208);
    float acc = 0.0F;
    for (std::size_t h = 0; h < 2; ++h) {
      const std::uint8_t* ql = row + base + h * 64;
      const std::uint8_t* qh = row + base + 128 + h * 32;
      const std::int8_t* sc =
          reinterpret_cast<const std::int8_t*>(row + base + 192 + h * 8);
      for (std::size_t part = 0; part < 4; ++part) {
        const std::size_t ql_off = (part & 1) ? 32 : 0;
        const std::size_t sh = 2 * part;
        for (std::size_t half = 0; half < 2; ++half) {
          const std::uint8_t* qlb = ql + ql_off + half * 16;
          const uint8x16_t qv =
              vld1q_u8(qlb);
          const uint8x16_t nib =
              (part < 2) ? vandq_u8(qv, low4) : vshrq_n_u8(qv, 4);
          const uint8x16_t hi = vandq_u8(
              vshlq_u8(vld1q_u8(qh + half * 16),
                       vdupq_n_s8(static_cast<std::int8_t>(-static_cast<int>(sh)))),
              vdupq_n_u8(0x03));
          const int8x16_t q = vsubq_s8(
              vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(hi, 4))), k32);
          const std::size_t a =
              b * 256 + h * 128 + part * 32 + half * 16;
          const int32x4_t p =
              vdotq_s32(vdupq_n_s32(0), q, vld1q_s8(ap + a));
          acc += static_cast<float>(vaddvq_s32(p)) *
                 static_cast<float>(sc[part * 2 + half]) * sp[a >> 5];
        }
      }
    }
    total += d * acc;
  }
  return total;
}

// ---- iq1_s, четыре строки: тот же layout, что в dot_iq1s; каждая строка —
// своя сетка (iq1s_grid) и активации, латентность vdot закрыта парами
void dot_rows_iq1s(const std::uint8_t* rows, std::size_t rb,
                   std::size_t nrows, const ActQ8& act, std::size_t in_dim,
                   float* out) noexcept {
  constexpr std::size_t kBs = 50;
  constexpr float kDelta = 0.125F;
  const std::size_t nb = in_dim / 256;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();

  std::size_t r = 0;
  for (; r + 4 <= nrows; r += 4) {
    const std::uint8_t* p0 = rows + (r + 0) * rb;
    const std::uint8_t* p1 = rows + (r + 1) * rb;
    const std::uint8_t* p2 = rows + (r + 2) * rb;
    const std::uint8_t* p3 = rows + (r + 3) * rb;
    out[r + 0] = 0.0F;
    out[r + 1] = 0.0F;
    out[r + 2] = 0.0F;
    out[r + 3] = 0.0F;
    for (std::size_t b = 0; b < nb; ++b) {
      const std::size_t base = b * kBs;
      const float d0 = load_f16(p0 + base);
      const float d1 = load_f16(p1 + base);
      const float d2 = load_f16(p2 + base);
      const float d3 = load_f16(p3 + base);
      const std::uint8_t* qs0 = p0 + base + 2;
      const std::uint8_t* qs1 = p1 + base + 2;
      const std::uint8_t* qs2 = p2 + base + 2;
      const std::uint8_t* qs3 = p3 + base + 2;
      const std::uint16_t* qh0 =
          reinterpret_cast<const std::uint16_t*>(p0 + base + 34);
      const std::uint16_t* qh1 =
          reinterpret_cast<const std::uint16_t*>(p1 + base + 34);
      const std::uint16_t* qh2 =
          reinterpret_cast<const std::uint16_t*>(p2 + base + 34);
      const std::uint16_t* qh3 =
          reinterpret_cast<const std::uint16_t*>(p3 + base + 34);
      float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
      for (std::size_t j = 0; j < 8; ++j) {
        const std::uint32_t q0 = qh0[j];
        const std::uint32_t q1 = qh1[j];
        const std::uint32_t q2 = qh2[j];
        const std::uint32_t q3 = qh3[j];
        const std::size_t a = b * 256 + j * 32;
        const int8x16_t x0 = vld1q_s8(ap + a);
        const int8x16_t x1 = vld1q_s8(ap + a + 16);
        const int8x8_t g00 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs0[4 * j + 0] | ((q0 << 8) & 0x700U)]));
        const int8x8_t g01 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs0[4 * j + 1] | ((q0 << 5) & 0x700U)]));
        const int8x8_t g02 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs0[4 * j + 2] | ((q0 << 2) & 0x700U)]));
        const int8x8_t g03 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs0[4 * j + 3] | ((q0 >> 1) & 0x700U)]));
        int32x4_t p0v = vdotq_s32(vdupq_n_s32(0), vcombine_s8(g00, g01), x0);
        p0v = vdotq_s32(p0v, vcombine_s8(g02, g03), x1);
        const int8x8_t g10 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs1[4 * j + 0] | ((q1 << 8) & 0x700U)]));
        const int8x8_t g11 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs1[4 * j + 1] | ((q1 << 5) & 0x700U)]));
        const int8x8_t g12 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs1[4 * j + 2] | ((q1 << 2) & 0x700U)]));
        const int8x8_t g13 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs1[4 * j + 3] | ((q1 >> 1) & 0x700U)]));
        int32x4_t p1v = vdotq_s32(vdupq_n_s32(0), vcombine_s8(g10, g11), x0);
        p1v = vdotq_s32(p1v, vcombine_s8(g12, g13), x1);
        const int8x8_t g20 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs2[4 * j + 0] | ((q2 << 8) & 0x700U)]));
        const int8x8_t g21 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs2[4 * j + 1] | ((q2 << 5) & 0x700U)]));
        const int8x8_t g22 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs2[4 * j + 2] | ((q2 << 2) & 0x700U)]));
        const int8x8_t g23 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs2[4 * j + 3] | ((q2 >> 1) & 0x700U)]));
        int32x4_t p2v = vdotq_s32(vdupq_n_s32(0), vcombine_s8(g20, g21), x0);
        p2v = vdotq_s32(p2v, vcombine_s8(g22, g23), x1);
        const int8x8_t g30 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs3[4 * j + 0] | ((q3 << 8) & 0x700U)]));
        const int8x8_t g31 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs3[4 * j + 1] | ((q3 << 5) & 0x700U)]));
        const int8x8_t g32 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs3[4 * j + 2] | ((q3 << 2) & 0x700U)]));
        const int8x8_t g33 = vld1_s8(reinterpret_cast<const std::int8_t*>(
            &dequant_tables::iq1s_grid[qs3[4 * j + 3] | ((q3 >> 1) & 0x700U)]));
        int32x4_t p3v = vdotq_s32(vdupq_n_s32(0), vcombine_s8(g30, g31), x0);
        p3v = vdotq_s32(p3v, vcombine_s8(g32, g33), x1);
        // 4 независимые цепочки: каждая заканчивает парой vdot + скалярный
        // вклад — латентность закрыта в 4 раза плотнее
        const int psum0 = vaddvq_s32(p0v);
        const int psum1 = vaddvq_s32(p1v);
        const int psum2 = vaddvq_s32(p2v);
        const int psum3 = vaddvq_s32(p3v);
        const int bsum =
            vaddlvq_s8(x0) + vaddlvq_s8(x1);
        const std::size_t a5 = a >> 5;
        const float ls0 = 2.0F * static_cast<float>((q0 >> 12) & 7U) + 1.0F;
        const float ls1 = 2.0F * static_cast<float>((q1 >> 12) & 7U) + 1.0F;
        const float ls2 = 2.0F * static_cast<float>((q2 >> 12) & 7U) + 1.0F;
        const float ls3 = 2.0F * static_cast<float>((q3 >> 12) & 7U) + 1.0F;
        const float sg0 = (q0 & 0x8000U) != 0 ? -1.0F : 1.0F;
        const float sg1 = (q1 & 0x8000U) != 0 ? -1.0F : 1.0F;
        const float sg2 = (q2 & 0x8000U) != 0 ? -1.0F : 1.0F;
        const float sg3 = (q3 & 0x8000U) != 0 ? -1.0F : 1.0F;
        a0 += (static_cast<float>(psum0) + kDelta * sg0 * static_cast<float>(bsum)) * ls0 * sp[a5];
        a1 += (static_cast<float>(psum1) + kDelta * sg1 * static_cast<float>(bsum)) * ls1 * sp[a5];
        a2 += (static_cast<float>(psum2) + kDelta * sg2 * static_cast<float>(bsum)) * ls2 * sp[a5];
        a3 += (static_cast<float>(psum3) + kDelta * sg3 * static_cast<float>(bsum)) * ls3 * sp[a5];
      }
      out[r + 0] += d0 * a0;
      out[r + 1] += d1 * a1;
      out[r + 2] += d2 * a2;
      out[r + 3] += d3 * a3;
    }
  }
  for (; r < nrows; ++r) {
    out[r] = dot_iq1s(rows + r * rb, act, in_dim);
  }
}

// ---- mxfp4, четыре строки: тот же layout, что в dot_mxfp4; активации
// загружаются один раз на блок, шкала e8m0 на блок
void dot_rows_mxfp4(const std::uint8_t* rows, std::size_t rb,
                    std::size_t nrows, const ActQ8& act, std::size_t in_dim,
                    float* out) noexcept {
  constexpr std::size_t kBs = 17;
  const std::size_t nb = in_dim / 32;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const int8x16_t tbl = vld1q_s8(
      reinterpret_cast<const std::int8_t*>(dequant_tables::kvalues_fp4.data()));
  const uint8x16_t mask4 = vdupq_n_u8(0x0FU);

  std::size_t r = 0;
  for (; r + 4 <= nrows; r += 4) {
    float32x4_t a0 = vdupq_n_f32(0.0F);
    float32x4_t a1 = vdupq_n_f32(0.0F);
    float32x4_t a2 = vdupq_n_f32(0.0F);
    float32x4_t a3 = vdupq_n_f32(0.0F);
    for (std::size_t b = 0; b < nb; ++b) {
      const int8x16_t x0 = vld1q_s8(ap + b * 32);
      const int8x16_t x1 = vld1q_s8(ap + b * 32 + 16);
      auto acc_row = [&](const std::uint8_t* p, float32x4_t& acc) {
        const float d = e8m0_half(p[b * kBs]);
        const uint8x16_t q = vld1q_u8(p + b * kBs + 1);
        const int8x16_t wlo = vqtbl1q_s8(tbl, vandq_u8(q, mask4));
        const int8x16_t whi = vqtbl1q_s8(tbl, vshrq_n_u8(q, 4));
        int32x4_t s = vdotq_s32(vdupq_n_s32(0), wlo, x0);
        s = vdotq_s32(s, whi, x1);
        acc = vfmaq_f32(acc, vcvtq_f32_s32(s),
                        vdupq_n_f32(d * sp[b]));
      };
      acc_row(rows + (r + 0) * rb, a0);
      acc_row(rows + (r + 1) * rb, a1);
      acc_row(rows + (r + 2) * rb, a2);
      acc_row(rows + (r + 3) * rb, a3);
    }
    out[r + 0] = vaddvq_f32(a0);
    out[r + 1] = vaddvq_f32(a1);
    out[r + 2] = vaddvq_f32(a2);
    out[r + 3] = vaddvq_f32(a3);
  }
  for (; r < nrows; ++r) {
    out[r] = dot_mxfp4(rows + r * rb, act, in_dim);
  }
}

// ---- q6_k, четыре строки: тот же layout, что в dot_q6k; активации загружаются
// один раз на итерацию (vdot + vfma на строку), сумма vaddvq по блоку
void dot_rows_q6k(const std::uint8_t* rows, std::size_t rb,
                  std::size_t nrows, const ActQ8& act, std::size_t in_dim,
                  float* out) noexcept {
  constexpr std::size_t kBs = 210;
  const std::size_t nb = in_dim / 256;
  const std::int8_t* ap = act.q.data();
  const float* sp = act.scales.data();
  const int8x16_t k32 = vdupq_n_s8(32);
  const uint8x16_t low4 = vdupq_n_u8(0x0F);

  auto row_w = [&](const std::uint8_t* p, std::size_t base, std::size_t h,
                   std::size_t part, std::size_t half, std::size_t ql_off,
                   std::size_t sh) -> int8x16_t {
    const uint8x16_t qv = vld1q_u8(p + base + h * 64 + ql_off + half * 16);
    const uint8x16_t nib = (part < 2) ? vandq_u8(qv, low4) : vshrq_n_u8(qv, 4);
    const uint8x16_t hi = vandq_u8(
        vshlq_u8(vld1q_u8(p + base + 128 + h * 32 + half * 16),
                 vdupq_n_s8(static_cast<std::int8_t>(-static_cast<int>(sh)))),
        vdupq_n_u8(0x03));
    return vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(hi, 4))), k32);
  };

  std::size_t r = 0;
  for (; r + 4 <= nrows; r += 4) {
    const std::uint8_t* p0 = rows + (r + 0) * rb;
    const std::uint8_t* p1 = rows + (r + 1) * rb;
    const std::uint8_t* p2 = rows + (r + 2) * rb;
    const std::uint8_t* p3 = rows + (r + 3) * rb;
    float32x4_t f0 = vdupq_n_f32(0.0F);
    float32x4_t f1 = vdupq_n_f32(0.0F);
    float32x4_t f2 = vdupq_n_f32(0.0F);
    float32x4_t f3 = vdupq_n_f32(0.0F);
    for (std::size_t b = 0; b < nb; ++b) {
      const std::size_t base = b * kBs;
      const float d0 = load_f16(p0 + base + 208);
      const float d1 = load_f16(p1 + base + 208);
      const float d2 = load_f16(p2 + base + 208);
      const float d3 = load_f16(p3 + base + 208);
      for (std::size_t h = 0; h < 2; ++h) {
        for (std::size_t part = 0; part < 4; ++part) {
          const std::size_t ql_off = (part & 1) ? 32 : 0;
          const std::size_t sh = 2 * part;
          for (std::size_t half = 0; half < 2; ++half) {
            const std::size_t aoff =
                b * 256 + h * 128 + part * 32 + half * 16;
            const float sc0 =
                static_cast<float>(
                    reinterpret_cast<const std::int8_t*>(p0 + base + 192 + h * 8)[part * 2 + half]) *
                sp[aoff >> 5];
            const float sc1 =
                static_cast<float>(
                    reinterpret_cast<const std::int8_t*>(p1 + base + 192 + h * 8)[part * 2 + half]) *
                sp[aoff >> 5];
            const float sc2 =
                static_cast<float>(
                    reinterpret_cast<const std::int8_t*>(p2 + base + 192 + h * 8)[part * 2 + half]) *
                sp[aoff >> 5];
            const float sc3 =
                static_cast<float>(
                    reinterpret_cast<const std::int8_t*>(p3 + base + 192 + h * 8)[part * 2 + half]) *
                sp[aoff >> 5];
            const int32x4_t p0v =
                vdotq_s32(vdupq_n_s32(0),
                          row_w(p0, base, h, part, half, ql_off, sh),
                          vld1q_s8(ap + aoff));
            const int32x4_t p1v =
                vdotq_s32(vdupq_n_s32(0),
                          row_w(p1, base, h, part, half, ql_off, sh),
                          vld1q_s8(ap + aoff));
            const int32x4_t p2v =
                vdotq_s32(vdupq_n_s32(0),
                          row_w(p2, base, h, part, half, ql_off, sh),
                          vld1q_s8(ap + aoff));
            const int32x4_t p3v =
                vdotq_s32(vdupq_n_s32(0),
                          row_w(p3, base, h, part, half, ql_off, sh),
                          vld1q_s8(ap + aoff));
            f0 = vfmaq_f32(f0, vcvtq_f32_s32(p0v), vdupq_n_f32(sc0));
            f1 = vfmaq_f32(f1, vcvtq_f32_s32(p1v), vdupq_n_f32(sc1));
            f2 = vfmaq_f32(f2, vcvtq_f32_s32(p2v), vdupq_n_f32(sc2));
            f3 = vfmaq_f32(f3, vcvtq_f32_s32(p3v), vdupq_n_f32(sc3));
          }
        }
      }
      out[r + 0] += d0 * vaddvq_f32(f0);
      out[r + 1] += d1 * vaddvq_f32(f1);
      out[r + 2] += d2 * vaddvq_f32(f2);
      out[r + 3] += d3 * vaddvq_f32(f3);
      f0 = vdupq_n_f32(0.0F);
      f1 = vdupq_n_f32(0.0F);
      f2 = vdupq_n_f32(0.0F);
      f3 = vdupq_n_f32(0.0F);
    }
  }
  for (; r < nrows; ++r) {
    out[r] = dot_q6k(rows + r * rb, act, in_dim);
  }
}

}  // namespace tokshit::kernels::neon

#endif  // __aarch64__ && TOKSHIT_HAVE_NEON_DOTPROD
