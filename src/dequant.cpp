#include "tokshit/dequant.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "tokshit/gguf_tables.hpp"

namespace tokshit {
namespace {

[[nodiscard]] uint16_t read_u16(const uint8_t* src, size_t off) noexcept {
  return static_cast<uint16_t>(static_cast<uint16_t>(src[off]) |
                               (static_cast<uint16_t>(src[off + 1]) << 8));
}

[[nodiscard]] uint32_t read_u32(const uint8_t* src, size_t off) noexcept {
  return static_cast<uint32_t>(src[off]) |
         (static_cast<uint32_t>(src[off + 1]) << 8) |
         (static_cast<uint32_t>(src[off + 2]) << 16) |
         (static_cast<uint32_t>(src[off + 3]) << 24);
}

[[nodiscard]] bool dequant_supported(QuantType quant) noexcept {
  switch (quant) {
    case QuantType::F32:
    case QuantType::F16:
    case QuantType::Bf16:
    case QuantType::Q8_0:
    case QuantType::Q4_0:
    case QuantType::Q4K:
    case QuantType::Q5K:
    case QuantType::Q6K:
    case QuantType::Iq2S:
    case QuantType::Iq3Xxs:
    case QuantType::Iq3S:
    case QuantType::Iq4Xs:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] float f16_to_f32_slow(uint16_t bits) noexcept {
  const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
  const uint32_t exp = (static_cast<uint32_t>(bits) >> 10) & 0x1Fu;
  const uint32_t mant = static_cast<uint32_t>(bits) & 0x3FFu;
  uint32_t out = 0;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      uint32_t e = 127 - 15 + 1;
      uint32_t m = mant;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFu;
      out = sign | (e << 23) | (m << 13);
    }
  } else {
    out = sign | 0x7F800000u | (mant << 13);
  }
  return std::bit_cast<float>(out);
}

// разложение квантов обратно в float. используется в vcheck, эмбеддингах и
// фолбэке, на горячем пути почти не вызывается
// dequant back to float. used by vcheck, embeddings and the fallback, almost
// never on the hot path
void dequant_f16(const uint8_t* src, size_t numel, float* out) noexcept {
  for (size_t i = 0; i < numel; ++i) {
    out[i] = f16_to_f32(read_u16(src, i * 2));
  }
}

void dequant_bf16(const uint8_t* src, size_t numel, float* out) noexcept {
  for (size_t i = 0; i < numel; ++i) {
    out[i] = std::bit_cast<float>(static_cast<uint32_t>(read_u16(src, i * 2))
                                  << 16);
  }
}

void dequant_q4_0(const uint8_t* src, size_t numel, float* out) noexcept {
  constexpr size_t kQk = 32;
  constexpr size_t kBs = 18;
  const size_t nb = numel / kQk;
  size_t y = 0;
  for (size_t b = 0; b < nb; ++b) {
    const size_t base = b * kBs;
    const float d = f16_to_f32(read_u16(src, base));
    for (size_t j = 0; j < 16; ++j) {
      const int32_t q = static_cast<int32_t>(src[base + 2 + j] & 0x0Fu) - 8;
      out[y++] = static_cast<float>(q) * d;
    }
    for (size_t j = 0; j < 16; ++j) {
      const int32_t q = static_cast<int32_t>(src[base + 2 + j] >> 4) - 8;
      out[y++] = static_cast<float>(q) * d;
    }
  }
}

void dequant_q8_0(const uint8_t* src, size_t numel, float* out) noexcept {
  constexpr size_t kQk = 32;
  constexpr size_t kBs = 34;
  const size_t nb = numel / kQk;
  size_t y = 0;
  for (size_t b = 0; b < nb; ++b) {
    const size_t base = b * kBs;
    const float d = f16_to_f32(read_u16(src, base));
    for (size_t j = 0; j < kQk; ++j) {
      out[y++] =
          d * static_cast<float>(static_cast<int8_t>(src[base + 2 + j]));
    }
  }
}

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

void dequant_q4_k(const uint8_t* src, size_t numel, float* out) noexcept {
  constexpr size_t kQk = 256;
  constexpr size_t kBs = 144;
  const size_t nb = numel / kQk;
  size_t y = 0;
  for (size_t b = 0; b < nb; ++b) {
    const size_t base = b * kBs;
    const float d = f16_to_f32(read_u16(src, base));
    const float dmin = f16_to_f32(read_u16(src, base + 2));
    const uint8_t* scales = src + base + 4;
    const uint8_t* qs = src + base + 16;
    for (size_t j = 0; j < 4; ++j) {
      uint8_t sc1 = 0;
      uint8_t m1 = 0;
      uint8_t sc2 = 0;
      uint8_t m2 = 0;
      get_scale_min_k4(2 * j, scales, sc1, m1);
      get_scale_min_k4(2 * j + 1, scales, sc2, m2);
      const float d1 = d * static_cast<float>(sc1);
      const float mm1 = dmin * static_cast<float>(m1);
      const float d2 = d * static_cast<float>(sc2);
      const float mm2 = dmin * static_cast<float>(m2);
      const uint8_t* chunk = qs + j * 32;
      for (size_t l = 0; l < 32; ++l) {
        out[y++] = d1 * static_cast<float>(chunk[l] & 0x0Fu) - mm1;
      }
      for (size_t l = 0; l < 32; ++l) {
        out[y++] = d2 * static_cast<float>(chunk[l] >> 4) - mm2;
      }
    }
  }
}

void dequant_q5_k(const uint8_t* src, size_t numel, float* out) noexcept {
  constexpr size_t kQk = 256;
  constexpr size_t kBs = 176;
  const size_t nb = numel / kQk;
  size_t y = 0;
  for (size_t b = 0; b < nb; ++b) {
    const size_t base = b * kBs;
    const float d = f16_to_f32(read_u16(src, base));
    const float dmin = f16_to_f32(read_u16(src, base + 2));
    const uint8_t* scales = src + base + 4;
    const uint8_t* qh = src + base + 16;
    const uint8_t* qs = src + base + 48;
    for (size_t j = 0; j < 4; ++j) {
      uint8_t sc1 = 0;
      uint8_t m1 = 0;
      uint8_t sc2 = 0;
      uint8_t m2 = 0;
      get_scale_min_k4(2 * j, scales, sc1, m1);
      get_scale_min_k4(2 * j + 1, scales, sc2, m2);
      const float d1 = d * static_cast<float>(sc1);
      const float mm1 = dmin * static_cast<float>(m1);
      const float d2 = d * static_cast<float>(sc2);
      const float mm2 = dmin * static_cast<float>(m2);
      const uint32_t u1 = 1u << (2 * j);
      const uint32_t u2 = 1u << (2 * j + 1);
      const uint8_t* chunk = qs + j * 32;
      for (size_t l = 0; l < 32; ++l) {
        const uint32_t qlo =
            static_cast<uint32_t>(chunk[l] & 0x0Fu) +
            ((static_cast<uint32_t>(qh[l]) & u1) != 0 ? 16u : 0u);
        out[y++] = d1 * static_cast<float>(qlo) - mm1;
      }
      for (size_t l = 0; l < 32; ++l) {
        const uint32_t qhi =
            static_cast<uint32_t>(chunk[l] >> 4) +
            ((static_cast<uint32_t>(qh[l]) & u2) != 0 ? 16u : 0u);
        out[y++] = d2 * static_cast<float>(qhi) - mm2;
      }
    }
  }
}

void dequant_q6_k(const uint8_t* src, size_t numel, float* out) noexcept {
  constexpr size_t kQk = 256;
  constexpr size_t kBs = 210;
  const size_t nb = numel / kQk;
  for (size_t b = 0; b < nb; ++b) {
    const size_t base = b * kBs;
    const float d = f16_to_f32(read_u16(src, base + 208));
    const uint8_t* ql = src + base;
    const uint8_t* qh = src + base + 128;
    const uint8_t* scales = src + base + 192;
    const size_t y = b * kQk;
    for (size_t npass = 0; npass < 2; ++npass) {
      const size_t ql_off = npass * 64;
      const size_t qh_off = npass * 32;
      const size_t sc_off = npass * 8;
      const size_t y_off = y + npass * 128;
      for (size_t l = 0; l < 32; ++l) {
        const size_t is = l / 16;
        const uint8_t qh_l = qh[qh_off + l];
        const int32_t q1 =
            (static_cast<int32_t>(ql[ql_off + l] & 0x0Fu) |
             ((static_cast<int32_t>(qh_l >> 0) & 3) << 4)) -
            32;
        const int32_t q2 =
            (static_cast<int32_t>(ql[ql_off + l + 32] & 0x0Fu) |
             ((static_cast<int32_t>(qh_l >> 2) & 3) << 4)) -
            32;
        const int32_t q3 =
            (static_cast<int32_t>(ql[ql_off + l] >> 4) |
             ((static_cast<int32_t>(qh_l >> 4) & 3) << 4)) -
            32;
        const int32_t q4 =
            (static_cast<int32_t>(ql[ql_off + l + 32] >> 4) |
             ((static_cast<int32_t>(qh_l >> 6) & 3) << 4)) -
            32;
        out[y_off + l] =
            d * static_cast<float>(static_cast<int8_t>(scales[sc_off + is])) *
            static_cast<float>(q1);
        out[y_off + l + 32] =
            d *
            static_cast<float>(static_cast<int8_t>(scales[sc_off + is + 2])) *
            static_cast<float>(q2);
        out[y_off + l + 64] =
            d *
            static_cast<float>(static_cast<int8_t>(scales[sc_off + is + 4])) *
            static_cast<float>(q3);
        out[y_off + l + 96] =
            d *
            static_cast<float>(static_cast<int8_t>(scales[sc_off + is + 6])) *
            static_cast<float>(q4);
      }
    }
  }
}

void dequant_iq2_s(const uint8_t* src, size_t numel, float* out) noexcept {
  constexpr size_t kQk = 256;
  constexpr size_t kBs = 82;
  const size_t nb = numel / kQk;
  size_t y = 0;
  for (size_t b = 0; b < nb; ++b) {
    const size_t base = b * kBs;
    const float d = f16_to_f32(read_u16(src, base));
    const uint8_t* qs = src + base + 2;
    const uint8_t* qh = src + base + 66;
    const uint8_t* scales = src + base + 74;
    const uint8_t* signs = qs + 32;
    size_t qs_off = 0;
    size_t sg_off = 0;
    for (size_t ib32 = 0; ib32 < 8; ++ib32) {
      const float db0 =
          d * (0.5F + static_cast<float>(scales[ib32] & 0x0Fu)) * 0.25F;
      const float db1 =
          d * (0.5F + static_cast<float>((scales[ib32] >> 4) & 0x0Fu)) * 0.25F;
      for (size_t l = 0; l < 4; ++l) {
        const float dl = l < 2 ? db0 : db1;
        const uint32_t idx =
            static_cast<uint32_t>(qs[qs_off + l]) |
            ((static_cast<uint32_t>(qh[ib32]) << (8 - 2 * l)) & 0x300u);
        const uint64_t grid = tables::iq2s_grid[idx];
        const uint8_t sign_byte = signs[sg_off + l];
        for (size_t j = 0; j < 8; ++j) {
          const float g =
              static_cast<float>((grid >> (8 * j)) & 0xFFu);
          const float sign =
              (sign_byte & tables::kmask_iq2xs[j]) != 0 ? -1.0F : 1.0F;
          out[y++] = dl * g * sign;
        }
      }
      qs_off += 4;
      sg_off += 4;
    }
  }
}

void dequant_iq3_xxs(const uint8_t* src, size_t numel, float* out) noexcept {
  constexpr size_t kQk = 256;
  constexpr size_t kBs = 98;
  const size_t nb = numel / kQk;
  size_t y = 0;
  for (size_t b = 0; b < nb; ++b) {
    const size_t base = b * kBs;
    const float d = f16_to_f32(read_u16(src, base));
    const uint8_t* qs = src + base + 2;
    const uint8_t* scales_and_signs = qs + 64;
    size_t qs_off = 0;
    for (size_t ib32 = 0; ib32 < 8; ++ib32) {
      const uint32_t aux32 = read_u32(scales_and_signs, ib32 * 4);
      const float db =
          d * (0.5F + static_cast<float>((aux32 >> 28) & 0xFu)) * 0.5F;
      for (size_t l = 0; l < 4; ++l) {
        const uint8_t signs =
            tables::ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
        const uint32_t g1 = tables::iq3xxs_grid[qs[qs_off + 2 * l]];
        const uint32_t g2 = tables::iq3xxs_grid[qs[qs_off + 2 * l + 1]];
        for (size_t j = 0; j < 4; ++j) {
          const float s1 =
              (signs & tables::kmask_iq2xs[j]) != 0 ? -db : db;
          out[y++] = s1 * static_cast<float>((g1 >> (8 * j)) & 0xFFu);
        }
        for (size_t j = 0; j < 4; ++j) {
          const float s2 =
              (signs & tables::kmask_iq2xs[j + 4]) != 0 ? -db : db;
          out[y++] = s2 * static_cast<float>((g2 >> (8 * j)) & 0xFFu);
        }
      }
      qs_off += 8;
    }
  }
}

void dequant_iq3_s(const uint8_t* src, size_t numel, float* out) noexcept {
  constexpr size_t kQk = 256;
  constexpr size_t kBs = 110;
  const size_t nb = numel / kQk;
  size_t y = 0;
  for (size_t b = 0; b < nb; ++b) {
    const size_t base = b * kBs;
    const float d = f16_to_f32(read_u16(src, base));
    const uint8_t* qs = src + base + 2;
    const uint8_t* qh = src + base + 66;
    const uint8_t* signs = src + base + 74;
    const uint8_t* scales = src + base + 106;
    size_t qs_off = 0;
    size_t sg_off = 0;
    size_t qh_off = 0;
    for (size_t ib = 0; ib < 4; ++ib) {
      const float db1 =
          d * (1.0F + 2.0F * static_cast<float>(scales[ib] & 0x0Fu));
      const float db2 =
          d * (1.0F + 2.0F * static_cast<float>(scales[ib] >> 4));
      for (size_t l = 0; l < 4; ++l) {
        const uint32_t idx1 =
            static_cast<uint32_t>(qs[qs_off + 2 * l]) |
            ((static_cast<uint32_t>(qh[qh_off]) << (8 - 2 * l)) & 256u);
        const uint32_t idx2 =
            static_cast<uint32_t>(qs[qs_off + 2 * l + 1]) |
            ((static_cast<uint32_t>(qh[qh_off]) << (7 - 2 * l)) & 256u);
        const uint32_t g1 = tables::iq3s_grid[idx1];
        const uint32_t g2 = tables::iq3s_grid[idx2];
        const uint8_t sign_byte = signs[sg_off + l];
        for (size_t j = 0; j < 4; ++j) {
          const float s1 =
              (sign_byte & tables::kmask_iq2xs[j]) != 0 ? -db1 : db1;
          out[y++] = s1 * static_cast<float>((g1 >> (8 * j)) & 0xFFu);
        }
        for (size_t j = 0; j < 4; ++j) {
          const float s2 =
              (sign_byte & tables::kmask_iq2xs[j + 4]) != 0 ? -db1 : db1;
          out[y++] = s2 * static_cast<float>((g2 >> (8 * j)) & 0xFFu);
        }
      }
      qs_off += 8;
      sg_off += 4;
      for (size_t l = 0; l < 4; ++l) {
        const uint32_t idx1 =
            static_cast<uint32_t>(qs[qs_off + 2 * l]) |
            ((static_cast<uint32_t>(qh[qh_off + 1]) << (8 - 2 * l)) & 256u);
        const uint32_t idx2 =
            static_cast<uint32_t>(qs[qs_off + 2 * l + 1]) |
            ((static_cast<uint32_t>(qh[qh_off + 1]) << (7 - 2 * l)) & 256u);
        const uint32_t g1 = tables::iq3s_grid[idx1];
        const uint32_t g2 = tables::iq3s_grid[idx2];
        const uint8_t sign_byte = signs[sg_off + l];
        for (size_t j = 0; j < 4; ++j) {
          const float s1 =
              (sign_byte & tables::kmask_iq2xs[j]) != 0 ? -db2 : db2;
          out[y++] = s1 * static_cast<float>((g1 >> (8 * j)) & 0xFFu);
        }
        for (size_t j = 0; j < 4; ++j) {
          const float s2 =
              (sign_byte & tables::kmask_iq2xs[j + 4]) != 0 ? -db2 : db2;
          out[y++] = s2 * static_cast<float>((g2 >> (8 * j)) & 0xFFu);
        }
      }
      qh_off += 2;
      qs_off += 8;
      sg_off += 4;
    }
  }
}

void dequant_iq4_xs(const uint8_t* src, size_t numel, float* out) noexcept {
  constexpr size_t kQk = 256;
  constexpr size_t kBs = 136;
  const size_t nb = numel / kQk;
  size_t y = 0;
  for (size_t b = 0; b < nb; ++b) {
    const size_t base = b * kBs;
    const float d = f16_to_f32(read_u16(src, base));
    const uint32_t scales_h = read_u16(src, base + 2);
    const uint8_t* scales_l = src + base + 4;
    const uint8_t* qs = src + base + 8;
    size_t qs_off = 0;
    for (size_t ib = 0; ib < 8; ++ib) {
      const uint32_t ls =
          (static_cast<uint32_t>(scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0Fu) |
          (((scales_h >> (2 * ib)) & 3u) << 4);
      const float dl = d * static_cast<float>(static_cast<int32_t>(ls) - 32);
      for (size_t j = 0; j < 16; ++j) {
        const uint8_t byte = qs[qs_off + j];
        out[y++] =
            dl * static_cast<float>(tables::kvalues_iq4nl[byte & 0x0Fu]);
      }
      for (size_t j = 0; j < 16; ++j) {
        const uint8_t byte = qs[qs_off + j];
        out[y++] =
            dl * static_cast<float>(tables::kvalues_iq4nl[byte >> 4]);
      }
      qs_off += 16;
    }
  }
}

}  // namespace

float f16_to_f32(uint16_t bits) noexcept {
  const uint32_t exp = (static_cast<uint32_t>(bits) >> 10) & 0x1Fu;
  if (exp != 0 && exp != 0x1Fu) {
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    const uint32_t e = (exp + 112u) << 23;
    const uint32_t m = (static_cast<uint32_t>(bits) & 0x3FFu) << 13;
    return std::bit_cast<float>(sign | e | m);
  }
  return f16_to_f32_slow(bits);
}

uint16_t f32_to_f16(float value) noexcept {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu);
  const uint32_t mant = bits & 0x7FFFFFu;
  if (exp == 0xFF) {
    return static_cast<uint16_t>(sign | 0x7C00u |
                                 (mant != 0 ? 0x0200u : 0u));
  }
  const int32_t e = exp - 127 + 15;
  if (e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
  if (e <= 0) {
    if (e < -10) return sign;
    const uint32_t m = mant | 0x800000u;
    const int32_t shift = 14 - e;
    uint32_t half = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    if (rem > (1u << (shift - 1)) ||
        (rem == (1u << (shift - 1)) && (half & 1u) == 1u)) {
      ++half;
    }
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(half));
  }
  uint32_t h = (static_cast<uint32_t>(e) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) == 1u)) ++h;
  return static_cast<uint16_t>(sign | static_cast<uint16_t>(h));
}

Result<void> dequant_into(QuantType quant, std::span<const uint8_t> src,
                          uint64_t numel, std::span<float> out) {
  const auto spec = block_spec(quant);
  if (!spec.has_value()) {
    return make_error(ErrCode::UnsupportedQuant, "dequant.quant",
                      "unsupported quant code " +
                          std::to_string(quant_code(quant)));
  }
  if (!dequant_supported(quant)) {
    return make_error(ErrCode::UnsupportedQuant, "dequant.quant",
                      "unsupported quant code " +
                          std::to_string(quant_code(quant)));
  }
  if (numel % spec->weights != 0) {
    return make_error(ErrCode::QuantBlockMismatch, "dequant.numel",
                      "element count " + std::to_string(numel) +
                          " is not a multiple of block weights " +
                          std::to_string(spec->weights));
  }
  const uint64_t blocks = numel / spec->weights;
  if (blocks > std::numeric_limits<uint64_t>::max() / spec->bytes) {
    return make_error(ErrCode::Overflow, "dequant.byte_len",
                      "quantized byte length overflows uint64");
  }
  const uint64_t need = blocks * spec->bytes;
  if (src.size() < need) {
    return make_error(ErrCode::Truncated, "dequant.src",
                      "need " + std::to_string(need) + " bytes, got " +
                          std::to_string(src.size()));
  }
  if (numel > out.size()) {
    return make_error(ErrCode::TensorSizeMismatch, "dequant.out",
                      "need " + std::to_string(numel) + " floats, got " +
                          std::to_string(out.size()));
  }

  const uint8_t* s = src.data();
  float* o = out.data();
  const size_t n = static_cast<size_t>(numel);

  switch (quant) {
    case QuantType::F32:
      for (size_t i = 0; i < n; ++i) {
        o[i] = std::bit_cast<float>(read_u32(s, i * 4));
      }
      break;
    case QuantType::F16:
      dequant_f16(s, n, o);
      break;
    case QuantType::Bf16:
      dequant_bf16(s, n, o);
      break;
    case QuantType::Q8_0:
      dequant_q8_0(s, n, o);
      break;
    case QuantType::Q4_0:
      dequant_q4_0(s, n, o);
      break;
    case QuantType::Q4K:
      dequant_q4_k(s, n, o);
      break;
    case QuantType::Q5K:
      dequant_q5_k(s, n, o);
      break;
    case QuantType::Q6K:
      dequant_q6_k(s, n, o);
      break;
    case QuantType::Iq2S:
      dequant_iq2_s(s, n, o);
      break;
    case QuantType::Iq3Xxs:
      dequant_iq3_xxs(s, n, o);
      break;
    case QuantType::Iq3S:
      dequant_iq3_s(s, n, o);
      break;
    case QuantType::Iq4Xs:
      dequant_iq4_xs(s, n, o);
      break;
    default:
      return make_error(ErrCode::UnsupportedQuant, "dequant.quant",
                        "unsupported quant code " +
                            std::to_string(quant_code(quant)));
  }
  return {};
}

Result<std::vector<float>> dequant(QuantType quant,
                                   std::span<const uint8_t> src,
                                   uint64_t numel) {
  if (numel > std::numeric_limits<size_t>::max()) {
    return make_error(ErrCode::Overflow, "dequant.numel",
                      "element count is larger than address space");
  }
  std::vector<float> out(static_cast<size_t>(numel));
  const auto result = dequant_into(quant, src, numel, out);
  if (!result.ok()) return result.error();
  return out;
}

Result<void> quantize_q8_0_into(std::span<const float> x,
                                std::span<uint8_t> dst) {
  constexpr size_t kQk = 32;
  constexpr size_t kBs = 34;
  if (x.size() % kQk != 0) {
    return make_error(ErrCode::QuantBlockMismatch, "quantize.numel",
                      "element count " + std::to_string(x.size()) +
                          " is not a multiple of 32");
  }
  const size_t blocks = x.size() / kQk;
  if (dst.size() < blocks * kBs) {
    return make_error(ErrCode::TensorSizeMismatch, "quantize.dst",
                      "need " + std::to_string(blocks * kBs) + " bytes, got " +
                          std::to_string(dst.size()));
  }
  for (size_t b = 0; b < blocks; ++b) {
    const float* src = x.data() + b * kQk;
    float amax = 0.0F;
    for (size_t j = 0; j < kQk; ++j) {
      const float a = src[j] < 0.0F ? -src[j] : src[j];
      if (a > amax) amax = a;
    }
    // та же схема что у активаций: симметричный int8 с шагом amax/127.
    // q8_0 хранит масштаб как f16, поэтому округляем через него же, иначе
    // вес и его масштаб разъедутся на последнем бите
    // same scheme as the activations: symmetric int8 with step amax/127.
    // q8_0 stores the scale as f16, so round through it, otherwise the weight
    // and its scale disagree in the last bit
    const uint16_t d_bits = f32_to_f16(amax / 127.0F);
    const float d = f16_to_f32(d_bits);
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    uint8_t* out = dst.data() + b * kBs;
    out[0] = static_cast<uint8_t>(d_bits & 0xFFU);
    out[1] = static_cast<uint8_t>(d_bits >> 8);
    for (size_t j = 0; j < kQk; ++j) {
      float q = src[j] * id;
      int v = static_cast<int>(q >= 0.0F ? q + 0.5F : q - 0.5F);
      if (v > 127) v = 127;
      if (v < -127) v = -127;
      out[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(v));
    }
  }
  return {};
}

Result<void> quantize_q4_0_into(std::span<const float> x,
                                std::span<uint8_t> dst) {
  constexpr size_t kQk = 32;
  constexpr size_t kBs = 18;
  if (x.size() % kQk != 0) {
    return make_error(ErrCode::QuantBlockMismatch, "quantize.q4.numel",
                      "element count " + std::to_string(x.size()) +
                          " is not a multiple of 32");
  }
  const size_t blocks = x.size() / kQk;
  if (dst.size() < blocks * kBs) {
    return make_error(ErrCode::TensorSizeMismatch, "quantize.q4.dst",
                      "need " + std::to_string(blocks * kBs) + " bytes, got " +
                          std::to_string(dst.size()));
  }
  for (size_t b = 0; b < blocks; ++b) {
    const float* src = x.data() + b * kQk;
    float amax = 0.0F;
    for (size_t j = 0; j < kQk; ++j) {
      const float a = src[j] < 0.0F ? -src[j] : src[j];
      if (a > amax) amax = a;
    }
    // Q4_0: (q - 8) * d, q in [0, 15]. d is positive here, so the range is
    // [-8d, 7d]. Rounding through f16 keeps the stored scale and the weights
    // consistent, the same way the Q8 path does.
    const uint16_t d_bits = f32_to_f16(amax / 8.0F);
    const float d = f16_to_f32(d_bits);
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    uint8_t* out = dst.data() + b * kBs;
    out[0] = static_cast<uint8_t>(d_bits & 0xFFU);
    out[1] = static_cast<uint8_t>(d_bits >> 8);
    std::memset(out + 2, 0, 16);
    for (size_t j = 0; j < kQk; ++j) {
      const float q = src[j] * id;
      int v = static_cast<int>(q >= 0.0F ? q + 0.5F : q - 0.5F) + 8;
      if (v > 15) v = 15;
      if (v < 0) v = 0;
      if (j < 16) {
        out[2 + j] = static_cast<uint8_t>((out[2 + j] & 0xF0U) |
                                          static_cast<uint8_t>(v));
      } else {
        out[2 + j - 16] = static_cast<uint8_t>(
            (out[2 + j - 16] & 0x0FU) | static_cast<uint8_t>(v << 4));
      }
    }
  }
  return {};
}

}  // namespace tokshit
