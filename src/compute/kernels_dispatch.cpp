#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/quant.hpp"

#if defined(__aarch64__)
#include <sys/auxv.h>
#include "kernels_internal.hpp"
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
// arm64 linux: FEAT_I8MM в AT_HWCAP2, бит 13. проверено пробником SMMLA на
// целевом cpu; заголовки Termux описывают только 32-битный arm, их биты
// совпадать не обязаны
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#endif

namespace tokshit::kernels {

namespace {

// env-гейт: не задано = 0 (старое поведение), "0" = 0, любое другое = 1
int env_flag(const char* name) noexcept {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0' || v[0] == '0') return 0;
  return 1;
}

}  // namespace

// проверяем есть ли на этом cpu dotprod. результат кэшируется
bool have_dotprod() noexcept {
#if defined(__aarch64__)
  static const bool detected =
      (::getauxval(AT_HWCAP) & static_cast<unsigned long>(HWCAP_ASIMDDP)) != 0;
  return detected;
#else
  return false;
#endif
}

bool have_i8mm() noexcept {
#if defined(__aarch64__)
  static const bool detected =
      (::getauxval(AT_HWCAP2) & static_cast<unsigned long>(HWCAP2_I8MM)) != 0;
  return detected;
#else
  return false;
#endif
}

// SMMLA-путь: только с TOKSHIT_I8MM=1 и наличием фичи. по умолчанию
// работает старый dotprod-путь
bool use_i8mm() noexcept {
#if defined(TOKSHIT_HAVE_NEON_I8MM)
  return have_dotprod() && have_i8mm() && env_flag("TOKSHIT_I8MM") != 0;
#else
  return false;
#endif
}

// K-квант ядра (q4_k): TOKSHIT_KQK=1, дефолт — старое поведение
bool use_kquant() noexcept {
#if defined(TOKSHIT_HAVE_NEON_DOTPROD)
  const char* v = std::getenv("TOKSHIT_KQK");
  return have_dotprod() && (v == nullptr || v[0] != '0');
#else
  return false;
#endif
}

// двухстрочные IQ-ядра: TOKSHIT_IQ2ROW=1 включает, "0" выключает; если
// замер покажет выигрыш, дефолт можно перевернуть на автодетект
bool use_iq2row() noexcept {
#if defined(TOKSHIT_HAVE_NEON_DOTPROD)
  return have_dotprod() && env_flag("TOKSHIT_IQ2ROW") != 0;
#else
  return false;
#endif
}

bool has_neon_dot(QuantType quant) noexcept {
#if defined(__aarch64__) && defined(TOKSHIT_HAVE_NEON_DOTPROD)
  if (!have_dotprod()) return false;
  switch (quant) {
    case QuantType::Q8_0:
    case QuantType::Q4_0:
    case QuantType::Iq4Xs:
    case QuantType::Iq2S:
    case QuantType::Iq3S:
    case QuantType::Iq3Xxs:
    case QuantType::Q2_0:
    case QuantType::Mxfp4:
    case QuantType::Nvfp4:
    case QuantType::Iq1S:
    case QuantType::Iq2Xxs:
    case QuantType::Iq2Xs:
    case QuantType::Q6K:
    case QuantType::Q4K:
      return true;
    default:
      return false;
  }
#else
  (void)quant;
  return false;
#endif
}

// выбираем neon или скаляр. никаких виртуальных вызовов, просто switch
float dot_row(QuantType quant, std::span<const uint8_t> row, size_t rb,
              std::span<const float> x, const ActQ8& act, size_t in_dim,
              std::vector<float>& scratch) {
#if defined(__aarch64__) && defined(TOKSHIT_HAVE_NEON_DOTPROD)
  if (have_dotprod() && row.size() >= rb) {
    switch (quant) {
      case QuantType::Q8_0:
        if (in_dim % 32 == 0) {
          return neon::dot_q8_0(row.data(), act, in_dim);
        }
        break;
      case QuantType::Q4_0:
        if (in_dim % 32 == 0) {
          return neon::dot_q4_0(row.data(), act, in_dim);
        }
        break;
      case QuantType::Iq4Xs:
        if (in_dim % 256 == 0) {
          return neon::dot_iq4xs(row.data(), act, in_dim);
        }
        break;
      case QuantType::Iq2S:
        if (in_dim % 256 == 0) {
          return neon::dot_iq2s(row.data(), act, in_dim);
        }
        break;
      case QuantType::Iq3S:
        if (in_dim % 256 == 0) {
          return neon::dot_iq3s(row.data(), act, in_dim);
        }
        break;
      case QuantType::Iq3Xxs:
        if (in_dim % 256 == 0) {
          return neon::dot_iq3xxs(row.data(), act, in_dim);
        }
        break;
      case QuantType::Q2_0:
        if (in_dim % 64 == 0) {
          return neon::dot_q2_0(row.data(), act, in_dim);
        }
        break;
      case QuantType::Iq2Xxs:
        if (in_dim % 256 == 0) {
          return neon::dot_iq2xxs(row.data(), act, in_dim);
        }
        break;
      case QuantType::Iq2Xs:
        if (in_dim % 256 == 0) {
          return neon::dot_iq2xs(row.data(), act, in_dim);
        }
        break;
      case QuantType::Iq1S:
        if (in_dim % 256 == 0) {
          return neon::dot_iq1s(row.data(), act, in_dim);
        }
        break;
      case QuantType::Mxfp4:
        if (in_dim % 32 == 0) {
          return neon::dot_mxfp4(row.data(), act, in_dim);
        }
        break;
      case QuantType::Nvfp4:
        if (in_dim % 64 == 0) {
          return neon::dot_nvfp4(row.data(), act, in_dim);
        }
        break;
      case QuantType::Q6K:
        if (in_dim % 256 == 0) {
          return neon::dot_q6k(row.data(), act, in_dim);
        }
        break;
      case QuantType::Q4K:
        if (use_kquant() && in_dim % 256 == 0) {
          return neon::dot_q4k(row.data(), act, in_dim);
        }
        break;
      default:
        break;
    }
  }
#endif

  return dot_fallback(quant, row, rb, x, act, in_dim, scratch);
}

void dot_row_multi(QuantType quant, std::span<const uint8_t> row, size_t rb,
                   const ActQ8* acts, size_t n, size_t in_dim, float* out,
                   std::vector<float>& scratch) {
  if (n == 0) return;

#if defined(__aarch64__) && defined(TOKSHIT_HAVE_NEON_DOTPROD)
  // если для кванта есть готовое neon-ядро, n вызовов дешевле одного
  // декванта в float: ядро читает упакованную строку из l1
  if (have_dotprod()) {
    bool handled = true;
    switch (quant) {
      case QuantType::Q8_0:
        if (in_dim % 32 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_q8_0(row.data(), acts[j], in_dim);
        break;
      case QuantType::Q4_0:
        if (in_dim % 32 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_q4_0(row.data(), acts[j], in_dim);
        break;
      case QuantType::Iq4Xs:
        if (in_dim % 256 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_iq4xs(row.data(), acts[j], in_dim);
        break;
      case QuantType::Iq2S:
        if (in_dim % 256 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_iq2s(row.data(), acts[j], in_dim);
        break;
      case QuantType::Iq3S:
        if (in_dim % 256 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_iq3s(row.data(), acts[j], in_dim);
        break;
      case QuantType::Iq3Xxs:
        if (in_dim % 256 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_iq3xxs(row.data(), acts[j], in_dim);
        break;
      case QuantType::Q6K:
        if (in_dim % 256 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_q6k(row.data(), acts[j], in_dim);
        break;
      case QuantType::Q4K:
        if (in_dim % 256 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_q4k(row.data(), acts[j], in_dim);
        break;
      case QuantType::Q2_0:
        if (in_dim % 64 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_q2_0(row.data(), acts[j], in_dim);
        break;
      case QuantType::Mxfp4:
        if (in_dim % 32 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_mxfp4(row.data(), acts[j], in_dim);
        break;
      case QuantType::Nvfp4:
        if (in_dim % 64 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_nvfp4(row.data(), acts[j], in_dim);
        break;
      case QuantType::Iq1S:
        if (in_dim % 256 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_iq1s(row.data(), acts[j], in_dim);
        break;
      case QuantType::Iq2Xxs:
        if (in_dim % 256 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_iq2xxs(row.data(), acts[j], in_dim);
        break;
      case QuantType::Iq2Xs:
        if (in_dim % 256 != 0) handled = false;
        else
          for (size_t j = 0; j < n; ++j)
            out[j] = neon::dot_iq2xs(row.data(), acts[j], in_dim);
        break;
      default:
        handled = false;
        break;
    }
    if (handled) return;
  }
#endif

  // общий путь: разложить строку один раз и умножить на каждую активацию.
  // арифметика ровно как в dot_fallback, поэтому числа совпадают
  const auto spec = block_spec(quant);
  const size_t expected =
      spec.has_value() && spec->weights != 0
          ? (in_dim / spec->weights) * spec->bytes
          : 0;
  const size_t needed = expected == 0 ? rb : std::min(rb, expected);
  if (row.size() < needed || spec == std::nullopt) {
    for (size_t j = 0; j < n; ++j) {
      out[j] = dot_fallback(
          quant, row, rb, std::span<const float>(), acts[j], in_dim, scratch);
    }
    return;
  }

  scratch.resize(in_dim);
  const auto r = dequant_into(quant, row.first(needed), in_dim, scratch);
  if (!r.ok()) {
    for (size_t j = 0; j < n; ++j) out[j] = 0.0F;
    return;
  }
  const float* w = scratch.data();
  for (size_t j = 0; j < n; ++j) {
    const int8_t* q = acts[j].q.data();
    const float* s = acts[j].scales.data();
    float sum = 0.0F;
    for (size_t i = 0; i < in_dim; ++i) {
      sum += w[i] * (static_cast<float>(q[i]) * s[i / 32]);
    }
    out[j] = sum;
  }
}

void dot_rows(QuantType quant, std::span<const uint8_t> rows, size_t rb,
              size_t nrows, std::span<const float> x, const ActQ8& act,
              size_t in_dim, float* out, std::vector<float>& scratch) {
  if (nrows == 0) return;
  if (in_dim == 0 || rb == 0 || rows.size() < nrows * rb) {
    for (size_t r = 0; r < nrows; ++r) out[r] = 0.0F;
    return;
  }
#if defined(__aarch64__) && defined(TOKSHIT_HAVE_NEON_DOTPROD)
  if (have_dotprod() && in_dim % 32 == 0) {
    // i8mm/SMMLA путь для q8_0: TOKSHIT_I8MM=1, дефолт — dotprod-ядро
    if (quant == QuantType::Q8_0 && use_i8mm()) {
      neon::dot_rows_q8_0_i8mm(rows.data(), rb, nrows, act, in_dim, out);
      return;
    }
    if (quant == QuantType::Q8_0) {
      neon::dot_rows_q8_0(rows.data(), rb, nrows, act, in_dim, out);
      return;
    }
    if (quant == QuantType::Q4_0) {
      neon::dot_rows_q4_0(rows.data(), rb, nrows, act, in_dim, out);
      return;
    }
    if (use_kquant() && quant == QuantType::Q4K && in_dim % 256 == 0) {
      neon::dot_rows_q4k(rows.data(), rb, nrows, act, in_dim, out);
      return;
    }
    if (quant == QuantType::Q2_0 && in_dim % 64 == 0) {
      neon::dot_rows_q2_0(rows.data(), rb, nrows, act, in_dim, out);
      return;
    }
    if (quant == QuantType::Nvfp4 && in_dim % 64 == 0) {
      neon::dot_rows_nvfp4(rows.data(), rb, nrows, act, in_dim, out);
      return;
    }
    if (use_iq2row() && in_dim % 256 == 0) {
      switch (quant) {
        case QuantType::Iq4Xs:
          neon::dot_rows_iq4xs(rows.data(), rb, nrows, act, in_dim, out);
          return;
        case QuantType::Iq3S:
          neon::dot_rows_iq3s(rows.data(), rb, nrows, act, in_dim, out);
          return;
        case QuantType::Iq3Xxs:
          neon::dot_rows_iq3xxs(rows.data(), rb, nrows, act, in_dim, out);
          return;
        default:
          break;
      }
    }
  }
#else
  (void)x;
#endif
  for (size_t r = 0; r < nrows; ++r) {
    const auto row = rows.subspan(r * rb, rb);
    out[r] = dot_row(quant, row, rb, x, act, in_dim, scratch);
  }
}

}  // namespace tokshit::kernels
