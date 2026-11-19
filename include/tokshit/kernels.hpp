#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "tokshit/quant.hpp"

namespace tokshit::kernels {

// активация проквантованная в int8 блоками по 32: x ~= q * scale
struct ActQ8 {
  std::vector<int8_t> q;
  std::vector<float> scales;  // одна на блок 32
  size_t len = 0;
};

// симметричный per-block int8 квант (max/127)
[[nodiscard]] ActQ8 quantize(std::span<const float> x);

// то же что quantize, но переиспользует буферы внутри `act`. горячий путь
// обязан звать это, иначе будет по аллокации на каждый сегмент matvec
void quantize_into(std::span<const float> x, ActQ8& act);

// скалярный фолбэк: разложить строку в `scratch`, потом скалярно умножить на ту
// же int8 активацию что и neon ядра
[[nodiscard]] float dot_fallback(QuantType quant, std::span<const uint8_t> row,
                                 size_t rb, std::span<const float> x,
                                 const ActQ8& act, size_t in_dim,
                                 std::vector<float>& scratch);

// главная точка входа: neon dotprod если есть, иначе скаляр
[[nodiscard]] float dot_row(QuantType quant, std::span<const uint8_t> row,
                            size_t rb, std::span<const float> x,
                            const ActQ8& act, size_t in_dim,
                            std::vector<float>& scratch);

// батч-версия: одна строка весов на n активаций. для квантов без neon-ядра
// строка раскладывается в float один раз и умножается на все активации — это
// экономит (n-1)/n декванта и чтений строки на verify-батче
void dot_row_multi(QuantType quant, std::span<const uint8_t> row, size_t rb,
                   const ActQ8* acts, size_t n, size_t in_dim, float* out,
                   std::vector<float>& scratch);

// Несколько подряд идущих строк весов против одной активации. q8_0/q4_0
// обрабатываются 4 строками за проход: активация читается один раз, а
// независимые цепочки vdot закрывают латентность. Остальные кванты идут
// обычным dot_row, поэтому результат совпадает с построчным вызовом.
void dot_rows(QuantType quant, std::span<const uint8_t> rows, size_t rb,
              size_t nrows, std::span<const float> x, const ActQ8& act,
              size_t in_dim, float* out, std::vector<float>& scratch);

// проверка фичи cpu на рантайме (HWCAP_ASIMDDP на aarch64)
[[nodiscard]] bool have_dotprod() noexcept;

// есть ли i8mm (FEAT_I8MM, бит 13 в AT_HWCAP2 на arm64 linux). SMMLA-ядра
// включаются отдельно флагом TOKSHIT_I8MM=1
[[nodiscard]] bool have_i8mm() noexcept;

// есть ли для этого кванта векторное ядро dot_row. для остальных квантов
// горячий путь платит за скалярный dequant+dot, поэтому загрузчик может
// переквантовать такие веса один раз при старте
[[nodiscard]] bool has_neon_dot(QuantType quant) noexcept;

}  // namespace tokshit::kernels
