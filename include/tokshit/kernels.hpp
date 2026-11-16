#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "tokshit/quant.hpp"

namespace tokshit::kernels {

// активация проквантованная в int8 блоками по 32: x ~= q * scale
// activation quantized to int8 in blocks of 32: x ~= q * scale
struct ActQ8 {
  std::vector<int8_t> q;
  std::vector<float> scales;  // одна на блок 32 / one per 32 elems
  size_t len = 0;
};

// симметричный per-block int8 квант (max/127), как в rust эталоне
// symmetric per-block int8 quant (max/127), same as the rust reference
[[nodiscard]] ActQ8 quantize(std::span<const float> x);

// то же что quantize, но переиспользует буферы внутри `act`. горячий путь
// обязан звать это, иначе будет по аллокации на каждый сегмент matvec
// same as quantize but reuses the buffers inside `act`. hot path should use
// this or it allocates once per matvec segment
void quantize_into(std::span<const float> x, ActQ8& act);

// скалярный фолбэк: разложить строку в `scratch`, потом скалярно умножить на ту
// же int8 активацию что и neon ядра
// scalar fallback, dequant the row into `scratch` then dot with the same int8
// activation the neon kernels use
[[nodiscard]] float dot_fallback(QuantType quant, std::span<const uint8_t> row,
                                 size_t rb, std::span<const float> x,
                                 const ActQ8& act, size_t in_dim,
                                 std::vector<float>& scratch);

// главная точка входа: neon dotprod если есть, иначе скаляр
// hot-path entry point, neon dotprod when available, scalar otherwise
[[nodiscard]] float dot_row(QuantType quant, std::span<const uint8_t> row,
                            size_t rb, std::span<const float> x,
                            const ActQ8& act, size_t in_dim,
                            std::vector<float>& scratch);

// батч-версия: одна строка весов на n активаций. для квантов без neon-ядра
// строка раскладывается в float один раз и умножается на все активации — это
// экономит (n-1)/n декванта и чтений строки на verify-батче
// batched dot: one weight row against n activations. for quants without a
// neon kernel the row is dequantized once and dotted with every activation,
// saving (n-1)/n of the dequant work and row reads on a verify batch
void dot_row_multi(QuantType quant, std::span<const uint8_t> row, size_t rb,
                   const ActQ8* acts, size_t n, size_t in_dim, float* out,
                   std::vector<float>& scratch);

// Несколько подряд идущих строк весов против одной активации. q8_0/q4_0
// обрабатываются 4 строками за проход: активация читается один раз, а
// независимые цепочки vdot закрывают латентность. Остальные кванты идут
// обычным dot_row, поэтому результат совпадает с построчным вызовом.
// Several consecutive weight rows against one activation. q8_0/q4_0 run four
// rows per pass: the activation is loaded once and independent vdot chains
// hide latency. Other quants fall back to dot_row, so results match the
// row-at-a-time path.
void dot_rows(QuantType quant, std::span<const uint8_t> rows, size_t rb,
              size_t nrows, std::span<const float> x, const ActQ8& act,
              size_t in_dim, float* out, std::vector<float>& scratch);

// проверка фичи cpu на рантайме (HWCAP_ASIMDDP на aarch64)
// runtime cpu feature check (HWCAP_ASIMDDP on aarch64)
[[nodiscard]] bool have_dotprod() noexcept;

// есть ли i8mm (FEAT_I8MM, бит 13 в AT_HWCAP2 на arm64 linux). SMMLA-ядра
// включаются отдельно флагом TOKSHIT_I8MM=1
// is i8mm present (FEAT_I8MM, bit 13 of AT_HWCAP2 on arm64 linux)? SMMLA
// kernels are enabled separately with TOKSHIT_I8MM=1
[[nodiscard]] bool have_i8mm() noexcept;

// есть ли для этого кванта векторное ядро dot_row. для остальных квантов
// горячий путь платит за скалярный dequant+dot, поэтому загрузчик может
// переквантовать такие веса один раз при старте
// is there a vector kernel for this quant in dot_row? everything else pays for
// the scalar dequant+dot on the hot path, so the loader may requantize those
// weights once at startup
[[nodiscard]] bool has_neon_dot(QuantType quant) noexcept;

}  // namespace tokshit::kernels
