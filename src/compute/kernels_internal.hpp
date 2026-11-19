#pragma once

#include <cstddef>
#include <cstdint>

#include "tokshit/kernels.hpp"

namespace tokshit::kernels::neon {

// aarch64 + dotprod only. Compiled in a separate translation unit with
// -march=armv8.2-a+dotprod so these instructions never leak into the base
// binary.
float dot_q8_0(const uint8_t* row, const ActQ8& act, size_t in_dim) noexcept;
float dot_q4_0(const uint8_t* row, const ActQ8& act, size_t in_dim) noexcept;
void dot_rows_q8_0(const uint8_t* rows, size_t rb, size_t nrows,
                   const ActQ8& act, size_t in_dim, float* out) noexcept;
void dot_rows_q4_0(const uint8_t* rows, size_t rb, size_t nrows,
                   const ActQ8& act, size_t in_dim, float* out) noexcept;
float dot_iq4xs(const uint8_t* row, const ActQ8& act, size_t in_dim) noexcept;
float dot_iq2s(const uint8_t* row, const ActQ8& act, size_t in_dim) noexcept;
float dot_iq3s(const uint8_t* row, const ActQ8& act, size_t in_dim) noexcept;
float dot_iq3xxs(const uint8_t* row, const ActQ8& act, size_t in_dim) noexcept;

// q4_k (6-бит шкалы + min на сабблок). dot считается точно против
// скалярного эталона через sum(v*q) и sum(q) на 32-блоке активации
float dot_q4k(const uint8_t* row, const ActQ8& act, size_t in_dim) noexcept;
void dot_rows_q4k(const uint8_t* rows, size_t rb, size_t nrows,
                  const ActQ8& act, size_t in_dim, float* out) noexcept;

// двухстрочные варианты IQ-ядер: пара строк за проход, общая загрузка
// активации, независимые цепочки деквант+dot
void dot_rows_iq4xs(const uint8_t* rows, size_t rb, size_t nrows,
                    const ActQ8& act, size_t in_dim, float* out) noexcept;
void dot_rows_iq3s(const uint8_t* rows, size_t rb, size_t nrows,
                   const ActQ8& act, size_t in_dim, float* out) noexcept;
void dot_rows_iq3xxs(const uint8_t* rows, size_t rb, size_t nrows,
                     const ActQ8& act, size_t in_dim, float* out) noexcept;

// i8mm (SMMLA), отдельный TU с -march=armv8.2-a+dotprod+i8mm
void dot_rows_q8_0_i8mm(const uint8_t* rows, size_t rb, size_t nrows,
                        const ActQ8& act, size_t in_dim, float* out) noexcept;

}  // namespace tokshit::kernels::neon
