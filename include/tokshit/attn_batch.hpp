#pragma once

#include <cstddef>
#include <cstdint>

namespace tokshit {

namespace attn {

// однострочный (одна позиция батча) attention по kv-слябам. строго та же
// математика, что в Model::layer::attention: scores = q·k с масштабом,
// sink добавляется как последний элемент softmax, в V-сумму НЕ входит
struct AttendArgs {
  // слябы текущего слоя, как отдаёт Kv
  const uint8_t* k = nullptr;  // cap × k_stride байт
  const uint8_t* v = nullptr;  // cap × v_stride байт
  std::size_t k_stride = 0;    // байт на токен
  std::size_t v_stride = 0;
  bool f16 = true;  // иначе слябы — plain f32
  // запросы: b × n_heads × head_dim, rope уже применён; строка j — q[j * sq]
  const float* q = nullptr;
  std::size_t sq = 0;              // шаг между строками в float
  const float* sinks = nullptr;    // n_heads значений (attn_sinks слоя)
  float* out = nullptr;            // b × n_heads × head_dim
  std::size_t b = 0;
  std::size_t n_heads = 0;
  std::size_t n_kvh = 0;           // kv-головы, group = n_heads / n_kvh
  std::size_t head_dim = 0;
  float scale = 1.0F;              // 1 / sqrt(head_dim)
  std::size_t first = 0;           // абсолютная позиция строки 0
  // ёмкость сляба в позициях; end не может её превысить
  std::size_t cap = static_cast<std::size_t>(-1);
  const std::size_t* win = nullptr;  // per row j: старт окна (absolute); 0 = без окна
};

void attend_row(const AttendArgs& a, std::size_t row);

}  // namespace attn
}  // namespace tokshit
