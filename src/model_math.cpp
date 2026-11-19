#include "tokshit/model.hpp"

#include <cmath>
#include <limits>

namespace tokshit {

void Model::rmsnorm(std::span<float> x, std::span<const float> w, float eps) {
  const std::size_t n = x.size();
  float ss = 0.0F;
  for (float v : x) ss += v * v;
  const float inv = 1.0F / std::sqrt(ss / static_cast<float>(n) + eps);
  for (std::size_t i = 0; i < n; ++i) x[i] *= inv * w[i];
}

// rope применяем половинками: первая половина с cos, вторая с sin
// apply rope in halves, first half with cos, second with sin
void Model::rope_apply(std::span<float> x,
                       std::span<const std::pair<float, float>> table) {
  const std::size_t half = x.size() / 2;
  for (std::size_t i = 0; i < half; ++i) {
    const float s = table[i].first;
    const float c = table[i].second;
    const float a = x[i];
    const float b = x[i + half];
    x[i] = a * c - b * s;
    x[i + half] = a * s + b * c;
  }
}

// обычный softmax на месте, через вычитание максимума
// plain in-place softmax, subtract the max first
void Model::softmax(std::span<float> x) {
  float max = -std::numeric_limits<float>::infinity();
  for (float v : x) {
    if (v > max) max = v;
  }
  float sum = 0.0F;
  for (float& v : x) {
    v = std::exp(v - max);
    sum += v;
  }
  for (float& v : x) v /= sum;
}

// swiglu: g = silu(g) * u, на месте по g
// swiglu: g = silu(g) * u, in place over g
void Model::swiglu(std::span<float> g, std::span<const float> u) {
  for (std::size_t i = 0; i < g.size(); ++i) {
    g[i] = g[i] / (1.0F + std::exp(-g[i])) * u[i];
  }
}

// один слой. сначала attention, потом ffn (плотный или moe). всё пишется в
// workspace, ни одной аллокации. kv растёт на один токен за вызов
// one layer. attention first, then ffn (dense or moe). everything goes into
}  // namespace tokshit
