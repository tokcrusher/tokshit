#include "tokshit/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace tokshit {
namespace {

struct Utf8Check {
  bool valid = true;
  std::size_t valid_up_to = 0;
  std::size_t error_len = 0;
};

// проверяем utf-8 вручную. отдаём до какого места валидно и
// длину ошибки
Utf8Check check_utf8(std::span<const uint8_t> s) {
  std::size_t i = 0;
  while (i < s.size()) {
    const uint8_t b0 = s[i];
    if (b0 < 0x80) {
      ++i;
      continue;
    }
    std::size_t len = 0;
    uint8_t lo = 0x80;
    uint8_t hi = 0xBF;
    if (b0 >= 0xC2 && b0 <= 0xDF) {
      len = 2;
    } else if (b0 == 0xE0) {
      len = 3;
      lo = 0xA0;
    } else if (b0 >= 0xE1 && b0 <= 0xEC) {
      len = 3;
    } else if (b0 == 0xED) {
      len = 3;
      hi = 0x9F;
    } else if (b0 >= 0xEE && b0 <= 0xEF) {
      len = 3;
    } else if (b0 == 0xF0) {
      len = 4;
      lo = 0x90;
    } else if (b0 >= 0xF1 && b0 <= 0xF3) {
      len = 4;
    } else if (b0 == 0xF4) {
      len = 4;
      hi = 0x8F;
    } else {
      return {false, i, 1};
    }
    if (i + len > s.size()) {
      return {false, i, 0};
    }
    const uint8_t b1 = s[i + 1];
    if (b1 < lo || b1 > hi) return {false, i, 1};
    bool ok = true;
    for (std::size_t k = 2; k < len; ++k) {
      const uint8_t bk = s[i + k];
      if (bk < 0x80 || bk > 0xBF) {
        ok = false;
        break;
      }
    }
    if (!ok) return {false, i, 1};
    i += len;
  }
  return {true, s.size(), 0};
}

// заменяем битые куски на U+FFFD, чтобы не сломать терминал
std::string utf8_lossy(std::span<const uint8_t> bytes) {
  std::string out;
  out.reserve(bytes.size());
  std::size_t i = 0;
  while (i < bytes.size()) {
    const auto remaining = bytes.subspan(i);
    const Utf8Check c = check_utf8(remaining);
    if (c.valid) {
      out.append(reinterpret_cast<const char*>(remaining.data()),
                 remaining.size());
      break;
    }
    if (c.valid_up_to > 0) {
      out.append(reinterpret_cast<const char*>(remaining.data()), c.valid_up_to);
      i += c.valid_up_to;
      continue;
    }
    out += "\xEF\xBF\xBD";
    if (c.error_len == 0) break;
    i += c.error_len;
  }
  return out;
}

}  // namespace

Rng::Rng() {
  const char* seed_env = std::getenv("TOKSHIT_SEED");
  if (seed_env != nullptr && *seed_env != '\0') {
    reseed(std::strtoull(seed_env, nullptr, 10));
  } else {
    std::random_device rd;
    state_.seed(rd());
  }
}

void Rng::reseed(uint64_t seed) {
  state_.seed(static_cast<std::mt19937::result_type>(seed));
}

float Rng::f32() { return dist_(state_); }

// если temperature <= 0 то честный greedy argmax (первый максимум). иначе
// softmax с температурой, top-k, top-p и один uniform выбор
uint32_t sample(std::span<const float> logits, const SampleParams& params,
                Rng& rng) {
  const std::size_t n = logits.size();
  if (n == 0) return 0;
  if (params.temperature <= 0.0F) {
    const auto best = std::max_element(logits.begin(), logits.end());
    return static_cast<uint32_t>(
        std::distance(logits.begin(), best));
  }
  const float inv_t = 1.0F / params.temperature;
  // Subtract the largest logit before exponentiation. Softmax is invariant to
  // this shift, and it prevents overflow for large (but valid) model logits.
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> probs(n);
  for (std::size_t i = 0; i < n; ++i) {
    probs[i] = std::exp((logits[i] - max_logit) * inv_t);
  }
  float sum = 0.0F;
  for (float p : probs) sum += p;
  if (sum > 0.0F) {
    for (float& p : probs) p /= sum;
  }

  const std::size_t k =
      (params.top_k > 0 && params.top_k < n) ? params.top_k : n;
  std::vector<uint32_t> idx(n);
  for (std::size_t i = 0; i < n; ++i) idx[i] = static_cast<uint32_t>(i);

  if (k < n) {
    std::nth_element(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k),
                     idx.end(), [&](uint32_t a, uint32_t b) {
                       return probs[a] > probs[b];
                     });
    idx.resize(k);
    std::vector<std::pair<uint32_t, float>> keep;
    keep.reserve(k);
    for (uint32_t i : idx) keep.emplace_back(i, probs[i]);
    std::fill(probs.begin(), probs.end(), 0.0F);
    for (const auto& [i, p] : keep) probs[i] = p;
    sum = 0.0F;
    for (float p : probs) sum += p;
    if (sum > 0.0F) {
      for (float& p : probs) p /= sum;
    }
  }

  std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
    return probs[a] > probs[b];
  });

  if (params.top_p > 0.0F && params.top_p < 1.0F) {
    float cum = 0.0F;
    std::size_t cutoff = idx.size();
    for (std::size_t pos = 0; pos < idx.size(); ++pos) {
      cum += probs[idx[pos]];
      if (cum >= params.top_p) {
        cutoff = pos + 1;
        break;
      }
    }
    for (std::size_t pos = cutoff; pos < idx.size(); ++pos) {
      probs[idx[pos]] = 0.0F;
    }
    sum = 0.0F;
    for (float p : probs) sum += p;
    if (sum > 0.0F) {
      for (float& p : probs) p /= sum;
    }
  }

  const float r = rng.f32();
  float cum = 0.0F;
  for (uint32_t i : idx) {
    cum += probs[i];
    if (r <= cum) return i;
  }
  return idx.empty() ? 0 : idx.front();
}

// копим байты и отдаём самый длинный целый utf-8 префикс. хвост остаётся
// внутри до следующего push
std::string Utf8Buffer::push(std::span<const uint8_t> bytes) {
  buf_.insert(buf_.end(), bytes.begin(), bytes.end());
  const Utf8Check c =
      check_utf8(std::span<const uint8_t>(buf_.data(), buf_.size()));
  const std::size_t valid_len = c.valid ? buf_.size() : c.valid_up_to;
  if (valid_len == 0) return {};
  std::string out(reinterpret_cast<const char*>(buf_.data()), valid_len);
  buf_.erase(buf_.begin(),
             buf_.begin() + static_cast<std::ptrdiff_t>(valid_len));
  return out;
}

// в конце генерации выплёвываем хвост, даже если он неполный
std::string Utf8Buffer::flush() {
  std::string out = utf8_lossy(std::span<const uint8_t>(buf_.data(), buf_.size()));
  buf_.clear();
  return out;
}

// пробуем сэмплить так чтобы токен не создал битый utf-8, максимум 8 попыток.
// если совсем не вышло берём lossy вариант
uint32_t sample_utf8_safe(std::span<const float> logits,
                          const SampleParams& params, const BpeTokenizer& tok,
                          Utf8Buffer& buf, Rng& rng) {
  for (int attempt = 0; attempt < 8; ++attempt) {
    const uint32_t id = sample(logits, params, rng);
    const auto& tb = tok.token_bytes(id);
    std::vector<uint8_t> test = buf.bytes();
    test.insert(test.end(), tb.begin(), tb.end());
    const Utf8Check c =
        check_utf8(std::span<const uint8_t>(test.data(), test.size()));
    if (c.valid) return id;
    if (c.valid_up_to + c.error_len == test.size()) return id;
  }
  return sample(logits, params, rng);
}

}  // namespace tokshit
