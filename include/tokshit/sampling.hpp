#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "tokshit/tokenizer.hpp"

namespace tokshit {

struct SampleParams {
  float temperature = 0.8F;
  std::size_t top_k = 0;
  float top_p = 0.9F;
};

// обёртка чтобы cli мог сидить сэмплинг через TOKSHIT_SEED
// little wrapper so the cli can seed sampling through TOKSHIT_SEED
class Rng {
 public:
  Rng();
  [[nodiscard]] float f32();
  void reseed(uint64_t seed);

 private:
  std::mt19937 state_;
  std::uniform_real_distribution<float> dist_{0.0F, 1.0F};
};

// сэмплинг точь в точь как в rust эталоне: exp(logit / T), top-k через
// nth_element, top-p по накопленной сумме, потом один uniform выбор
// sampling exactly like the rust reference: exp(logit / T), top-k via
// nth_element, top-p cumulative cut, then one uniform draw
[[nodiscard]] uint32_t sample(std::span<const float> logits,
                              const SampleParams& params, Rng& rng);

// копит байты токена и отдаёт только целые utf-8 префиксы
// accumulates token bytes and emits only complete utf-8 prefixes
class Utf8Buffer {
 public:
  [[nodiscard]] std::string push(std::span<const uint8_t> bytes);
  [[nodiscard]] std::string flush();
  [[nodiscard]] const std::vector<uint8_t>& bytes() const noexcept {
    return buf_;
  }

 private:
  std::vector<uint8_t> buf_;
};

// не даём токену создать битый utf-8, максимум 8 попыток
// reject tokens that would make invalid utf-8, max 8 attempts
[[nodiscard]] uint32_t sample_utf8_safe(std::span<const float> logits,
                                        const SampleParams& params,
                                        const BpeTokenizer& tok,
                                        Utf8Buffer& buf, Rng& rng);

}  // namespace tokshit
