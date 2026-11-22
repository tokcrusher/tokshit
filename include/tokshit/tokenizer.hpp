#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokshit/result.hpp"

namespace tokshit {

inline constexpr uint32_t NO_TOKEN = 0xFFFFFFFFU;

struct TokenizerMeta {
  std::vector<std::string> tokens;
  std::vector<std::string> merges;
  std::vector<uint32_t> token_types;
  std::vector<uint32_t> eog_tokens;
  std::string pre;
  uint32_t bos_token = NO_TOKEN;
  uint32_t eos_token = NO_TOKEN;
  bool add_bos = false;
  bool add_eos = false;
};

// GPT-2 byte-level BPE tokenizer with the Flint pre-tokenizer by default and
// the GPT-4o (inkling2) pre-tokenizer when requested via tokenizer.ggml.pre.
class BpeTokenizer {
 public:
  explicit BpeTokenizer(const TokenizerMeta& meta);

  [[nodiscard]] const std::vector<uint8_t>& token_bytes(uint32_t id) const;
  [[nodiscard]] std::string decode(std::span<const uint32_t> ids) const;
  [[nodiscard]] std::string decode(const std::vector<uint32_t>& ids) const {
    return decode(std::span<const uint32_t>(ids.data(), ids.size()));
  }

  [[nodiscard]] std::optional<uint32_t> token_id(std::string_view token) const;
  [[nodiscard]] std::string_view pre() const noexcept { return pre_; }

  [[nodiscard]] bool is_eog(uint32_t id) const;
  [[nodiscard]] const std::vector<uint32_t>& eog_tokens() const noexcept {
    return eog_tokens_;
  }
  [[nodiscard]] uint32_t bos_token() const noexcept { return bos_token_; }
  [[nodiscard]] uint32_t eos_token() const noexcept { return eos_token_; }
  [[nodiscard]] bool add_bos() const noexcept { return add_bos_; }
  [[nodiscard]] bool add_eos() const noexcept { return add_eos_; }

  [[nodiscard]] std::vector<uint32_t> encode(std::string_view text) const;
  [[nodiscard]] std::vector<uint32_t> encode_full(std::string_view text) const;

 private:
  enum class Cls { Letter, Number, Space, Other };

  [[nodiscard]] static Cls classify(char32_t c);
  [[nodiscard]] std::pair<size_t, size_t> next_word(
      const std::vector<char32_t>& cpts, size_t pos) const;
  [[nodiscard]] std::pair<size_t, size_t> next_word_gpt4o(
      const std::vector<char32_t>& cpts, size_t pos) const;
  [[nodiscard]] std::vector<uint32_t> encode_word(std::string_view word) const;

  std::unordered_map<std::string, uint32_t> tokens_;
  std::unordered_map<uint8_t, uint32_t> byte_tokens_;
  std::unordered_map<std::string, uint32_t> merges_;
  std::vector<std::vector<uint8_t>> token_bytes_by_id_;
  std::vector<uint32_t> eog_tokens_;
  std::string pre_;
  uint32_t bos_token_ = NO_TOKEN;
  uint32_t eos_token_ = NO_TOKEN;
  bool gpt4o_pre_ = false;
  bool add_bos_ = false;
  bool add_eos_ = false;
};

// Build the tokenizer from GGUF metadata keys.
[[nodiscard]] Result<BpeTokenizer> build_tokenizer(const class Gguf& gguf);

}  // namespace tokshit
