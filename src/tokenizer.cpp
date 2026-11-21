#include "tokshit/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokshit/gguf.hpp"

namespace tokshit {
namespace {

struct ByteMap {
  std::array<char32_t, 256> to_char{};
  std::unordered_map<char32_t, uint8_t> to_byte;

  ByteMap() {
    uint32_t extra = 0;
    for (uint32_t b = 0; b < 256; ++b) {
      char32_t c = 0;
      if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
          (b >= 0xAE && b <= 0xFF)) {
        c = static_cast<char32_t>(b);
      } else {
        c = static_cast<char32_t>(0x100 + extra);
        ++extra;
      }
      to_char[b] = c;
      to_byte[c] = static_cast<uint8_t>(b);
    }
  }

  [[nodiscard]] char32_t byte_to_char(uint8_t b) const { return to_char[b]; }
  [[nodiscard]] uint8_t char_to_byte(char32_t c) const {
    const auto it = to_byte.find(c);
    return it == to_byte.end() ? 0 : it->second;
  }
};

const ByteMap& byte_map() {
  static const ByteMap map;
  return map;
}

void append_utf8(char32_t c, std::string& out) {
  if (c <= 0x7F) {
    out.push_back(static_cast<char>(c));
  } else if (c <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (c >> 6)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else if (c <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (c >> 12)));
    out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (c >> 18)));
    out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  }
}

std::vector<char32_t> decode_utf8(std::string_view text) {
  std::vector<char32_t> out;
  out.reserve(text.size());
  size_t i = 0;
  while (i < text.size()) {
    const auto b0 = static_cast<uint8_t>(text[i]);
    if (b0 < 0x80) {
      out.push_back(static_cast<char32_t>(b0));
      ++i;
      continue;
    }
    size_t len = 0;
    char32_t cp = 0;
    if ((b0 & 0xE0) == 0xC0) {
      len = 2;
      cp = b0 & 0x1F;
    } else if ((b0 & 0xF0) == 0xE0) {
      len = 3;
      cp = b0 & 0x0F;
    } else if ((b0 & 0xF8) == 0xF0) {
      len = 4;
      cp = b0 & 0x07;
    } else {
      out.push_back(0xFFFD);
      ++i;
      continue;
    }
    if (i + len > text.size()) {
      out.push_back(0xFFFD);
      ++i;
      continue;
    }
    bool ok = true;
    for (size_t k = 1; k < len; ++k) {
      const auto bk = static_cast<uint8_t>(text[i + k]);
      if ((bk & 0xC0) != 0x80) {
        ok = false;
        break;
      }
      cp = (cp << 6) | (bk & 0x3F);
    }
    if (!ok) {
      out.push_back(0xFFFD);
      ++i;
      continue;
    }
    out.push_back(cp);
    i += len;
  }
  return out;
}

bool is_unicode_space(char32_t c) {
  if (c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0B || c == 0x0C ||
      c == 0x0D || c == 0x85 || c == 0xA0 || c == 0x1680 ||
      (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
      c == 0x202F || c == 0x205F || c == 0x3000) {
    return true;
  }
  return false;
}

bool is_unicode_alpha(char32_t c) {
  if (c < 0x80) {
    return std::isalpha(static_cast<int>(c)) != 0;
  }
  if (is_unicode_space(c)) return false;
  if (c >= 0x2000 && c <= 0x206F) return false;
  if (c >= 0x3000 && c <= 0x303F) return false;
  if (c >= 0x1F000) return false;
  return true;
}

bool is_unicode_number(char32_t c) {
  if (c < 0x80) return std::isdigit(static_cast<int>(c)) != 0;
  return c >= 0xFF10 && c <= 0xFF19;
}

size_t unicode_len_utf8(uint8_t b) {
  if (b <= 0x7F) return 1;
  if (b >= 0xC0 && b <= 0xDF) return 2;
  if (b >= 0xE0 && b <= 0xEF) return 3;
  if (b >= 0xF0 && b <= 0xF7) return 4;
  return 1;
}

char32_t to_ascii_lower(char32_t c) {
  if (c >= U'A' && c <= U'Z') return c + 32;
  return c;
}

std::string utf8_lossy(std::span<const uint8_t> bytes) {
  std::string out;
  out.reserve(bytes.size());
  size_t i = 0;
  while (i < bytes.size()) {
    const uint8_t b0 = bytes[i];
    if (b0 < 0x80) {
      out.push_back(static_cast<char>(b0));
      ++i;
      continue;
    }
    size_t len = 0;
    if ((b0 & 0xE0) == 0xC0) {
      len = 2;
    } else if ((b0 & 0xF0) == 0xE0) {
      len = 3;
    } else if ((b0 & 0xF8) == 0xF0) {
      len = 4;
    } else {
      out += "\xEF\xBF\xBD";
      ++i;
      continue;
    }
    if (i + len > bytes.size()) {
      out += "\xEF\xBF\xBD";
      break;
    }
    bool ok = true;
    for (size_t k = 1; k < len; ++k) {
      if ((bytes[i + k] & 0xC0) != 0x80) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      out += "\xEF\xBF\xBD";
      ++i;
      continue;
    }
    out.append(reinterpret_cast<const char*>(bytes.data() + i), len);
    i += len;
  }
  return out;
}

}  // namespace

// строим словарь токенов и merge-таблицу. тут можно чуть-чуть аллоцировать,
// это холодный путь
BpeTokenizer::BpeTokenizer(const TokenizerMeta& meta) {
  token_bytes_by_id_.resize(meta.tokens.size());
  tokens_.reserve(meta.tokens.size() * 2);
  for (size_t id = 0; id < meta.tokens.size(); ++id) {
    const auto cpts = decode_utf8(meta.tokens[id]);
    std::vector<uint8_t> bytes;
    bytes.reserve(cpts.size());
    for (char32_t c : cpts) bytes.push_back(byte_map().char_to_byte(c));
    tokens_.emplace(std::string(reinterpret_cast<const char*>(bytes.data()),
                                bytes.size()),
                    static_cast<uint32_t>(id));
    token_bytes_by_id_[id] = std::move(bytes);
  }

  for (uint32_t b = 0; b < 256; ++b) {
    const char key = static_cast<char>(b);
    const auto it = tokens_.find(std::string(1, key));
    if (it != tokens_.end()) byte_tokens_[static_cast<uint8_t>(b)] = it->second;
  }

  merges_.reserve(meta.merges.size() * 2);
  for (size_t rank = 0; rank < meta.merges.size(); ++rank) {
    const std::string& merge = meta.merges[rank];
    const size_t space = merge.find(' ');
    const std::string_view first =
        space == std::string::npos ? std::string_view(merge)
                                   : std::string_view(merge).substr(0, space);
    const std::string_view second =
        space == std::string::npos ? std::string_view()
                                   : std::string_view(merge).substr(space + 1);
    std::string left;
    std::string right;
    for (char32_t c : decode_utf8(first)) {
      left.push_back(static_cast<char>(byte_map().char_to_byte(c)));
    }
    for (char32_t c : decode_utf8(second)) {
      right.push_back(static_cast<char>(byte_map().char_to_byte(c)));
    }
    std::string key;
    key.reserve(left.size() + 1 + right.size());
    key.append(left);
    key.push_back('\0');
    key.append(right);
    merges_.emplace(std::move(key), static_cast<uint32_t>(rank));
  }

  eog_tokens_ = meta.eog_tokens;
  pre_ = meta.pre;
  gpt4o_pre_ = pre_ == "inkling2" || pre_ == "gpt4o";
  bos_token_ = meta.bos_token;
  eos_token_ = meta.eos_token;
  add_bos_ = meta.add_bos;
  add_eos_ = meta.add_eos;

  if (std::getenv("TOKSHIT_DEBUG_TOK") != nullptr) {
    for (uint32_t id : {56892U, 24653U, 1U, 100U}) {
      const auto& tb = token_bytes(id);
      std::string decoded = decode(std::vector<uint32_t>{id});
      std::fprintf(stderr, "[TOK] id=%u token=%s bytes=", id,
                   id < meta.tokens.size() ? meta.tokens[id].c_str() : "?");
      for (uint8_t b : tb) std::fprintf(stderr, "%02x", b);
      std::fprintf(stderr, " decoded=%s\n", decoded.c_str());
    }
  }
}

const std::vector<uint8_t>& BpeTokenizer::token_bytes(uint32_t id) const {
  static const std::vector<uint8_t> empty;
  if (id >= token_bytes_by_id_.size()) return empty;
  return token_bytes_by_id_[id];
}

std::string BpeTokenizer::decode(std::span<const uint32_t> ids) const {
  std::vector<uint8_t> bytes;
  for (uint32_t id : ids) {
    const auto& tb = token_bytes(id);
    bytes.insert(bytes.end(), tb.begin(), tb.end());
  }
  return utf8_lossy(bytes);
}

bool BpeTokenizer::is_eog(uint32_t id) const {
  for (uint32_t e : eog_tokens_) {
    if (e == id) return true;
  }
  return false;
}

std::optional<uint32_t> BpeTokenizer::token_id(std::string_view token) const {
  const auto it = tokens_.find(std::string(token));
  if (it == tokens_.end()) return std::nullopt;
  return it->second;
}

std::vector<uint32_t> BpeTokenizer::encode_full(std::string_view text) const {
  std::vector<uint32_t> ids = encode(text);
  if (add_bos_) ids.insert(ids.begin(), bos_token_);
  if (add_eos_) ids.push_back(eos_token_);
  return ids;
}

BpeTokenizer::Cls BpeTokenizer::classify(char32_t c) {
  if (is_unicode_space(c)) return Cls::Space;
  if (is_unicode_alpha(c)) return Cls::Letter;
  if (is_unicode_number(c)) return Cls::Number;
  return Cls::Other;
}

std::pair<size_t, size_t> BpeTokenizer::next_word(
    const std::vector<char32_t>& cpts, size_t pos) const {
  if (gpt4o_pre_) return next_word_gpt4o(cpts, pos);

  const size_t n = cpts.size();
  const char32_t cp = cpts[pos];
  const Cls cls = classify(cp);
  const bool is_newline = cp == U'\r' || cp == U'\n';

  if (cp == U'\'' && pos + 1 < n) {
    const char32_t n1 = to_ascii_lower(cpts[pos + 1]);
    if (n1 == U's' || n1 == U't' || n1 == U'm' || n1 == U'd') {
      return {pos, pos + 2};
    }
    if (pos + 2 < n) {
      const char32_t n2 = to_ascii_lower(cpts[pos + 2]);
      if ((n1 == U'r' && n2 == U'e') || (n1 == U'v' && n2 == U'e') ||
          (n1 == U'l' && n2 == U'l')) {
        return {pos, pos + 3};
      }
    }
  }

  auto is_letter_at = [&](size_t p) {
    return p < n && classify(cpts[p]) == Cls::Letter;
  };
  if (cls == Cls::Letter) {
    size_t end = pos + 1;
    while (end < n && is_letter_at(end)) ++end;
    return {pos, end};
  }
  if (cls != Cls::Number && !is_newline && pos + 1 < n &&
      is_letter_at(pos + 1)) {
    size_t end = pos + 1;
    while (end < n && is_letter_at(end)) ++end;
    return {pos, end};
  }

  if (cls == Cls::Number) {
    size_t end = pos + 1;
    while (end < n && end - pos < 3 && classify(cpts[end]) == Cls::Number) {
      ++end;
    }
    return {pos, end};
  }

  if (cls == Cls::Other) {
    size_t end = pos;
    bool has_space = false;
    if (cp == U' ' && pos + 1 < n && classify(cpts[pos + 1]) == Cls::Other) {
      has_space = true;
      end = pos + 1;
    }
    if (!has_space && end == pos) end = pos + 1;
    while (end < n && classify(cpts[end]) == Cls::Other &&
           !is_unicode_space(cpts[end])) {
      ++end;
    }
    while (end < n && (cpts[end] == U'\r' || cpts[end] == U'\n')) ++end;
    return {pos, end};
  }

  size_t end = pos + 1;
  while (end < n && classify(cpts[end]) == Cls::Space) ++end;
  const size_t run_len = end - pos;
  if (end == n) return {pos, end};
  if (run_len >= 2) return {pos, end - 1};
  return {pos, end};
}

// ASCII subset of the GPT-4o (inkling2) pre-tokenizer regex:
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//     (?i:'s|'t|'re|'ve|'m|'ll|'d)?
// | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
//     (?i:'s|'t|'re|'ve|'m|'ll|'d)?
// | \p{N}{1,3}
// | ?[^\s\p{L}\p{N}]+[\r\n/]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
// Multibyte letters are treated as lower-case-ish so non-ASCII words are not
// torn apart.
std::pair<size_t, size_t> BpeTokenizer::next_word_gpt4o(
    const std::vector<char32_t>& cpts, size_t pos) const {
  const size_t n = cpts.size();
  const char32_t cp = cpts[pos];

  auto is_letter = [](char32_t c) {
    return BpeTokenizer::classify(c) == Cls::Letter;
  };
  auto is_number = [](char32_t c) {
    return BpeTokenizer::classify(c) == Cls::Number;
  };
  auto is_newline = [](char32_t c) { return c == U'\r' || c == U'\n'; };
  auto is_upper = [](char32_t c) { return c >= U'A' && c <= U'Z'; };
  auto is_lower = [](char32_t c) {
    return (c >= U'a' && c <= U'z') ||
           (c >= 0x80 && BpeTokenizer::classify(c) == Cls::Letter);
  };

  auto with_contraction = [&](size_t end) {
    if (end >= n || cpts[end] != U'\'') return end;
    if (end + 1 < n) {
      const char32_t c1 = to_ascii_lower(cpts[end + 1]);
      if (c1 == U's' || c1 == U't' || c1 == U'm' || c1 == U'd') {
        return end + 2;
      }
      if (end + 2 < n) {
        const char32_t c2 = to_ascii_lower(cpts[end + 2]);
        if ((c1 == U'r' && c2 == U'e') || (c1 == U'v' && c2 == U'e') ||
            (c1 == U'l' && c2 == U'l')) {
          return end + 3;
        }
      }
    }
    return end;
  };

  // Optional prefix is anything but CR/LF, a letter or a digit. That keeps
  // " Hello", "(foo", "'s" in one pre-token, matching the regex alternatives.
  const bool prefixed = !is_letter(cp) && !is_number(cp) && !is_newline(cp) &&
                        pos + 1 < n && is_letter(cpts[pos + 1]);
  if (is_letter(cp) || prefixed) {
    const size_t start = is_letter(cp) ? pos : pos + 1;

    // upper* lower+ -- lowercase and TitleCase words.
    size_t end = start;
    while (end < n && is_upper(cpts[end])) ++end;
    if (end < n && is_lower(cpts[end])) {
      do {
        ++end;
      } while (end < n && is_lower(cpts[end]));
      return {pos, with_contraction(end)};
    }

    // upper+ lower* -- ALLCAPS and lone capitals.
    end = start;
    if (is_upper(cpts[end])) {
      do {
        ++end;
      } while (end < n && is_upper(cpts[end]));
      while (end < n && is_lower(cpts[end])) ++end;
      return {pos, with_contraction(end)};
    }
  }

  if (is_number(cp)) {
    size_t end = pos + 1;
    while (end < n && end - pos < 3 && is_number(cpts[end])) ++end;
    return {pos, end};
  }

  auto is_punct = [](char32_t c) {
    return !is_unicode_space(c) && BpeTokenizer::classify(c) == Cls::Other;
  };
  if (is_punct(cp) ||
      (cp == U' ' && pos + 1 < n && is_punct(cpts[pos + 1]))) {
    size_t end = cp == U' ' ? pos + 1 : pos;
    while (end < n && is_punct(cpts[end])) ++end;
    while (end < n && (is_newline(cpts[end]) || cpts[end] == U'/')) ++end;
    return {pos, end};
  }

  size_t run_end = pos;
  while (run_end < n && is_unicode_space(cpts[run_end])) ++run_end;

  // \s*[\r\n]+ -- whitespace up to and including the last newline in the run.
  size_t last_newline = pos;
  for (size_t i = run_end; i > pos; --i) {
    if (is_newline(cpts[i - 1])) {
      last_newline = i;
      break;
    }
  }
  if (last_newline > pos) return {pos, last_newline};

  // \s+(?!\S) leaves one space for the next word; \s+ eats trailing runs.
  if (run_end == n) return {pos, run_end};
  if (run_end - pos >= 2) return {pos, run_end - 1};
  return {pos, run_end};
}

std::vector<uint32_t> BpeTokenizer::encode_word(std::string_view word) const {
  const auto* bytes = reinterpret_cast<const uint8_t*>(word.data());
  const size_t len = word.size();
  std::vector<std::string> symbols;
  size_t offset = 0;
  while (offset < len) {
    const size_t char_len = unicode_len_utf8(bytes[offset]);
    const size_t end = std::min(offset + char_len, len);
    symbols.emplace_back(reinterpret_cast<const char*>(bytes + offset),
                         end - offset);
    offset = end;
  }
  if (symbols.empty()) return {};

  for (;;) {
    bool have_best = false;
    uint32_t best_rank = 0;
    size_t best_i = 0;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      if (symbols[i].empty() || symbols[i + 1].empty()) continue;
      std::string key;
      key.reserve(symbols[i].size() + 1 + symbols[i + 1].size());
      key.append(symbols[i]);
      key.push_back('\0');
      key.append(symbols[i + 1]);
      const auto it = merges_.find(key);
      if (it != merges_.end()) {
        if (!have_best || it->second < best_rank) {
          have_best = true;
          best_rank = it->second;
          best_i = i;
        }
      }
    }
    if (!have_best) break;
    symbols[best_i].append(symbols[best_i + 1]);
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
  }

  std::vector<uint32_t> ids;
  for (const std::string& sym : symbols) {
    if (sym.empty()) continue;
    const auto it = tokens_.find(sym);
    if (it != tokens_.end()) {
      ids.push_back(it->second);
      continue;
    }
    for (char ch : sym) {
      const auto bit = byte_tokens_.find(static_cast<uint8_t>(ch));
      if (bit != byte_tokens_.end()) ids.push_back(bit->second);
    }
  }
  return ids;
}

// энкодим текст в токены: сначала по словам/границам, потом bpe внутри слова
std::vector<uint32_t> BpeTokenizer::encode(std::string_view text) const {
  const std::vector<char32_t> cpts = decode_utf8(text);
  std::vector<uint32_t> out;
  const size_t n = cpts.size();
  size_t start = 0;
  size_t pos = 0;
  while (pos < n) {
    const auto [a, b] = next_word(cpts, pos);
    std::string word;
    for (size_t i = a; i < b; ++i) append_utf8(cpts[i], word);
    const auto ids = encode_word(word);
    out.insert(out.end(), ids.begin(), ids.end());
    start = b;
    pos = b;
  }
  if (start < n) {
    std::string word;
    for (size_t i = start; i < n; ++i) append_utf8(cpts[i], word);
    const auto ids = encode_word(word);
    out.insert(out.end(), ids.begin(), ids.end());
  }
  return out;
}

// тащим токенизатор из метадаты gguf
Result<BpeTokenizer> build_tokenizer(const Gguf& gguf) {
  TOKSHIT_TRY(tokens, gguf.strings("tokenizer.ggml.tokens"));
  TOKSHIT_TRY(merges, gguf.strings("tokenizer.ggml.merges"));
  TOKSHIT_TRY(token_types, gguf.u32s("tokenizer.ggml.token_type"));

  uint32_t bos = NO_TOKEN;
  if (auto r = gguf.u64_kv("tokenizer.ggml.bos_token_id"); r.ok()) {
    bos = static_cast<uint32_t>(r.value());
  }
  uint32_t eos = NO_TOKEN;
  if (auto r = gguf.u64_kv("tokenizer.ggml.eos_token_id"); r.ok()) {
    eos = static_cast<uint32_t>(r.value());
  }

  std::vector<uint32_t> eog;
  if (auto r = gguf.get("tokenizer.ggml.eog_token_ids"); r.ok()) {
    if (const auto* arr = r.value()->as_array(); arr != nullptr) {
      for (const Value& v : *arr) {
        if (const auto id = v.as_u64(); id.has_value()) {
          eog.push_back(static_cast<uint32_t>(*id));
        }
      }
    }
  }
  if (eog.empty()) eog.push_back(eos);

  std::string pre;
  if (auto r = gguf.get("tokenizer.ggml.pre"); r.ok()) {
    if (const auto s = r.value()->as_str(); s.has_value()) {
      pre = std::string(*s);
    }
  }
  bool add_bos = false;
  if (auto r = gguf.get("tokenizer.ggml.add_bos_token"); r.ok()) {
    if (const auto b = r.value()->as_bool(); b.has_value()) add_bos = *b;
  }
  bool add_eos = false;
  if (auto r = gguf.get("tokenizer.ggml.add_eos_token"); r.ok()) {
    if (const auto b = r.value()->as_bool(); b.has_value()) add_eos = *b;
  }

  TokenizerMeta meta;
  meta.tokens = std::move(tokens);
  meta.merges = std::move(merges);
  meta.token_types = std::move(token_types);
  meta.eog_tokens = std::move(eog);
  meta.pre = std::move(pre);
  meta.bos_token = bos;
  meta.eos_token = eos;
  meta.add_bos = add_bos;
  meta.add_eos = add_eos;
  return BpeTokenizer(meta);
}

}  // namespace tokshit
