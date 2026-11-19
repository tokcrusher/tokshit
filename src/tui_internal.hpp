#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace tokshit::tui::detail {

inline std::string_view trim(std::string_view s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' ||
                   s[b] == '\n')) {
    ++b;
  }
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' ||
                   s[e - 1] == '\n')) {
    --e;
  }
  return s.substr(b, e - b);
}

inline std::string to_lower(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c >= 'A' && c <= 'Z') {
      out.push_back(static_cast<char>(c - 'A' + 'a'));
    } else {
      out.push_back(c);
    }
  }
  return out;
}

// Ширина в колонках: кодовые точки UTF-8, не байты.
inline std::size_t columns(std::string_view s) {
  std::size_t n = 0;
  for (unsigned char c : s) {
    if ((c & 0xC0U) != 0x80U) ++n;
  }
  return n;
}

inline void pad_to(std::string& out, std::size_t col) {
  const std::size_t nl = out.rfind('\n');
  const std::size_t from = nl == std::string::npos ? 0 : nl + 1;
  while (columns(std::string_view(out).substr(from)) < col) {
    out.push_back(' ');
  }
}

inline std::vector<std::string> wrap(std::string_view text,
                                     std::size_t width) {
  std::vector<std::string> lines;
  std::string cur;
  std::size_t i = 0;
  while (i < text.size()) {
    std::size_t j = i;
    while (j < text.size() && text[j] != ' ') ++j;
    const std::string_view word = text.substr(i, j - i);
    if (!word.empty()) {
      const std::size_t extra =
          cur.empty() ? columns(word) : columns(word) + 1;
      if (!cur.empty() && columns(cur) + extra > width) {
        lines.push_back(cur);
        cur.assign(word);
      } else {
        if (!cur.empty()) cur.push_back(' ');
        cur.append(word);
      }
    }
    i = j;
    while (i < text.size() && text[i] == ' ') ++i;
  }
  if (!cur.empty()) lines.push_back(std::move(cur));
  if (lines.empty()) lines.emplace_back();
  return lines;
}

inline void append_wrapped(std::string& out, std::string_view text,
                           std::size_t first_col, std::size_t rest_col,
                           std::size_t width) {
  const std::vector<std::string> lines = wrap(text, width - rest_col);
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (i > 0) out.push_back('\n');
    pad_to(out, i == 0 ? first_col : rest_col);
    out.append(lines[i]);
  }
}

inline void append_line(std::string& out, std::string_view text) {
  out.append(text);
  out.push_back('\n');
}

}  // namespace tokshit::tui::detail
