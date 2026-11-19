#include "tokshit/tui.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

#include <sys/ioctl.h>
#include <unistd.h>

#include "tui_internal.hpp"

namespace tokshit::tui {

using detail::append_line;
using detail::append_wrapped;
using detail::columns;
using detail::pad_to;
using detail::to_lower;
using detail::trim;
using detail::wrap;

std::uint64_t mem_available_mb() noexcept {
  std::FILE* f = std::fopen("/proc/meminfo", "r");
  if (f == nullptr) return 0;
  char line[256];
  unsigned long long kb = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
  }
  std::fclose(f);
  return static_cast<std::uint64_t>(kb / 1024ULL);
}

double cache_hit_rate(std::uint64_t hit, std::uint64_t miss) noexcept {
  const std::uint64_t total = hit + miss;
  if (total == 0) return 0.0;
  return 100.0 * static_cast<double>(hit) / static_cast<double>(total);
}

RateColor rate_color(double percent) noexcept {
  if (percent >= 60.0) return RateColor::Green;
  if (percent >= 30.0) return RateColor::Yellow;
  return RateColor::Red;
}

std::string_view rate_color_code(RateColor color) noexcept {
  switch (color) {
    case RateColor::Green:
      return "\x1b[32m";
    case RateColor::Yellow:
      return "\x1b[33m";
    case RateColor::Red:
      return "\x1b[31m";
    case RateColor::None:
      break;
  }
  return "";
}

// hit-rate строкой: цветное значение при color, «—» без данных
std::string hit_rate_text(std::uint64_t hit, std::uint64_t miss,
                          bool color) {
  if (hit + miss == 0) return "—";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.1f%%",
                cache_hit_rate(hit, miss));
  std::string out;
  if (color) {
    out.append(rate_color_code(rate_color(cache_hit_rate(hit, miss))));
    out.append(buf);
    out.append("\x1b[0m");
  } else {
    out.append(buf);
  }
  return out;
}

std::string format_status(const Status& status, bool color) {
  char buf[512];
  const std::string rate = hit_rate_text(status.cache_hit, status.cache_miss, color);
  std::string out = "┌─ ПОСЛЕДНИЙ ОТВЕТ ───────────────────────────────\n";
  std::snprintf(buf, sizeof(buf), "│  %.1f tok/s  ·  %llu токенов  ·  контекст %zu",
                status.tok_per_s,
                static_cast<unsigned long long>(status.gen_tokens), status.ctx_pos);
  out.append(buf);
  out.push_back('\n');
  if (status.mem_avail_mb != 0) {
    std::snprintf(buf, sizeof(buf),
                  "│  кэш %s  ·  RSS %llu MB  ·  свободно %llu MB\n"
                  "│  бюджет кэша %llu MB  ·  spec %s",
                  rate.c_str(),
                  static_cast<unsigned long long>(status.rss_mb),
                  static_cast<unsigned long long>(status.mem_avail_mb),
                  static_cast<unsigned long long>(status.cache_budget_mb),
                  status.spec ? "вкл" : "выкл");
  } else {
    std::snprintf(buf, sizeof(buf),
                  "│  кэш %s  ·  RSS %llu MB  ·  свободно ?\n"
                  "│  бюджет кэша %llu MB  ·  spec %s",
                  rate.c_str(),
                  static_cast<unsigned long long>(status.rss_mb),
                  static_cast<unsigned long long>(status.cache_budget_mb),
                  status.spec ? "вкл" : "выкл");
  }
  out.append(buf);
  out.append("\n└────────────────────────────────────────────────");
  return out;
}

std::string format_stats(const Stats& stats, bool color) {
  const Status& s = stats.status;
  constexpr std::size_t kWidth = 78;
  char buf[512];
  std::string out;
  std::snprintf(buf, sizeof(buf),
                "╭─ СТАТИСТИКА СЕССИИ ────────────────────────────\n");
  out.append(buf);

  // скорость / speed
  out.append("│ скорость\n");
  std::snprintf(buf, sizeof(buf),
                "│   последняя: %.1f tok/s  ·  средняя: %.1f tok/s\n",
                s.tok_per_s, stats.avg_tok_per_s);
  out.append(buf);
  std::snprintf(buf, sizeof(buf),
                "│   prefill %.2fs  ·  сгенерировано %llu токенов всего\n",
                stats.last_prefill_s,
                static_cast<unsigned long long>(stats.total_tokens));
  out.append(buf);

  // кэш экспертов / expert cache
  out.append("│ кэш экспертов\n");
  std::snprintf(buf, sizeof(buf),
                "│   кэш %s (hit %llu / miss %llu)  ·  union %.2f/проход\n",
                hit_rate_text(s.cache_hit, s.cache_miss, color).c_str(),
                static_cast<unsigned long long>(s.cache_hit),
                static_cast<unsigned long long>(s.cache_miss),
                stats.union_per_pass);
  out.append(buf);

  // память / memory
  out.append("│ память\n");
  std::snprintf(buf, sizeof(buf),
                "│   RSS %llu MB  ·  MemAvailable %llu MB  ·  кэш %llu MB\n",
                static_cast<unsigned long long>(s.rss_mb),
                static_cast<unsigned long long>(s.mem_avail_mb),
                static_cast<unsigned long long>(s.cache_budget_mb));
  out.append(buf);

  // сессия / session
  out.append("│ сессия\n");
  std::snprintf(buf, sizeof(buf),
                "│   %llu ходов  ·  контекст %zu  ·  spec %s",
                static_cast<unsigned long long>(stats.turns), s.ctx_pos,
                s.spec ? "вкл" : "выкл (в чате не используется)");
  out.append(buf);
  out.push_back('\n');

  // сэмплинг / sampling
  out.append("│ сэмплинг\n");
  std::snprintf(buf, sizeof(buf),
                "│   temp %.2f  ·  top-k %zu  ·  top-p %.2f  ·  reasoning-budget: %d",
                stats.temperature, stats.top_k, stats.top_p,
                stats.reasoning_budget);
  out.append(buf);
  out.push_back('\n');
  const std::string label = "│   системная инструкция: ";
  out.append(label);
  const std::string body =
      stats.has_system ? stats.system_prompt : std::string("нет");
  const std::vector<std::string> lines = wrap(body, kWidth - columns(label));
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (i > 0) {
      out.push_back('\n');
      pad_to(out, columns(label));
    }
    out.append(lines[i]);
  }
  out.push_back('\n');
  out.append("╰────────────────────────────────────────────────");
  return out;
}

namespace {

// кадры спиннера живого индикатора декода
constexpr std::string_view kFrames[] = {"◜", "◠", "◝", "◞", "◡", "◟"};

}  // namespace


std::size_t spinner_frames() noexcept {
  return sizeof(kFrames) / sizeof(kFrames[0]);
}

std::string_view spinner_frame(std::size_t index) noexcept {
  return kFrames[index % spinner_frames()];
}

// добавляет « · текст» только если строка остаётся в пределах width
void append_fit(std::string& out, std::string_view piece, std::size_t width) {
  if (width == 0) {
    out.append(piece);
    return;
  }
  const std::size_t nl = out.rfind('\n');
  const std::size_t from = nl == std::string::npos ? 0 : nl + 1;
  std::string candidate = out.substr(from);
  candidate.append(piece);
  if (columns(candidate) <= width) out.append(piece);
}

std::string format_decode(const DecodeInfo& info, std::size_t width) {
  const double rate =
      info.seconds > 1e-9
          ? static_cast<double>(info.tokens) / info.seconds
          : 0.0;
  char buf[192];
  // скорость печатаем только когда её можно измерить: на первом токене
  // elapsed близок к нулю и «2030 tok/s» только сбивает
  if (info.seconds > 0.3) {
    std::snprintf(buf, sizeof(buf), "%s генерирую · %llu ток · %.1f tok/s",
                  std::string(spinner_frame(info.frame)).c_str(),
                  static_cast<unsigned long long>(info.tokens), rate);
  } else {
    std::snprintf(buf, sizeof(buf), "%s генерирую · %llu ток",
                  std::string(spinner_frame(info.frame)).c_str(),
                  static_cast<unsigned long long>(info.tokens));
  }
  std::string out = buf;
  if (info.cache_hit + info.cache_miss > 0) {
    std::snprintf(buf, sizeof(buf), " · кэш %.0f%%",
                  cache_hit_rate(info.cache_hit, info.cache_miss));
    append_fit(out, buf, width);
  }
  if (info.mem_avail_mb != 0) {
    std::snprintf(buf, sizeof(buf), " · свб %llu MB",
                  static_cast<unsigned long long>(info.mem_avail_mb));
    append_fit(out, buf, width);
  }
  return out;
}

std::string format_prefill(const PrefillInfo& info, std::size_t width) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "предобработка %zu/%zu ток", info.done,
                info.total);
  std::string out = buf;
  if (info.total != 0) {
    const std::size_t pct = info.done >= info.total
                                ? 100
                                : 100 * info.done / info.total;
    std::snprintf(buf, sizeof(buf), " · %zu%%", pct);
    append_fit(out, buf, width);
  }
  if (info.seconds > 1e-9) {
    std::snprintf(buf, sizeof(buf), " · %.1fs", info.seconds);
    append_fit(out, buf, width);
  }
  return out;
}

std::size_t terminal_width(int fd) noexcept {
  if (::isatty(fd) != 1) return 0;
  winsize ws{};
  if (::ioctl(fd, TIOCGWINSZ, &ws) != 0) return 0;
  return static_cast<std::size_t>(ws.ws_col);
}
}  // namespace tokshit::tui
