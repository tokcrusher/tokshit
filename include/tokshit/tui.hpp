#pragma once

// Вспомогательные вещи для cli: справка, строка статуса и разбор команд
// интерактивного режима. Здесь нет ни модели, ни ANSI — только чистые строки,
// поэтому всё это тестируется без загрузки gguf.
// cli helpers: help text, status line and interactive command parsing. no model
// and no ANSI in here, only plain strings, so all of it is testable without
// loading a gguf.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace tokshit::tui {

// группы, по которым --help раскладывает флаги
// groups used to lay the flags out in --help
enum class FlagGroup {
  Run,
  Memory,
  Cache,
  Spec,
  Sampling,
  Debug,
  Preset,
  Count,
};

struct FlagDoc {
  const char* flag;
  const char* env;
  const char* arg;  // подсказка про значение, "" у булевых
  FlagGroup group;
  const char* desc;
  const char* def;
};

// единственный источник правды и для --help, и для разбора флагов
// single source of truth for both --help and flag parsing
[[nodiscard]] std::span<const FlagDoc> flag_docs() noexcept;

[[nodiscard]] std::string_view group_title(FlagGroup group) noexcept;

// полный текст --help. default_model подставляется в описание пути к модели
// full --help text. default_model is used in the model path description
[[nodiscard]] std::string help_text(std::string_view default_model);

// MemAvailable из /proc/meminfo, МБ; 0 если прочитать не удалось
// MemAvailable from /proc/meminfo in MB; 0 when unavailable
[[nodiscard]] std::uint64_t mem_available_mb() noexcept;

struct Status {
  double tok_per_s = 0.0;
  std::uint64_t gen_tokens = 0;
  std::uint64_t cache_hit = 0;
  std::uint64_t cache_miss = 0;
  std::uint64_t rss_mb = 0;
  std::uint64_t mem_avail_mb = 0;
  std::size_t ctx_pos = 0;
  std::uint64_t cache_budget_mb = 0;
  bool spec = false;
};

// hit / (hit + miss) в процентах; 0 когда обращений ещё не было
// hit / (hit + miss) as a percentage; 0 before the first access
[[nodiscard]] double cache_hit_rate(std::uint64_t hit,
                                    std::uint64_t miss) noexcept;

// две короткие строки, влезают на телефон шириной ~80 колонок.
// color=true добавляет цвет hit-rate (только он; остальное остаётся dim),
// color=false — чистый текст без ANSI
// two short lines that fit an ~80 column phone screen. color=true adds the
// hit-rate color only (the rest stays plain); color=false is plain text
[[nodiscard]] std::string format_status(const Status& status,
                                        bool color = false);

// ── живой индикатор декода и прогресс префилла ────────────────────────────
// все форматтеры ниже возвращают ОДНУ строку без \r и без ANSI: вызывающий
// сам рисует \r, тусклый цвет и очистку хвоста, чтобы TUI оставался
// тестируемым
// live decode indicator and prefill progress. every formatter below returns
// ONE line without \r and without ANSI: the caller draws \r, the dim color
// and the tail erase itself, keeping the tui testable

// кадры спиннера; индекс циклический, строка всегда непустая
// spinner frames; cyclic index, always a non-empty line
[[nodiscard]] std::string_view spinner_frame(std::size_t index) noexcept;

// сколько кадров у спиннера
[[nodiscard]] std::size_t spinner_frames() noexcept;

struct DecodeInfo {
  std::size_t frame = 0;          // индекс кадра спиннера
  std::uint64_t tokens = 0;       // сгенерировано к этому моменту
  double seconds = 0.0;           // от начала генерации
  std::uint64_t cache_hit = 0;    // дельта попаданий кэша за этот ответ
  std::uint64_t cache_miss = 0;   // дельта промахов за этот ответ
  std::uint64_t mem_avail_mb = 0; // 0 = поле не показывать
};

// «◜ генерирую · 12 ток · 2.3 tok/s · кэш 64% · свб 3120 MB». скорость
// появляется только после ~0.3s (раньше она бессмысленна), кэш — только
// когда есть обращения (hit+miss>0), свб — когда известно.
// width>0 урезает хвост, чтобы строка не ломалась на узком терминале;
// width==0 означает «без ограничения»
// "◜ generating · 12 tok · 2.3 tok/s · cache 64% · free 3120 MB". the rate
// appears only after ~0.3s (before that it is meaningless), the cache
// field only with observed accesses (hit+miss>0), free memory only when
// known. width>0 trims the tail so the line survives a narrow terminal;
// width==0 means "no limit"
[[nodiscard]] std::string format_decode(const DecodeInfo& info,
                                        std::size_t width = 0);

struct PrefillInfo {
  std::size_t done = 0;
  std::size_t total = 0;
  double seconds = 0.0;
};

// «предобработка 12/48 ток · 25% · 1.2s»: сначала счётчик, затем проценты и
// секунды, но только то, что влезает в width (width==0 — всё)
// "prefill 12/48 tok · 25% · 1.2s": counter first, then percent and seconds,
// but only what fits into width (width==0 keeps everything)
[[nodiscard]] std::string format_prefill(const PrefillInfo& info,
                                         std::size_t width = 0);

// цветовой класс hit-rate: зелёный ≥60%, жёлтый 30–60%, красный <30%.
// отсутствие данных различает вызывающий (hit+miss==0 → поле не печатается)
// hit-rate color class: green ≥60%, yellow 30–60%, red <30%. callers
// distinguish "no data" (hit+miss==0 → the field is not printed)
enum class RateColor { None, Red, Yellow, Green };

[[nodiscard]] RateColor rate_color(double percent) noexcept;
[[nodiscard]] std::string_view rate_color_code(RateColor color) noexcept;

// ширина терминала в колонках для fd; 0 если не tty или ioctl не сработал.
// используется, чтобы живая строка не ломалась на узком экране
// terminal width in columns for fd; 0 for non-tty or failed ioctl. used to
// keep the live line intact on narrow screens
[[nodiscard]] std::size_t terminal_width(int fd) noexcept;

struct Stats {
  Status status;
  std::uint64_t turns = 0;
  std::uint64_t total_tokens = 0;
  double avg_tok_per_s = 0.0;
  double last_prefill_s = 0.0;
  double union_per_pass = 0.0;
  float temperature = 0.8F;
  std::size_t top_k = 0;
  float top_p = 0.9F;
  int reasoning_budget = -1;
  bool has_system = false;
  std::string system_prompt;
};

[[nodiscard]] std::string format_stats(const Stats& stats, bool color = false);

enum class CommandKind {
  None,
  Chat,
  Help,
  Quit,
  Clear,
  Stats,
  Reset,
  Temp,
  TopK,
  TopP,
  ReasoningBudget,
  System,
  Save,
  Unknown,
};

struct Command {
  CommandKind kind = CommandKind::None;
  // текст для /system и /save, сырой аргумент для остальных
  // text for /system and /save, raw argument otherwise
  std::string arg;
  float number = 0.0F;     // /temp, /topp
  std::size_t count = 0;   // /topk
  int budget = -1;         // /reasoning-budget, signed
  bool valid = true;       // числовой аргумент разобрался
};

// пустая строка -> None, "/cmd ..." -> команда, всё остальное -> Chat
// empty line -> None, "/cmd ..." -> command, anything else -> Chat
[[nodiscard]] Command parse_command(std::string_view line);

// краткая справка по командам чата
// short command reference for the interactive mode
[[nodiscard]] std::string command_help();

}  // namespace tokshit::tui
