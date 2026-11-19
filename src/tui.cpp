#include "tokshit/tui.hpp"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sys/ioctl.h>
#include <unistd.h>

namespace tokshit::tui {
namespace {

constexpr FlagDoc kFlags[] = {
    // запуск и модель / run and model
    {"model", "TOKSHIT_MODEL", "FILE", FlagGroup::Run,
     "путь к .gguf-модели; иначе TOKSHIT_MODEL или вшитый", "встроенный"},
    {"prompt-file", "TOKSHIT_PROMPT_FILE", "FILE", FlagGroup::Run,
     "читать промпт из файла, а не из argv", "нет"},
    {"layers", "GGUF_NLAYERS", "N", FlagGroup::Run,
     "ограничить число слоёв (только отладка)", "все"},

    // производительность и память / performance and memory
    {"hot", "TOKSHIT_HOT", "", FlagGroup::Memory,
     "копировать плотные веса в RAM; 0 = mmap", "1"},
    {"hot-conv", "TOKSHIT_HOT_CONV", "", FlagGroup::Memory,
     "переквантовывать hot-веса в Q8_0 (нужен neon dot); 0 = как в gguf",
     "вкл"},
    {"drop-cache", "TOKSHIT_DROP_CACHE", "", FlagGroup::Memory,
     "сбросить page cache после копии hot-весов", "1"},
    {"mlock", "TOKSHIT_MLOCK", "", FlagGroup::Memory,
     "зарезервировано: best-effort mlock весов (не подключён)", "выкл"},
    {"pool-workers", "TOKSHIT_POOL_WORKERS", "N", FlagGroup::Memory,
     "потоков вычислительного пула", "ядра-1"},
    {"pool-pin", "TOKSHIT_POOL_PIN", "", FlagGroup::Memory,
     "привязать потоки пула к ядрам", "выкл"},
    {"pool-rt", "TOKSHIT_POOL_RT", "", FlagGroup::Memory,
     "потоки пула в realtime-приоритет", "выкл"},
    {"main-cpu", "TOKSHIT_MAIN_CPU", "N", FlagGroup::Memory,
     "ядро главного потока при --pool-pin", "7"},

    // экспертный кэш и ввод-вывод / expert cache and io
    {"cache-mb", "TOKSHIT_EXPERT_CACHE_MB", "MB", FlagGroup::Cache,
     "бюджет кэша экспертов, МБ; 0 = не кэшировать", "1024"},
    {"expert-cache-mb", "TOKSHIT_EXPERT_CACHE_MB", "MB", FlagGroup::Cache,
     "то же, что --cache-mb", "как --cache-mb"},
    {"o-direct", "TOKSHIT_O_DIRECT", "", FlagGroup::Cache,
     "O_DIRECT при чтении экспертов; 0 = выкл", "вкл"},
    {"io-threads", "TOKSHIT_IO_THREADS", "N", FlagGroup::Cache,
     "потоков чтения экспертов; 0/1 = один", "4"},
    {"prefetch", "TOKSHIT_PREFETCH", "", FlagGroup::Cache,
     "предзагрузка экспертов на слой вперёд", "выкл"},
    {"prefetch-idle", "TOKSHIT_PREFETCH_IDLE", "", FlagGroup::Cache,
     "спек-заказы только на простое диска: gate/up, ≤4 эксперта, слой+1",
     "выкл"},
    {"io-merge", "TOKSHIT_IO_MERGE", "", FlagGroup::Cache,
     "сливать соседних экспертов одного proj в один pread", "выкл"},
    {"freq-persist", "TOKSHIT_FREQ_PERSIST", "", FlagGroup::Cache,
     "писать профиль частот роутинга после генерации", "выкл"},
    {"freq-file", "TOKSHIT_FREQ_FILE", "FILE", FlagGroup::Cache,
     "путь профиля частот (загрузка при старте и запись)", "$TMPDIR"},
    {"pin-experts", "TOKSHIT_PIN_EXPERTS", "K", FlagGroup::Cache,
     "закрепить K горячих экспертов после промпта", "0"},
    {"mmap-willneed", "TOKSHIT_MMAP_WILLNEED", "", FlagGroup::Cache,
     "madvise(WILLNEED) на mmap модели", "выкл"},
    {"fadv-dontneed", "TOKSHIT_FADV_DONTNEED", "", FlagGroup::Cache,
     "fadvise(DONTNEED) после чтения эксперта", "выкл"},
    {"fadv-random", "TOKSHIT_FADV_RANDOM", "", FlagGroup::Cache,
     "fadvise(RANDOM) на файл модели", "выкл"},

    // спекуляция и черновик / speculation and draft
    {"spec", "TOKSHIT_SPEC", "", FlagGroup::Spec,
     "спекулятивный декод: lossless, только greedy", "выкл"},
    {"spec-k", "TOKSHIT_SPEC_K", "K", FlagGroup::Spec,
     "черновиков за раунд; 0/1 = выключено", "4"},
    {"spec-verify-topk", "TOKSHIT_SPEC_VERIFY_TOPK", "K", FlagGroup::Spec,
     "top-k экспертов на verify; 0 = как в модели", "0"},
    {"spec-maxb", "TOKSHIT_SPEC_MAXB", "B", FlagGroup::Spec,
     "максимум строк verify-батча, минимум 2", "8"},
    {"ctx", "TOKSHIT_SPEC_CTX", "N", FlagGroup::Spec,
     "ёмкость KV для спекуляции, токенов", "2048"},
    {"spec-ctx", "TOKSHIT_SPEC_CTX", "N", FlagGroup::Spec,
     "то же, что --ctx", "как --ctx"},
    {"draft-topk", "TOKSHIT_DRAFT_TOPK", "K", FlagGroup::Spec,
     "экспертов у черновика; меньше = быстрее, LOSSY", "0"},
    {"verify-ids", "TOKSHIT_VERIFY_IDS", "FILE", FlagGroup::Spec,
     "эталонные id: замер acceptance черновика", "нет"},

    // сэмплинг / sampling
    {"temperature", "TOKSHIT_TEMPERATURE", "X", FlagGroup::Sampling,
     "температура; 0 = жадный argmax", "0.8"},
    {"top-k", "TOKSHIT_TOP_K", "N", FlagGroup::Sampling,
     "top-k сэмплинга; 0 = без отсечения", "0"},
    {"top-p", "TOKSHIT_TOP_P", "X", FlagGroup::Sampling,
     "top-p (nucleus); 1 = выключено", "0.9"},
    {"seed", "TOKSHIT_SEED", "N", FlagGroup::Sampling,
     "зерно ГПСЧ", "случайное"},
    {"reasoning-budget", "TOKSHIT_REASONING_BUDGET", "N", FlagGroup::Sampling,
     "бюджет thinking-токенов: N>0 кап, 0 закрыть сразу, -1 без лимита", "-1"},

    // отладка и бенчмарки / debug and benchmarks
    {"print-ids", "TOKSHIT_PRINT_IDS", "", FlagGroup::Debug,
     "печатать id сгенерированных токенов", "выкл"},
    {"dump-logits", "TOKSHIT_DUMP_LOGITS", "FILE", FlagGroup::Debug,
     "писать f32-логиты в файл (паритетный гейт)", "нет"},
    {"expert-stats", "TOKSHIT_EXPERT_STATS", "", FlagGroup::Debug,
     "итоговая статистика экспертов", "выкл"},
    {"pool-stats", "TOKSHIT_POOL_STATS", "", FlagGroup::Debug,
     "печать статистики пула", "выкл"},
    {"bench-mm", "TOKSHIT_BENCH_MM", "", FlagGroup::Debug,
     "бенчмарк matvec плотных весов", "выкл"},
    {"bench-passes", "TOKSHIT_BENCH_PASSES", "N", FlagGroup::Debug,
     "проходов бенчмарка", "5"},
    {"bench-breakdown", "TOKSHIT_BENCH_BREAKDOWN", "", FlagGroup::Debug,
     "разбивка q/k/v/o в бенчмарке", "выкл"},
    {"debug-tok", "TOKSHIT_DEBUG_TOK", "", FlagGroup::Debug,
     "отладка токенизатора", "выкл"},
    {"seg-debug", "TOKSHIT_SEG_DEBUG", "", FlagGroup::Debug,
     "отладка сегментов matvec", "выкл"},

    // пресеты / presets
    {"turbo", "TOKSHIT_TURBO", "", FlagGroup::Preset,
     "LOSSY: top-2 + Q4_0 dense, кэш 1024 и O_DIRECT", "выкл"},
    {"spec-fast", "TOKSHIT_SPEC_FAST", "", FlagGroup::Preset,
     "LOSSY: спекуляция K=4, черновик top-2, verify top-2", "выкл"},};

std::string_view trim(std::string_view s) {
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

std::string to_lower(std::string_view s) {
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

// ширина в колонках: считаем кодовые точки, а не байты utf-8
// column width: count utf-8 code points instead of bytes
std::size_t columns(std::string_view s) {
  std::size_t n = 0;
  for (unsigned char c : s) {
    if ((c & 0xC0U) != 0x80U) ++n;
  }
  return n;
}

void pad_to(std::string& out, std::size_t col) {
  // отступ считаем от начала текущей строки, а не всего буфера
  // indent is measured from the start of the current line, not the buffer
  const std::size_t nl = out.rfind('\n');
  const std::size_t from = nl == std::string::npos ? 0 : nl + 1;
  while (columns(std::string_view(out).substr(from)) < col) {
    out.push_back(' ');
  }
}

// перенос по словам: каждая строка не шире width колонок
// word wrap: no line wider than width columns
std::vector<std::string> wrap(std::string_view text, std::size_t width) {
  std::vector<std::string> lines;
  std::string cur;
  std::size_t i = 0;
  while (i < text.size()) {
    std::size_t j = i;
    while (j < text.size() && text[j] != ' ') ++j;
    const std::string_view word = text.substr(i, j - i);
    if (!word.empty()) {
      const std::size_t extra = cur.empty() ? columns(word) : columns(word) + 1;
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

void append_wrapped(std::string& out, std::string_view text,
                    std::size_t first_col, std::size_t rest_col,
                    std::size_t width) {
  const std::vector<std::string> lines = wrap(text, width - rest_col);
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (i > 0) out.push_back('\n');
    pad_to(out, i == 0 ? first_col : rest_col);
    out.append(lines[i]);
  }
}

void append_flag(std::string& out, const FlagDoc& f) {
  constexpr std::size_t kDescCol = 24;
  constexpr std::size_t kWidth = 78;
  std::string head = "  --";
  head.append(f.flag);
  if (f.arg != nullptr && *f.arg != '\0') {
    head.push_back(' ');
    head.append(f.arg);
  }
  out.append(head);
  append_wrapped(out, f.desc, kDescCol, kDescCol, kWidth);
  out.push_back('\n');
  std::string tail = "по умолчанию ";
  tail.append(f.def);
  tail.append(" · env ");
  tail.append(f.env);
  append_wrapped(out, tail, kDescCol, kDescCol, kWidth);
  out.push_back('\n');
}

void append_line(std::string& out, std::string_view text) {
  out.append(text);
  out.push_back('\n');
}

}  // namespace

std::span<const FlagDoc> flag_docs() noexcept {
  return std::span<const FlagDoc>(kFlags, sizeof(kFlags) / sizeof(kFlags[0]));
}

std::string_view group_title(FlagGroup group) noexcept {
  switch (group) {
    case FlagGroup::Run:
      return "Запуск и модель";
    case FlagGroup::Memory:
      return "Производительность и память";
    case FlagGroup::Cache:
      return "Экспертный кэш и ввод-вывод";
    case FlagGroup::Spec:
      return "Спекуляция и черновик";
    case FlagGroup::Sampling:
      return "Сэмплинг";
    case FlagGroup::Debug:
      return "Отладка и бенчмарки";
    case FlagGroup::Preset:
      return "Пресеты (наборы флагов)";
    case FlagGroup::Count:
      break;
  }
  return "";
}

std::string help_text(std::string_view default_model) {
  std::string out;
  append_line(out,
              "токшит — MoE-рантайм для Flint-40B-A2.5B на C++20 (Termux).");
  out.push_back('\n');
  append_line(out, "использование:");
  append_line(out, "  tokshit [флаги] <model.gguf> [промпт] [max_new]");
  append_line(out, "  tokshit [флаги] <model.gguf> --prompt-file <файл> [max_new]");
  append_line(out, "  tokshit [флаги]                 # интерактивный чат");
  out.push_back('\n');
  append_line(out, "без промпта запускается чат; модель по умолчанию:");
  append_line(out, "  " + std::string(default_model));
  out.push_back('\n');
  append_line(out,
              "флаги: --имя значение или --имя=значение. У каждого флага есть");
  append_line(out,
              "env-фолбэк TOKSHIT_*; флаги выставляют те же переменные. Логи и");
  append_line(out, "метрики идут в stderr, ответ модели — в stdout.");
  out.push_back('\n');
  append_line(out,
              "булевы флаги не съедают следующий аргумент: пишите --hot=0,");
  append_line(out, "а не --hot 0 (иначе 0 станет промптом).");

  constexpr FlagGroup kOrder[] = {
      FlagGroup::Run,      FlagGroup::Memory,   FlagGroup::Cache,
      FlagGroup::Spec,     FlagGroup::Sampling, FlagGroup::Debug,
      FlagGroup::Preset,
  };
  for (FlagGroup g : kOrder) {
    out.push_back('\n');
    std::string title = "── ";
    title.append(group_title(g));
    title.append(" ──");
    append_line(out, title);
    for (const FlagDoc& f : kFlags) {
      if (f.group == g) append_flag(out, f);
    }
  }

  out.push_back('\n');
  append_line(out, "── Команды чата ──");
  out.append(command_help());
  out.push_back('\n');
  append_line(out, "── Примеры ──");
  append_line(out, "  # обычный чат");
  append_line(out, "  tokshit");
  out.push_back('\n');
  append_line(out, "  # один ответ, 64 токена");
  append_line(out, "  tokshit model.gguf \"привет\" 64");
  out.push_back('\n');
  append_line(out, "  # безопасный замер: жадный сэмплинг под сторожем памяти");
  append_line(out, "  TOKSHIT_EXPERT_CACHE_MB=512 TOKSHIT_MIN_AVAIL_MB=1400 \\");
  append_line(out,
              "    scripts/safe_run.sh build/tokshit --temperature 0 \\");
  append_line(out, "    model.gguf \"привет\" 1");
  out.push_back('\n');
  append_line(out, "  # мало памяти: без hot-копии, маленький кэш, 1 io-поток");
  append_line(out,
              "  tokshit --hot=0 --cache-mb=256 --io-threads=1 --temperature 0 \\");
  append_line(out, "    model.gguf \"привет\" 8");
  out.push_back('\n');
  append_line(out, "  # спекуляция: жадный декод, K=4, verify top-2");
  append_line(out,
              "  tokshit --spec --spec-k 4 --spec-verify-topk 2 \\");
  append_line(out, "    --temperature 0 model.gguf \"привет\" 64");
  out.push_back('\n');
  append_line(out, "  # LOSSY turbo: top-2 + Q4_0 dense, кэш 1024 и O_DIRECT");
  append_line(out, "  tokshit --turbo --temperature 0 model.gguf \"привет\" 64");
  out.push_back('\n');
  append_line(out, "  # эта справка");
  append_line(out, "  tokshit --help");
  return out;
}

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
// hit-rate as text: colored value when color, "—" without data
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
// spinner frames of the live decode indicator
constexpr std::string_view kFrames[] = {"◜", "◠", "◝", "◞", "◡", "◟"};

}  // namespace

std::size_t spinner_frames() noexcept {
  return sizeof(kFrames) / sizeof(kFrames[0]);
}

std::string_view spinner_frame(std::size_t index) noexcept {
  return kFrames[index % spinner_frames()];
}

// добавляет « · текст» только если строка остаётся в пределах width
// appends " · text" only while the line still fits into width
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
  // show the rate only once it is measurable: right after the first token
  // elapsed is near zero and "2030 tok/s" is just noise
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

Command parse_command(std::string_view line) {
  Command cmd;
  std::string_view raw = line;
  while (!raw.empty() && (raw.back() == '\r' || raw.back() == '\n')) {
    raw.remove_suffix(1);
  }
  const std::string_view t = trim(raw);
  if (t.empty()) return cmd;
  if (t.front() != '/') {
    const std::string lower = to_lower(t);
    if (lower == "exit" || lower == "quit") {
      cmd.kind = CommandKind::Quit;
      return cmd;
    }
    cmd.kind = CommandKind::Chat;
    cmd.arg.assign(raw);
    return cmd;
  }

  std::size_t sp = 1;
  while (sp < t.size() && t[sp] != ' ' && t[sp] != '\t') ++sp;
  const std::string name = to_lower(t.substr(1, sp - 1));
  const std::string_view rest = trim(t.substr(sp));
  cmd.arg.assign(rest);

  if (name.empty()) {
    cmd.kind = CommandKind::Unknown;
    return cmd;
  }
  if (name == "help" || name == "?") {
    cmd.kind = CommandKind::Help;
    return cmd;
  }
  if (name == "exit" || name == "quit" || name == "q") {
    cmd.kind = CommandKind::Quit;
    return cmd;
  }
  if (name == "clear" || name == "cls") {
    cmd.kind = CommandKind::Clear;
    return cmd;
  }
  if (name == "stats") {
    cmd.kind = CommandKind::Stats;
    return cmd;
  }
  if (name == "reset") {
    cmd.kind = CommandKind::Reset;
    return cmd;
  }
  if (name == "system" || name == "sys") {
    cmd.kind = CommandKind::System;
    return cmd;
  }
  if (name == "save") {
    cmd.kind = CommandKind::Save;
    if (rest.empty()) cmd.valid = false;
    return cmd;
  }
  if (name == "temp" || name == "temperature") {
    cmd.kind = CommandKind::Temp;
    if (!rest.empty()) {
      char* end = nullptr;
      const std::string tmp(rest);
      const float v = std::strtof(tmp.c_str(), &end);
      if (end != nullptr && static_cast<std::size_t>(end - tmp.c_str()) == tmp.size()) {
        cmd.number = v;
      } else {
        cmd.valid = false;
      }
    } else {
      cmd.valid = false;
    }
    return cmd;
  }
  if (name == "topk" || name == "top-k") {
    cmd.kind = CommandKind::TopK;
    if (!rest.empty() && rest.front() != '-') {
      char* end = nullptr;
      const std::string tmp(rest);
      const unsigned long long v = std::strtoull(tmp.c_str(), &end, 10);
      if (end != nullptr && static_cast<std::size_t>(end - tmp.c_str()) == tmp.size()) {
        cmd.count = static_cast<std::size_t>(v);
      } else {
        cmd.valid = false;
      }
    } else {
      cmd.valid = false;
    }
    return cmd;
  }
  if (name == "topp" || name == "top-p") {
    cmd.kind = CommandKind::TopP;
    if (!rest.empty()) {
      char* end = nullptr;
      const std::string tmp(rest);
      const float v = std::strtof(tmp.c_str(), &end);
      if (end != nullptr && static_cast<std::size_t>(end - tmp.c_str()) == tmp.size()) {
        cmd.number = v;
      } else {
        cmd.valid = false;
      }
    } else {
      cmd.valid = false;
    }
    return cmd;
  }
  if (name == "reasoning-budget") {
    cmd.kind = CommandKind::ReasoningBudget;
    if (!rest.empty()) {
      char* end = nullptr;
      errno = 0;
      const std::string tmp(rest);
      const long v = std::strtol(tmp.c_str(), &end, 10);
      const bool whole =
          end != nullptr &&
          static_cast<std::size_t>(end - tmp.c_str()) == tmp.size();
      // допустимы только -1, 0 и положительные значения
      // only -1, 0 and positive values are valid
      if (whole && errno != ERANGE && v >= -1 && v <= INT_MAX) {
        cmd.budget = static_cast<int>(v);
      } else {
        cmd.valid = false;
      }
    } else {
      cmd.valid = false;
    }
    return cmd;
  }
  cmd.kind = CommandKind::Unknown;
  cmd.arg = name;
  return cmd;
}

std::string command_help() {
  std::string out;
  append_line(out, "  /help              эта справка по командам");
  append_line(out, "  /exit, /quit       выйти (или Ctrl-D)");
  append_line(out, "  /clear             очистить контекст (KV), в TTY — и экран");
  append_line(out, "  /stats             скорость, кэш, память, сессия, параметры");
  append_line(out, "  /reset             вернуть сэмплинг к умолчаниям");
  append_line(out, "  /temp <x>          температура; 0 = жадный argmax");
  append_line(out, "  /topk <n>          top-k; 0 = без отсечения");
  append_line(out, "  /topp <x>          top-p; 1 = выключено");
  append_line(out,
              "  /reasoning-budget N  thinking: N>0 кап, 0 закрыть сразу, -1 без лимита");
  append_line(out, "  /system <текст>    системная инструкция: текст подставляется");
  append_line(out, "                     в начало следующего сообщения (пусто = убрать)");
  append_line(out, "  /save <файл>       дописать диалог в файл");
  append_line(out, "  пустая строка      ничего не делает");
  append_line(out, "  Ctrl-C             прервать ответ, не выходить из чата");
  append_line(out, "  Ctrl-D             аккуратный выход");
  out.push_back('\n');
  append_line(out, "  примеры:");
  append_line(out, "    /temp 0.2              мягкий сэмплинг вместо жадного");
  append_line(out, "    /topk 40               выбор из 40 кандидатов");
  append_line(out, "    /reasoning-budget 0    отвечать без длинных размышлений");
  append_line(out, "    /system отвечай кратко  задать системную инструкцию");
  append_line(out, "    /stats                  проверить скорость и hit-rate кэша");
  return out;
}

}  // namespace tokshit::tui
