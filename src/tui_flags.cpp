#include "tokshit/tui.hpp"

#include <span>
#include <string>
#include <string_view>

#include "tui_internal.hpp"

namespace tokshit::tui {

using detail::append_line;
using detail::append_wrapped;
using detail::columns;
using detail::pad_to;
using detail::to_lower;
using detail::trim;
using detail::wrap;

namespace {
constexpr FlagDoc kFlags[] = {
    // Запуск и модель
    {"model", "TOKSHIT_MODEL", "FILE", FlagGroup::Run,
     "путь к .gguf-модели; иначе TOKSHIT_MODEL или вшитый", "встроенный"},
    {"prompt-file", "TOKSHIT_PROMPT_FILE", "FILE", FlagGroup::Run,
     "читать промпт из файла, а не из argv", "нет"},
    {"layers", "GGUF_NLAYERS", "N", FlagGroup::Run,
     "ограничить число слоёв (только отладка)", "все"},

    // Производительность и память
    {"hot", "TOKSHIT_HOT", "", FlagGroup::Memory,
     "копировать плотные веса в RAM; 0 = mmap", "1"},
    {"hot-conv", "TOKSHIT_HOT_CONV", "", FlagGroup::Memory,
     "переквантовывать hot-веса в Q8_0 (нужен neon dot); 0 = как в gguf",
     "вкл"},
    {"drop-cache", "TOKSHIT_DROP_CACHE", "", FlagGroup::Memory,
     "сбросить page cache после копии hot-весов", "1"},
    {"drop-siblings", "TOKSHIT_DROP_SIBLINGS", "", FlagGroup::Memory,
     "выкинуть из page cache соседние .gguf (mmproj, другие кванты)", "выкл"},
    {"prefault", "TOKSHIT_PREFAULT", "", FlagGroup::Memory,
     "трогать все страницы модели при загрузке; 0 = demand paging", "вкл"},
    {"kv-ctx", "TOKSHIT_KV_CTX", "N", FlagGroup::Memory,
     "KV-позиции Inkling2; меньше = меньше памяти", "2048"},
    {"mlock", "TOKSHIT_MLOCK", "", FlagGroup::Memory,
     "зарезервировано: best-effort mlock весов (не подключён)", "выкл"},
    {"mlock-dense", "TOKSHIT_MLOCK_DENSE_MB", "MB", FlagGroup::Memory,
     "mlock плотных q/k/v/o/gate/out, МБ; осторожно с памятью", "0"},
    {"pool-workers", "TOKSHIT_POOL_WORKERS", "N", FlagGroup::Memory,
     "потоков вычислительного пула", "4 (aarch64), иначе ядра-1"},
    {"pool-pin", "TOKSHIT_POOL_PIN", "", FlagGroup::Memory,
     "привязать потоки пула к ядрам", "выкл"},
    {"pool-rt", "TOKSHIT_POOL_RT", "", FlagGroup::Memory,
     "потоки пула в realtime-приоритет", "выкл"},
    {"main-cpu", "TOKSHIT_MAIN_CPU", "N", FlagGroup::Memory,
     "ядро главного потока при --pool-pin", "7"},

    // Экспертный кэш и ввод-вывод
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

    // Спекуляция и черновик
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
    {"dspark", "TOKSHIT_DSPARK", "FILE", FlagGroup::Spec,
     "DSpark draft GGUF для Inkling2; greedy, lossless", "нет"},
    {"dspark-k", "TOKSHIT_DSPARK_K", "K", FlagGroup::Spec,
     "черновиков DSpark за раунд, 1..7", "3"},
    {"passes", "TOKSHIT_INK_PASSES", "N", FlagGroup::Spec,
     "LOSSY: core-проходов Inkling2, 1..3; меньше = быстрее", "3"},
    {"topk-experts", "TOKSHIT_DRAFT_TOPK", "K", FlagGroup::Spec,
     "LOSSY: экспертов на токен; меньше = быстрее", "как в модели"},
    {"dspark-batch", "TOKSHIT_DSPARK_BATCH", "0|1", FlagGroup::Spec,
     "батч-verify DSpark: один проход плотных весов на K кандидатов", "1"},

    // Сэмплинг
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

    // Отладка и бенчмарки
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

    // Пресеты
    {"turbo", "TOKSHIT_TURBO", "", FlagGroup::Preset,
     "LOSSY: top-2 + Q4_0 dense, кэш 1024 и O_DIRECT", "выкл"},
    {"spec-fast", "TOKSHIT_SPEC_FAST", "", FlagGroup::Preset,
     "LOSSY: спекуляция K=4, черновик top-2, verify top-2", "выкл"},};

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
}  // namespace tokshit::tui
