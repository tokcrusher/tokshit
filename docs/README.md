# Документация tokshit

`tokshit` — специализированный C++20-инференс-движок для ARM64/Snapdragon,
оптимизированный под конкретное устройство и модель. Он запускает
MoE-модели, которые больше оперативной памяти устройства, читает GGUF,
считает forward на NEON dot-product и стримит экспертов с флеша через
послойный LRU-кэш.

Этот файл — карта чтения. Начните с быстрого старта, затем выберите тему.

## Быстрый старт

1. [README.ru.md](../README.ru.md) — установка, первый запуск, статус.
2. [CLI.md](CLI.md) — флаги, пресеты, TUI, `safe_run.sh`.

## Архитектура

3. [ARCHITECTURE.ru.md](ARCHITECTURE.ru.md) — модули, загрузка, инварианты.
4. [RUNTIME.md](RUNTIME.md) — сквозной путь токена, MoE, кэш, I/O, AHB, пул.
5. [SPEC_KV.md](SPEC_KV.md) — KV-кеш, батч-матвек, спекулятивный цикл.

## Кванты и ядра

6. [QUANTIZATION.md](QUANTIZATION.md) — Q8_0, Q4_0, K-кванты, IQ, деквант.
7. [KERNELS.md](KERNELS.md) — NEON dotprod, fallback, диспетчеризация,
   замеры полосы.
8. [GGUF.md](GGUF.md) — формат, метаданные, смещения, mmap/pread, ошибки.
9. [TOKENIZER.md](TOKENIZER.md) — BPE, байтовый fallback, сэмплинг.

## Результаты и планы

10. [PERFORMANCE.ru.md](PERFORMANCE.ru.md) — методика, декод, спекуляция,
    микробенчмарки, исторические прогоны.
11. [PERF_SESSION2.md](PERF_SESSION2.md) — baseline сессии 16.11.2026,
    отрицательные результаты, план.
12. [BENCH.ru.md](BENCH.ru.md) — воспроизводимый харнесс decode/prefill/A/B,
    JSON/CSV, regression gate.
13. [ROADMAP.md](ROADMAP.md) — этапы, карта компонентов, риски, идеи.
14. [CONTRIBUTING.md](CONTRIBUTING.md) — правила, гигиена модели, коммиты.

## English

- [README.md](../README.md) — English entry point.
- [ARCHITECTURE.en.md](ARCHITECTURE.en.md) — modules and data flow.
- [PERFORMANCE.md](PERFORMANCE.md) — measurements and methodology.

## Обозначения

| метка | смысл |
|---|---|
| **LOSSY** | режим меняет выход модели, не сравним с обычным декодом |
| `tok/s` | сгенерированных токенов в секунду |
| `ГБ/с` | гигабайты в секунду, если не указано иное |
| hot-веса | плотные веса, читаемые на каждом токене |
| эксперт | срез MoE-FFN, выбираемый роутером |

## Правила чисел

- Все замеры — целевой телефон 8 ядер, 15.2 ГБ RAM, Android/Termux.
- Телефон троттлит: одиночные прогоны шумные, сравнение — через
  `scripts/bench_matrix.sh` (см. [BENCH.ru.md](BENCH.ru.md)).
- Микробенчмарки не предсказывают tok/s полного декода.
- Идеи и незавершённые эксперименты помечены в [ROADMAP.md](ROADMAP.md).
