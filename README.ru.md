# tokshit

Специализированный инференс-движок на C++20 для ARM64/Snapdragon. Запускает
MoE-модели, которые больше оперативной памяти устройства. Целевая связка:
Flint-40B-A2.5B (GGUF, 18.5 ГБ) на телефоне с 15.2 ГБ RAM.

Универсальные движки платят кроссплатформенный налог: динамическая
диспетчеризация, общие аллокаторы, компромиссы под x86 и CUDA. `tokshit`
убирает этот слой: фиксированный горячий путь, NEON dot-product, AHB
(dma-buf) для плотных весов, `O_DIRECT` для экспертов, послойный LRU-кэш с
частотным профилем роутинга и батчевый префилл.

English entry point: [README.md](README.md).

## Возможности

- **MoE больше RAM.** 256 экспертов на слой, top-6. Эксперты читаются с
  флеша в кэш; плотные веса закреплены в AHB и не уходят в ZRAM.
- **NEON dot-product.** Ядра для Q8_0, Q4_0 и IQ-форматов. i8mm-путь
  собирается, но на этом устройстве проигрывает и включается вручную
  (`TOKSHIT_I8MM=1`). Двухстрочные IQ-ядра (`TOKSHIT_IQ2ROW=1`) тоже
  экспериментальные.
- **Спекулятивный декод.** Черновик на урезанном роутере, батч-verify,
  откат KV. Полный verify сохраняет жадный выход; на замерах медленнее
  обычного декода.
- **Частотный профиль роутинга.** Профиль хранится на диске и прогревает
  кэш экспертов до первого токена (`TOKSHIT_FREQ_PERSIST=1`).
- **Батчевый префилл.** Веса читаются один раз на чанк, эксперты — union
  роутинга чанка.
- **TUI.** Живой прогресс префилла, строка статуса, `/stats`, `/temp`,
  `/topk`, `/system`, `/save`.
- **Сторож памяти.** `safe_run.sh` завершает процесс при низком
  `MemAvailable` и не даёт OOM-killer перезагрузить телефон.

## Производительность

Замеры на целевом устройстве: 8 ядер (1× Cortex-X4 + 7× Cortex-A720),
15.2 ГБ RAM, UFS. Телефон троттлит, поэтому одиночные прогоны шумные;
конфиги сравниваются вперемешку через `scripts/bench_matrix.sh`.

| режим | tok/s |
|---|---:|
| декод top-6, 8 токенов, 47 °C, кэш 1024 МБ, io 8 | **2.40** |
| декод top-6, 16 токенов, 60 °C, тот же конфиг | 0.96 |
| декод top-6, исторические длинные прогоны | 1.0–2.7 |
| `--turbo` (top-2 + Q4_0, **LOSSY**) | 3.52 |
| префилл union, T=224, батч 32 | 4.53 |
| префилл union, T=128 | 6.01 |
| prefill-bulk, T=224 | 2.32 |
| спекуляция, полный verify (lossless) | 0.93–1.0 |

Температура меняет картину: при 60 °C частоты падают до 1.2–1.7 ГГц,
поток плотных весов — с 12.6 до 4.2 ГБ/с. Запускайте с
`termux-wake-lock`, проверяйте температуру и давайте телефону остыть.
`--turbo` меняет выход модели и не заменяет обычный декод.

Микробенчмарки: Q8_0 matvec на сегментах — 25–35 ГБ/с; поток hot-весов
1.74 ГБ через AHB — 5.2 ГБ/с; последовательное чтение с флеша — около
1.4 ГБ/с. Узкое место декода — полный поток плотных весов.

Бюджет токена: плотные веса около 1.74 ГБ (после переквантовки в Q8_0),
эксперты при полном промахе около 0.38 ГБ (6 из 256 × 3 проекции ×
47 слоёв). Методика и прогоны — в [docs/PERFORMANCE.ru.md](docs/PERFORMANCE.ru.md).

## Модель

| параметр | значение |
|---|---|
| блоки | 49: 2 плотных + 47 MoE |
| эксперты | 256 на слой, top-6 |
| hidden / ffn эксперта | 2048 / 512 |
| shared expert | 1024 |
| внимание | 32 q-головы, 4 kv, head_dim 128, SWA 513 |
| словарь / контекст | 150144 / 262144 |
| кванты | эксперты IQ3_S/XXS и IQ4_XS; внимание Q8_0; роутер BF16 |

## Быстрый старт

```sh
git clone https://github.com/tokcrusher/tokshit.git
cd tokshit
./install.sh
```

`install.sh` проверяет тулчейн, ставит недостающие пакеты, собирает
Release/Ninja и кладёт бинарь в `$PREFIX/bin` (Termux) или
`/usr/local/bin` (Linux).

Модель в комплект не входит. Положите GGUF рядом и укажите путь:

```sh
# один ответ
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  ~/models/flint-UD-Q3_K_XL.gguf "привет" 32

# интерактивный чат
scripts/safe_run.sh build/tokshit --cache-mb 512 \
  ~/models/flint-UD-Q3_K_XL.gguf

# справка по флагам
build/tokshit --help
```

`scripts/safe_run.sh` раз в секунду проверяет `MemAvailable` и завершает
процесс ниже `TOKSHIT_MIN_AVAIL_MB` (по умолчанию 1200). Запускайте только
один процесс модели.

### Сборка вручную

```sh
cmake --preset release
cmake --build build --parallel 2
```

Нужны CMake 3.20+, Ninja и компилятор с C++20. На aarch64 NEON-файлы
компилируются отдельно: dot-product-ядро с
`-march=armv8.2-a+dotprod+fp16+lse`, i8mm-ядро — с
`+dotprod+i8mm+fp16+lse`. i8mm-код собирается опцией CMake
`TOKSHIT_I8MM` (по умолчанию ON), а на исполнении включается переменной
`TOKSHIT_I8MM=1`.

## Использование

```sh
build/tokshit [флаги] <model.gguf> "<промпт>" [max_new]
```

Без промпта открывается интерактивный чат. Основные флаги:

| флаг | что делает |
|---|---|
| `--cache-mb N` | бюджет кэша экспертов (512 МБ — рабочая точка) |
| `--io-threads N` | фоновые читатели диска |
| `--temperature/--top-k/--top-p` | сэмплинг |
| `--turbo` | top-2 + Q4_0 dense, быстрее и **LOSSY** |
| `--spec --spec-k N` | спекулятивный декод (lossless с полным verify) |
| `--expert-stats` | hit/miss кэша экспертов в конце прогона |

Пресеты:

```sh
# LOSSY: top-2 из 6 и Q4_0 dense
scripts/safe_run.sh build/tokshit --turbo --temperature 0 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# спекуляция с lossless-verify
scripts/safe_run.sh build/tokshit --spec --spec-k 3 --temperature 0 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 24
```

Полная таблица флагов, переменных окружения и чат-команд —
в [docs/CLI.md](docs/CLI.md).

## Память

Один AHB-бюджет (`TOKSHIT_AHB_MAX_MB`, по умолчанию 5120 МиБ) делят
hot-веса и кэш экспертов. Hot-веса пакуются в сегменты по 32 МБ
(`TOKSHIT_HOT_SEG_MB`); 512 МБ кэша — рабочая точка, 2 ГБ добавляют
попадания, но уходят в reclaim.

Если памяти мало:

```sh
scripts/safe_run.sh build/tokshit --hot=0 --cache-mb 256 --io-threads 1 \
  --temperature 0 ~/models/flint-UD-Q3_K_XL.gguf "hi" 8
```

## Структура

```text
include/tokshit/   публичные заголовки
src/               рантайм
src/compute/       скалярные, NEON и батчевые ядра
src/tools/         qstat, kbench, vcheck, bwbench, pbench, ahbtest, densebench
scripts/           safe_run.sh, device_lock.sh, bench_matrix.sh и другие
docs/              тематическая документация
install.sh         установщик Termux/Linux
CMakePresets.json  пресеты release/debug
LICENSE            кастомная WTF 1.2
```

## Документация

Индекс и порядок чтения — [docs/README.md](docs/README.md).

| документ | о чём |
|---|---|
| [docs/CLI.md](docs/CLI.md) | флаги, переменные, пресеты, TUI, `safe_run` |
| [docs/ARCHITECTURE.ru.md](docs/ARCHITECTURE.ru.md) | модули и поток данных |
| [docs/RUNTIME.md](docs/RUNTIME.md) | загрузка, forward, MoE, кэш, AHB, пул |
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md) | форматы блоков и деквант |
| [docs/KERNELS.md](docs/KERNELS.md) | NEON dotprod, i8mm, диспетчеризация |
| [docs/GGUF.md](docs/GGUF.md) | парсинг, метаданные, mmap/pread |
| [docs/TOKENIZER.md](docs/TOKENIZER.md) | BPE и сэмплинг |
| [docs/PERFORMANCE.ru.md](docs/PERFORMANCE.ru.md) | замеры и методика |
| [docs/SPEC_KV.md](docs/SPEC_KV.md) | KV, батч-матвек, спекуляция |
| [docs/ROADMAP.md](docs/ROADMAP.md) | ограничения и следующие шаги |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | сборка и правила разработки |

## Лицензия

Кастомная WTF License 1.2, см. [LICENSE](LICENSE). Код можно использовать,
менять и распространять, в том числе коммерчески. При распространении
софта или его существенной части нужно приложить текст лицензии. Гарантий
и обязательств по поддержке нет.
