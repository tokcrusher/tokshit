# tokshit

Специализированный инференс-движок на C++20 для ARM64/Snapdragon: запускает
MoE-модели, которые больше оперативной памяти устройства. Целевая связка —
Flint-40B-A2.5B (GGUF, 18.5 ГБ) на телефоне с 15.2 ГБ RAM.

Универсальные движки платят кроссплатформенный налог: динамическая
диспетчеризация, общие аллокаторы, компромиссы под x86 и CUDA. `tokshit`
убирает этот слой: фиксированный горячий путь, NEON dot-product и i8mm,
AHB (dma-buf) для плотных весов, `O_DIRECT` для экспертов, послойный
LRU-кэш с частотным профилем роутинга и батчевый GEMM-конвейер префилла.

English version: [README.md](README.md).

## Возможности

- **MoE больше RAM.** Эксперты (256 на слой, top-6) читаются с флеша в
  кэш; плотные веса закреплены в несвопируемой AHB-памяти — без ZRAM и
  без резких задержек.
- **NEON/i8m ядра.** Dot-product-ядра для Q8_0/Q4_0/IQ-форматов,
  SMMLA-путь для Q8_0 (`TOKSHIT_I8MM=1`), батчевые матвеки и
  GEMM-конвейер префилла.
- **Спекулятивный декод.** Черновик на урезанном роутере, батч-verify,
  lossless-жадный режим; acceptance до ~75 % на коротких раундах.
- **Частотный профиль роутинга.** Профиль сохраняется между запусками и
  прогревает кэш экспертов до первого токена.
- **Батчевый префилл.** Веса читаются один раз на чанк токенов, эксперты —
  union'ом роутинга чанка.
- **TUI.** Живой индикатор декода (скорость, hit-rate кэша, свободная
  память), прогресс префилла, дашборд `/stats`, чат-команды `/help`,
  `/temp`, `/topk`, `/system`, `/save` и другие.
- **Сторож памяти.** `safe_run.sh` следит за `MemAvailable` и не даёт
  OOM-killer перезагрузить телефон.

## Производительность

Замеры на целевом устройстве (8 ядер: 1× Cortex-X4 + 7× Cortex-A720,
15.2 ГБ RAM, UFS ~1.4 ГБ/с на рабочих чанках, 40–55 °C). Телефон троттлит:
конфиги сравниваются вперемешку через `scripts/ab.sh`, одиночные прогоны
шумные.

| режим | tok/s |
|---|---:|
| декод top-6, кэш 512 МБ, `O_DIRECT` | 2.0–2.7 |
| декод top-6, кэш 1024 МБ, io 8 | 2.43 (hit 60 % на длинных прогонах) |
| префилл батчами (`TOKSHIT_PREFILL_BATCH=32`) | 4–5.3 |
| `--fast` (top-2, **LOSSY**) | ~4.5 |
| спекуляция, lossless verify | acceptance до 75 % |

Микробенчмарки:

| что | результат |
|---|---:|
| Q8_0 matvec, отдельные сегменты | 25–35 ГБ/с |
| hot-область из page cache (`bwbench`) | 39.5 ГБ/с |
| AHB против malloc, блоки 328 КБ | 25.7 против 21.7 ГБ/с |
| поток всех hot-весов 1.74 ГБ через AHB | 5.2 ГБ/с |
| сырое чтение с флеша (`dd iflag=direct`) | ~1.4 ГБ/с |

Бюджет токена: плотные веса ~1.74 ГБ (переквантованные в Q8_0), эксперты
при полном промахе ~0.37 ГБ (6 из 256 × 3 проекции × 47 слоёв). Методика,
фазы и подробные прогоны — в
[docs/PERFORMANCE.ru.md](docs/PERFORMANCE.ru.md).

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
git clone git@github.com:tokcrusher/tokshit.git
cd tokshit

# тулчейн, сборка, установка бинаря в $PREFIX/bin
./install.sh

# модель не входит в комплект: положите GGUF рядом и укажите путь
# один ответ
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  ~/models/flint-UD-Q3_K_XL.gguf "привет" 32

# интерактивный чат
scripts/safe_run.sh build/tokshit --cache-mb 512 \
  ~/models/flint-UD-Q3_K_XL.gguf

# справка по всем флагам
build/tokshit --help
```

`./install.sh` проверяет тулчейн, ставит недостающие пакеты, собирает
Release/Ninja и кладёт бинарь в `$PREFIX/bin` (Termux) или
`/usr/local/bin` (Linux).

`scripts/safe_run.sh` раз в секунду проверяет `MemAvailable` и завершает
процесс ниже `TOKSHIT_MIN_AVAIL_MB` (по умолчанию 1200). Запускайте модель
только через него и только одним процессом.

### Сборка вручную

```sh
cmake --preset release
cmake --build build --parallel 2
```

Нужны CMake 3.20+, Ninja и компилятор с C++20. На aarch64 NEON-файлы
компилируются отдельно: dot-product-ядро с `-march=armv8.2-a+dotprod+fp16+lse`,
i8mm-ядро — с `+dotprod+i8mm+fp16+lse`; флаги не протекают в скалярные
объекты. i8mm-варианты включаются опцией `TOKSHIT_I8MM` (по умолчанию ON).

## Использование

```sh
build/tokshit [флаги] <model.gguf> "<промпт>" [max_new]
```

Без промпта открывается интерактивный чат. Ключевые флаги:

| флаг | что делает |
|---|---|
| `--cache-mb N` | бюджет кэша экспертов (512 МБ — рабочая точка) |
| `--io-threads N` | фоновые читатели диска |
| `--temperature/--top-k/--top-p` | сэмплинг |
| `--fast` | top-2 вместо top-6, быстрее и **LOSSY** |
| `--spec --spec-k N` | спекулятивный декод (lossless с полным verify) |
| `--expert-stats` | hit/miss кэша экспертов в конце прогона |

Пресеты:

```sh
# быстрый LOSSY-прогон: top-2 из 6, page cache
scripts/safe_run.sh build/tokshit --fast --temperature 0 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# спекуляция с lossless-verify
scripts/safe_run.sh build/tokshit --spec --spec-k 3 --temperature 0 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 24
```

Полная таблица флагов, переменных окружения и чат-команд —
в [docs/CLI.md](docs/CLI.md).

## Память

Один общий AHB-бюджет (`TOKSHIT_AHB_MAX_MB`, по умолчанию 5120 МиБ) делят
hot-веса и кэш экспертов. 512 МБ кэша — рабочая точка; 2 ГБ добавляют
попадания, но память уходит в reclaim. Один буфер на все hot-веса gralloc
не даёт, поэтому 391 буфер по тензору.

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
src/tools/         qstat, kbench, vcheck, bwbench, pbench, ahbtest
scripts/           safe_run.sh, ab.sh, sweep.sh, deploy.sh
docs/              тематическая документация
install.sh         установщик Termux/Linux
CMakePresets.json  пресеты release/debug
LICENSE            кастомная WTF 1.2
```

## Документация

Полный индекс и порядок чтения — [docs/README.md](docs/README.md).

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
| [docs/ROADMAP.md](docs/ROADMAP.md) | этапы, риски, идеи |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | сборка и правила разработки |

## Лицензия

Кастомная WTF License 1.2, см. [LICENSE](LICENSE). Код можно использовать,
менять и распространять, в том числе коммерчески. При распространении
софта или его существенной части нужно приложить текст лицензии. Гарантий
и обязательств по поддержке нет.
