# CLI, флаги и окружение

`tokshit` — один бинарь: загрузка GGUF, forward, сэмплинг и интерактивный
чат. Флаги выставляют те же переменные `TOKSHIT_*`, поэтому скрипты могут
пользоваться любым способом.

```text
tokshit [флаги] <model.gguf> [промпт] [max_new]
tokshit [флаги] <model.gguf> --prompt-file <файл> [max_new]
tokshit [флаги]                 # интерактивный чат
```

Без пути берётся `TOKSHIT_MODEL` или вшитый путь. Без промпта запускается
чат. Ответ идёт в stdout, логи и метрики — в stderr.

## 1. Быстрый старт

```sh
# сборка
cmake --preset release
cmake --build build --parallel 2

# один ответ, жадный сэмплинг, под сторожем памяти
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  ~/models/flint-UD-Q3_K_XL.gguf "привет" 32

# интерактивный чат
scripts/safe_run.sh build/tokshit --cache-mb 512 \
  ~/models/flint-UD-Q3_K_XL.gguf

# справка
build/tokshit --help
```

`safe_run.sh` раз в секунду проверяет `MemAvailable` и завершает процесс
ниже `TOKSHIT_MIN_AVAIL_MB` (по умолчанию 1200). Модель запускается только
через него.

## 2. Флаги

### Запуск и модель

| флаг | по умолчанию | что делает |
|---|---|---|
| `--model FILE` | `TOKSHIT_MODEL` | путь к GGUF |
| `--prompt-file FILE` | нет | читать промпт из файла |
| `--layers N` | все | считать только первые N слоёв (отладка) |

### Память и потоки

| флаг | по умолчанию | что делает |
|---|---|---|
| `--hot`, `--hot=0` | вкл | копировать плотные веса в RAM/AHB |
| `--hot-conv`, `--hot-conv=0` | вкл | переквантовывать hot-веса в Q8_0 (нужен neon dot) |
| `--drop-cache`, `--drop-cache=0` | вкл | сбросить page cache после hot-копии |
| `--drop-siblings` | выкл | выкинуть из page cache соседние `.gguf` (mmproj, другие кванты) |
| `--prefault`, `--prefault=0` | вкл | трогать все страницы модели при загрузке; `0` — demand paging (только Inkling2) |
| `--kv-ctx N` | 2048 | KV-позиции Inkling2; `512` экономит ~300 МБ |
| `--mlock` | выкл | зарезервировано: best-effort `mlock` (не подключён) |
| `--mlock-dense MB` | 0 | mlock плотных q/k/v/o/gate/out, МБ; осторожно с памятью |
| `--pool-workers N` | ядра-1 | воркеры вычислительного пула |
| `--pool-pin` | выкл | прибить воркеры к ядрам; ядра сортируются по max frequency |
| `--pool-rt` | выкл | `SCHED_FIFO` для пула (обычно отказ) |
| `--main-cpu N` | самое быстрое ядро | ядро главного потока при `--pool-pin` |

`--prefault` включён по умолчанию: загрузчик трогает все страницы модели и
печатает `prefault: X GB touched, rss=..., MemAvailable=...`. Если RSS
больше чем на 512 МБ ниже размера модели, печатается WARNING: модель не
resident, page cache будет thrash. Лечение — освободить память,
`--drop-siblings`, `--kv-ctx 512`; `--prefault=0` переключает на demand
paging (на тёплом кеше RSS был 2.2–4.3 ГБ, скорость шумная).

`--main-cpu` по умолчанию не фиксирован: при `--pool-pin` берётся самое
быстрое ядро (в `--help` пока напечатано 7; фактический выбор делает
`pool.cpp`).

### Кэш экспертов и I/O

| флаг | по умолчанию | что делает |
|---|---|---|
| `--cache-mb N`, `--expert-cache-mb N` | 1024 | бюджет кэша экспертов, 0 = без кэша |
| `--o-direct`, `--o-direct=0` | вкл | `O_DIRECT` для чтения экспертов |
| `--io-threads N` | 4 | фоновые читатели экспертов |
| `--prefetch` | выкл | предсказание экспертов следующего слоя |
| `--prefetch-idle` | выкл | спек-заказы только на простоях диска (1 слой) |
| `--io-merge` | выкл | сливать смежные чтения экспертов в один pread |
| `--pin-experts K` | 0 | закрепить K горячих экспертов после префилла |
| `--freq-persist` | выкл | сохранять частотный профиль роутинга между запусками |
| `--freq-file FILE` | `$TMPDIR` | путь профиля частот: загрузка при старте и запись |
| `--mmap-willneed` | выкл | `madvise(WILLNEED)` на весь файл |
| `--fadv-dontneed` | выкл | `POSIX_FADV_DONTNEED` после чтения эксперта |
| `--fadv-random` | выкл | `POSIX_FADV_RANDOM` на файл модели |

### Спекуляция и черновик

| флаг | по умолчанию | что делает |
|---|---|---|
| `--spec` | выкл | спекулятивный декод, только greedy |
| `--spec-k K` | 4 | черновиков за раунд; 0/1 = выключено |
| `--spec-verify-topk K` | 0 | **lossy**: урезать роутер на verify |
| `--spec-maxb B` | 8 | максимум строк verify-батча |
| `--ctx N`, `--spec-ctx N` | 2048 | ёмкость KV спекулятивного пути |
| `--passes N` | 3 | **LOSSY**: core-проходов Inkling2, 1..3; `2` = без третьего прохода |
| `--topk-experts K` | как в модели | **LOSSY**: экспертов на токен; `2` = turbo-роутер |
| `--draft-topk K` | 0 | алиас `--topk-experts`; 0 = как в модели |
| `--verify-ids FILE` | нет | эталонные id для замера acceptance |
| `--dspark FILE` | нет | экспериментальный DSpark draft для Inkling-2-Mini; только greedy, lossless на проверенных промптах |
| `--dspark-k K` | 3 | кандидатов DSpark за раунд, 1..7 |
| `--dspark-batch 0\|1` | 1 | батч-verify DSpark; 0 = последовательный |

`--passes` и `--topk-experts` — **LOSSY**-ручки скорости. `--passes 2`
пропускает третий core-проход, но эпилог всё равно использует слоты
42..47; текст деградирует. `--topk-experts 2` — тот же top-2, что у
`--turbo`, только без Q4_0 dense; `--draft-topk` — его старый алиас.
DSpark требует полные 3 прохода (`--passes 3`, default).

`--dspark` работает только с Inkling-2-Mini, только в single-shot и только
при `--temperature 0`. При другой температуре main печатает warning и
уходит в обычный декод. Измеренные режимы на этом телефоне (warm cache, K=3):

- plain greedy: ~1.65 tok/s (24-токенный код-промпт),
- DSpark `--dspark-batch=0`: ~1.53 tok/s,
- DSpark `--dspark-batch=1`: ~1.13–1.23 tok/s, acceptance 47.6% (mean
  accepted 2.29), шумный и не лосслесс. Батч-режим читает веса MoE по
  top-8 и работает, но draft и verify дорогие; 10 tok/s на текущих
  низкобитных GEMV не выйдет. Устройство и замеры — в [DSPARK.md](DSPARK.md).

```sh
# Inkling-2-Mini + DSpark, жадный декод
scripts/safe_run.sh build/tokshit --temperature 0 \
  --dspark ~/models/dspark-Inkling-2-Mini-Q8_0.gguf --dspark-k 3 \
  ~/models/Inkling-2-Mini-UD-Q2_K_XL.gguf "hi" 24

# teacher-forced проверка первого draft-токена
TOKSHIT_DSPARK_DEBUG=~/models/dspark-Inkling-2-Mini-Q8_0.gguf \
  scripts/safe_run.sh build/tokshit --temperature 0 \
  ~/models/Inkling-2-Mini-UD-Q2_K_XL.gguf "hi" 8
```

### Сэмплинг

| флаг | по умолчанию | что делает |
|---|---|---|
| `--temperature X` | 0.8 | 0 = жадный argmax |
| `--top-k N` | 0 | 0 = без отсечения |
| `--top-p X` | 0.9 | nucleus, 1 = выключено |
| `--seed N` | случайный | зерно `std::mt19937` |
| `--reasoning-budget N` | -1 | бюджет thinking-токенов: N>0 кап, 0 закрыть сразу, -1 без лимита |

`--reasoning-budget` ограничивает блок `<think/>…</think/>`:
`N > 0` — после N сгенерированных токенов внутри блока следующим токеном
принудительно ставится `</think/>`; `0` — блок закрывается сразу после
открывающего токена; `-1` — без лимита, обычное поведение. Бюджет действует
на генерацию, не на prefill, и применяется к каждому ответу отдельно.
Принудительный токен печатается, попадает в KV и учитывается в статистике
как обычный.

### Отладка и бенчмарки

| флаг | по умолчанию | что делает |
|---|---|---|
| `--print-ids` | выкл | id токенов в stderr |
| `--dump-logits FILE` | нет | f32-логиты для паритета |
| `--expert-stats` | выкл | union экспертов и hit-rate |
| `--pool-stats` | выкл | статистика пула |
| `--bench-mm` | выкл | matvec q/k/v/o attention-слоя |
| `--bench-passes N` | 5 | проходов `--bench-mm` |
| `--bench-breakdown` | выкл | разбивка q/k/v/o |
| `--debug-tok` | выкл | отладка токенизатора |
| `--seg-debug` | выкл | печать сегментов matvec |

Булевы флаги не съедают следующий аргумент. Выключение — `--hot=0`,
`--o-direct=0` и т.д. `--` завершает разбор флагов: промпт может начинаться
с дефиса.

## 3. Пресеты

`--turbo` — **LOSSY**-пресет для длинного decode:

```text
--topk-experts 2 --o-direct=1 --cache-mb 1024 --io-threads 8 --drop-cache=0
```

Роутер берёт 2 эксперта (из 6 у Flint, из 8 у Inkling2), плотные веса
переквантовываются в Q4_0, но кэш экспертов и `O_DIRECT` остаются
включёнными. Это не normal top-6/top-8: выход модели отличается.
`--draft-topk 2` — старый алиас `--topk-experts 2`.

`--spec-fast` — отдельный **LOSSY**-пресет:

```text
--o-direct=0 --cache-mb 0 --topk-experts 2 --drop-cache=0
--spec --spec-k 4 --spec-maxb 8 --spec-verify-topk 2
```

Переменные `TOKSHIT_TURBO=1` и `TOKSHIT_SPEC_FAST=1` включают те же
пресеты без флага.

## 4. Переменные окружения

| переменная | по умолчанию | что делает |
|---|---|---|
| `TOKSHIT_MODEL` | вшитый путь | модель |
| `TOKSHIT_PROMPT_FILE` | нет | промпт из файла |
| `GGUF_NLAYERS` | все | ограничить число слоёв |
| `TOKSHIT_EXPERT_CACHE_MB` | 1024 | бюджет кэша экспертов |
| `TOKSHIT_HOT` | вкл | `0` — не копировать плотные веса |
| `TOKSHIT_HOT_CONV` | вкл | `0` — не переквантовывать hot-веса в Q8_0 |
| `TOKSHIT_DROP_CACHE` | вкл | `0` — не сбрасывать page cache |
| `TOKSHIT_DROP_SIBLINGS` | выкл | выкинуть из page cache соседние `.gguf` |
| `TOKSHIT_PREFAULT` | вкл | `0` — demand paging вместо префолта (Inkling2) |
| `TOKSHIT_KV_CTX` | 2048 | KV-позиции Inkling2; `512` экономит ~300 МБ |
| `TOKSHIT_O_DIRECT` | вкл | `0` — буферизованные чтения |
| `TOKSHIT_IO_THREADS` | 4 | фоновые читатели |
| `TOKSHIT_PREFETCH` | выкл | предсказание экспертов |
| `TOKSHIT_MMAP_WILLNEED` | выкл | `madvise(WILLNEED)` |
| `TOKSHIT_FADV_DONTNEED` | выкл | дропать page cache экспертов |
| `TOKSHIT_FADV_RANDOM` | выкл | `POSIX_FADV_RANDOM` |
| `TOKSHIT_AHB` | вкл | `0` — полностью выключить AHB |
| `TOKSHIT_AHB_MAX_MB` | 5120 | общий бюджет AHB, МиБ |
| `TOKSHIT_AHB_SLAB_MB` | 16 | размер слэба экспертного AHB-пула, МиБ |
| `TOKSHIT_AHB_USAGE` | `rare` | `rare` или `often` |
| `TOKSHIT_AHB_DEBUG` | выкл | печать аллокаций AHB |
| `TOKSHIT_POOL_WORKERS` | ядра-1 | воркеры пула |
| `TOKSHIT_POOL_PIN` | выкл | пиннинг воркеров; сортировка CPU по max frequency |
| `TOKSHIT_POOL_RT` | выкл | realtime-приоритет |
| `TOKSHIT_POOL_STATS` | выкл | статистика пула |
| `TOKSHIT_MAIN_CPU` | самое быстрое ядро | CPU главного потока при `--pool-pin` |
| `TOKSHIT_MLOCK` | выкл | зарезервировано, не подключено |
| `TOKSHIT_MLOCK_DENSE_MB` | 0 | mlock плотных q/k/v/o/gate/out, МБ |
| `TOKSHIT_EXPERT_STATS` | выкл | union и hit-rate |
| `TOKSHIT_EXPERT_TRACE` | нет | файл трассы роутинга: слой и top-k на каждый проход |
| `TOKSHIT_DRAFT_TOPK` | 0 | **LOSSY**: экспертов на токен; 0 = как в модели (у Inkling2 — 8) |
| `TOKSHIT_PIN_EXPERTS` | 0 | пиннинг частых экспертов |
| `TOKSHIT_FREQ_PERSIST` | выкл | писать частотный профиль после генерации |
| `TOKSHIT_FREQ_FILE` | авто | путь профиля (по умолчанию `$TMPDIR/tokshit-freq-<модель>.bin`) |
| `TOKSHIT_PREFETCH_IDLE` | выкл | prefetch только при простое диска |
| `TOKSHIT_IO_MERGE` | выкл | слитые pread смежных экспертов |
| `TOKSHIT_I8MM` | выкл | SMMLA-путь для Q8_0-матвеков (нужна фича i8mm; экспериментальный) |
| `TOKSHIT_IQ2ROW` | выкл | двухстрочные IQ-ядра (экспериментальный) |
| `TOKSHIT_KQK` | выкл | Q4_K NEON-ядро (экспериментальный) |
| `TOKSHIT_PREFILL_BATCH` | 32 | размер чанка батч-префилла; 0 — по одному токену |
| `TOKSHIT_PREFILL_BULK` | выкл | sequential-чтение региона слоя на батч-префилле |
| `TOKSHIT_HOT_SEG_MB` | 32 | размер сегмента hot-весов, МиБ |
| `TOKSHIT_HOT_Q4` | выкл | переквантовать dense в Q4_0 (turbo) |
| `TOKSHIT_PIN_EVERY` | нет | повторять пиннинг каждые N токенов |
| `TOKSHIT_SPEC` | выкл | спекулятивный цикл |
| `TOKSHIT_SPEC_K` | 4 | черновиков за раунд |
| `TOKSHIT_SPEC_VERIFY_TOPK` | 0 | lossy verify top-k |
| `TOKSHIT_SPEC_CTX` | 2048 | ёмкость KV |
| `TOKSHIT_SPEC_MAXB` | 8 | размер verify-батча |
| `TOKSHIT_SPEC_DEBUG` | выкл | раунды спекуляции: кандидаты, argmax verify, отказы |
| `TOKSHIT_INK_PASSES` | 3 | **LOSSY**: core-проходов Inkling2, 1..3 |
| `TOKSHIT_DSPARK` | нет | DSpark draft GGUF для Inkling-2-Mini |
| `TOKSHIT_DSPARK_K` | 3 | кандидатов DSpark за раунд, 1..7 |
| `TOKSHIT_DSPARK_BATCH` | 1 | батч-verify; 0 = последовательный |
| `TOKSHIT_DSPARK_DEBUG` | нет | teacher-forced проверка первого draft-токена |
| `TOKSHIT_DSPARK_TRACE` | выкл | кандидаты DSpark по шагам |
| `TOKSHIT_DSPARK_PROF` | выкл | тайминги draft-блока, LM head и головы |
| `TOKSHIT_INK_BATCH_PROF` | выкл | разбивка `[batch prof]` (dense/moe/attn/conv/head) на батч |
| `TOKSHIT_INK_BATCH_DUMP_DIR` | нет | дамп различных танцев |
| `TOKSHIT_INK_BATCH_TEST` | нет | паритет батч-логитов и тапов с последовательным проходом |
| `TOKSHIT_MLOCK_DENSE_MB` | 0 | best-effort mlock dense-весов; на Android часто отказывается |
| `TOKSHIT_TEMPERATURE` | 0.8 | температура |
| `TOKSHIT_TOP_K` | 0 | top-k |
| `TOKSHIT_TOP_P` | 0.9 | top-p |
| `TOKSHIT_SEED` | случайный | сид |
| `TOKSHIT_REASONING_BUDGET` | -1 | бюджет thinking-токенов |
| `TOKSHIT_PROF` | выкл | обнулить метрики после префилла |
| `TOKSHIT_DUMP_LOGITS` | нет | файл логитов |
| `TOKSHIT_PRINT_IDS` | нет | id в stderr |
| `TOKSHIT_VERIFY_IDS` | нет | teacher-forced verify |
| `TOKSHIT_DEBUG_TOK` | выкл | отладка токенизатора |
| `TOKSHIT_SEG_DEBUG` | выкл | отладка сегментов |
| `TOKSHIT_SEG_DEBUG_ALL` | выкл | отлаживать все сегменты, не только первые |
| `TOKSHIT_LAYER_DEBUG` | выкл | тайминги слоя и экспертов |
| `TOKSHIT_TOKEN_DEBUG` | выкл | `[tok]`-строка: `w` (МБ весов на токен), `minf/majf`, RSS, hit/miss |
| `TOKSHIT_DEBUG_NRUN` | выкл | печатать число выполненных слоёв |
| `TOKSHIT_POOL_WAIT_STATS` | выкл | статистика барьера пула |
| `TOKSHIT_BENCH_MM` | выкл | matvec-бенчмарк |
| `TOKSHIT_BENCH_PASSES` | 5 | проходов бенчмарка |
| `TOKSHIT_BENCH_BREAKDOWN` | выкл | разбивка q/k/v/o |
| `TOKSHIT_MIN_AVAIL_MB` | 1200 | порог `safe_run.sh` |
| `GGUF_TIME` | выкл | таймеры фаз |
| `TOKSHIT_SPEC_FAST` | выкл | LOSSY-пресет |
| `TOKSHIT_TURBO` | выкл | LOSSY-пресет |

## 5. Интерактивный чат

Запуск без промпта открывает REPL. В TTY есть цветные роли, потоковая
выдача и строка статуса; в пайпе — обычный построчный режим без ANSI.

Во время работы терминал показывает два живых индикатора (только когда
stderr — TTY; в пайпе тишина):

- во время префилла: `предобработка 12/48 ток · 25% · 1.3s` — одна строка,
  обновляется по `\r`, по завершении стирается;
- во время декода: `◜ генерирую · 12 ток · 2.3 tok/s · кэш 64% · свб 3120 MB`
  — спиннер тикает по времени, hit-rate кэша считается за текущий ответ,
  `свб` — `MemAvailable`; на узком экране хвост строки отрезается, чтобы
  перерисовка не ломалась (при ширине < 40 остаётся счётчик и скорость).

| команда | действие |
|---|---|
| `/help` | справка по командам |
| `/exit`, `/quit` | выйти (или Ctrl-D) |
| `/clear` | очистить KV и начать заново; в TTY чистит и экран |
| `/stats` | дашборд: скорость, кэш, память, сессия, параметры |
| `/reset` | вернуть сэмплинг к умолчаниям |
| `/temp <x>` | температура, 0 = жадный |
| `/topk <n>` | top-k, 0 = выключено |
| `/topp <x>` | top-p, 1 = выключено |
| `/reasoning-budget N` | thinking-токенов: N>0 кап, 0 закрыть сразу, -1 без лимита |
| `/system <текст>` | системная инструкция для следующих сообщений |
| `/save <файл>` | дописать диалог в файл |

`Ctrl-C` прерывает ответ, но не выходит из чата. `Ctrl-D` — аккуратный
выход. Пустая строка ничего не делает. `/clear` сбрасывает и utf8-буфер:
он живёт один ход, так что после очистки контекста остатков не остаётся.

`/stats` печатает дашборд секциями (скорость / кэш экспертов / память /
сессия / сэмплинг); в TTY hit-rate кэша подсвечен цветом: зелёный ≥60%,
жёлтый 30–60%, красный <30%. Ошибки аргументов (`/temp abc`) печатают
подсказку и текущее значение, сессия продолжается.

Пример дашборда:

```text
╭─ СТАТИСТИКА СЕССИИ ────────────────────────────
│ скорость
│   последняя: 2.3 tok/s  ·  средняя: 2.1 tok/s
│   prefill 6.81s  ·  сгенерировано 214 токенов всего
│ кэш экспертов
│   кэш 61.4% (hit 3210 / miss 2018)  ·  union 5.87/проход
│ память
│   RSS 6214 MB  ·  MemAvailable 3120 MB  ·  кэш 512 MB
│ сессия
│   5 ходов  ·  контекст 431  ·  spec выкл (в чате не используется)
│ сэмплинг
│   temp 0.80  ·  top-k 0  ·  top-p 0.90  ·  reasoning-budget: -1
│   системная инструкция: нет
╰────────────────────────────────────────────────
```

Пример сессии:

```text
вы › объясни, что такое MoE
токшит: ...
вы › /temp 0.2
температура: 0.20
вы › /save chat.txt
диалог дописан в chat.txt (1234 байт)
вы › /exit
```

## 6. Примеры

```sh
# обычный декод с кэшем 512 МБ
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  --expert-stats ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# без кэша, mmap
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 0 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# мало памяти
scripts/safe_run.sh build/tokshit --hot=0 --cache-mb 256 \
  --io-threads 1 --temperature 0 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 8

# LOSSY: быстрый Inkling-2-Mini — 2 прохода, top-2, маленький KV,
# demand paging и без соседних GGUF в page cache. Текст деградирует.
scripts/safe_run.sh build/tokshit --temperature 0 \
  --passes 2 --topk-experts 2 --kv-ctx 512 --prefault=0 --drop-siblings \
  ~/models/Inkling-2-Mini-UD-Q2_K_XL.gguf "hi" 24

# профиль фаз и I/O
GGUF_TIME=1 scripts/safe_run.sh build/tokshit --temperature 0 \
  --cache-mb 512 --expert-stats ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# спекуляция, полный verify (lossless при greedy)
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  --spec --spec-k 3 ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# Inkling-2-Mini + экспериментальный DSpark
scripts/safe_run.sh build/tokshit --temperature 0 \
  --dspark ~/models/dspark-Inkling-2-Mini-Q8_0.gguf --dspark-k 3 \
  ~/models/Inkling-2-Mini-UD-Q2_K_XL.gguf "hi" 24

# LOSSY turbo: top-2 + Q4_0 dense, с кэшем и O_DIRECT
scripts/safe_run.sh build/tokshit --turbo --temperature 0 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 24
```

## 7. Безопасный запуск

```sh
TOKSHIT_MIN_AVAIL_MB=1400 TOKSHIT_EXPERT_CACHE_MB=512 \
  scripts/safe_run.sh build/tokshit --temperature 0 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 8
```

Правила:

- один процесс модели за раз;
- только через `scripts/safe_run.sh`;
- `--cache-mb` не больше 4096;
- перед замером `MemAvailable` > 4000 МБ;
- для Inkling2 при нехватке памяти: `--kv-ctx 512 --prefault=0 --drop-siblings`;
- не включать `TOKSHIT_AHB_USAGE=often` без замера;
- сравнивать конфиги через `scripts/ab.sh`, а не одиночными прогонами.

## 8. Диагностика

| симптом | что смотреть |
|---|---|
| медленно | `GGUF_TIME=1`, `TOKSHIT_TOKEN_DEBUG=1` (`w`, `minf/majf`, RSS), `--expert-stats`, температура |
| мало памяти / page cache thrash | prefault WARNING про RSS, `--drop-siblings`, `--kv-ctx 512`, `--prefault=0` |
| мало памяти | `--cache-mb 256`, `--hot=0`, `TOKSHIT_AHB_DEBUG=1` |
| AHB не выдаётся | `TOKSHIT_AHB_DEBUG=1`, затем `TOKSHIT_AHB=0` |
| спекуляция не ускоряет | `[spec: ...]`, acceptance, `expert_hit` |
| DSpark не ускоряет | `[dspark: ...]`, acceptance, `TOKSHIT_DSPARK_PROF=1`; см. [PERF_SESSION3.md](PERF_SESSION3.md) |
| текст отличается | `--temperature 0`, `--print-ids`, `--dump-logits` |

Метрики и их смысл — в [RUNTIME.md](RUNTIME.md). Замеры — в
[PERFORMANCE.ru.md](PERFORMANCE.ru.md).
