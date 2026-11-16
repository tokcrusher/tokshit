# бенчмарк-харнесс (ru)

`scripts/bench_matrix.sh` — единый способ мерить decode, prefill и A/B на
целевом устройстве. Он фиксирует одни и те же правила для всех агентов:
`device_lock`, `safe_run`, проверка памяти, таймаут, повторы, медиана,
температура до/после, JSON/CSV и regression gate.

`./bench.sh "<флаги>" [max_new] [runs]` — совместимая обёртка над ним.

## быстрый старт

```sh
# обычный decode: 3 повтора по 24 токена, кэш 512 МБ, io4/pool6
scripts/bench_matrix.sh

# короткий smoke, 4 токена, 1 повтор
scripts/bench_matrix.sh --tokens 4 --runs 1 --cooldown 0

# prefill 128 и 512 токенов, max_new=1
scripts/bench_matrix.sh --mode prefill --prompt-tokens 128,512

# decode + prefill
scripts/bench_matrix.sh --mode both --prompt-tokens 128

# A/B: первый конфиг — reference, порядок чередуется по раундам
scripts/bench_matrix.sh --runs 3 --tokens 24 \
  --config 'base|--cache-mb 512' \
  --config 'pin8|--cache-mb 512 --pin-experts 8' \
  --config 'io8|--cache-mb 512 --io-threads 8'
```

Перед запуском сборки, тестов и замеров берите `device_lock`:

```sh
TOKSHIT_DEVICE_LOCK_WAIT=3600 scripts/device_lock.sh \
  sh -c 'cmake --preset release && cmake --build build --parallel 2'
```

Сам харнесс тоже держит `tokshit-device.lock` на всё время матрицы, поэтому
вложенный `device_lock.sh` не нужен.

## что измеряется

| режим | промпт | max_new | что парсится |
|---|---|---|---|
| `decode` | `--prompt` (по умолчанию `hi`) | `--tokens` | `[prefill: ...]`, `[gen-only]`, `[matvec]` |
| `prefill` | `--prompt-file`, цель `--prompt-tokens` | 1 | `[prefill: N tok за Xs ...]` |
| `both` | оба | — | оба набора |

- **decode tok/s** — поле `= X tok/s` из строки `[prefill: ... | генерация:
  ... = X tok/s]`; это чистая генерация, без prefill и загрузки модели.
- **prefill tok/s** — `N / Xs` из `[prefill: N tok за Xs ...]`. Время
  печатается с точностью 0.1 с; на T=128 этого достаточно, на очень коротком
  промпте относительная ошибка растёт.
- **hit %** — `rate=` из `[gen-only]`: попадания в expert-кэш только за
  генерацию, prefill исключён.
- **dense GB/s** — `[matvec: ... = X GB/s]` при `GGUF_TIME=1`. Счётчики
  обнуляются после prefill (`TOKSHIT_PROF=1`), поэтому это decode-профиль.
- **mem / temp** — `MemAvailable` и `$TOKSHIT_TEMP_ZONE` до и после каждого
  прогона; `[guard]` даёт пиковый RSS и минимум `MemAvailable`.
- **prompt_tokens** — фактическое число токенов после chat-обёртки. Цель
  `--prompt-tokens` приблизительная: файл собирается из английских слов,
  BPE может дать другое число. В отчёте всегда фактическое значение.

Загрузка модели (`loaded in ...`) и prefill в decode-число не входят.

## параметры

| флаг | по умолчанию | смысл |
|---|---:|---|
| `--mode decode\|prefill\|both` | `decode` | что мерить |
| `--runs N` | `3` | повторов на конфиг |
| `--tokens N` | `24` | новых токенов в decode |
| `--cooldown SEC` | `15` | пауза между прогонами |
| `--cache-mb N` | `512` | expert-кэш; `> 4096` запрещено |
| `--io-threads N` | `4` | потоки чтения экспертов |
| `--pool-workers N` | `6` | потоки пула |
| `--flags "..."` | — | общие дополнительные флаги |
| `--config 'name\|flags'` | `base` | конфиг A/B; можно повторять |
| `--prompt TEXT` | `hi` | короткий decode-промпт |
| `--prompt-tokens N[,N]` | `128` | цели prefill |
| `--timeout SEC` | `1200` | timeout одного прогона |
| `--min-start-mb N` | `4000` | минимум `MemAvailable` перед прогоном |
| `--min-avail-mb N` | `1200` | порог watchdog `safe_run.sh` |
| `--out DIR` | `$TMPDIR/tokshit-bench` | JSON/CSV/логи |
| `--json FILE`, `--csv FILE` | авто | явные пути отчётов |
| `--no-logs` | выкл | не писать сырые логи |
| `--gate` | выкл | exit 3 при регрессе |
| `--gate-pct N` | `10` | порог регресса, % |
| `--baseline-decode N` | `1.1` | baseline decode tok/s |
| `--baseline-dense N` | `12.55` | baseline dense GB/s |
| `--baseline-hit N` | `27.5` | baseline hit % |
| `--lock-wait SEC` | `3600` | ожидание device-lock |
| `--busy-wait SEC` | `120` | ожидание чужого `tokshit` |
| `--mem-wait SEC` | `120` | ожидание памяти |

Флаги конфига дописываются после общих, поэтому могут переопределять
`--cache-mb`, `--io-threads`, `--pool-workers`. Значения флагов разделяются
пробелами; кавычки внутри строки не поддерживаются.

## методика

1. Проверяется, что модель read-only (`scripts/model_guard.sh`), бинарь и
   `safe_run.sh` на месте, `cache <= 4096`.
2. Берётся `tokshit-device.lock`; без него не запускается ни один прогон.
3. Перед каждым прогоном: `pgrep -x tokshit` пуст, `MemAvailable >=
   --min-start-mb`, температура фиксируется.
4. Прогон идёт через `scripts/safe_run.sh` с `timeout -k 15`; при падении
   `MemAvailable` ниже `--min-avail-mb` watchdog убивает процесс.
5. Один и тот же конфиг повторяется `--runs` раз. Сводка — медиана, минимум
   и максимум; медиана устойчивее к троттлингу, чем среднее.
6. Конфиги в A/B чередуются: в раунде 1 порядок `A B C`, в раунде 2 —
   `B C A`, в раунде 3 — `C A B`. Так каждый конфиг получает и холодный, и
   тёплый участок сессии.
7. Между прогонами пауза `--cooldown`. Температура и память пишутся до и
   после, поэтому можно отличить «стало быстрее» от «стало холоднее».

A/B-дельты считаются по медианам относительно **первого** `--config`:
decode/prefill/dense — в процентах, hit — в процентных пунктах.

## regression gate

`--gate` сравнивает медиану первого конфига с baseline и возвращает exit 3,
если значение ниже `baseline * (1 - gate_pct/100)`:

| метрика | baseline | порог при 10% |
|---|---:|---:|
| decode | 1.1 tok/s | 0.99 tok/s |
| dense | 12.55 GB/s | 11.30 GB/s |
| hit | 27.5 % | 24.75 % |

По умолчанию gate **выключен**: харнесс только печатает статус, exit 0.
Это важно для экспериментов: новый конфиг может быть медленнее baseline, но
должен спокойно дойти до отчёта. Для CI/приёмки добавьте `--gate`.

Gate применяется только к reference-конфигу (первому), а не ко всем A/B
вариантам. Метрика, которой нет в прогоне (например dense в `--mode
prefill`), пропускается. Если decode-данных нет в decode/both-режиме, gate
падает.

Baseline — историческая рабочая точка: короткие прогоны (< 16 токенов) часто
дают hit ниже 27.5 %, поэтому для gate используйте `--tokens 24` или больше.

## выход

В `--out` (по умолчанию `$TMPDIR/tokshit-bench`) пишутся:

- `bench-<mode>-<stamp>.json` — `schema: tokshit.bench.v1`, параметры,
  summary по конфигам, A/B-дельты, gate и все сырые прогоны;
- `bench-<mode>-<stamp>.csv` — те же сырые прогоны таблицей;
- `logs-<mode>-<stamp>/` — stdout+stderr каждого прогона.

Сырой прогон содержит: `suite, config, round, order, prompt_target, rc,
decode_tps, prefill_tps, prompt_tokens, prefill_s, gen_tokens, gen_s,
hit_rate, dense_gbps, io_gb, rss_mb, guard_min_avail_mb, temp_before_c,
temp_after_c, mem_before_mb, mem_after_mb, duration_s, load_s, flags, log,
timestamp`.

Человекочитаемый вывод — таблица прогонов, медианы `[min..max]`, A/B-дельты
и gate. `json`/`csv` пути печатаются в конце.

## коды выхода

| код | значение |
|---:|---|
| 0 | всё прошло; gate выключен или пройден |
| 2 | ошибка аргументов/окружения |
| 3 | gate включён и регресс > `--gate-pct` |
| 4 | небезопасно: чужой `tokshit` или мало `MemAvailable` |
| 75 | не дождались device-lock |
| 124 | отдельный прогон упёрся в timeout (матрица продолжается) |

## ограничения

- Один процесс модели. Харнесс ждёт освобождения lock и `pgrep`, но
  параллельный запуск в обход lock всё равно испортит цифры.
- `prefill` с целью 512 даёт ~515 токенов после chat-обёртки; это край
  sliding-window (513). Для проверки длинного контекста смотрите
  предупреждения в [ROADMAP.md](ROADMAP.md).
- Время prefill печатается с точностью 0.1 с. Для T=32 и меньше ошибка
  заметна; для T=128/512 — нет.
- `GGUF_TIME=1` включает per-op таймеры. Накладные расходы малы по сравнению
  с 0.3–0.5 с на токен, но для абсолютного рекорда можно убрать их отдельным
  прогоном без `GGUF_TIME`.
- Page cache и AHB прогреваются от прогона к прогону; медиана из 3 повторов
  это учитывает, но «холодный» и «тёплый» режимы — разные измерения.
- `--fast`, `--spec-*`, `--draft-topk` — lossy-режимы. Харнесс не запрещает
  их, но в отчёте они должны быть помечены **LOSSY** и не сравниваться с
  обычным top-6 как с равным.

Связанные документы: [PERFORMANCE.ru.md](PERFORMANCE.ru.md) — измеренные
цифры; [RUNTIME.md](RUNTIME.md) — что происходит внутри прогона;
[KERNELS.md](KERNELS.md) — микробенчмарки ядер.
