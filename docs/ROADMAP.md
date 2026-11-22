# Roadmap и инженерные заметки

`tokshit` — специализированный C++20 ARM64-рантайм для MoE-моделей больше
оперативной памяти. Горячий путь фиксирован под Snapdragon: NEON
dot-product, AHB для плотных весов, `O_DIRECT` для экспертов, послойный
LRU-кэш.

## 1. Текущий статус

| параметр | значение |
|---|---|
| обычный декод top-6, 8 токенов, 47 °C | **2.40 tok/s** |
| обычный декод top-6, 16 токенов, 60 °C | 0.96 tok/s |
| `--turbo` (top-2 + Q4_0) | 3.52 tok/s, **LOSSY** |
| батчевый префилл, T=224 | **4.53 tok/s** |
| prefill-bulk, T=224 | 2.32 tok/s |
| спекуляция, полный verify | 0.93–1.0 tok/s, acceptance 50–78 % |
| Inkling-2-Mini, plain greedy | 1.61–1.72 tok/s (2026-11-22, шумно) |
| Inkling-2-Mini, top-2 plain | 2.28 tok/s (2026-11-22) |
| Inkling-2-Mini top-2/passes=2 | ≈5.2 tok/s, одиночный прогон, **LOSSY** |
| Inkling-2-Mini, `--turbo` | 0.26–0.30 tok/s (63–74 °C, 2026-11-22) |
| DSpark, sequential verify (Inkling-2-Mini) | 0.72 против plain 1.61–1.72 (2026-11-22) |
| DSpark, batched verify (Inkling-2-Mini) | 0.52 tok/s, acceptance 47.6 % (2026-11-22) |
| hot-веса | 52 сегмента по 32 МБ |
| поток hot-весов через AHB | 5.2 ГБ/с |
| тесты | 186 кейсов / 9951 проверка |

Узкое место Flint — суммарные байты токена: 1.74 ГБ dense + 0.377 ГБ экспертов.
На 47 °C dense идёт 12.6 ГБ/с, на 60 °C — 4.2 ГБ/с. Цель 15 tok/s требует
31.8 ГБ/с устойчиво; измеренный потолок ниже. Подробности и методика —
в [PERFORMANCE.ru.md](PERFORMANCE.ru.md).

## 2. Ограничения

- Две модели: Flint-40B-A2.5B (Q3_K_XL, top-6) и Inkling-2-Mini
  (UD-Q2_K_XL, 24 физических блока / 48 слотов, top-8).
- Одно устройство: Snapdragon 8s Gen 4, 15.2 ГБ RAM, UFS.
- Один процесс модели. Все замеры через `scripts/safe_run.sh` и
  `scripts/device_lock.sh`.
- `normal` — top-6 у Flint, top-8 у Inkling2, lossless. `--turbo`,
  `--spec-fast`, `--topk-experts`, `--passes 1/2` и `--spec-verify-topk`
  помечаются как LOSSY.
- `.gguf` открывается только на чтение.

## 3. Ближайшие задачи

### 3.1 Поток плотных весов

- слить последовательные matvec без лишних барьеров пула;
- укрупнить row-чанки задач: per-op накладные расходы съедают полный проход;
- проверить Q6_K/Q5_K NEON-ядра вместо конверсии в Q8_0: 1.36 ГБ вместо
  1.74 ГБ плотных, минус 22 % полосы;
- измерить `-flto` и `-O3` с `-mcpu` под конкретные ядра;
- f16 KV на декоде: минус 50 % полосы attention на длинном контексте.

### 3.2 Кэш экспертов

- cross-layer gate prediction: гейт слоя i+1 по hidden слоя i;
- shallow-favoring кэш: слои 0–3 целиком, eviction ARC/Least-Stale вместо
  чистого LRU;
- sidecar-layout: gate|up|down одного эксперта слитно, один pread около
  2 МБ вместо трёх мелких;
- проверить `--pin-experts` и `TOKSHIT_PIN_EVERY` на длинном прогоне.

### 3.3 Prefill

- bulk-streaming: стриминг слой-блоков около 333 МБ последовательно плюс
  батч-GEMM 8×8;
- union-GEMM по экспертам: строка веса деквантуется один раз на чанк;
- считать prefill-попадания в метриках.

### 3.4 Спекуляция

Полный verify lossless, но расширяет union экспертов и проигрывает
обычному декоду. DSpark для Inkling-2-Mini: greedy-ids совпадают с plain
greedy; sequential 0.72, batched 0.52 (2026-11-22, 16 токенов, warm,
acceptance 47.6 %). Выигрыша над plain нет. Остаток:

- batched MoE по union-экспертов; сейчас токенами;
- acceptance и MoE-union на батче: нужна высокая цепочка, иначе экономии нет;
- draft с кэшем горячих экспертов, отдельным от verify;
- verify top-k как LOSSY-компромисс;
- откат KV без повторной записи принятых строк.

### 3.5 Сравнение с универсальными движками

Одинаковый GGUF, промпт, `temperature 0`, число потоков и кэш. Сравнивать
tok/s и качество (id/логиты). Цифры должны воспроизводиться.

## 4. Идеи вне текущего результата

- CECAR-подобное смещение роутинга в сторону закэшированных экспертов.
  Меняет выход модели, поэтому только LOSSY-путь.
- Меньше ядер ради энергии: декод memory-bound, те же tok/s на 3 ядрах
  дадут меньше тепла и троттлинга.
- Q2-формат для экспериментов.
- `TOKN`-секция для совместимости.

## 5. Карта компонентов

| компонент | файлы | ответственность |
|---|---|---|
| GGUF и mmap | `src/gguf.cpp`, `src/mmap.cpp` | метаданные, тензоры, отображение |
| Кванты | `src/quant.cpp`, `src/dequant.cpp` | коды, block spec, деквант |
| Ядра | `src/compute/` | NEON dotprod, скалярный fallback, батч |
| Модель | `src/model_*.cpp` | загрузка, forward, prefill, эксперты |
| Inkling-2-Mini | `src/inkling2_*.cpp` | loop-слоты, conv/depth-mix, тапы |
| DSpark | `src/dspark_*.cpp`, `include/tokshit/dspark.hpp` | draft `dflash`, инъекция, голова |
| KV и внимание | `src/kv.cpp`, `src/compute/attn_batch.cpp` | кэш, causal attention |
| Кэш экспертов и I/O | `src/expert_io.cpp` | LRU, demand/prefetch, `O_DIRECT` |
| AHB | `src/ahb.cpp` | pinned dma-buf, бюджет, hot-веса |
| Пул потоков | `src/pool.cpp` | воркеры, барьеры, задачи |
| CLI и TUI | `src/cli*.cpp`, `src/tui_*.cpp` | флаги, чат, статус |
| Спекуляция | `src/spec_loop.cpp` | draft, verify, откат KV |

## 6. Риски

| риск | мера |
|---|---|
| ошибка в NEON-ядре | тест против скаляра до интеграции, `vcheck` |
| нет `FEAT_DotProd` | runtime-детект `HWCAP_ASIMDDP`, скалярный fallback |
| разный `exp`/softmax | без `-ffast-math`, сравнение логитов |
| расхождение IQ-декванта | тесты на известных блоках |
| Android-ограничения | best-effort `madvise`, `safe_run.sh` |
| OOM на 15.2 ГБ | кэш 512 МБ, один процесс, AHB ≤ 4096 МБ |
