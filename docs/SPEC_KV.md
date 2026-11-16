# KV-кеш, батч-матвек и спекулятивный цикл
# KV cache, batched matvec and the speculative loop

Коротко о том, зачем появились `kv.hpp`, `batch.hpp`, `attn_batch.hpp` и
`spec_loop.hpp`, как они устроены и что уже проверено.
A short note on why `kv.hpp`, `batch.hpp`, `attn_batch.hpp` and `spec_loop.hpp`
exist, how they work and what is already verified.

## 1. Зачем / Why

Одиночный декод упирается в плотные веса: ~1.55 ГБ/токен ≈ 0.19 с, то есть
~5 tok/s даже при идеальном кэше. Чтобы подойти к 7 tok/s, нужно читать веса
реже, чем раз на токен, — то есть батчить и спекулировать. Батчинг даёт
смысл только с полноценным KV-кешем фиксированной ёмкости.
Single-token decode is bound by the dense weights: ~1.55 GB/token ≈ 0.19 s,
i.e. ~5 tok/s even with a perfect cache. To approach 7 tok/s the weights must
be read less often than once per token, i.e. batching plus speculation.
Batching only makes sense with a real fixed-capacity KV cache.

## 2. KV-кеш / KV cache

`include/tokshit/kv.hpp`, `src/kv.cpp`.

- два сплошных сляба `k` и `v`, слой `l` начинается с `l * cap * stride`;
  two flat slabs `k`/`v`, layer `l` starts at `l * cap * stride`;
- `stride = kv_dim * (f16 ? 2 : 4)` байт; `kv_dim = n_kv_heads * head_dim`
  (у флинта 4 × 128 = 512);
- позиции абсолютные, строки перезаписываемые;
  absolute positions, rewritable rows;
- `tokens(l)` — high-water mark, `truncate(l, n)` / `truncate_all(n)`
  откатывают кэш без пересылок;
  `tokens(l)` is a high-water mark; `truncate*` roll back without data moves;
- f16 по умолчанию: 2К контекста ≈ 49 × 2048 × 512 × 4 байта ≈ 195 МБ
  (k+v вместе), f32 вдвое дороже.
  f16 by default: 2k context ≈ 195 MB for k+v, f32 doubles it.

Почему f16: attention читает весь K/V на каждый токен, и полоса важнее
точности — f16 вдвое дешевле f32 и достаточно точен для softmax.
Why f16: attention streams all K/V per token, bandwidth beats precision here,
and f16 is accurate enough for the softmax.

## 3. Батч-матвек / Batched matvec

`include/tokshit/batch.hpp`, `src/compute/kernels_batch.cpp`.

`BatchSeg` — это прямоугольник весов (`rows × in_dim`) плюс `b` активаций.
`batchlib::run_bsegs` режет сегменты на задачи «пачка строк × все b
активаций» и кидает их в общий пул. Строка весов при этом остаётся в L1 на
время b проходов, поэтому DRAM-трафик на выход делится примерно на b.
`BatchSeg` is a weight rectangle (`rows × in_dim`) plus `b` activations.
`batchlib::run_bsegs` flattens segments into “chunk of rows × all b
activations” tasks for the shared pool. A weight row stays in L1 for the b
passes, so DRAM traffic per output divides by roughly b.

Первый вариант переиспользует `kernels::dot_row` (на aarch64 — dotprod
варианты), поэтому семантика совпадает со скалярным путём байт в байт, а
быстрые batched-ядра можно добавлять позже без изменения API.
The first version reuses `kernels::dot_row` (dotprod on aarch64), so the
semantics match the scalar path exactly; faster batched kernels can be added
later without changing the API.

## 4. Attention батчем / Batched attention

`include/tokshit/attn_batch.hpp`, `src/compute/attn_batch.cpp`.

`attend_row` считает одну строку батча: scores по `[start, end)`, softmax с
sink (sink только в знаменателе, в V-сумму не входит), взвешенная сумма V.
Обход вложен по kv-головам: K/V-сегмент читается один раз на группу из
`group = n_heads / n_kvh` q-голов. На 2К это главная экономия полосы.
`attend_row` handles one batch row: scores over `[start, end)`, softmax with
the sink (sink only in the denominator), weighted V sum. The loop nests by
kv head: a K/V segment is read once for `group = n_heads / n_kvh` q heads.
That is the main bandwidth saving at 2k.

- `first` — абсолютная позиция строки 0; строка `j` видит `[.., first+j]`;
  `first` is row 0's absolute position; row `j` sees `[.., first+j]`;
- `win[j]` — начало sliding-window; `cap` не даёт выйти за сляб;
  `win[j]` is the sliding-window start; `cap` clamps the end;
- f16 и f32 сравниваются с наивной скалярной
  ссылкой, включая sink и окно.
  f16 is checked against a naive scalar reference;erence,
  including the sink and the window.

## 5. Спекуляция / Speculation

`include/tokshit/spec_loop.hpp`, `src/spec_loop.cpp`.

Раунд: `draft → verify → accept → truncate`.

1. Черновик: первый токен берём из головы прошлого раунда, следующие `K-1`
   генерит батч из одного токена с урезанным роутером
   (`TOKSHIT_DRAFT_TOPK`). Это дешевле по чтению экспертов.
   Draft: the first token comes from the last round's head; the next `K-1`
   are produced one-token-at-a-time with a reduced router
   (`TOKSHIT_DRAFT_TOPK`), cheaper in expert reads.
2. Verify: те же кандидаты прогоняются батчем с полным роутером; KV
   перезаписывает их строки.
   Verify: the same candidates run as a batch with the full router; KV
   rewrites their rows.
3. Цепочка: `argmax(row j)` должен совпасть с `cand[j+1]`; принят минимум
   один токен. Логиты последнего принятого — голова следующего раунда.
   Chain: `argmax(row j)` must equal `cand[j+1]`; at least one token is
   accepted. The last accepted row seeds the next round's head.
4. `kv.truncate_all(pos)` выбрасывает непринятый хвост.
   `kv.truncate_all(pos)` drops the unaccepted tail.

Только жадный семплинг: при greedy выход обязан совпадать с обычным
декодом, иначе это не lossless-спекуляция. Env-ручки: `TOKSHIT_SPEC`,
`TOKSHIT_SPEC_K`, `TOKSHIT_DRAFT_TOPK`, `TOKSHIT_SPEC_VERIFY_TOPK`,
`TOKSHIT_SPEC_CTX`, `TOKSHIT_SPEC_MAXB`.
Greedy sampling only: with greedy the output must match plain decode, or it
is not lossless speculation. Env knobs: `TOKSHIT_SPEC`, `TOKSHIT_SPEC_K`,
`TOKSHIT_DRAFT_TOPK`, `TOKSHIT_SPEC_VERIFY_TOPK`, `TOKSHIT_SPEC_CTX`,
`TOKSHIT_SPEC_MAXB`.

## 6. Что проверено / What is verified

- KV: capacity, high-water, rollback, f16/f32 roundtrip;
- attention: f16/f32 vs the scalar reference, sink and SWA;
- деквант Q8_0 плюс `dot_row_multi` на нескольких активациях;
- batched matvec vs `dot_row` for F32 and Q8_0, plus `dot_row_multi` over
  several activations;
- полный набор 161 кейс / 7716 проверок зелёный локально; на устройстве
  проверяется при следующем деплое.
  the full 161 cases / 7716 assertions are green locally; the device run
  happens on the next deploy.
- паритет на устройстве: greedy-иды `--spec` и обычного декода совпали
  байт в байт (`[ids: 150006 27177 11 269 1464 950]`).
  on-device parity: greedy ids from `--spec` and plain decode matched exactly.

### Замер на устройстве (короткий прогон, ctx 512)

| режим | генерация | принято |
|---|---|---|
| plain greedy, top-6 | 1.49 tok/s | — |
| `--spec --spec-k 2 --draft-topk 2` | 0.98 tok/s | 75 % черновиков |
| plain greedy, `--draft-topk 2` (lossy) | 1.61 tok/s | — |

Вывод: на коротких сериях verify-батч из 2 строк расширяет union экспертов
примерно во столько же раз, во сколько батч экономит плотные веса, поэтому
эффект нейтральный. Спекуляция даёт выигрыш только на длинных сериях
принятых токенов; оценка ускорения остаётся ~1.1–1.35×. 7 tok/s на этой
модели без lossy-шагов требуют другого потолка: 4 слоя дают 28 tok/s,
полные 49 — ~1.5–2.3 tok/s, то есть ~12 мс на слой-проход.

On short runs a 2-row verify batch widens the expert union by about the same
factor it saves on dense weights, so there is no gain. Speculation only gains
on long accepted runs; the estimated range stays ~1.1–1.35×. 7 tok/s is out
of reach without lossy steps: 4 layers run at 28 tok/s while all 49 run at
~1.5–2.3 tok/s, i.e. ~12 ms per layer pass.

### Более поздние замеры и статус багов / Later numbers and bug status

| режим | tok/s | принято |
|---|---|---|
| `--spec --spec-k 3` (полный top-6 verify), «hi»/24 | 0.93 | 77.8 % |
| `--spec --spec-k 3` (полный verify), RU-промпт/12 | 1.00 | 50.0 % |
| `--spec-fast` (k=4, maxb=8, verify top-2), «hi»/24 | 2.09 | 83.3 % |

- Паритет подтверждён: greedy-иды `--spec` и обычного декода совпали байт
  в байт на обоих промптах (`[ids: 150006 27177 11 269 1464 950 ...]`).
- Ранние нули acceptance в этой таблице не воспроизводятся: на коммите
  `62bdc5c` прогон «hi»/24 даёт 54.5 %, на текущем коде — 77.8 %. Вероятная
  причина старых нулей — сборка с гонкой `ExpertIo::load` (ожидающий поток
  двигал `slot->data` параллельно с читателем) и недомером AHB-пула под
  давлением памяти; обе проблемы закрыты коммитами `8e9ee40` и `c65be2d`.
- `[spec: expert_hit=0]` был дырой метрик, а не обходом кэша: батч-путь
  шёл через `demand_async`/`take_expert` и не увеличивал
  `c_expert_hit`/`c_expert_miss`. Теперь `model_batch.cpp` считает их по
  каждому заказанному срезу: hit = блок уже в кэше или вне диапазона,
  miss = заказано чтение, дупликаат в полёте — miss, как в single-пути.

**Открытый баг:** батч-путь ломается на контексте длиннее sliding-window
(у флинта 513). Prefill промпта ~1040 токенов даёт взорванные логиты
(~50 вместо ~0) и мусорный декод; на коротких промптах (до ~500 токенов)
путь корректен. Баг в батчевой обработке SWA-слоёв — до локализации
длинные спекулятивные прогоны нельзя считать валидными.

Подробные таблицы и методика — в [PERFORMANCE.ru.md](PERFORMANCE.ru.md).
Архитектура KV и батч-матвека — в [RUNTIME.md](RUNTIME.md).

## 7. Известные ограничения / Known limits

- Батч-путь пока зовёт `dot_row` построчно; batched NEON-ядра — следующий шаг.
  The batch path still calls `dot_row` per row; batched NEON is the next step.
- Спекуляция расширяет union экспертов verify-батча, поэтому оценка
  ускорения ~1.15–1.35× на флеш-устройстве, а не paper-3.7×.
  Verify widens the expert union, so the estimated speedup is ~1.15–1.35× on
  flash, not the paper's 3.7×.
- 7 tok/s без lossy-шагов (CECAR/top-k) требует другого потолка:
  однотокового декода ~5 tok/s, батчинг помогает только на сериях принятых
  токенов. 7 tok/s without lossy steps may stay out of reach: single-token
  decode tops out near 5 tok/s and batching only helps on accepted runs.

## 8. Пакет под 10 tok/s / The 10 tok/s package

Что добавлено после первых замеров. Ничего из этого не меняет формат модели
и не трогает квантизацию.
What was added after the first measurements. None of it changes the model
format or the quantization.

| ручка | что делает |
|---|---|
| `--spec-verify-topk N` | top-k на verify; меньше union экспертов (lossy) |
| `--pin-experts N` | пин N частых экспертов на слой после префилла |
| `--expert-stats` | union экспертов и hit-rate кэша |
| `--mlock` | best-effort `mlock` плотных весов |
| `--fast` | **LOSSY** top-2: `o-direct=0`, `cache-mb=0`, `draft-topk=2` |
| `--spec-fast` | `--fast` + `--spec --spec-k 4 --spec-maxb 8 --spec-verify-topk 2` |

Внутри также: `dot_row_multi` — для квантов без NEON-ядра строка весов
раскладывается в float один раз и умножается на все активации батча; в
батч-пути убраны per-layer аллокации (`segs`, `idx`, пары эксперт-строка
теперь плоские); `collect_hot` просит huge pages для плотных весов.

Inside: `dot_row_multi` dequantizes a weight row once for all batch
activations when there is no NEON kernel; the batch path no longer allocates
per layer (`segs`, `idx`, expert-member pairs are flat now); `collect_hot`
asks for huge pages on the dense weights.

Ожидаемая вилка: если `--pin-experts` действительно удержит рабочий набор
(47 MoE-слоёв из 49 блоков × 2–8 экспертов × 3 тензора ≈ 92–370 МБ),
экспертный диск уходит почти в ноль, и остаётся плотный матвек. 10 tok/s
требует ~16 ГБ/с на плотных весах; замеренный потолок на телефоне
4.5–8 ГБ/с, поэтому цифра требует подтверждения на устройстве.
Expected range: if `--pin-experts` really holds the working set (47 MoE layers
out of 49 blocks × 2–8 experts × 3 tensors ≈ 92–370 MB), expert disk traffic nearly vanishes and
dense matvec dominates. 10 tok/s needs ~16 GB/s on dense weights; the phone
measured 4.5–8 GB/s, so the number must be confirmed on the device, not
promised.
