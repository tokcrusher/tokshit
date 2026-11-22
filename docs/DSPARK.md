# DSpark-спекуляция для Inkling-2-Mini

Статус: **эксперимент**, включается флагом `--dspark FILE` (или
`TOKSHIT_DSPARK`). Цель —
lossless-ускорение жадного декода: draft-модель
`dspark-Inkling-2-Mini-Q8_0.gguf` предлагает до 7 токенов, target проверяет
их и принимает префикс. На проверенных промптах ids совпадают с обычным
greedy один в один.

Статус: не даёт ускорения над plain при K=3, B=3. Последовательный verify
0.72 tok/s, батч 0.52; plain (2026-11-22) — 1.61–1.72. Причины — даже
высокий acceptance оплачивает один батч вместо трёх последовательных
forward, а MoE по top-8 читает всё равно по токенам. Цифры и условия — в
разделе «Замеры»; флаг `--dspark` остаётся экспериментальным.

## Что лежит в draft GGUF

`general.architecture = dflash`, 54 тензора, 402 МБ. 9 глобальных тензоров
плюс по 15 на каждый из 3 блоков.

Глобальные:

- `dspark_fuse.weight` [4,4] F32 — softmax-веса смешивания 4 target-тапов;
  строки 0..2 — draft-слои, строка 3 — голова.
- `markov_w1.weight` [256, 201024] BF16 — эмбеддинг предыдущего токена,
  markov rank 256.
- `joint_proj.weight` [5120,768] BF16 + `joint_proj.bias` [768] F32 —
  вход 2×256 + 2×2304, выход 3×256.
- `dspark_hidden_out.weight` [256,2304] BF16 + bias [2304] F32 — коррекция
  hidden из markov-состояния.
- `mask_embd.weight` [2304] F32 — эмбеддинг mask-позиций.
- `output_norm.weight` [2304] F32 — финальная норма draft-блока.
- `dspark_norm.weight` [2304] F32 — норма fused-контекста для головы.

На блок `blk.N.*`: `attn_q/k/v/output` и `ffn_gate/up/down` (Q8_0),
`attn_k_ctx`, `attn_v_ctx` (Q8_0), нормы `attn_norm`, `ffn_norm`,
`attn_q_norm`, `attn_k_norm`, `attn_k_ctx_norm`, `dflare_norm` (F32).

Метаданные:

- `dflash.target_layers = [5,23,41,48]`, `block_size = 8`,
  `sample_from_anchor = false`, `has_confidence_head = false`;
- `attention.causal = false`, `sliding_window = 1024`,
  `sliding_window_pattern = [true,false,true]`;
- `rope.dimension_count = 128`, `freq_base = 1e6`,
  `final_logit_softcapping = 30`;
- `dspark.markov_rank = 256`, `vocab_size = 201024`,
  `vocab_size_unpadded = 200058`.

Draft не имеет своей эмбеддинг-матрицы и LM head: он заимствует у target
`token_embd` + `token_embd_norm` и `output.weight`.

## Target-тапы: маппинг по визиту, не по физическому слою

Inkling2 исполняет 48 KV-слотов, но физических блоков только 24. Слоты
раскладываются так:

```text
слоты 0..5    -> физические 0..5          (прелюдия)
слоты 6..41   -> физические 6..17 × 3     (core, 3 прохода)
слоты 42..47  -> физические 18..23        (эпилог)
слот 48       -> финальный normed hidden
```

Формула для core: `slot = 6 + pass*12 + (layer - 6)`, `pass = 0..2`,
`layer = 6..17`.

`target_layers` — это индексы **визитов** (слотов), а не физических слоёв.
Захватывается hidden **на входе** слота, то есть `x_` до pre-norm блока:

| tap | что это | физический блок |
|---:|---|---|
| 5 | вход в слот 5 | 5, прелюдия |
| 23 | вход в слот 23 | 11, pass 1 (`6 + 12 + 5`) |
| 41 | вход в слот 41 | 17, pass 2 (`6 + 24 + 11`) |
| 48 | после output depth-mix и `output_norm` | — |

Захват делает `Inkling2::set_tap_slots()` / `capture_tap()`: перед запуском
слота копируется `x_`, для 48 — уже нормализованный `mix2_` после
`rmsnorm`. На каждый target-токен получается 4 вектора по 2304 float.

## Фаза инъекции

Для каждого draft-слоя `il`:

```text
w      = softmax(dspark_fuse[il, 0..3])
ctx_il = rmsnorm(sum_k w[k] * tap[k], dflare_norm[il])
K      = attn_k_ctx[il] @ ctx_il        // 8*128
V      = attn_v_ctx[il] @ ctx_il        // 8*128
K      = rmsnorm_head(K, attn_k_ctx_norm[il])
K      = rope(K, pos)
```

K/V пишутся в draft-KV слоя `il` на позицию target-токена `pos` в f16.
Слои 0 и 2 — SWA, слой 1 — полный; окно задаётся паттерном из GGUF.

Для головы:

```text
w       = softmax(dspark_fuse[3, 0..3])
context = rmsnorm(sum_k w[k] * tap[k], dspark_norm)
```

`context` хранится в кольце на `block_size + 1 = 9` позиций. Раунду нужен
контекст последнего committed-токена, `context(pos0 - 1)`; если его нет,
draft в этом раунде не выдаёт кандидатов.

## Draft-блок: backbone параллельный, голова цепочная

Вход блока — 8 позиций (`block_size`):

- позиция `pos0` — эмбеддинг последнего committed-токена (target-позиция
  `pos0 - 1`, эмбеддинг через target `token_embd` + `token_embd_norm`);
- позиции `pos0+1 .. pos0+7` — `mask_embd`.

Draft сдвинут на единицу относительно target: mask-позиция `pos0+1+i`
предсказывает target-позицию `pos0+i`. Поэтому из 8 позиций блока
получается максимум 7 кандидатов.

Три слоя считаются **одним батчем на все 8 позиций** (`run_bsegs`), а не
авторегрессивно:

- Q/K/V из hidden блока, Q/K — RMS-норма с весами, V — без нормы;
- RoPE на Q/K;
- внимание некаузальное: слой 0/2 видят окно `[pos0-1024, pos0+7]`,
  слой 1 — всё;
- блок видит инъектированные target-позиции `< pos0` и все 8 своих
  позиций;
- FFN: `down(silu(gate(x)) * up(x))`, residual.

После `output_norm` для всех 8 позиций target `lm_head_batch` параллельно
считает base-логиты для `n` mask-позиций. Это тот же `output.weight`
(Q4_K), что у target.

Цепочная только RNN-голова. На каждый шаг `i = 0..n-1`:

```text
state_0  = 0
prev_0   = anchor
prev_row = markov_w1[prev_i]
joint    = [state_i, prev_row, hidden_{i+1}, context]
joint    = joint_proj @ joint + bias            // 3*256
gate     = sigmoid(joint[0:256])
cand     = tanh(joint[256:512])
state    = cand + gate * (state_i - cand)
markov   = tanh(joint[512:768])
corr     = dspark_hidden_out @ markov + hidden_out_bias
logits   = base_logits[i] + lm_head @ corr
logits   = softcap(truncate(logits, 200058), 30)
prev     = argmax(logits)
```

То есть 8-токенный блок — это один проход по весам draft-слоёв и один
батчевый проход LM head для base-логитов. Но голова цепочная: на каждом
из `n` шагов она делает полный `lm_head @ corr` (vocab 200058 × hidden
2304), поэтому за раунд получается `n` полных LM-head матвеков, а не
один. Маленькое обновление состояния (rank 256) — только часть шага.

## Спекулятивный цикл (greedy)

1. Префилл промпта: target `token_logits` по токену, после каждого —
   `draft.inject(pos, taps)`.
2. Раунд:
   - `pos0 = pos`, `anchor = last_token`;
   - draft-блок + голова дают `cand[0..K-1]`;
   - target проверяет кандидатов батчем (`--dspark-batch=1`, по умолчанию)
     или последовательно (`--dspark-batch=0`);
   - принимается префикс, где `argmax(target_logits_after_cand[j]) ==
     cand[j+1]`; первый несовпавший кандидат отбрасывается, вместо него
     target выдаёт свой токен;
   - для всех обработанных токенов (принятых и корректирующего) инъектятся
     их тапы в draft.
3. Жадный выход lossless: при `temperature=0` ids совпали с plain greedy
   на проверенных промптах.

В последовательном режиме откат не нужен: target обрабатывает ровно
принятый префикс и свой токен. Батчевый путь откатывает K/V и conv-state
до принятого префикса через `restore_batch` и досчитывает только
корректирующий токен.

## Замеры

Все числа — один и тот же телефон, coding-промпт, 16 выходных токенов,
`--temperature 0`, K=3. Warm cache. Target-only и DSpark разделены
намеренно; greedy-ids plain и batched DSpark совпали, цепочка lossless.

| режим | tok/s | acceptance |
|---|---:|---:|
| target only, plain greedy | **1.61–1.72** (2026-11-22) | — |
| DSpark, sequential verify (`--dspark-batch=0`) | **0.72** (2026-11-22) | 47.6 % chain (10/21) |
| DSpark, batched verify (`--dspark-batch=1`) | **0.52** (2026-11-22) | 47.6 % chain (10/21) |

2026-11-22 прогоны: телефон 63–74 °C, модель не resident
(«page cache will thrash»), 4 серии по 16 выходных токенов через
`safe_run.sh`, один процесс модели. Оба пути приняли 10 из 21
предложенных (47.6 %), средняя длина цепочки 2.29. Plain работает
быстрее — ускорения над plain нет ни в одном режиме DSpark на этом
телефоне.

Причины — draft и correction-forwards на каждом раунде, MoE по top-8,
interrupting peaks; на B=3 union-экспертов растёт, и union при высокой
acceptance давит батч-читание.

### Оптимизация 2026-11-22: деквант vs барьеры vs локальность

Батч-роутер `moe_batch` (`src/inkling2_batch.cpp`): раньше F32/BF16
`gate_inp` шёл через `quantize_into` + `run_bsegs` (одна полоса на все b
активаций, диспетчеризация через пул, деквант линий inline); теперь —
`matvec()` на каждый токен, совпадает с `moe()`.

- **Деквант:** роутер больше не деквантует — линия `blk.0.ffn_gate_inp`
  F32 128×2304 (1.2 МБ/токен) идёт через `matvec`; kbench — доставка
  ~6.5 ГБ/с ≈ 0.19 мс/токен. Доля декванта в роутере — нулевая по
  конструкции.
- **Барьеры:** `run_bsegs` — полосы на задачи через пул с барьерами —
  ушла; `matvec` — прямые вызовы. Паритет: `moe_timing` (layer 0) —
  `moe_batch` b=3 = 17.6 мс vs `moe` = 36.3 мс, `max|dout| = 0`,
  union-эксперты = 11 на 3 токена.
- **Локальность:** `Mmap::advise_hugepage()` (MADV_HUGEPAGE — 2 МБ
  страницы для gather-доступа к экспертам; ошибку не фиксируем,
  обязательное ограничение). Вызывающих пока нет — API добавлен,
  подключить в горячий путь осталось.

На коротком текстовом промпте (8 токенов) первый draft-токен совпал с
target в 7/8 случаев = 87.5 %. Это acceptance **только первого токена**
(teacher-forced `TOKSHIT_DSPARK_DEBUG`), а не цепочки; переносить эту
цифру на раунд нельзя.

Lossless: batched ids = plain ids на код-промпте 16 токенов. Sequential
ids совпали с plain на 24 выходных токена. Это единственное, что DSpark
сейчас гарантирует.

## Почему sequential медленнее

1. **Последовательный verify.** Каждый кандидат — полный forward target,
   плотные веса читаются заново. Экономии на чтении весов нет вообще.
2. **Цепочка короткая.** 30.8 % дают 1.85 токена за раунд. Target делает
   примерно один forward на выданный токен — столько же, сколько plain
   greedy, — а draft добавляет работу сверху.
3. **Draft не бесплатный.** 402 МБ весов, 3 слоя, LM head и голова на
   каждый раунд.
4. **Голова читает LM head на каждом шаге.** `n` полных матвеков
   `output.weight` за раунд; это не «маленькая поправка» к base-логитам.

## Батч-верификация: что уже есть и что осталось

Батч-путь реализован в `src/inkling2_batch.cpp`:
`Inkling2::token_logits_batch()` делает layer-major проход по B токенам,
плотные matvec идут через `batchlib::run_bsegs` (`dot_row_multi`), K/V и
conv-state откатываются через `restore_batch`. Включается
`TOKSHIT_DSPARK_BATCH=1` (по умолчанию), выключается `--dspark-batch=0`.

Что он экономит и что нет:

- dense-проекции (q/k/v/o, плотные FFN, LM head) читаются один раз на B
  строк — это главный смысл батча;
- роутер (`gate_inp`, F32/BF16) теперь тоже `matvec` на токен — совпадает
  с `moe()`; деквант-стайп `run_bsegs` роутера пропал (паритет
  `moe_timing`, `max|dout| = 0`).
- attention и shortconv всё равно идут по токенам: состояние слота
  обновляется строго по позициям;
- MoE тоже пока по токенам. Если батчить его по union экспертов, трафик
  вырастет: при top-8 из 128 union быстро приближается к полному набору.
  Это отдельный компромисс, а не бесплатное ускорение.

Осталось:

- batched head: батчить `lm_head @ corr` по `n` шагам головы;
  сейчас `head_step` делает LM head на каждый шаг, то есть почти полный
  forward поверх принятых токенов.
- вопрос acceptance: при реальных промптах B=3 не окупает потерю
  union-экспертов; лосслесная версия (top-2/2-pass) не в зоне комфорта.
- после batched head: может возникнуть ещё +10–20% в random
  correction-forwards; измерить после честного A/B.

## Ограничения

- Только greedy. При `temperature > 0` main печатает warning, и DSpark-путь
  не запускается — дальше идёт обычный декод. Нужен явный
  `--temperature 0`.
- Только single-shot с промптом. В интерактивном чате DSpark не
  используется.
- Confidence head отсутствует (`has_confidence_head=false`) — порог `p_min`
  не применяется, и нет раннего отсечения: голова считает все `n`
  кандидатов.
- DSpark требует полные 3 core-прохода (`--passes 3`, default). При
  `--passes 1` не захватываются тапы 23 и 41, при `--passes 2` — тап 41;
  недостающие тапы остаются нулевыми, спек-путь невалиден.
- Память: draft 402 МБ; target 9.14 ГБ. Inkling2 остаётся в mmap, hot-пул
  AHB для него не используется. Запускать через `scripts/safe_run.sh`.
- Батч-verify можно выключить: `--dspark-batch=0` (или
  `TOKSHIT_DSPARK_BATCH=0`); тогда работает старый последовательный путь.
- Диагностика: `TOKSHIT_DSPARK_DEBUG=FILE` — teacher-forced проверка
  первого токена, `TOKSHIT_DSPARK_TRACE=1` — кандидаты по шагам,
  `TOKSHIT_DSPARK_PROF=1` — тайминги блока, LM head и головы,
  `TOKSHIT_INK_BATCH_TEST=1` — паритет батч-логитов и тапов против
  последовательного прохода.
