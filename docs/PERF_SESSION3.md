# PERF_SESSION3: Inkling-2-Mini, отчёт по сессии

Дата: 2026-11-21. Устройство: Snapdragon 8s Gen 4, 15.2 ГБ RAM, Termux.

## Цель

Загрузить `Inkling-2-Mini-UD-Q2_K_XL.gguf` (9.14 ГБ) в память целиком,
реализовать forward и получить корректный текст; затем скорость.

## Что сделано

- Модель грузится полностью в RAM: prefault всех страниц при старте (9.4 с),
  fadvise(DONTNEED) отключён, expert streaming не используется.
- Arch-spec диспетчеризация: `general.architecture=inkling2` → отдельный
  бэкенд, Flint не затронут.
- 24 физических блока, 48 KV-слотов (3 прохода середины), свой KV и
  conv-state на каждый слот.
- Новые кванты: MXFP4, NVFP4, Q2_0, IQ1_S, IQ2_XXS, IQ2_XS (dequant
  сверен с ggml-референсом, 174 теста).
- Токенайзер: `pre=inkling2` (GPT4O), спецтокены по строке, промпт по
  chat-template (effort 0.9). Токенизация промпта сверена с llama.cpp
  id-в-id.
- Forward: relative bias, shortconv (static+dynamic), qk/rel clip, sinks,
  tau log-scaling, depth-mix residual, loop-контроллеры (legacy-раскладка
  Unsloth), fused gate_up + per-expert scale, SiTU, final softcap.

## Найденные баги (исправлены)

1. Softcap считался как `x*cap*tanh(x/cap)` вместо `cap*tanh(x/cap)`.
2. Dynamic conv taps: порядок был head-major, нужен lag-major (`lag + 4*head`).
3. Q/K/V входы SWA-слоя при loop-проходе ссылались на один буфер — control
   применялся трижды к одному и тому же.
4. `ffn_in` затирался буфером gate_up в MoE.
5. Переполнение `norm_states_` (5 состояний в буфере на 4).
6. MoE: `out` и буфер down-выхода эксперта были одним и тем же `ffn_` —
   каждый эксперт затирал накопленную сумму, в FFN шёл только последний.

## Проверки

- Инкрементальный декод == полный пересчёт: maxdiff 0.0000.
- Независимый numpy-референс слоёв 0..17 (attention/conv/depth-mix, FFN
  отключён): maxdiff ≤ 0.012, cos = 1.0 на позиции 8.
- Токенайзер: ids промпта совпали с llama.cpp.
- Кванты: dequant против ggml, 1800 случайных блоков, worst rel 5e-9.

## Статус

**Модель работает.** Корневая причина расхождения: в этом GGUF relative
bias и sink применяются **без масштаба** (семантика MLX/transformers), а не
делятся на sqrt(head_dim), как в mainline llama.cpp. После переключения
наши logprobs совпали с официальным Unsloth-бинарем до 0.03 на одном тексте
(например, «dog» после «The quick brown fox jumps over the lazy»: −0.072
против −0.071). Генерация осмысленная:
«The user asks "What is 2+2? Answer with one number." ...».

Полностью подтверждены numpy-референсом: attention/conv/depth-mix всех 48
применений (maxdiff ≤ 0.012), loop-сообщения и контроллеры pass1/2, MoE на
слоях 0 (IQ1_S/Q2_0) и 9 (IQ2_S/NVFP4), output mix и голова Q4_K.
Dequant Q4_K и IQ4_XS сверены с ggml побитово (maxdiff 0).

## Скорость

Текущая: **0.72 tok/s decode, ~0.6 tok/s prefill** после Q8_0 hot-конверсии
dense+output и NEON-ядер для поддерживаемых квантов. Модель целиком в RAM.
Осталось узкое место — экспертные IQ1_S/IQ2_XXS/IQ2_XS/Q2_0/NVFP4/MXFP4:
для них нет NEON dotprod, работает скалярный dequant. Потолок по байтам
~1.8 ГБ/токен при 25.7 ГБ/с — ~12 tok/s; нужны ядра, expert cache и
батч-префилл.
