# voice_chat.py — голосовой чат с tokshit (persistent + streaming)

`mic → whisper.cpp → tokshit persistent chat → Piper TTS → speaker`

Движок запускается **один раз на сессию**, модель и KV живут между
ходами. Ответ читается по мере генерации, режется на предложения, и TTS
начинает говорить, пока движок ещё декодирует. Голос — один ffplay на
ход, сырой PCM в stdin, поэтому чанки не накладываются и не щёлкают.

```
scripts/voice_chat.py
scripts/voice_chat.md   ← этот файл
```

## Быстрый старт

```bash
cd ~/tokshit

scripts/voice_chat.py --check          # зависимости, движок, модели, плеер
scripts/voice_chat.py                  # интерактивно: Enter/q, Ctrl-C выход
scripts/voice_chat.py --seconds 5      # запись ровно 5 с
scripts/voice_chat.py --hands-free --seconds 5

# без микрофона: WAV через ASR -> движок -> TTS -> play
scripts/voice_chat.py --wav "$TMPDIR/question.wav"

# без ASR, два хода в одной сессии (контекст сохраняется)
scripts/voice_chat.py --text "What is the capital of France?" \
                      --text "And Italy?" --verbose

# синтез без воспроизведения + логи чанков
scripts/voice_chat.py --text "Hello" --no-play

# пропустить probe движка (если он уже проверен на этой сборке)
scripts/voice_chat.py --no-guard --text "Hello"
```

Ответ идёт в **stdout**, все логи/статусы/метрики — в **stderr**.
Можно писать `> answer.txt` и получать только текст.

## Как это устроено

### Движок

```
safe_run.sh build/tokshit --temperature 0 <model.gguf>     # без prompt
```

Окружение ребёнка:

```
TOKSHIT_CHAT_PROTOCOL=1     # stdout: <answer>\n\x1e на каждый ход
TOKSHIT_CHAT_MAX_NEW=64     # --max-new
TOKSHIT_RAW_PROMPT=1        # без шаблона Inkling
TOKSHIT_CHAT_UNESCAPE=1     # литеральные \n в строке stdin -> newline
TOKSHIT_STOP_AT=Q:          # не давать модели продолжить следующий Q&A
TOKSHIT_STOP_AT2=\n         # стоп на end_message/переводе строки
TOKSHIT_POOL_WORKERS=4      # A/B: 4 быстрее 5 на persistent chat
```

Промпты (одна строка stdin, `\n` — литеральные два символа):

* первый ход:
  `Q: 2 + 2?\nA: 4.\nQ: <user>\nA:`
* следующие:
  `\nQ: <user>\nA:`

Короткий few-shot — 15–17 токенов вместо 32 у длинного; prefill первого
хода заметно дешевле. Нейтральный пример (`2+2`) выбран после теста:
Germany-пример отвечал «Paris» на вопрос, но на statement про Францию
выдавал «The capital of Germany is Berlin». Нейтральный отвечает «Paris»
и на вопрос, и на statement. Свой префикс — `VOICE_FEWSHOT='Q: ...\nA: ...'`.
`Q:` и `\n` — стоп-маркеры: движок отдаёт ответ до маркера и не печатает
сам маркер. Если в ответе легитимно есть `Q:`, он обрежется — это
сознательный трейд-офф для голоса.

`--chat-template` — opt-in нативный шаблон Inkling (system prompt +
`TOKSHIT_THINK_EFFORT=0.2`). Он обычно печатает видимую reasoning-
преамбулу («The user asks...»), поэтому по умолчанию выключен.

### Стриминг TTS

```
engine stdout ──> SentenceStreamer ──> очередь предложений
                                         │
                                   TTS worker (Piper in-process)
                                         │
                                   очередь WAV
                                         │
                                   один ffplay (raw s16le)
```

* `SentenceStreamer` режет по `. ! ?` + пробел/newline, склеивает
  нумерованные списки (`1. one 2. two.` — один чанк), выкидывает fenced-
  код (в речь идёт «Code block omitted»), обрезает хвост few-shot-эха
  (`Q:`/`A:` после первого предложения).
* TTS worker синтезирует чанки по очереди, пока движок ещё пишет.
  По умолчанию Piper работает **в процессе** `voice_chat.py`: ONNX-сессия
  из `~/audio/lib/piper_tts.py` и та же модель загружаются один раз
  (~1.1 s) и живут всю сессию. Короткое предложение — 0.10–0.35 s.
  `VOICE_TTS=script` возвращает `~/audio/tts.sh` (новый python + nice/
  ionice на чанк, 1.8–2.7 s под нагрузкой) как fallback.
* Плеер один: `ffplay -f s16le -ar 22050 -ch_layout mono pipe:0`.
  Он стартует на первом чанке и получает PCM следующих — без повторного
  `ffplay` на каждое предложение и без наложения голосов.
* `--no-play` всё равно синтезирует и печатает тексты чанков — удобно
  проверять нарезку.
* `VOICE_PLAYER=termux` переключает на `termux-media-player` (по WAV на
  чанк), `none` — без звука.

### Вывод

Нормальный режим — стадии и одна сводка на ход, без стены метрик:

```
[engine] ready in 4.3s
[tts] in-process Piper ready in 1.14s
[wav] question.wav · 1.59 s
[asr] base · 3278 ms · 0.5× realtime · 1.59 s audio
🗣  What is the capital of France?
[engine] first token 6.19s
 Paris.
[tts] first chunk at 12.08s (0.67s audio)
[play] first audio at 12.08s (turn) · 6.89s after engine start
[summary] asr 3278 ms (base) · engine 6673 ms · tts 213 ms · total 13611 ms
[stats]   3 tok · 6.30 tok/s · 8 chars · 1 tts chunks · 0.7 s audio
[latency] first token 6.19s · first tts 12.08s (turn) · first audio 12.08s (turn) / 6.89s (engine)
```

`--verbose` добавляет тексты TTS-чанков, границы предложений и хвост
stderr движка. `[engine] loading model... / waiting for model lock...` —
живой статус, пока нет первого токена.

## Замеры

Условия: Termux, 16 ГБ LPDDR5X, `Inkling-2-Mini-UD-Q2_K_XL.gguf`,
e60da2c (batched prefill B=16), persistent chat, W=4, короткий нейтральный
few-shot, `--max-new 64`, Piper `en_US-lessac-medium` в процессе, ffplay,
whisper `base`. Page cache тёплый, чужого `safe_run` lock нет.

Guard: **13.5 s**, ответ `Paris, a major city in Europe.` (ok).
Загрузка движка: **4.3 s**. TTS-сессия Piper: **1.14 s**.

| ход | prefill | first token | engine | first tts | tts | first audio | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. «What is the capital of France?» → `Paris.` | 27 tok / 6.6 s | 6.61 s | 7.14 s | 11.82 s | 0.33 s | **11.83 s** | 13.56 s |
| 2. «And Italy?» → `Rome.` | 9 tok / 2.1 s | 2.10 s | 2.57 s | 2.67 s | 0.10 s | **2.67 s** | 3.70 s |

`first audio` первого хода включает 4.3 s загрузки и 6.6 s prefill; от
старта движка это **7.48 s**. Второй ход — **2.67 s** от реплики до звука:
prefill 9 токенов, декод 3 токена, TTS 0.10 s. Ответ — одно предложение,
поэтому TTS не может стартовать раньше его конца.

Стриминг на двух предложениях («Tell me two short facts about Paris»):
engine **12.17 s**, first token 7.95 s, first tts **15.97 s** при turn
start с загрузкой 5.4 s — то есть до engine end (~17.6 s) примерно на
**1.6 s**. TTS: 0.70 s + 0.20 s на два чанка, first audio 15.97 s, всего
21.0 s (из них 5.1 s — проигрывание 3.7 s аудио).

WAV-путь (`--wav`, 1.59 s, «What is the capital of France?»):
ASR base **3.28 s**, engine 6.67 s, first token 6.19 s, first tts
12.08 s, TTS 0.21 s, first audio **12.08 s**, total 13.61 s, ответ
`Paris.`

ASR: base — дефолт. На 11-секундной фразе ~3× realtime; на короткой
1.6-секундной доминирует фиксированное 30-секундное окно whisper, поэтому
0.5×. `small` (~0.65× realtime) забирает большие ядра и замедляет ход —
только opt-in `--asr-model small`/`VOICE_ASR_MODEL=small`.

TTS: in-process Piper даёт **0.10–0.35 s** на короткое предложение
(0.9 s на два) против 1.8–2.7 s у `tts.sh` с новым процессом и
`nice/ionice`. Prefill: 27 токенов за 6.2–6.6 s, 9 токенов за 2.1 s
(B=16, ~4–6 tok/s); короткий few-shot + вопрос дают 27 токенов, не 15–17.

> Прогоны в момент, когда другой агент держал `safe_run` lock, дают
> `[engine] waiting for model lock...` и десятки секунд сверху. Это не
> загрузка модели; в таблицу такие прогоны не включены.

## Переменные окружения

| переменная | по умолчанию | смысл |
|---|---|---|
| `VOICE_ENGINE` / `--engine` | `~/tokshit/build/tokshit` | бинарь для persistent chat (не launcher) |
| `VOICE_MODEL` | `~/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf` | GGUF |
| `VOICE_MAX_NEW` | `64` | токенов на ход |
| `VOICE_ASR_MODEL` | auto: base → tiny; small только явно | whisper; small медленный |
| `VOICE_FEWSHOT` | `Q: 2 + 2?\\nA: 4.` | короткий префикс первого хода |
| `VOICE_POOL_WORKERS` | `4` | не поднимать: 5 медленнее (A/B) |
| `VOICE_PREFILL_BATCH` | `16` | `TOKSHIT_INK_PREFILL_BATCH` |
| `VOICE_TTS` | `inproc` | `inproc` (Piper в процессе) или `script` (tts.sh) |
| `VOICE_STOP_AT` | `Q:` | первый стоп-маркер (пусто = выкл) |
| `VOICE_STOP_AT2` | `\n` | второй стоп-маркер (пусто = выкл) |
| `VOICE_TEMPERATURE` | `0` | сэмплинг |
| `VOICE_PLAYER` | `auto` | `ffplay`, `termux`, `none` |
| `VOICE_REC_ENCODER` | `aac` | `aac`, `opus`, `amr_wb`, `amr_nb` |
| `VOICE_REC_RATE` | `16000` | если устройство не умеет 16k — `44100` |
| `VOICE_REC_BITRATE` | `48` | кбит/с |
| `VOICE_THINK_EFFORT` | `0.2` | только с `--chat-template` |
| `TTS_VOICE` | `en_US-lessac-medium` | `.onnx` в `~/audio/models` |
| `ASR_THREADS`, `ASR_LANG` | 6, auto | читает `asr.sh` |

## Зависимости

- `~/tokshit/build/tokshit`, `scripts/safe_run.sh`, GGUF;
- `~/audio/asr.sh`, `whisper-cli`;
- `~/audio/models/ggml-base.bin` (дефолт) или `ggml-tiny.bin`;
  `ggml-small.bin` — только opt-in (медленный на этом телефоне);
- `~/audio/lib/piper_tts.py` + `~/audio/models/en_US-lessac-medium.onnx`
  + `.json` — in-process TTS; `~/audio/tts.sh` — fallback;
- `ffmpeg`/`ffplay`, `python` + `onnxruntime` + `numpy`, `espeak-ng`;
- `termux-api` + APK Termux:API для микрофона;
- `flock` для `safe_run.sh`.

## Quality guard

Перед голосовой сессией скрипт один раз на сборку (по mtime бинаря)
запускает:

```
TOKSHIT_RAW_PROMPT=1 scripts/run_inkling2.sh \
  "Question: What is the capital of France?\nAnswer:" 8
```

и проверяет, что в ответе есть `Paris`. Результат кэшируется в
`$TMPDIR/voice-chat-guard.json`; после пересборки probe повторится.
Плохой ответ — предупреждение `[guard] ⚠ ...`, но сессия продолжается.
`--no-guard` пропускает probe. Это защита от ситуации, когда параллельный
перф-агент пересобрал бинарь и движок стал отвечать «Neon»/«Napoleon».

## Отказы и ограничения

| симптом | причина / лечение |
|---|---|
| `[engine] waiting for model lock` | другой tokshit держит `safe_run` lock. Ждать или завершить его; это не загрузка модели. |
| `termux-microphone-record завис` | нет Termux:API/разрешения RECORD_AUDIO. Выдать Microphone в Android Settings → Apps → Termux:API → Permissions. В этом окружении живой микрофон не протестирован. |
| `микрофон не создал файл` | разрешение или 16 kHz AAC не поддержан. `VOICE_REC_RATE=44100`. |
| ASR `rc=1`, но текст есть | баг `asr.sh`: EXIT-трап перебивает код. Скрипт читает stdout и терпит. |
| ответ обрезан на `Q:` | сработал стоп-маркер. `VOICE_STOP_AT=` или `VOICE_STOP_AT2=` отключает. |
| ответ обрывается на полуслове | `--max-new`; поднять. |
| `[guard] ⚠` | сборка движка плохая; пересобрать HEAD и убрать кэш guard. |
| TTS коверкает русский | голос английский; поставить русскую Piper-модель и `TTS_VOICE`. |
| ffplay молчит / занят | `VOICE_PLAYER=termux` (Android media session) или `none`. |
| эхо микрофон↔колонка | AEC нет; наушники, короче запись, пауза в hands-free. |

Осталось для живого микрофона: Termux:API + RECORD_AUDIO, audio focus
(ffplay его не запрашивает; `termux-media-player` — запрашивает), AEC.
Live VAD невозможен: Termux:API не отдаёт уровень сигнала; сейчас Enter,
`--seconds`/`--max-seconds` и обрезка тишины после записи.

## Ручные проверки

```bash
scripts/voice_chat.py --check
scripts/voice_chat.py --clean-text '**Paris** ```code``` 1. one 2. two'
scripts/voice_chat.py --no-guard --no-play --text "Hello" --text "Again"
scripts/voice_chat.py --text "What is the capital of France?" --verbose
scripts/voice_chat.py --wav "$TMPDIR/question.wav"
```
