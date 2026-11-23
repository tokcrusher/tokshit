# voice_chat.py — голосовой чат с tokshit

`mic → whisper.cpp (ASR) → tokshit/Inkling-2-Mini → Piper VITS (TTS) → speaker`

Оркестратор, а не новый движок: аудио-часть — это существующие
`~/audio/asr.sh` и `~/audio/tts.sh`, ответ генерирует
`scripts/run_inkling2.sh` (lossless-профиль). Ничего в `~/audio` не
меняется, `llama.cpp` не используется.

```
scripts/voice_chat.py
scripts/voice_chat.md   ← этот файл
```

## Быстрый старт

```bash
cd ~/tokshit

# проверить зависимости (модели, бинарь, плеер, микрофон)
scripts/voice_chat.py --check

# интерактивно: Enter — начать запись, Enter — остановить, q — выход
scripts/voice_chat.py

# запись ровно 5 секунд, без второго Enter
scripts/voice_chat.py --seconds 5

# полностью hands-free цикл (Ctrl-C — выход)
scripts/voice_chat.py --hands-free --seconds 5

# без микрофона: готовый WAV через весь пайплайн
scripts/voice_chat.py --wav "$TMPDIR/question.wav"

# без ASR: текст сразу в движок
scripts/voice_chat.py --text "What is the capital of France?"
```

Полезные флаги: `--max-new N` (по умолчанию 32), `--no-play`
(синтез без воспроизведения), `--once`, `--keep` (оставить WAV),
`--verbose`, `--chat-template` (нативный шаблон движка вместо raw few-shot).

## Что происходит в ходе

1. `termux-microphone-record` пишет AAC 16 kHz mono в `$TMPDIR/voice-chat-*`.
2. `ffmpeg` конвертирует в 16 kHz mono WAV и обрезает тишину по краям
   (`silenceremove`, порог −40 dB, хвост 0.35 с).
3. `~/audio/asr.sh <wav> <model>` — whisper.cpp, CPU/NEON.
4. `scripts/run_inkling2.sh "<prompt>" <max_new>` — движок tokshit.
   stdout — только ответ, stderr — метрики; ANSI вырезается.
5. Ответ печатается **как есть**, а для TTS чистится: fenced-код
   заменяется на «Code block omitted», снимается markdown, ссылки,
   таблицы, html, эхо few-shot (`Q:`/`A:` после первой строки).
6. `~/audio/tts.sh` синтезирует очищенный текст (длинный — по предложениям,
   WAV-чанки склеиваются), `ffplay -nodisp -autoexit` играет.
7. Цикл до `q` / Ctrl-C; два пустых (тихих) хода подряд тоже завершают
   сессию. В конце — средние времена по сессии.

## Замеры (этот телефон, 2026-11-23)

Условия: Termux, 16 ГБ LPDDR5X, модель
`Inkling-2-Mini-UD-Q2_K_XL.gguf` (8.5 GiB), lossless-лаунчер
`run_inkling2.sh`, whisper `tiny` (small в момент теста перекачивался),
Piper `en_US-lessac-medium`, ffplay. Телефон не в термобоксе, но и не
«холодный»: цифры плавают ±20 %, главное — порядок величин.

### TTS (Piper, один запуск = одна загрузка модели)

| текст | аудио | wall | комментарий |
|---|---:|---:|---|
| «The capital of France is Paris» | 1.73 с | 1.60 с | 1 предложение |
| «The capital of France is Paris.» | 1.9 с | 1.94 с | в пайплайне |
| 3 предложения (Paris/Eiffel/Louvre/Seine) | 5.45 с | 1.77 с | ~0.6 с на предложение, модель грузится один раз |

То есть синтез быстрее реального времени примерно в 3 раза; накладные
~0.8 с на старт onnxruntime. `speak()` делает один вызов на ответ до
320 символов, дальше режет по предложениям (каждый чанк — ещё один
старт модели).

### ASR (whisper.cpp, CPU/NEON, 6 потоков)

| аудио | модель | wall | realtime factor | текст |
|---|---:|---:|---:|---|
| 1.73 с | tiny | 1.06–1.38 с | 1.25–1.6× | «The capital of France is Paris.» |
| 5.45 с | tiny | 1.67 с | 3.3× | 3 предложения, WER ~4 % (Seine → scene) |

Whisper кодирует фиксированное 30-секундное окно, поэтому короткие
фразы дают всего 1.3–1.6×, а на 5-секундной фразе уже 3.3×. `small`
точнее, но на этом телефоне он ещё докачивался; скрипт сам выберет
`small`, когда файл целиком на месте (проверка по известному размеру,
`.aria2`-сайдкар игнорируется, если размер совпал).

### Движок (lossless, `run_inkling2.sh`)

| прогон | токенов | prefill | генерация | tok/s | wall |
|---|---:|---:|---:|---:|---:|
| `say exactly: hello` | 8 | 10.4 с | 3.0 с | 2.64 | 25.3 с |
| вопрос + raw few-shot | 24 | 14.6 с | 6.9 с | 3.47 | 33.8 с |
| вопрос + raw, 32 токена | 32 | 15.0 с | ~10 с | 3.21 | 37.0 с |
| вопрос + raw, 48 токенов | 48 | 11.7 с | 15.1 с | 3.17 | 34.8 с |

Постоянные издержки на каждый ход: загрузка модели **9–11 с**
(9.1 GB prefault, RSS ~7–8 GB) + prefill 5–15 с. Генерация ~3.2 tok/s.
Один прогон `safe_run.sh` держит эксклюзивный lock: если рядом крутится
другой tokshit, скрипт честно пишет `waited for model lock`, и wall
включает ожидание (наблюдал +18 с).

> ⚠️ Финальная проверка (10:41–10:44) попала на параллельную пересборку
> `build/tokshit` с перф-правками в `src/compute/*` и
> `src/inkling2_forward.cpp`. На этой сборке движок стал отвечать бредом
> («The capital of France is Neon», «1. Napoleon») **даже при прямом
> вызове без голосового слоя**; GGUF-модель не менялась. Замеры с
> правильным «Paris» сделаны на сборке 09:31. Если видите бред —
> сверьте mtime бинаря, пересоберите HEAD и повторите; `voice_chat.py`
> здесь ни при чём, он только передаёт промпт.

### Полный ход без микрофона

`--wav question.wav` (WAV 1.73 с, «What is the capital of France?»):

```
ASR 1.38 с → движок 37.0 с (32 tok @ 3.21 tok/s) → TTS 2.3 с → play 2.97 с
итого 43.4 с, wall процесса 47.6 с
```

Ответ: «The capital of France is Paris.» (сборка 09:31) — движок после
ответа ещё продолжает few-shot (`Q: What is the capital of Germany?`), но
в речь это не попадает: `strip_echoed_prompt` режет по первой строке
`Q:`/`A:`, а печать остаётся оригинальной.

## Переменные окружения

| переменная | по умолчанию | смысл |
|---|---|---|
| `VOICE_ENGINE` | `scripts/run_inkling2.sh` | лаунчер движка |
| `VOICE_MAX_NEW` | `32` | токенов на ответ |
| `VOICE_ASR_MODEL` | auto: small → base → tiny | модель whisper |
| `VOICE_PROMPT_TEMPLATE` | English few-shot | шаблон, обязан содержать `{text}` |
| `VOICE_TEMPERATURE` | как в лаунчере (0) | переопределить сэмплинг |
| `VOICE_PLAYER` | `auto` | `ffplay`, `termux`, `none` |
| `VOICE_REC_ENCODER` | `aac` | `aac`, `opus`, `amr_wb`, `amr_nb` |
| `VOICE_REC_RATE` | `16000` | если устройство не умеет 16k — `44100` |
| `VOICE_REC_BITRATE` | `48` | кбит/с |
| `TTS_VOICE` | `en_US-lessac-medium` | имя `.onnx` в `~/audio/models` |
| `ASR_THREADS`, `ASR_LANG` | 6, auto | читает `asr.sh` |

Быстрый, но **LOSSY** профиль (качество заметно хуже, ~10 tok/s по
докам репозитория):

```bash
VOICE_ENGINE=$HOME/tokshit/scripts/run_inkling2_fast.sh scripts/voice_chat.py
```

## Зависимости

- `~/audio/asr.sh`, `~/audio/tts.sh`, `~/audio/whisper.cpp/build/bin/whisper-cli`;
- `~/audio/models/ggml-{small,base,tiny}.bin` — хотя бы один целиком;
- `~/audio/models/en_US-lessac-medium.onnx` + `.json`;
- `~/tokshit/build/tokshit` и `~/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf`;
- `ffmpeg` (конвертация и `ffplay`), `python` + `onnxruntime` (для Piper);
- `termux-api` + приложение **Termux:API** (микрофон и `termux-media-player`);
- `flock` (входит в Termux `util-linux`, нужен `safe_run.sh`).

## Почему raw prompt, а не чат-шаблон

Нативный шаблон Inkling-2 добавляет system-сообщение
`Thinking effort level: 0.9`, и Q2_K_XL на нём охотно печатает
рассуждения прямо в ответ:

```
The user asks: "What is the capital of France?" Need answer one short
sentence. Capital of France is Paris. Need final concise...
```

`--chat-template` это воспроизводит. По умолчанию скрипт шлёт короткий
English few-shot в `TOKSHIT_RAW_PROMPT=1` — ответ прямой:
«The capital of France is Paris.» Но у raw-режима есть цена: без
английского якоря модель однажды ответила по-китайски («巴黎»), поэтому
в шаблоне есть английский пример. Если меняете шаблон — проверяйте язык.

## Отказы и что делать

| симптом | причина / лечение |
|---|---|
| `termux-microphone-record завис` (10 с) | нет Termux:API, сервис умер или висит диалог разрешения. Поставить `termux-api` + APK, выдать **Microphone** в Android Settings → Apps → Termux:API → Permissions. В этом окружении команда висела даже на `-q`, поэтому живой микрофон не протестирован. |
| `микрофон не создал файл` | разрешение RECORD_AUDIO не выдано, или устройство не поддержало 16 kHz AAC. Пробовать `VOICE_REC_RATE=44100`. |
| ASR вернул `rc=1`, но текст есть | баг `asr.sh`: EXIT-трап заканчивается на `[ -n "" ]` и перебивает код возврата. Скрипт это терпит: смотрит на stdout. |
| `no complete whisper model` | файл ещё качается/битый. Проверить размер: tiny 77 691 713, base 147 951 465, small 487 601 967 байт. Не запускать второй параллельный download. |
| `engine failed (rc=137)` | `safe_run.sh` убил движок: MemAvailable ниже 1200 MB. Закрыть приложения, не держать два tokshit. |
| `waited for model lock` | другой процесс tokshit держит lock. Ждать (до 600 с) или завершить его. |
| ответ обрывается на полуслове | упёрлись в `--max-new`. Поднять до 48–64 (дороже по времени). |
| ответ не по-английски / бред | Q2_K_XL нестабилен. Повторить, сменить шаблон, поднять `--max-new`; lossless-профиль не гарантирует качество. |
| TTS читает код/разметку | это `--no-play`/`--clean-text` диагностика; в норме код заменяется на «Code block omitted». Проверить: `scripts/voice_chat.py --clean-text '...'`. |
| TTS коверкает русский | голос `en_US-lessac-medium` англоязычный. Поставить русскую Piper-модель в `~/audio/models` и задать `TTS_VOICE`; промпт тоже перевести на русский. |
| ffplay молчит / устройство занято | `VOICE_PLAYER=termux` (Android media session, лучше с audio focus) или `none`. |
| эхо: микрофон ловит TTS | AEC нет. Наушники; в hands-free увеличить паузу (сейчас 0.4 с) или `--seconds` короче. |

## Что осталось для живого микрофона/колонки

- **Termux:API + разрешение микрофона** — главный блокер: в этом окружении
  `termux-microphone-record` зависает на старте и на `-q`, файл не
  создаётся. Скрипт корректно отваливается по таймауту с подсказкой.
- **Audio focus**: `termux-media-player` играет через Android media
  session и обычно ставит на паузу чужую музыку; `ffplay` (SDL/OpenSL)
  этого не делает и может не открыть устройство, если оно занято.
  Поэтому есть fallback и `VOICE_PLAYER`.
- **Live VAD**: Termux:API не отдаёт уровень сигнала, поэтому «стоп по
  тишине» в реальном времени невозможен. Сейчас: Enter, `--seconds N`
  или `-l max-seconds` (20 с), а тишина по краям срезается после записи.
- **AEC/эхо** и разблокировка экрана во время записи — на стороне Android.
- **Термалка**: после нескольких ходов телефон греется, tok/s падает;
  lossless-профиль и без того ~3.2 tok/s.

## Ручные проверки

```bash
# TTS → WAV → ASR
~/audio/tts.sh "The capital of France is Paris" "$TMPDIR/tts.wav"
~/audio/asr.sh "$TMPDIR/tts.wav" tiny

# движок напрямую (lossless)
scripts/run_inkling2.sh "say exactly: hello" 8

# весь пайплайн без микрофона, с таймингами
scripts/voice_chat.py --wav "$TMPDIR/tts.wav" --keep --verbose

# что услышит Piper (без синтеза)
scripts/voice_chat.py --clean-text '**Paris** is [here](https://x.y). ```code```'
```
