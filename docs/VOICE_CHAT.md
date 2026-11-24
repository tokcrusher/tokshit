# Voice chat with tokshit (TUI, EN/RU)

`mic -> whisper.cpp ASR -> tokshit persistent chat -> Piper TTS -> speaker`

One user-facing command: `scripts/chat`. It opens the curses TUI and keeps
one engine process for the whole session, so the model and KV cache survive
between turns. Tokens appear as they are generated. TTS starts at the first
completed clause or sentence, not after the whole answer.

There is no startup prefix. The engine loads the model and is ready; the
first turn is load-bound only. `VOICE_FEWSHOT` defaults to empty and exists
only as a power-user override.

```
scripts/chat                 # TUI, mic, persistent engine
scripts/voice_chat.py        # backend; do not call directly for voice chat
docs/VOICE_CHAT.md           # this file
```

`scripts/chat` is a pass-through wrapper:

```bash
#!/data/data/com.termux/files/usr/bin/bash
exec python3 "$(dirname "$0")/voice_chat.py" --tui "$@"
```

All arguments after the script name reach the backend unchanged.

## Model and runtime tuning

`scripts/chat --settings` prints the resolved model and runtime config without
loading the model. Every setting has a CLI flag; the voice wrapper passes them
to the engine child:

| flag | engine knob | default | notes |
|---|---|---|---|
| `--model NAME\|PATH` | `TOKSHIT_MODEL` | Inkling | `inkling`, `flint`, `dspark`, basename, or path |
| `--experts K` | `TOKSHIT_DRAFT_TOPK` | 8 / fast 4 EN, 6 RU | lower = faster, LOSSY |
| `--passes N` | `TOKSHIT_INK_PASSES` | 3 | 2 is LOSSY and loops |
| `--workers N` | `TOKSHIT_POOL_WORKERS` | 4 | 4 is the stable voice value |
| `--cache-mb MB` | `TOKSHIT_EXPERT_CACHE_MB` | 512 | expert cache |
| `--kv-ctx N` | `TOKSHIT_KV_CTX` | engine 2048 | context positions |
| `--layers N` | `GGUF_NLAYERS` | all | debug/limit layers |
| `--hot 0\|1` | `TOKSHIT_HOT` | 0 | 1 = dense weights in RAM |
| `--q4` | `TOKSHIT_HOT_Q4` | off | **LOSSY** dense Q4_0 |
| `--prefill-batch N` | `TOKSHIT_INK_PREFILL_BATCH` | 16 | prompt batch |
| `--max-new N` | `TOKSHIT_CHAT_MAX_NEW` | 64 | tokens per turn |
| `--temperature X` | `TOKSHIT_TEMPERATURE` | 0 | sampling |

Example:

```sh
scripts/chat --model inkling --experts 8 --passes 3 --kv-ctx 1024 \
  --cache-mb 512 --workers 4 --settings
scripts/chat --model inkling --experts 8 --passes 3 --kv-ctx 1024
```

Explicit flags win over `--fast` presets. The TUI header shows the active
`CFG top/passes/w/cache/kv` values.

## Quick start

```bash
cd ~/tokshit

scripts/chat --check          # dependencies, engine, models, player; no TUI
scripts/chat --models         # list .gguf files and exit
scripts/chat --model inkling  # select by name: inkling, flint, dspark, path
scripts/chat --model ~/models/MiniCPM5-2B-Q4_K_M.gguf --no-tui --text "hi"
scripts/chat                  # interactive TUI, English
scripts/chat --ru             # Russian ASR + Russian TTS + RU fast experts
scripts/chat --lang ru        # same language switch
scripts/chat --fast           # fast voice tier; see the LOSSY warning below
scripts/chat --ru --fast      # Russian fast tier: top-6 experts
scripts/chat --one-sentence   # opt-in: stop after the first sentence
scripts/chat --seconds 5      # record exactly 5 s, then process the turn
scripts/chat --hands-free --seconds 5

# no microphone: WAV -> ASR -> engine -> TTS -> play
scripts/chat --wav "$TMPDIR/question.wav"
scripts/chat --ru --wav "$TMPDIR/question.wav"

# no ASR: text turns in the same persistent session
scripts/chat --text "What is the capital of France?"
scripts/chat --ru --text "Привет, как дела?"
scripts/chat --ru --text "Привет, как дела?" \
             --text "Расскажи о Париже" --verbose

# synthesize without playback
scripts/chat --text "Hello" --no-play

# skip the engine probe if this build was already checked
scripts/chat --no-guard --text "Hello"
```

The backend accepts `--max-new`, `--max-seconds`, `--hands-free`,
`--one-sentence`, `--once`, `--keep`, `--asr-model`, `--engine`,
`--chat-template`, `--clean-text`, `--ru`, `--lang en|ru`, `--fast`,
`--tui`/`--no-tui`, and the environment variables listed below.

## TUI

The screen has a header (phase, mode, language, load, workers, thermal,
RSS/context/mic), the transcript, a details panel with per-turn timings and
memory, and a button row with key hints.

```
+----------------------------------------------------------------------+
| TOKSHIT VOICE | READY | FAST | turn 2 | max_new 64                  |
| lang ru | load 7.8s | workers 6 | batch 16 | therm 39.4C            |
| rss 310/7100 MB | ctx 42 | mic idle | turn 2                        |
+----------------------------------------------------------------------+
| YOU  Привет, как дела?                                               |
| INK  Привет! Хорошо, спасибо. А у тебя?                               |
| YOU  Расскажи о Париже                                               |
| INK  В Париже много музеев и театров.                                |
|                                                                      |
+----------------------------------------------------------------------+
| mic 1.6 s | ASR 2.9 s | first token 2.7 s | decode 6.2 tok/s         |
| TTS 1.1 s | audio 3.2 s | total 5.4 s | RSS 7.1 GB                   |
| experts hit 87% / miss 13%                                           |
+----------------------------------------------------------------------+
| [ RECORD ] [ STOP ] [ FAST ] [ CLEAR ] [ QUIT ]                      |
| Enter/Space record  s stop  f fast  c clear  q quit  PgUp/PgDn scroll|
+----------------------------------------------------------------------+
```

Header fields:

- phase: `WARMING`, `READY`, `RECORDING`, `THINKING` or `ERROR`.
- mode: `LOSSLESS` or `FAST`.
- language: `en` or `ru`.
- load, workers, batch, thermal: startup and phone state.
- RSS/context/mic: process memory, KV position and microphone state.

Details panel, per turn:

- `mic`: seconds recorded.
- `ASR`: whisper wall time.
- `first token`: seconds from the question to the first generated token.
- `decode`: generated tokens per second.
- `TTS`: synthesis wall time.
- `audio`: seconds of speech produced.
- `total`: wall time for the turn.
- `RSS`: resident memory of the engine.
- `experts hit / miss`: expert cache counters from the engine.

Buttons and keys:

| key | action |
|---|---|
| `Enter`, `Space` | start recording |
| `s` | stop the active stage (recording, generation or playback) |
| `f` | toggle fast mode |
| `c` | send `/clear`: clear engine KV and transcript, reset the turn |
| `q` | quit |
| `PgUp`, `PgDn` | scroll the transcript |

The transcript keeps the original user text and the full answer. The answer
is appended token by token as it arrives.

## Language: English and Russian

The default language is English. Three equivalent switches select Russian:

```bash
scripts/chat --ru
scripts/chat --lang ru
VOICE_LANG=ru scripts/chat
```

| setting | English default | Russian mode |
|---|---|---|
| `ASR_LANG` | `en` | `ru` |
| `TTS_VOICE` | `en_US-lessac-medium` | `ru_RU-irina-medium` |
| fast experts | top-4 | top-6 |
| language directive | `Answer in English.` | `Отвечай по-русски.` |

`ru_RU-irina-medium` is installed: 63 MB, 22 kHz. If the model is missing,
the backend warns and keeps the current English voice.

The language is anchored inside each user turn, not in a startup prefix.
The engine payload is:

```
English: \nQ: Answer in English. <user text>\nA:
Russian: \nQ: Отвечай по-русски. <user text>\nA:
```

The leading `\n` is used on every turn, including the first. The directive
stays inside the engine payload; the transcript shows the original ASR or
typed text without it. If the text has no terminal punctuation, the backend
adds a period in the payload only. The per-turn directive fixes English
drifting to French and stabilizes Russian.

## Streaming and TTS

The engine runs with `TOKSHIT_CHAT_PROTOCOL=1`: each turn is framed on
stdout with `0x1e`, and tokens are forwarded to the TUI as they are decoded.
The transcript therefore grows during generation.

By default the engine generates until EOG or `--max-new`; it does not stop
at a newline. `--one-sentence` (`VOICE_ONE_SENTENCE=1`) is the opt-in that
sets `TOKSHIT_STOP_SENTENCE=1` and stops after the first sentence.
`TOKSHIT_STOP_SENTENCE` is unset by default. Default stops are `Q:`, `\nQ`
and `\nA`; they cut next-turn echo, not ordinary words.

The token stream feeds the TTS splitter:

```
engine stdout -> SentenceStreamer -> clause/sentence queue
                                      |
                                TTS worker (Piper in-process)
                                      |
                                WAV queue
                                      |
                                one ffplay (raw s16le)
```

- `SentenceStreamer` splits on `. ! ?` followed by whitespace or newline.
  It joins numbered list items (`1. one 2. two.` stays one chunk), drops
  fenced code (speech gets `Code block omitted.`), and flushes at a comma,
  semicolon or colon once at least 26 characters are buffered. A chunk is
  capped at 84 characters.
- TTS starts at the first completed clause or sentence while the engine is
  still decoding. A short sentence takes 0.10-0.35 s with in-process Piper.
  Russian chunks use `ru_RU-irina-medium` at 22 kHz. `VOICE_TTS=script`
  falls back to `~/audio/tts.sh` with a new process per chunk.
- A single `ffplay -f s16le -ar 22050 -ch_layout mono pipe:0` starts on the
  first chunk and receives the PCM of later chunks. There is no second
  player and no overlapping voice.
- `--no-play` still synthesizes and logs chunk texts, which is useful for
  checking the split.

### TTS duplicate guard

The guard normalizes each sentence (case, punctuation and zero-width
characters are ignored) and compares it with the previous spoken sentence.

- an identical consecutive sentence is dropped;
- after two duplicate sentences in a row, TTS stops for the rest of the
  turn;
- the transcript keeps the full generated text; only audio is cut;
- generation is not stopped, so `--max-new` and EOG still apply.

This is separate from `--one-sentence`, which changes generation.

## Fast mode

`scripts/chat --fast` (or `VOICE_FAST=1`) selects the faster voice tier:
3 loop passes, original dense weights, 6 workers. Expert count depends on
language. Both variants are LOSSY.

| mode | experts/token | passes | dense weights | measured decode | quality |
|---|---|---:|---|---:|---|
| default | top-8 | 3 | original | ~5.5 tok/s | reference, lossless |
| EN `--fast` | top-4 | 3 | original | 6.2-6.7 tok/s | readable; **LOSSY** (4 of 8) |
| RU `--fast` | top-6 | 3 | original | not separately benched | readable; **LOSSY** (6 of 8); top-4 loops on turn 2 |
| forbidden | top-2 | 2 | Q4 | 13.3 tok/s | loops: "Capital capital capital..." |

Fast mode is not lossless. The dense path stays original, but expert
pruning changes model output. Do not use fast mode for code, benchmarks or
quality comparisons. Russian fast uses top-6 because top-4 looped on the
second Russian turn. `VOICE_TOPK` is an explicit override; leave it unset
to get top-4 for English and top-6 for Russian.

The quality cliff is the 2-pass loop, not only the expert count:

- top-8 or top-4/top-6 with 3 passes: coherent.
- top-8 or top-4/top-6 with 2 passes: 6.4-9.2 tok/s, loops on many prompts.
- top-2 with 2 passes: garbage.

top-2/2-pass configurations are forbidden for voice and for code.

## Measurements

Startup is load-only. All older latency numbers were measured with the
removed startup warmup stage and are obsolete; those tables were deleted.

| mode | ready (load-only) | notes |
|---|---:|---|
| English default/fast | 7-10 s | no warmup stage |
| Russian fast | 6.8-9.1 s | no warmup stage |

The TUI allows recording during `WARMING`; the turn queues and runs when
the engine is ready. The first turn waits only for model load.

Verified Russian fast turns:

| turn | user text | answer | tokens | TTS chunks |
|---|---|---|---:|---:|
| 1 | `Привет, как дела?` | `Привет! Хорошо, спасибо. А у тебя?` | 15 | 3 |
| 2 | `Расскажи о Париже` | `В Париже много музеев и театров.` | 16 | 1 |

TTS chunks are sentence chunks; the first chunk starts playback while the
rest of the answer is still being generated.

Thermal caveat: this phone throttles. The TUI header shows temperature;
compare runs at a similar thermal state. Single runs are not A/B medians.
ASR, TTS and decode share the big cores, and a hot phone can halve
throughput.

## How it works

### Engine

```
safe_run.sh build/tokshit --temperature 0 <model.gguf>     # no prompt
```

Child environment:

```
TOKSHIT_CHAT_PROTOCOL=1     # stdout: <answer>\n\x1e per turn
TOKSHIT_CHAT_MAX_NEW=64     # --max-new
TOKSHIT_RAW_PROMPT=1        # no Inkling template
TOKSHIT_CHAT_UNESCAPE=1     # literal \n in the stdin line -> newline
TOKSHIT_STOP_AT=Q:          # default stop marker
TOKSHIT_STOP_AT2=\nQ        # next bare Q line
TOKSHIT_STOP_AT3=\nA        # next bare A line
TOKSHIT_POOL_WORKERS=4      # A/B: 4 is faster than 5 on persistent chat
TOKSHIT_INK_PREFILL_BATCH=16
```

There is no startup prefix: `TOKSHIT_CHAT_PREFIX` is not set. The engine
loads the model, then reports ready:

```
[engine] loaded in 6.8s
[engine] ready in 7.3s
```

`VOICE_FEWSHOT` defaults to empty and is a power-user override only. If set,
the old few-shot path and its startup stage return, but that is not the
default or recommended voice setup.

Each turn is one stdin line (`\n` means two literal characters):

```
\nQ: Answer in English. <user text>\nA:
```

The leading newline is required even on the first turn; without it the
model drifts. `Q:` and the two newline markers are stop markers: the engine
returns the answer up to the marker and does not print the marker itself.

`--chat-template` enables the native Inkling template (system prompt plus
`TOKSHIT_THINK_EFFORT=0.2`) instead of the raw Q/A path. It usually prints
a visible reasoning preamble, so it is off by default.

## Clear context

`c` or `[ CLEAR ]` sends `/clear` to the engine, drains the reply, clears
the KV cache and transcript, and resets the turn counter. Verified in a
Russian session: the model recalled `Алекс` before clear and did not recall
it after clear.

If a turn or recording is active, clear is refused with a notice:
`Cannot clear context during a turn or recording`. Finish the turn first.
If the engine does not answer `/clear`, the transcript is still cleared and
the TUI reports that engine clearing failed.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `VOICE_ENGINE` / `--engine` | `~/tokshit/build/tokshit` | persistent chat binary (not the launcher) |
| `VOICE_MODEL` / `--model` | `~/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf` | GGUF; accepts `inkling`, `flint`, `dspark`, basename, or path |
| `--models` | - | list available `.gguf` and exit |
| `VOICE_MAX_NEW` | `64` | tokens per turn |
| `VOICE_LANG` | `en` | `en` or `ru`; `--ru` wins, then `--lang` |
| `VOICE_ASR_MODEL` | auto: base -> tiny; small only if explicit | whisper model; small is slow |
| `VOICE_FEWSHOT` | empty | power-user few-shot override; normally unset |
| `VOICE_FAST` | `0` | `1` = 3 passes, original dense, 6 workers; top-4 EN / top-6 RU; LOSSY |
| `VOICE_TOPK` | auto | expert override; normally unset (4 EN, 6 RU) |
| `VOICE_FAST_WORKERS` | `6` | workers in `VOICE_FAST` |
| `VOICE_POOL_WORKERS` | `4` | lossless voice; do not raise: 5 was slower in A/B |
| `VOICE_PREFILL_BATCH` | `16` | `TOKSHIT_INK_PREFILL_BATCH` for the question |
| `VOICE_TTS` | `inproc` | `inproc` (Piper in-process) or `script` (`tts.sh`) |
| `VOICE_STOP_AT` | `Q:` | first stop marker (empty disables) |
| `VOICE_STOP_AT2` | `\nQ` | second stop marker (empty disables) |
| `VOICE_STOP_AT3` | `\nA` | third stop marker (empty disables) |
| `VOICE_ONE_SENTENCE` | `0` | `1` sets `TOKSHIT_STOP_SENTENCE=1` |
| `VOICE_TEMPERATURE` | `0` | sampling |
| `VOICE_PLAYER` | `auto` | `ffplay`, `termux`, `none` |
| `VOICE_REC_ENCODER` | `aac` | `aac`, `opus`, `amr_wb`, `amr_nb` |
| `VOICE_REC_RATE` | `16000` | use `44100` if the device cannot record 16 kHz |
| `VOICE_REC_BITRATE` | `48` | kbit/s |
| `VOICE_THINK_EFFORT` | `0.2` | only with `--chat-template` |
| `TTS_VOICE` | `en_US-lessac-medium` | `.onnx` in `~/audio/models`; RU mode uses `ru_RU-irina-medium` |
| `ASR_LANG` | `en` | `ru` in Russian mode |

## Dependencies

- `~/tokshit/build/tokshit`, `scripts/safe_run.sh`, the GGUF model;
- `~/audio/asr.sh`, `whisper-cli`;
- `ffmpeg`/`ffplay`, `python` with `onnxruntime` and `numpy`,
  `espeak-ng`;
- `termux-api` CLI plus the Termux:API APK for the microphone;
- `flock` for `safe_run.sh`.

Voice and ASR models:

| component | model | size | rate | notes |
|---|---|---|---:|---|
| ASR default | `ggml-base.bin` | 283 MB (74M params) | 16 kHz input | ~3x realtime on an 11 s phrase; whisper pads to 30 s |
| ASR fallback | `ggml-tiny.bin` | not listed | 16 kHz input | used when base is unavailable |
| ASR optional | `ggml-small.bin` | 244M params | 16 kHz input | ~0.65x realtime; takes the big cores; opt-in |
| TTS English | `en_US-lessac-medium.onnx` | not listed | 22.05 kHz | default |
| TTS Russian | `ru_RU-irina-medium.onnx` | 63 MB | 22 kHz | installed; used by `--ru`/`--lang ru` |

## Microphone

Android needs both the Termux:API APK and the Microphone permission:

1. Install the Termux:API APK (the APK, not only the `termux-api` CLI
   package). The downloaded copy is
   `~/storage/downloads/termux-api-0.53.0.apk`.
2. Grant Microphone: Android Settings -> Apps -> Termux:API ->
   Permissions -> Microphone.
3. Test the recorder before blaming the chat:

```bash
termux-microphone-record -f "$TMPDIR/mic-test.m4a" -l 5
```

Wait for the recorder to finalize the file. `-l 5` is the recording limit;
the command must return and the `.m4a` must reach its final size before ASR
starts. If it hangs, the APK or the permission is missing. If it returns
but no file appears, try `VOICE_REC_RATE=44100`.

## Quality guard

Before a voice session, once per build (keyed by the binary mtime), the
backend runs a short capital-of-France probe and checks that the answer
contains the expected capital. The result is cached in
`$TMPDIR/voice-chat-guard.json`; a rebuild repeats the probe. A bad answer
prints `[guard] WARN ...` but the session continues. `--no-guard` skips the
probe. This catches the case where a parallel perf agent rebuilt the binary
and the engine started producing broken answers.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `[engine] waiting for model lock` | another tokshit process holds the `safe_run` lock. Wait or stop it; this is not model loading. |
| ASR `small` slows the turn | `base` is the default (3x realtime on 11 s); `small` is ~0.65x and takes the big cores. Use `--asr-model small` only for accuracy. |
| `termux-microphone-record` hangs | Termux:API APK or RECORD_AUDIO permission is missing. Install the APK and grant Microphone. Test with `termux-microphone-record -f "$TMPDIR/mic-test.m4a" -l 5`. |
| microphone produced no file | permission problem or 16 kHz AAC is unsupported. Try `VOICE_REC_RATE=44100`. |
| ASR `rc=1` but text exists | `asr.sh` bug: the EXIT trap overwrites the status. The backend reads stdout and tolerates it. |
| answer cut at `Q:` or a newline marker | a stop marker fired. Tune `VOICE_STOP_AT` / `VOICE_STOP_AT2` / `VOICE_STOP_AT3`; empty disables one. |
| answer stops after 3-5 tokens | a stop marker matched too early. Check the three stop values; leave the defaults unless you have a measured reason. |
| answer stops mid-word | `--max-new` is too low; raise it. |
| English answer drifts to French | use the default English directive; leave `VOICE_FEWSHOT` empty and do not inject a non-English directive. |
| Russian answer drifts or mixes English | start with `scripts/chat --ru` or `--lang ru`; check `ASR_LANG=ru` and `TTS_VOICE=ru_RU-irina-medium`. |
| Russian fast loops on turn 2 | top-4 is too small for Russian; use `--ru --fast` (top-6) and leave `VOICE_TOPK` unset. |
| TTS sounds English in Russian mode | `ru_RU-irina-medium` is missing; install it in `~/audio/models`, or set `TTS_VOICE` explicitly. |
| TTS stops mid-answer | the duplicate guard dropped repeated sentences and stopped audio for the turn. The transcript keeps the full text; this is not a generation stop. |
| `Cannot clear context during a turn or recording` | finish the current turn or stop recording, then press `c` again. |
| `[guard] WARN` | bad engine build. Rebuild HEAD and remove the guard cache. |
| ffplay is silent or busy | use `VOICE_PLAYER=termux` (Android media session) or `none`. |
| echo between microphone and speaker | there is no AEC. Use headphones, record shorter, or pause in hands-free mode. |

Live microphone status is still limited: Termux:API does not expose the
input level, so there is no VAD. Recording is controlled by Enter/Space,
`--seconds`/`--max-seconds`, and silence trimming after the recording.
Audio focus is also not requested by `ffplay`; `termux-media-player` does
request it.

## Manual checks

```bash
scripts/chat --check
scripts/chat --clean-text '**Hello** ```code``` 1. one 2. two'
scripts/chat --no-guard --no-play --text "Hello" --text "Again"
scripts/chat --ru --text "Привет, как дела?"
scripts/chat --ru --text "Расскажи о Париже"
scripts/chat --wav "$TMPDIR/question.wav"
```
