# Voice chat with tokshit (TUI)

`mic -> whisper.cpp -> tokshit persistent chat -> Piper TTS -> speaker`

One user-facing command: `scripts/chat`. It opens the curses TUI and keeps
one engine process for the whole session, so the model and KV cache survive
between turns. Tokens appear as they are generated. TTS starts at the first
completed clause or sentence, not after the whole answer.

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

## Quick start

```bash
cd ~/tokshit

scripts/chat --check          # dependencies, engine, models, player; no TUI
scripts/chat                  # interactive TUI
scripts/chat --fast           # fast voice tier; see the LOSSY warning below
scripts/chat --one-sentence   # opt-in: stop after the first sentence
scripts/chat --seconds 5      # record exactly 5 s, then process the turn
scripts/chat --hands-free --seconds 5

# no microphone: WAV -> ASR -> engine -> TTS -> play
scripts/chat --wav "$TMPDIR/question.wav"

# no ASR: text turn in the same persistent session
scripts/chat --text "What is the capital of France?"
scripts/chat --text "What is the capital of France?" \
             --text "And Italy?" --verbose

# synthesize without playback
scripts/chat --text "Hello" --no-play

# skip the engine probe if this build was already checked
scripts/chat --no-guard --text "Hello"
```

The backend accepts `--max-new`, `--max-seconds`, `--hands-free`,
`--one-sentence`, `--once`, `--keep`, `--asr-model`, `--engine`,
`--chat-template`, `--clean-text`, and the environment variables listed
below. Only the flags above are part of the normal TUI workflow.

## TUI

The screen has four regions: a header with engine, mode and thermal state;
the transcript; a details panel with per-turn timings and memory; and a
button row with key hints.

```
+----------------------------------------------------------------------+
| tokshit voice chat   engine: build/tokshit   mode: lossless   38 C   |
+----------------------------------------------------------------------+
| YOU  What is the capital of France?                                  |
| INK  Paris is the capital of France. It has the Eiffel Tower.        |
| YOU  And Italy?                                                      |
| INK  Rome is the capital of Italy. It has the Colosseum.             |
|                                                                      |
+----------------------------------------------------------------------+
| mic 1.6 s | ASR 2.9 s | prefill 13 tok / 2.8 s | first token 2.7 s   |
| decode 6.8 tok/s | TTS 1.1 s | audio 3.2 s | total 5.8 s             |
| RSS 7.1 GB | experts hit 87% / miss 13%                              |
+----------------------------------------------------------------------+
| [ RECORD ] [ STOP ] [ FAST ] [ CLEAR ] [ QUIT ]                      |
| Enter/Space record  s stop  f fast  c clear  q quit  PgUp/PgDn scroll|
+----------------------------------------------------------------------+
```

Header fields:

- `engine`: the backend binary in use.
- `mode`: `lossless` (default) or `fast` (LOSSY, 4 of 8 experts).
- `thermal`: phone temperature when the backend can read it.

Details panel fields, per turn:

- `mic`: seconds recorded.
- `ASR`: whisper wall time.
- `prefill`: question tokens and seconds.
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
| `c` | clear the transcript |
| `q` | quit |
| `PgUp`, `PgDn` | scroll the transcript |

The transcript keeps the user question and the full answer. The answer is
appended token by token as it arrives, so streaming text grows in place.

## Streaming

The engine runs with `TOKSHIT_CHAT_PROTOCOL=1`: each turn is framed on
stdout with `0x1e`, and tokens are forwarded to the TUI as they are decoded.
The transcript therefore grows during generation.

By default the engine generates until EOG or `--max-new`; it does not stop
at a newline. `--one-sentence` (`VOICE_ONE_SENTENCE=1`) is the opt-in that
sets `TOKSHIT_STOP_SENTENCE=1` and stops after the first sentence. In both
cases the visible answer is fed to TTS as clauses and sentences complete.

The same token stream feeds the TTS splitter:

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
  `VOICE_TTS=script` falls back to `~/audio/tts.sh` with a new process per
  chunk (1.8-2.7 s under load).
- A single `ffplay -f s16le -ar 22050 -ch_layout mono pipe:0` starts on the
  first chunk and receives the PCM of later chunks. There is no second
  player and no overlapping voice.
- `--no-play` still synthesizes and logs chunk texts, which is useful for
  checking the split.

## Fast mode

`scripts/chat --fast` (or `VOICE_FAST=1`) selects the faster voice tier:
top-4 of the 8 experts selected per token, 3 loop passes, original dense
weights (unchanged), 6 workers. The model has 128 experts and normally uses
top-8.

| mode | experts/token | passes | dense weights | measured decode | quality |
|---|---|---:|---|---:|---|
| default | top-8 | 3 | original | ~5.5 tok/s | reference, lossless |
| `--fast` | top-4 | 3 | original | 6.2-6.7 tok/s | readable; **LOSSY** (4 of 8) |
| forbidden | top-2 | 2 | Q4 | 13.3 tok/s | loops: "Capital capital capital..." |

Fast mode is not lossless. Dropping 4 of 8 experts changes the model output;
it stays readable for voice, but do not use it for code, benchmarks or
quality comparisons. The dense path remains the original weights, but that
does not make the mode lossless.

The quality cliff is the 2-pass loop, not only the expert count:

- top-8 or top-4 with 3 passes: coherent, 5.5-6.7 tok/s.
- top-8 or top-4 with 2 passes: 6.4-9.2 tok/s, loops on many prompts.
- top-2 with 2 passes: garbage.

top-2/2-pass configurations are forbidden for voice and for code.

## Measurements

Conditions: Termux on the 16 GB LPDDR5X phone, warm page cache,
`Inkling-2-Mini-UD-Q2_K_XL.gguf`, current build (batched prefill B=16 plus
`TOKSHIT_CHAT_PREFIX`), persistent chat, `W=4`, `--max-new 64`, Piper
`en_US-lessac-medium` in-process, `ffplay`, whisper `base`. The table is a
historical run with the older short prefix `Q: 2 + 2?\nA: 4.`; the current
default prefix is the Paris two-sentence example, so its prefix is longer.
Per-turn decode and TTS timings remain representative.

Guard: **13.5 s**, answer `Paris, a major city in Europe.` (ok, cached).
Engine start in that run: **load 3.8 s + prefix 14 tok / 3.3 s** = ready
in 7.1 s. The run reports `[engine] ready in 8.2 s`; about 1 s is another
process holding the model lock. Piper session: **1.19 s**.

| turn | prefix | question prefill | first token (engine/turn) | engine | first TTS | TTS | first audio | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. "What is the capital of France?" -> `Paris.` | 14 tok / 3.3 s | 13 tok / 3.3 s | 3.32 / 11.56 s | 3.88 s | 12.33 s | 0.21 s | **12.34 s** | 14.12 s |
| 2. "And Italy?" -> `Rome.` | - | 9 tok / 2.0 s | 2.02 / 2.03 s | 2.48 s | 2.59 s | 0.10 s | **2.59 s** | 3.61 s |

The first turn is measured from process start: turn-relative
`first token 11.56 s` includes load + prefix (7.1 s) and the question
prefill (3.3 s). In a live mic session, load + prefix overlap the user's
speech and ASR, so after speech ends only the question prefill (3.3 s) and
TTS (0.2 s) remain. The second turn is **2.59 s** from speech end to audio.
In that run both answers were one sentence, so TTS started after the
sentence ended.

Streaming on two sentences ("Tell me two short facts about Paris", same
in-process TTS): engine **12.17 s**, first TTS **15.97 s** while generation
ends at ~17.6 s, so first audio arrives **1.6 s before** generation ends.
TTS took 0.70 + 0.20 s for two chunks; total 21.0 s, including 5.1 s of
playback for 3.7 s of audio.

WAV path (`--wav`, 1.59 s clip, "What is the capital of France?"):
ASR base **2.99 s** (0.5x realtime on a short clip because whisper pads to
a 30 s window), engine load 5.2 s + prefix 4.2 s (overlapped with ASR),
question prefill 13 tok / 2.8 s, first token 2.82 s (engine) / 12.33 s
(turn), TTS 0.24 s, first audio **13.02 s** (3.51 s after engine start),
total 14.55 s. Answer: `Paris.`

ASR: `base` is the default. On an 11 s phrase it runs at about 3x realtime;
on a 1.6 s clip the fixed 30 s whisper window dominates, hence 0.5x.
`small` (~0.65x realtime) takes the big cores and slows the turn down.
Use `--asr-model small` / `VOICE_ASR_MODEL=small` only when accuracy
matters more than latency.

TTS: in-process Piper gives 0.10-0.35 s per short sentence (0.9 s for two)
against 1.8-2.7 s for `tts.sh` with a new process and `nice/ionice`.
Prefill: prefix 14 tok / 3.3 s, question 13 tok / 2.8-3.3 s, repeat turn
9 tok / 2.0 s (B=16, ~4-6 tok/s).

> If another agent holds the `safe_run` lock, the backend logs
> `[engine] waiting for model lock...` and tens of seconds are added. This
> is not model loading and such runs are excluded from the table. The first
> run above includes about 1 s of lock wait.

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
TOKSHIT_CHAT_PREFIX=...     # hidden prefill at startup, no generation
TOKSHIT_STOP_AT=Q:          # only default stop marker
# TOKSHIT_STOP_AT2 is unset by default: no newline stop
# TOKSHIT_STOP_SENTENCE=1   # opt-in via --one-sentence / VOICE_ONE_SENTENCE=1
TOKSHIT_POOL_WORKERS=4      # A/B: 4 is faster than 5 on persistent chat
TOKSHIT_INK_PREFILL_BATCH=16
```

`TOKSHIT_CHAT_PREFIX` is prefilled into the KV cache before stdin is read;
the engine generates nothing for it. Startup is:

```
[engine] loaded in 3.8s
[engine] prefix <N> tok / <T>s
[engine] ready
```

The prefix token count and time depend on `VOICE_FEWSHOT`. The historical
run below used a 14-token prefix (3.3 s); the Paris default is longer.
In mic mode prefixing overlaps speech and ASR. The first question is
prefilled separately (9-13 tokens).

Each turn is one stdin line (`\n` means two literal characters):

```
\nQ: <user>\nA:
```

The first turn also starts with `\nQ:` because the few-shot text is in the
prefix. The current default prefix teaches a two-sentence answer with a
Paris example:

```
Q: Hello\nA: Hi! How can I help you today?\nQ: Tell me about Paris.\nA: Paris is the capital of France. It is known for the Eiffel Tower and the Louvre.
```

It is sent as one `TOKSHIT_CHAT_PREFIX` value with literal `\n` separators.
Override with `VOICE_FEWSHOT='Q: ...\nA: ...'`; an empty value disables the
prefix and the first turn starts with `Q:`.

`Q:` is the only default stop marker: the engine returns the answer up to
`Q:` and does not print the marker itself. `TOKSHIT_STOP_AT2` is empty by
default. A newline second marker cut live answers to 3-5 tokens, so do not
set one. `--one-sentence` (`VOICE_ONE_SENTENCE=1`) is the explicit opt-in:
it sets `TOKSHIT_STOP_SENTENCE=1` and stops after the first sentence.
Without it, generation runs to EOG or `--max-new` and is streamed to TTS
as clauses and sentences complete, so a two-sentence answer starts speaking
before it finishes. A legitimate `Q:` inside an answer is still truncated; this is a
deliberate trade-off for voice.

`--chat-template` enables the native Inkling template (system prompt plus
`TOKSHIT_THINK_EFFORT=0.2`). It usually prints a visible reasoning preamble
("The user asks..."), so it is off by default.

### TTS streaming

See the diagram and splitter rules in the Streaming section.

### Output

The TUI owns the screen. The answer is written to the transcript as it
arrives, and the details panel summarizes the turn. The backend keeps its
full logs on stderr; `--verbose` enables the detailed backend log, including
TTS chunk texts, sentence boundaries and the engine's stderr tail.

`--no-play` still runs synthesis so the TTS timings and chunking can be
verified without audio.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `VOICE_ENGINE` / `--engine` | `~/tokshit/build/tokshit` | persistent chat binary (not the launcher) |
| `VOICE_MODEL` | `~/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf` | GGUF |
| `VOICE_MAX_NEW` | `64` | tokens per turn |
| `VOICE_ASR_MODEL` | auto: base -> tiny; small only if explicit | whisper model; small is slow |
| `VOICE_FEWSHOT` | `Q: Hello\\nA: Hi! How can I help you today?\\nQ: Tell me about Paris.\\nA: Paris is the capital of France. It is known for the Eiffel Tower and the Louvre.` | hidden startup prefix (`TOKSHIT_CHAT_PREFIX`); empty disables |
| `VOICE_FAST` | `0` | `1` = top-4 experts, 3 passes, original dense weights, 6 workers; readable LOSSY mode |
| `VOICE_TOPK` | `4` | experts per token in `VOICE_FAST` (not 2: top-2 loops) |
| `VOICE_FAST_WORKERS` | `6` | workers in `VOICE_FAST` |
| `VOICE_POOL_WORKERS` | `4` | lossless voice; do not raise: 5 was slower in A/B |
| `VOICE_PREFILL_BATCH` | `16` | `TOKSHIT_INK_PREFILL_BATCH` |
| `VOICE_TTS` | `inproc` | `inproc` (Piper in-process) or `script` (`tts.sh`) |
| `VOICE_STOP_AT` | `Q:` | default stop marker (empty disables) |
| `VOICE_STOP_AT2` | empty | optional second stop marker; do not set a newline stop |
| `VOICE_ONE_SENTENCE` | `0` | `1` sets `TOKSHIT_STOP_SENTENCE=1`: stop after the first sentence |
| `VOICE_TEMPERATURE` | `0` | sampling |
| `VOICE_PLAYER` | `auto` | `ffplay`, `termux`, `none` |
| `VOICE_REC_ENCODER` | `aac` | `aac`, `opus`, `amr_wb`, `amr_nb` |
| `VOICE_REC_RATE` | `16000` | use `44100` if the device cannot record 16 kHz |
| `VOICE_REC_BITRATE` | `48` | kbit/s |
| `VOICE_THINK_EFFORT` | `0.2` | only with `--chat-template` |
| `TTS_VOICE` | `en_US-lessac-medium` | `.onnx` in `~/audio/models` |
| `ASR_THREADS`, `ASR_LANG` | 6, auto | read by `asr.sh` |

## Dependencies

- `~/tokshit/build/tokshit`, `scripts/safe_run.sh`, the GGUF model;
- `~/audio/asr.sh`, `whisper-cli`;
- `~/audio/models/ggml-base.bin` (default) or `ggml-tiny.bin`;
  `ggml-small.bin` only as an explicit opt-in (slow on this phone);
- `~/audio/lib/piper_tts.py`, `~/audio/models/en_US-lessac-medium.onnx`
  and `.json` for in-process TTS; `~/audio/tts.sh` as fallback;
- `ffmpeg`/`ffplay`, `python` with `onnxruntime` and `numpy`,
  `espeak-ng`;
- `termux-api` CLI plus the Termux:API APK for the microphone;
- `flock` for `safe_run.sh`.

### Microphone

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
backend runs:

```
TOKSHIT_RAW_PROMPT=1 scripts/run_inkling2.sh \
  "Question: What is the capital of France?\nAnswer:" 8
```

and checks that the answer contains `Paris`. The result is cached in
`$TMPDIR/voice-chat-guard.json`; a rebuild repeats the probe. A bad answer
prints `[guard] warning: ...` but the session continues. `--no-guard`
skips the probe. This catches the case where a parallel perf agent rebuilt
the binary and the engine started answering "Neon" or "Napoleon".

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `[engine] waiting for model lock` | another tokshit process holds the `safe_run` lock. Wait or stop it; this is not model loading. |
| prefix stage hangs | the binary does not support `TOKSHIT_CHAT_PREFIX`, or the prefix did not parse. `VOICE_FEWSHOT=` disables it; the first turn starts with `Q:`. |
| ASR `small` slows the turn | `base` is the default (3x realtime on 11 s); `small` is ~0.65x and takes the big cores. Use `--asr-model small` only for accuracy. |
| `termux-microphone-record` hangs | Termux:API APK or RECORD_AUDIO permission is missing. Install the APK and grant Microphone. Test with `termux-microphone-record -f "$TMPDIR/mic-test.m4a" -l 5`. |
| microphone produced no file | permission problem or 16 kHz AAC is unsupported. Try `VOICE_REC_RATE=44100`. |
| ASR `rc=1` but text exists | `asr.sh` bug: the EXIT trap overwrites the status. The backend reads stdout and tolerates it. |
| answer cut at `Q:` | a stop marker fired. `VOICE_STOP_AT=` disables the default; leave `VOICE_STOP_AT2` empty. |
| answer cut after 3-5 tokens | a second stop marker is set, usually a newline. `VOICE_STOP_AT2` is empty by default; unset it. Use `--one-sentence` only when a one-sentence answer is intended. |
| answer stops mid-word | `--max-new` is too low; raise it. |
| `[guard] warning` | bad engine build. Rebuild HEAD and remove the guard cache. |
| TTS mangles non-English text | the voice is English. Install a matching Piper model and set `TTS_VOICE`. |
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
scripts/chat --clean-text '**Paris** ```code``` 1. one 2. two'
scripts/chat --no-guard --no-play --text "Hello" --text "Again"
scripts/chat --text "What is the capital of France?" \
             --text "And Italy?" --verbose
scripts/chat --wav "$TMPDIR/question.wav"
```
