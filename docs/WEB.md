# tokshit web UI

Local, phone-friendly web chat for the same persistent `build/tokshit`
engine. No external Python packages, no cloud, no API keys.

```sh
cd ~/tokshit
scripts/web --open          # http://127.0.0.1:8765/
scripts/web --port 9000     # different port
```

`--open` calls `termux-open-url`; if the browser did not open, tap:

```
http://127.0.0.1:8765/
```

## What works

- Text chat with token streaming (`POST /api/chat`).
- Russian/English toggle: `ASR_LANG=ru|en`, `ru_RU-irina-medium` or
  `en_US-lessac-medium` TTS, per-turn language directive.
- Browser microphone: MediaRecorder -> `POST /api/asr` -> whisper.cpp base.
  Needs microphone permission for the browser and a secure context
  (`127.0.0.1` counts as secure).
- Sentence/clause TTS: the page requests `/api/tts` as text arrives and plays
  chunks in order; answers do not wait for the full message.
- Settings drawer: model, experts, passes, workers, cache MB, KV ctx, layers,
  hot dense, Q4. Apply restarts the engine and reloads the model.
- Clear sends `/clear` to the engine, so KV is actually reset.

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/` | page |
| GET | `/api/status` | ready/loading, model, settings |
| GET | `/api/models` | `tokshit --models` output |
| POST | `/api/chat` | `{text, lang}` -> streamed text |
| POST | `/api/tts` | `{text, lang}` -> `audio/wav` |
| POST | `/api/asr?lang=ru` | raw webm/m4a body -> `{text}` |
| POST | `/api/rec/start` | start phone mic recording |
| POST | `/api/rec/stop?lang=ru` | stop, transcribe, return text |
| POST | `/api/clear` | restart engine and reset KV |
| POST | `/api/settings` | restart with model/runtime settings |

## Microphone

The composer has a source switch:

- **mic: phone** (default) — the server records with `termux-microphone-record`,
  then converts the m4a to 16 kHz WAV and runs whisper. This works even when
  the browser cannot access the microphone. Requires the Termux:API APK and
  `RECORD_AUDIO` permission.
- **mic: browser** — `MediaRecorder` in the browser; sends WebM to
  `/api/asr`. Requires Chrome microphone permission for
  `http://127.0.0.1:8765/`.

Both paths show a timer and stop on the mic button or **stop**. If the phone
mic errors, the page prints the exact Termux:API message; switch to browser
mic or fix the permission.

## Answer length

By default the site asks for a full 2-3 sentence answer; the settings drawer
has **stop at first sentence** (default off). The engine uses a minimal
language anchor (`Q: Отвечай по-русски. A: Хорошо.`, not the old persona
few-shot) plus a per-turn "2-3 sentences" instruction. Without the anchor the
model drifts to French for Paris; without the length instruction it answers
in 3-5 words. Turn on **stop at first sentence** if a weak prompt starts to
loop.

## Speed

The web server starts in the **quality** tier: top-8 experts, 3 passes,
4 workers, full answers, whisper `small` for Russian. The settings drawer
exposes every knob. `fast` is top-6 RU / top-4 EN with 6 workers; `turbo`
(4 experts, 2 passes) reaches ~12 tok/s but loops and is not usable.
Measured on this phone: 3-7 tok/s depending on temperature and tier; 10 tok/s
is not reachable without dropping passes until output breaks. On a 90+ C
phone even the quality tier falls toward 3-4 tok/s; cool it before judging.


## Notes

- The server binds `127.0.0.1` only. Do not expose it to the LAN without auth.
- Only one engine process; stop the TUI/other tokshit before starting the web
  server, or `safe_run` will make it wait for the model lock.
- The first load is ~7-15 s. The page shows `loading` and enables chat when
  ready.
- `cpu temp` uses CPU/SoC thermal zones only. Battery/charger/camera zones can
  read 90+ C while the phone is cool; the old gauge used the global maximum.
- `ram free` is `MemAvailable`. The Inkling GGUF is ~8.7 GiB; if free RAM is
  below that, part of the experts is read from flash every token and speed
  drops to 2-3 tok/s. Close other apps or use a smaller model; code cannot
  cache weights that do not fit.
- `scripts/web` is a thin wrapper around `scripts/voice_web.py`; `--models`,
  `--model`, `--lang`, `--port`, `--max-new` are supported by the backend.
