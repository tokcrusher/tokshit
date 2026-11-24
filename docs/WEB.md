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

The site stops at the first sentence by default (`TOKSHIT_STOP_SENTENCE=1`).
This is deliberate: without it the model repeats the same sentence until
`max_new` on weak prompts. The settings drawer has **stop at first sentence**;
disable it for full text answers. The language anchor
(`Отвечай по-русски.` / `Answer in English.`) is a two-token prefix, not the
old few-shot persona. Without it the model drifts to French for Paris.

## Speed

The web server starts in the coherent fast tier: top-6 experts for Russian,
top-4 for English, 3 passes, 6 workers. The settings drawer exposes every
knob. 2 passes is the quality cliff and loops; it is available for
experiments but is not a usable default. Measured on this phone: 4-7 tok/s
depending on temperature and tier; 10 tok/s is not reachable without dropping
passes/experts until output breaks.


## Notes

- The server binds `127.0.0.1` only. Do not expose it to the LAN without auth.
- Only one engine process; stop the TUI/other tokshit before starting the web
  server, or `safe_run` will make it wait for the model lock.
- The first load is ~7-15 s. The page shows `loading` and enables chat when
  ready.
- `scripts/web` is a thin wrapper around `scripts/voice_web.py`; `--models`,
  `--model`, `--lang`, `--port`, `--max-new` are supported by the backend.
