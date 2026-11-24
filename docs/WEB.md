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
| POST | `/api/clear` | engine `/clear` |
| POST | `/api/settings` | restart with model/runtime settings |

## Notes

- The server binds `127.0.0.1` only. Do not expose it to the LAN without auth.
- Only one engine process; stop the TUI/other tokshit before starting the web
  server, or `safe_run` will make it wait for the model lock.
- The first load is ~7-15 s. The page shows `loading` and enables chat when
  ready.
- `scripts/web` is a thin wrapper around `scripts/voice_web.py`; `--models`,
  `--model`, `--lang`, `--port`, `--max-new` are supported by the backend.
