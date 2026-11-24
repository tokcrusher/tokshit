#!/usr/bin/env python3
"""Local web UI for tokshit voice/text chat.

Serves a phone-friendly page at http://127.0.0.1:8765. The persistent engine
is the same one used by scripts/voice_chat.py; text streams over HTTP, TTS is
synthesized per sentence with in-process Piper, and the browser microphone is
transcribed by whisper.cpp.

No external Python packages: http.server + the existing voice_chat module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import voice_chat as vc  # noqa: E402

DEFAULT_PORT = 8765
DEFAULT_MODEL = "inkling"


def env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.console = vc.Console(verbose=False)
        self.workdir = Path(tempfile.mkdtemp(prefix="voice-web-"))
        (self.workdir / "tts").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.chat_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.tts_lock = threading.Lock()
        self.tts_models: dict[str, vc.PiperTTS] = {}
        self.rec_lock = threading.Lock()
        self.rec_active = False
        self.rec_path = self.workdir / "phone-rec.m4a"
        self.model = args.model or DEFAULT_MODEL
        self.engine: vc.EngineSession | None = None
        self.last_stats: dict = {}
        self.lang = args.lang or os.environ.get("VOICE_LANG", "ru")
        self.settings = {
            "model": self.model,
            # Coherent fast tier by default: top-6 RU / top-4 EN, 3 passes.
            # Clear the field or set 8 for lossless. 2 passes loops; never
            # make it the default.
            "experts": env_int("VOICE_EXPERTS", 8),
            "passes": env_int("VOICE_PASSES", 3),
            "workers": env_int("VOICE_WORKERS", 4),
            "cache_mb": env_int("VOICE_CACHE_MB", 512),
            "kv_ctx": env_int("VOICE_KV_CTX", None),
            "layers": env_int("VOICE_LAYERS", None),
            "hot": env_int("VOICE_HOT", None),
            "q4": os.environ.get("VOICE_Q4", "0") not in ("", "0"),
            "one_sentence": os.environ.get("VOICE_ONE_SENTENCE", "0") not in ("", "0"),
            "asr_model": os.environ.get("VOICE_ASR_MODEL", "small" if self.lang == "ru" else "base"),
        }
        self.engine_args = SimpleNamespace(
            max_new=args.max_new,
            engine=None,
            chat_template=False,
            one_sentence=False,
            lang=self.lang,
            text=[],
            wav=None,
            verbose=False,
            seconds=0,
            max_seconds=20,
            hands_free=False,
            once=False,
            keep=False,
            no_play=True,
            no_guard=True,
            asr_model=None,
            fast=False,
        )

    # ---- engine -----------------------------------------------------
    def _apply_env(self) -> None:
        s = self.settings
        mapping = {
            "experts": "TOKSHIT_DRAFT_TOPK",
            "passes": "TOKSHIT_INK_PASSES",
            "workers": "TOKSHIT_POOL_WORKERS",
            "cache_mb": "TOKSHIT_EXPERT_CACHE_MB",
            "kv_ctx": "TOKSHIT_KV_CTX",
            "layers": "GGUF_NLAYERS",
            "hot": "TOKSHIT_HOT",
        }
        for key, env_name in mapping.items():
            value = s.get(key)
            if value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = str(value)
        if s.get("q4"):
            os.environ["TOKSHIT_HOT_Q4"] = "1"
        else:
            os.environ.pop("TOKSHIT_HOT_Q4", None)
        if s.get("one_sentence", False):
            os.environ["VOICE_ONE_SENTENCE"] = "1"
            os.environ["TOKSHIT_STOP_SENTENCE"] = "1"
        else:
            os.environ["VOICE_ONE_SENTENCE"] = "0"
            os.environ.pop("TOKSHIT_STOP_SENTENCE", None)
        model = vc.resolve_model_name(str(s.get("model") or DEFAULT_MODEL))
        os.environ["VOICE_MODEL"] = str(model)
        os.environ["VOICE_LANG"] = self.lang
        os.environ["TOKSHIT_CHAT_MAX_NEW"] = str(self.engine_args.max_new)
        # Minimal language anchor, not a persona few-shot. Without it the
        # model drifts to French for "Paris"; with the old canned prefix it
        # parrots the assistant persona. One instruction is enough.
        if self.lang == "ru":
            os.environ["VOICE_FEWSHOT"] = r"Q: Отвечай по-русски.\nA: Хорошо."
        else:
            os.environ["VOICE_FEWSHOT"] = r"Q: Answer in English.\nA: Okay."

    def start_engine(self) -> None:
        with self.lock:
            if self.engine is not None:
                try:
                    self.engine.stop()
                except Exception:
                    pass
            self._apply_env()
            self.engine_args.lang = self.lang
            eng = vc.EngineSession(self.engine_args, self.console)
            self.engine = eng
            eng.start()

    def status(self) -> dict:
        with self.lock:
            eng = self.engine
        ready = bool(eng and eng.ready_event.is_set())
        load = eng.child_load_s if eng and eng.child_load_s else (eng.load_s if eng else 0.0)
        err = eng.start_error if eng else None
        engine_stats: dict = {}
        if eng is not None and ready and not self.chat_lock.locked():
            try:
                with self.stats_lock:
                    engine_stats = eng.query_stats(timeout=0.15) or {}
            except Exception:
                engine_stats = {}
        thermal = None
        try:
            thermal = vc.thermal_max_c()
        except Exception:
            thermal = None
        return {
            "ready": ready,
            "loading": not ready and not err,
            "error": err,
            "load_s": round(load, 2),
            "model": str(vc.resolve_model_name(str(self.settings.get("model") or DEFAULT_MODEL))),
            "lang": self.lang,
            "settings": self.settings,
            "max_new": self.engine_args.max_new,
            "last": self.last_stats,
            "engine": engine_stats,
            "thermal": thermal,
        }

    def clear(self) -> None:
        # /clear wipes the prefix too; a restart re-prefills the language
        # anchor. The model reload is ~8-10 s, shown as loading in the UI.
        with self.chat_lock:
            with self.lock:
                self.last_stats = {}
            self.start_engine()

    # ---- TTS --------------------------------------------------------
    def tts(self, text: str, lang: str) -> bytes:
        text = text.strip()
        if not text:
            raise ValueError("empty text")
        voice = "ru_RU-irina-medium" if lang == "ru" else "en_US-lessac-medium"
        key = hashlib.sha256((voice + "\0" + text).encode("utf-8")).hexdigest()[:24]
        out = self.workdir / "tts" / f"{key}.wav"
        if out.is_file() and out.stat().st_size > 44:
            return out.read_bytes()
        with self.tts_lock:
            piper = self.tts_models.get(voice)
            if piper is None:
                old = os.environ.get("TTS_VOICE")
                os.environ["TTS_VOICE"] = voice
                try:
                    piper = vc.PiperTTS(vc.Console(verbose=False))
                    piper.start()
                    piper.ready_event.wait(timeout=90)
                finally:
                    if old is None:
                        os.environ.pop("TTS_VOICE", None)
                    else:
                        os.environ["TTS_VOICE"] = old
                if piper.error:
                    raise RuntimeError(piper.error)
                self.tts_models[voice] = piper
            piper.synth(text, out)
        return out.read_bytes()

    # ---- recording (phone mic via Termux:API) -----------------------
    def start_rec(self) -> dict:
        with self.rec_lock:
            if self.rec_active:
                return {"ok": True, "already": True}
            # clear a stale recorder from an earlier crashed session
            subprocess.run(["termux-microphone-record", "-q"],
                           capture_output=True, text=True, timeout=4)
            try:
                self.rec_path.unlink()
            except OSError:
                pass
            p = subprocess.run(
                ["termux-microphone-record", "-f", str(self.rec_path),
                 "-e", "aac", "-r", "16000", "-c", "1", "-b", "48", "-l", "0"],
                capture_output=True, text=True, timeout=12,
            )
            output = (p.stdout or "") + (p.stderr or "")
            if p.returncode != 0 or "Please grant" in output or '"error"' in output:
                raise RuntimeError(output.strip() or "termux-microphone-record failed")
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not self.rec_path.is_file():
                time.sleep(0.05)
            if not self.rec_path.is_file():
                raise RuntimeError("microphone did not create a file")
            self.rec_active = True
            return {"ok": True, "path": str(self.rec_path)}

    def stop_rec(self, lang: str) -> str:
        with self.rec_lock:
            p = subprocess.run(["termux-microphone-record", "-q"],
                               capture_output=True, text=True, timeout=8)
            self.rec_active = False
            path = self.rec_path
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(
                    (p.stderr or p.stdout or "empty recording").strip()
                )
            # wait for the m4a to finalize
            last = -1
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                size = path.stat().st_size if path.is_file() else 0
                if size == last and size > 0:
                    break
                last = size
                time.sleep(0.15)
            return self.asr_file(path, lang)

    # ---- ASR --------------------------------------------------------
    def asr_file(self, src: Path, lang: str) -> str:
        wav = src.with_suffix(".conv.wav")
        try:
            conv = subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-y", "-i", str(src), "-ar", "16000", "-ac", "1",
                 "-c:a", "pcm_s16le", str(wav)],
                capture_output=True, text=True, timeout=60,
            )
            if conv.returncode != 0 or not wav.is_file():
                raise RuntimeError((conv.stderr or "ffmpeg failed")[-500:])
            env = os.environ.copy()
            env["ASR_LANG"] = "ru" if lang == "ru" else "en"
            p = subprocess.run(
                [str(vc.ASR_SH), str(wav), str(self.settings.get("asr_model", "base"))],
                env=env, capture_output=True, text=True, timeout=180,
            )
            text = (p.stdout or "").strip()
            if text in ("[BLANK_AUDIO]", "[NO SPEECH]"):
                text = ""
            if not text and p.returncode != 0:
                raise RuntimeError((p.stderr or p.stdout or "ASR failed")[-500:])
            return text
        finally:
            for path in (wav, src):
                try:
                    path.unlink()
                except OSError:
                    pass

    def asr(self, raw: bytes, suffix: str, lang: str) -> str:
        src = self.workdir / f"asr-{int(time.time() * 1000)}{suffix or '.webm'}"
        src.write_bytes(raw)
        return self.asr_file(src, lang)


def json_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def clean_model_text(text: str, lang: str) -> str:
    s = vc.strip_invisible(text)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    if lang != "zh":
        s = CJK_RE.sub("", s)
    return s


class Handler(BaseHTTPRequestHandler):
    server_version = "tokshit-web/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write("[web] " + (fmt % args) + "\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json_bytes(obj), "application/json; charset=utf-8")

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/", "/index.html"):
            try:
                page = WEB_UI.read_bytes()
            except OSError:
                self._json(500, {"error": f"missing {WEB_UI}"})
                return
            self._send(200, page, "text/html; charset=utf-8")
            return
        if url.path == "/api/status":
            self._json(200, self.app.status())
            return
        if url.path == "/api/models":
            try:
                p = subprocess.run(
                    [str(vc.engine_path(SimpleNamespace(engine=None))), "--models"],
                    capture_output=True, text=True, timeout=10,
                )
                self._send(200, (p.stdout + p.stderr).encode("utf-8"),
                           "text/plain; charset=utf-8")
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        try:
            if url.path == "/api/chat":
                self._chat(json.loads(self._body() or b"{}"))
            elif url.path == "/api/tts":
                self._tts(json.loads(self._body() or b"{}"))
            elif url.path == "/api/asr":
                qs = urllib.parse.parse_qs(url.query)
                lang = (qs.get("lang") or [self.app.lang])[0]
                text = self.app.asr(self._body(), ".webm", lang)
                self._json(200, {"text": text})
            elif url.path == "/api/rec/start":
                self._json(200, self.app.start_rec())
            elif url.path == "/api/rec/stop":
                qs = urllib.parse.parse_qs(url.query)
                lang = (qs.get("lang") or [self.app.lang])[0]
                text = self.app.stop_rec(lang)
                self._json(200, {"text": text})
            elif url.path == "/api/clear":
                self.app.clear()
                self._json(200, {"ok": True})
            elif url.path == "/api/settings":
                data = json.loads(self._body() or b"{}")
                with self.app.lock:
                    if data.get("max_new") not in (None, "", "auto"):
                        self.app.engine_args.max_new = int(data["max_new"])
                    for key in ("model", "experts", "passes", "workers",
                                "cache_mb", "kv_ctx", "layers", "hot"):
                        if key in data:
                            value = data[key]
                            self.app.settings[key] = (
                                None if value in (None, "", "auto") else int(value)
                                if key != "model" else value
                            )
                    if "q4" in data:
                        self.app.settings["q4"] = bool(data["q4"])
                    if "one_sentence" in data:
                        self.app.settings["one_sentence"] = bool(data["one_sentence"])
                    if data.get("asr_model") in ("tiny", "base", "small"):
                        self.app.settings["asr_model"] = data["asr_model"]
                    if "lang" in data:
                        self.app.lang = "ru" if data["lang"] == "ru" else "en"
                self.app.start_engine()
                self._json(200, self.app.status())
            else:
                self._json(404, {"error": "not found"})
        except BrokenPipeError:
            pass
        except Exception as exc:
            sys.stderr.write("[web] POST %s: %s\n%s\n" % (
                url.path, exc, traceback.format_exc()))
            try:
                self._json(500, {"error": str(exc)})
            except Exception:
                pass

    def _chat(self, data: dict) -> None:
        text = str(data.get("text") or "").strip()
        lang = "ru" if data.get("lang") == "ru" else "en"
        if not text:
            self._json(400, {"error": "empty text"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            with self.app.chat_lock:
                eng = self.app.engine
                if eng is None:
                    raise RuntimeError("engine not started")
                eng.wait_ready(timeout=180)
                stats = vc.TurnStats()
                def on_text(chunk: str) -> None:
                    chunk = clean_model_text(chunk, lang)
                    if not chunk:
                        return
                    try:
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                quality = (
                    " Ответь по-русски, подробно и по делу, 2-3 предложения."
                    if lang == "ru" else
                    " Answer in English, detailed and to the point, 2-3 sentences."
                )
                rc, err_lines = eng.ask(text + quality, on_text, stats)
                with self.app.lock:
                    self.app.last_stats = vc.parse_engine_stats(
                        "\n".join(err_lines)
                    )
        except Exception as exc:
            try:
                self.wfile.write(f"\n[error] {exc}\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    def _tts(self, data: dict) -> None:
        text = str(data.get("text") or "").strip()
        lang = "ru" if data.get("lang") == "ru" else "en"
        wav = self.app.tts(text, lang)
        self._send(200, wav, "audio/wav")


WEB_UI = Path(__file__).with_name("web_ui.html")



def main() -> int:
    ap = argparse.ArgumentParser(description="tokshit local web UI")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=os.environ.get("VOICE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--lang", default=os.environ.get("VOICE_LANG", "ru"),
                    choices=("ru", "en"))
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--open", action="store_true", help="open the URL in the browser")
    args = ap.parse_args()

    app = App(args)
    app.start_engine()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.app = app  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}/"
    print(f"[web] serving {url} (model={app.model}, lang={app.lang})", flush=True)
    if args.open:
        def opener() -> None:
            time.sleep(1.0)
            try:
                subprocess.run(["termux-open-url", url], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass
        threading.Thread(target=opener, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with app.lock:
            if app.engine is not None:
                app.engine.stop()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
