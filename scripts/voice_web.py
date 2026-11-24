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
        self.tts_lock = threading.Lock()
        self.tts_models: dict[str, vc.PiperTTS] = {}
        self.model = args.model or DEFAULT_MODEL
        self.engine: vc.EngineSession | None = None
        self.lang = args.lang or os.environ.get("VOICE_LANG", "ru")
        self.settings = {
            "model": self.model,
            "experts": env_int("VOICE_EXPERTS", None),
            "passes": env_int("VOICE_PASSES", None),
            "workers": env_int("VOICE_WORKERS", None),
            "cache_mb": env_int("VOICE_CACHE_MB", None),
            "kv_ctx": env_int("VOICE_KV_CTX", None),
            "layers": env_int("VOICE_LAYERS", None),
            "hot": env_int("VOICE_HOT", None),
            "q4": os.environ.get("VOICE_Q4", "0") not in ("", "0"),
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
        model = vc.resolve_model_name(str(s.get("model") or DEFAULT_MODEL))
        os.environ["VOICE_MODEL"] = str(model)
        os.environ["VOICE_LANG"] = self.lang
        os.environ["TOKSHIT_CHAT_MAX_NEW"] = str(self.engine_args.max_new)

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
        return {
            "ready": ready,
            "loading": not ready and not err,
            "error": err,
            "load_s": round(load, 2),
            "model": str(vc.resolve_model_name(str(self.settings.get("model") or DEFAULT_MODEL))),
            "lang": self.lang,
            "settings": self.settings,
            "max_new": self.engine_args.max_new,
        }

    def clear(self) -> None:
        with self.lock:
            if self.engine is not None:
                self.engine.clear_context()

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

    # ---- ASR --------------------------------------------------------
    def asr(self, raw: bytes, suffix: str, lang: str) -> str:
        src = self.workdir / f"asr-{int(time.time() * 1000)}{suffix or '.webm'}"
        src.write_bytes(raw)
        env = os.environ.copy()
        env["ASR_LANG"] = "ru" if lang == "ru" else "en"
        try:
            p = subprocess.run(
                [str(vc.ASR_SH), str(src), "base"],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
        finally:
            try:
                src.unlink()
            except OSError:
                pass
        text = (p.stdout or "").strip()
        if not text and p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout or "ASR failed")[-500:])
        return text


def json_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


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
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
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
            elif url.path == "/api/clear":
                self.app.clear()
                self._json(200, {"ok": True})
            elif url.path == "/api/settings":
                data = json.loads(self._body() or b"{}")
                with self.app.lock:
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
        directive = "Отвечай по-русски. " if lang == "ru" else "Answer in English. "
        try:
            with self.app.chat_lock:
                eng = self.app.engine
                if eng is None:
                    raise RuntimeError("engine not started")
                eng.wait_ready(timeout=180)
                stats = vc.TurnStats()
                def on_text(chunk: str) -> None:
                    if not chunk:
                        return
                    try:
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                eng.ask(directive + text, on_text, stats)
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


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>tokshit</title>
<style>
:root{color-scheme:dark;--bg:#0b0e13;--panel:#131820;--panel2:#0f141b;--line:#243040;--text:#e8edf5;--muted:#8b98a9;--accent:#5eead4;--accent2:#38bdf8;--warn:#fbbf24;--user:#1b2a3a;--bot:#131a24;--danger:#fb7185}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{background:radial-gradient(1200px 600px at 50% -20%,#14202b 0%,var(--bg) 60%);color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--line);background:rgba(11,14,19,.9);backdrop-filter:blur(12px);position:sticky;top:0;z-index:5}
.logo{font-weight:750;letter-spacing:-.02em;font-size:16px}
.logo span{color:var(--accent)}
#status{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:12px;min-width:0}
.dot{width:8px;height:8px;border-radius:50%;background:var(--warn);box-shadow:0 0 0 3px rgba(251,191,36,.12)}
.dot.ok{background:var(--accent);box-shadow:0 0 0 3px rgba(94,234,212,.12)}
.dot.err{background:var(--danger)}
#model{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:38vw}
.spacer{flex:1}
button,select,input,textarea{font:inherit;color:inherit}
button{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 12px;cursor:pointer;transition:border-color .15s,background .15s}
button:hover{border-color:#3b4d63;background:#18212c}
button.primary{background:var(--accent);color:#04211d;border-color:var(--accent);font-weight:700}
button.primary:disabled{opacity:.5;cursor:default}
button.ghost{background:transparent}
#settingsBtn{font-size:12px;padding:7px 10px}
#chat{flex:1;overflow-y:auto;padding:18px 14px 8px;scroll-behavior:smooth}
.wrap{max-width:980px;margin:0 auto}
.msg{display:flex;gap:10px;margin:0 0 14px;align-items:flex-start}
.msg .who{flex:0 0 38px;height:38px;border-radius:11px;display:grid;place-items:center;font-size:11px;font-weight:750;border:1px solid var(--line)}
.msg.user .who{background:var(--user);color:var(--accent2)}
.msg.bot .who{background:var(--bot);color:var(--accent)}
.bubble{padding:11px 14px;border-radius:14px;border:1px solid var(--line);background:var(--panel2);max-width:min(760px,86vw);white-space:pre-wrap;overflow-wrap:anywhere}
.msg.user .bubble{background:var(--user);border-color:#24384d}
.bubble.streaming:after{content:"";display:inline-block;width:7px;height:15px;background:var(--accent);margin-left:3px;vertical-align:-2px;animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
.hint{color:var(--muted);font-size:12px;text-align:center;padding:10px 0 16px}
#bar{position:sticky;bottom:0;border-top:1px solid var(--line);background:rgba(11,14,19,.94);backdrop-filter:blur(12px);padding:10px 12px calc(10px + env(safe-area-inset-bottom))}
.bar-inner{max-width:980px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
#text{flex:1;min-height:44px;max-height:140px;resize:none;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:11px 13px;outline:none}
#text:focus{border-color:#3b556e}
.iconbtn{width:46px;height:46px;padding:0;display:grid;place-items:center;border-radius:14px;flex:0 0 auto}
.iconbtn.rec{background:var(--danger);border-color:var(--danger);color:#2a0710;animation:pulse 1.2s infinite}
@keyframes pulse{50%{box-shadow:0 0 0 8px rgba(251,113,133,.12)}}
#send{width:auto;height:46px;padding:0 18px;border-radius:14px}
#meta{max-width:980px;margin:6px auto 0;display:flex;gap:10px;align-items:center;color:var(--muted);font-size:11px;flex-wrap:wrap}
#meta label{display:flex;gap:5px;align-items:center;cursor:pointer}
#drawer{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;z-index:20;align-items:center;justify-content:center;padding:14px}
#drawer.open{display:flex}
.card{width:min(560px,100%);max-height:90vh;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 30px 80px rgba(0,0,0,.5)}
.card h2{margin:0 0 4px;font-size:18px}
.card p{margin:0 0 14px;color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.grid .full{grid-column:1/-1}
label.f{display:flex;flex-direction:column;gap:5px;font-size:12px;color:var(--muted)}
label.f input,label.f select{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:9px 10px;color:var(--text)}
.checks{display:flex;gap:16px;margin:12px 0;color:var(--text);font-size:13px}
.checks label{display:flex;gap:7px;align-items:center}
.actions{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
#toast{position:fixed;left:50%;bottom:90px;transform:translateX(-50%);background:#1f2937;border:1px solid var(--line);border-radius:12px;padding:10px 14px;font-size:13px;opacity:0;pointer-events:none;transition:opacity .2s;z-index:30;max-width:90vw}
#toast.show{opacity:1}
@media(max-width:600px){.msg .who{display:none}.bubble{max-width:100%}#model{max-width:30vw}#send{padding:0 14px}}
</style>
</head>
<body>
<header>
  <div class="logo">tok<span>shit</span></div>
  <div id="status"><span class="dot" id="dot"></span><span id="statustext">loading</span></div>
  <div id="model"></div>
  <div class="spacer"></div>
  <select id="lang" title="language" style="background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:7px 9px">
    <option value="ru">RU</option><option value="en">EN</option>
  </select>
  <button id="settingsBtn" class="ghost">settings</button>
</header>

<main id="chat"><div class="wrap" id="messages">
  <div class="hint">Text chat works immediately. Mic needs browser permission.<br>Engine loads once; answers stream token by token.</div>
</div></main>

<div id="bar"><div class="bar-inner">
  <button id="mic" class="iconbtn" title="record">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8"/></svg>
  </button>
  <textarea id="text" rows="1" placeholder="Type a message..."></textarea>
  <button id="send" class="primary">Send</button>
</div>
<div id="meta">
  <label><input type="checkbox" id="tts" checked> speak answers</label>
  <label><input type="checkbox" id="autosend" checked> auto-send transcript</label>
  <span id="cfg"></span>
</div></div>

<div id="drawer"><div class="card">
  <h2>Model / runtime</h2>
  <p>Applying restarts the engine and reloads the model (~8-15 s). Explicit values win over presets.</p>
  <div class="grid">
    <label class="f full">model<input id="s_model" placeholder="inkling"></label>
    <label class="f">experts<input id="s_experts" type="number" min="1" max="8" placeholder="auto"></label>
    <label class="f">passes<input id="s_passes" type="number" min="1" max="3" placeholder="3"></label>
    <label class="f">workers<input id="s_workers" type="number" min="1" max="8" placeholder="4"></label>
    <label class="f">cache MB<input id="s_cache" type="number" min="0" placeholder="512"></label>
    <label class="f">kv ctx<input id="s_kv" type="number" min="128" placeholder="2048"></label>
    <label class="f">layers<input id="s_layers" type="number" min="1" placeholder="all"></label>
    <label class="f">hot dense<input id="s_hot" type="number" min="0" max="1" placeholder="0"></label>
  </div>
  <div class="checks">
    <label><input type="checkbox" id="s_q4"> Q4 dense (LOSSY)</label>
  </div>
  <div class="actions">
    <button id="modelsBtn" class="ghost">list models</button>
    <button id="applyBtn" class="primary">Apply + restart</button>
    <button id="closeBtn">Close</button>
  </div>
  <pre id="modelsOut" style="white-space:pre-wrap;color:var(--muted);font-size:11px;max-height:180px;overflow:auto"></pre>
</div></div>
<div id="toast"></div>

<script>
const $ = (s) => document.querySelector(s);
const messages = $("#messages");
const chat = $("#chat");
let lang = "ru";
let busy = false;
let ttsQueue = Promise.resolve();
let rec = null, chunks = [], recording = false;

function toast(text, ms=2600){ const t=$("#toast"); t.textContent=text; t.classList.add("show"); clearTimeout(t._t); t._t=setTimeout(()=>t.classList.remove("show"),ms); }
function scroll(){ chat.scrollTop = chat.scrollHeight; }
function add(role, text=""){
  const wrap=document.createElement("div"); wrap.className="msg "+role;
  const who=document.createElement("div"); who.className="who"; who.textContent= role==="user"?"YOU":"INK";
  const b=document.createElement("div"); b.className="bubble"; b.textContent=text;
  wrap.appendChild(who); wrap.appendChild(b); messages.appendChild(wrap); scroll(); return b;
}
function setStatus(st){
  const dot=$("#dot"), tx=$("#statustext"), m=$("#model");
  dot.className="dot"+(st.ready?" ok":st.error?" err":"");
  tx.textContent=st.error?"error":st.ready?"ready":"loading";
  if(st.error) tx.title=st.error;
  m.textContent=(st.model||"").split("/").pop()+" · "+(st.settings?.experts??"auto")+" exp · "+(st.settings?.passes??3)+" pass";
  $("#cfg").textContent = "kv "+(st.settings?.kv_ctx??"auto")+" · cache "+(st.settings?.cache_mb??"auto")+"MB · w "+(st.settings?.workers??4);
}
async function statusLoop(){
  try{ const r=await fetch("/api/status"); setStatus(await r.json()); }catch(e){}
  setTimeout(statusLoop, 2000);
}
async function synth(text){
  if(!$("#tts").checked || !text.trim()) return;
  ttsQueue = ttsQueue.then(async()=>{
    try{
      const r=await fetch("/api/tts",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,lang})});
      if(!r.ok) return;
      const blob=await r.blob(); const url=URL.createObjectURL(blob);
      await new Promise(res=>{ const a=new Audio(url); a.onended=()=>{URL.revokeObjectURL(url);res()}; a.onerror=()=>{URL.revokeObjectURL(url);res()}; a.play().catch(()=>res()); });
    }catch(e){}
  });
}
function feedTts(buf, final=false){
  // split on sentence ends and clauses; keep a small tail
  let out=[];
  const re=/([^.!?\n]{12,}[.!?]+|[^,;:\n]{28,}[,;:])/g;
  let last=0,m;
  while((m=re.exec(buf))){ out.push(m[0]); last=re.lastIndex; }
  if(final && buf.slice(last).trim()) out.push(buf.slice(last).trim());
  for(const s of out) synth(s);
  return out.length?buf.slice(last):buf;
}
async function send(){
  const ta=$("#text"); const text=ta.value.trim(); if(!text||busy) return;
  ta.value=""; ta.style.height="auto"; add("user",text);
  const bubble=add("bot",""); bubble.classList.add("streaming"); busy=true; $("#send").disabled=true;
  let ttsBuf="";
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,lang})});
    if(!r.ok || !r.body) throw new Error("HTTP "+r.status);
    const rd=r.body.getReader(); const dec=new TextDecoder();
    while(true){ const {done,value}=await rd.read(); if(done) break; const s=dec.decode(value,{stream:true}); bubble.textContent+=s; scroll(); ttsBuf=feedTts(ttsBuf+s); }
    ttsBuf=feedTts(ttsBuf,true); if(ttsBuf.trim()) synth(ttsBuf);
  }catch(e){ bubble.textContent+="\n[error] "+e; }
  bubble.classList.remove("streaming"); busy=false; $("#send").disabled=false; ta.focus();
}
async function toggleMic(){
  if(recording){ rec.stop(); return; }
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
    rec=new MediaRecorder(stream,{mimeType:MediaRecorder.isTypeSupported("audio/webm;codecs=opus")?"audio/webm;codecs=opus":""});
    chunks=[];
    rec.ondataavailable=e=>{ if(e.data.size) chunks.push(e.data); };
    rec.onstop=async()=>{
      recording=false; $("#mic").classList.remove("rec"); stream.getTracks().forEach(t=>t.stop());
      const blob=new Blob(chunks,{type:rec.mimeType||"audio/webm"});
      if(blob.size<800){ toast("recording too short"); return; }
      toast("transcribing...");
      try{
        const r=await fetch("/api/asr?lang="+lang,{method:"POST",headers:{"Content-Type":"application/octet-stream"},body:blob});
        const j=await r.json(); if(j.error) throw new Error(j.error);
        if(!j.text){ toast("silence / not recognized"); return; }
        $("#text").value=j.text; if($("#autosend").checked) send();
      }catch(e){ toast("ASR: "+e); }
    };
    rec.start(); recording=true; $("#mic").classList.add("rec"); toast("recording — tap mic to stop");
  }catch(e){ toast("mic: "+e); }
}
function openSettings(){
  const s=window._status?.settings||{};
  $("#s_model").value=s.model||"inkling"; $("#s_experts").value=s.experts??"";
  $("#s_passes").value=s.passes??""; $("#s_workers").value=s.workers??"";
  $("#s_cache").value=s.cache_mb??""; $("#s_kv").value=s.kv_ctx??"";
  $("#s_layers").value=s.layers??""; $("#s_hot").value=s.hot??"";
  $("#s_q4").checked=!!s.q4; $("#drawer").classList.add("open");
}
async function applySettings(){
  const val=id=>{ const v=$(id).value.trim(); return v===""?null:(id==="#s_model"?v:Number(v)); };
  const body={model:val("#s_model"),experts:val("#s_experts"),passes:val("#s_passes"),workers:val("#s_workers"),cache_mb:val("#s_cache"),kv_ctx:val("#s_kv"),layers:val("#s_layers"),hot:val("#s_hot"),q4:$("#s_q4").checked,lang};
  $("#applyBtn").disabled=true; toast("restarting engine...",5000);
  try{ const r=await fetch("/api/settings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}); const j=await r.json(); window._status=j; setStatus(j); toast("settings applied"); }
  catch(e){ toast("settings: "+e); }
  finally{ $("#applyBtn").disabled=false; }
}
$("#send").onclick=send;
$("#text").addEventListener("keydown",e=>{ if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();} });
$("#text").addEventListener("input",e=>{ e.target.style.height="auto"; e.target.style.height=Math.min(140,e.target.scrollHeight)+"px"; });
$("#mic").onclick=toggleMic;
$("#lang").onchange=e=>{ lang=e.target.value; toast("language: "+lang.toUpperCase()); };
$("#settingsBtn").onclick=openSettings; $("#closeBtn").onclick=()=>$("#drawer").classList.remove("open");
$("#applyBtn").onclick=applySettings;
$("#modelsBtn").onclick=async()=>{ $("#modelsOut").textContent="loading..."; try{ const r=await fetch("/api/models"); $("#modelsOut").textContent=await r.text(); }catch(e){ $("#modelsOut").textContent=String(e); } };
$("#drawer").addEventListener("click",e=>{ if(e.target.id==="drawer") $("#drawer").classList.remove("open"); });
statusLoop();
fetch("/api/status").then(r=>r.json()).then(j=>{window._status=j;setStatus(j);});
</script>
</body></html>
"""


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
