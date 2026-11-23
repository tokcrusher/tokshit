#!/data/data/com.termux/files/usr/bin/env python
"""voice_chat.py — голосовой чат с tokshit (persistent chat + streaming TTS).

mic -> whisper.cpp ASR -> tokshit persistent chat -> Piper TTS -> speaker

Движок запускается ОДИН раз на сессию:
    safe_run.sh build/tokshit --temperature 0 <model.gguf>
и общается по протоколу TOKSHIT_CHAT_PROTOCOL=1: одна строка в stdin,
ответ + "\\n\\x1e" в stdout. Модель и KV живут между ходами.

Ответ читается по мере генерации, режется на предложения и сразу уходит
в TTS; синтез и проигрывание идут в фоновых потоках, пока движок ещё
декодирует. Один TTS-воркер и один плеер — голоса не накладываются.

Режимы:
  scripts/voice_chat.py                       # интерактивно (Enter/q)
  scripts/voice_chat.py --seconds 5           # запись ровно 5 с
  scripts/voice_chat.py --hands-free --seconds 5
  scripts/voice_chat.py --wav in.wav          # без микрофона
  scripts/voice_chat.py --text "..."          # без ASR
  scripts/voice_chat.py --text "..." --text "..."   # два хода, один движок
  scripts/voice_chat.py --check               # зависимости
  scripts/voice_chat.py --no-guard ...        # пропустить probe движка

Аудио — ~/audio/{asr.sh,tts.sh}; они только вызываются, не меняются.
"""

from __future__ import annotations

import argparse
import codecs
import html
import importlib.util
import json
import os
import queue
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path(os.environ.get("HOME", "/data/data/com.termux/files/home"))
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", HOME / "audio"))
TOKSHIT_ROOT = Path(os.environ.get("TOKSHIT_ROOT", HOME / "tokshit"))
ASR_SH = AUDIO_DIR / "asr.sh"
TTS_SH = AUDIO_DIR / "tts.sh"
MODELS_DIR = AUDIO_DIR / "models"
WHISPER_DIR = AUDIO_DIR / "whisper.cpp"
DEFAULT_MODEL = HOME / "Inkling-2-Mini-GGUF" / "Inkling-2-Mini-UD-Q2_K_XL.gguf"
DEFAULT_ENGINE = TOKSHIT_ROOT / "build" / "tokshit"
SAFE_RUN = TOKSHIT_ROOT / "scripts" / "safe_run.sh"
GUARD_LAUNCHER = TOKSHIT_ROOT / "scripts" / "run_inkling2.sh"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")
ENGINE_RE = re.compile(
    r"\[prefill:\s*(\d+)\s+tok.*?([\d.]+)s\s*\|\s*генерация:\s*(\d+)\s+tok"
    r".*?([\d.]+)s\s*=\s*([\d.]+)\s+tok/s\]"
)
LOADED_RE = re.compile(r"loaded in\s+([\d.]+)s")
PREFIX_RE = re.compile(r"\[prefix\]\s*(\d+)\s+tok.*?([\d.]+)s")
RS = "\x1e"
TMPDIR = Path(tempfile.gettempdir())


class VoiceError(RuntimeError):
    pass


# ────────────────────────────── processes ──────────────────────────────

_PROC_LOCK = threading.Lock()
_PROCS: set[subprocess.Popen] = set()


def _register(p: subprocess.Popen) -> None:
    with _PROC_LOCK:
        _PROCS.add(p)


def _unregister(p: subprocess.Popen) -> None:
    with _PROC_LOCK:
        _PROCS.discard(p)


def _kill_group(p: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    if p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            p.send_signal(sig)
        except ProcessLookupError:
            pass


def kill_proc(p: subprocess.Popen, wait: float = 2.0) -> None:
    _kill_group(p, signal.SIGTERM)
    try:
        p.wait(timeout=wait)
        return
    except subprocess.TimeoutExpired:
        pass
    _kill_group(p, signal.SIGKILL)
    try:
        p.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def shutdown_procs() -> None:
    with _PROC_LOCK:
        procs = list(_PROCS)
    for p in procs:
        _kill_group(p, signal.SIGTERM)
    deadline = time.monotonic() + 3.0
    for p in procs:
        while p.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
    for p in procs:
        if p.poll() is None:
            _kill_group(p, signal.SIGKILL)


def run_cmd(
    cmd: list[str],
    *,
    timeout: float | None,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    stdin_data: str | None = None,
) -> tuple[int, str, str]:
    try:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=env,
            cwd=str(cwd) if cwd else None,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise VoiceError(f"command not found: {cmd[0]}")
    _register(p)
    try:
        try:
            out, err = p.communicate(stdin_data, timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_proc(p)
            try:
                out, err = p.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                out, err = "", ""
            return 124, out or "", (err or "") + f"\n[timeout after {timeout:.0f}s]"
        return p.returncode, out or "", err or ""
    finally:
        _unregister(p)


# ────────────────────────────── console ──────────────────────────────

class Console:
    """Ответ — в stdout, логи/статусы — в stderr, без мешанины."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._lock = threading.Lock()
        self._answer_open = False
        self._turn_open = False
        self._deferred: list[str] = []
        self._status_open = False

    def answer(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._turn_open = True
            sys.stdout.write(text)
            sys.stdout.flush()
            self._answer_open = not text.endswith("\n")

    def end_answer(self) -> None:
        with self._lock:
            if self._answer_open:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._answer_open = False
            self._turn_open = False
            self._flush_locked()

    def log(self, msg: str, *, defer: bool = False) -> None:
        with self._lock:
            self._clear_status_locked()
            if defer and self._turn_open:
                self._deferred.append(msg)
                return
            self._flush_locked()
            print(msg, file=sys.stderr, flush=True)

    def status(self, msg: str) -> None:
        with self._lock:
            self._flush_locked()
            if sys.stderr.isatty():
                sys.stderr.write("\r\x1b[K" + msg)
                sys.stderr.flush()
                self._status_open = True
            else:
                print(msg, file=sys.stderr, flush=True)

    def clear_status(self) -> None:
        with self._lock:
            self._clear_status_locked()

    def _clear_status_locked(self) -> None:
        if self._status_open:
            sys.stderr.write("\r\x1b[K")
            sys.stderr.flush()
            self._status_open = False

    def _flush_locked(self) -> None:
        self._clear_status_locked()
        while self._deferred:
            print(self._deferred.pop(0), file=sys.stderr, flush=True)


# ────────────────────────────── text ──────────────────────────────

def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def clean_for_speech(text: str) -> str:
    """Markdown/code -> текст, который не стыдно отдать Piper."""
    s = text
    parts = s.split("```")
    out = []
    for i, part in enumerate(parts):
        out.append(part if i % 2 == 0 else " Code block omitted. ")
    s = "".join(out)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r"<\|[^|>]*\|>", " ", s)
    s = re.sub(r"</?[A-Za-z][^>]*>", " ", s)
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"https?://\S+", "link", s)
    s = html.unescape(s)
    s = re.sub(r"(?m)^\s*\|?[\s:|-]+\|?\s*$", " ", s)
    s = s.replace("|", ", ")
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
    s = re.sub(r"(?m)^\s{0,3}>\s?", "", s)
    s = re.sub(r"(?m)^\s*[-*+•]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+[.)]\s+", "", s)
    s = re.sub(r"[*_~]{1,3}", "", s)
    s = s.replace("*", " ").replace("#", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", ". ", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\.{2,}", ".", s)
    s = re.sub(r"\s*\.\s*\.", ".", s)
    s = re.sub(r":\s*\.", ".", s)
    s = re.sub(r",\s*\.", ".", s)
    s = re.sub(r"\.\s*,", ".", s)
    s = re.sub(r",\s*,+", ",", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,.;:!?])([A-Za-zА-Яа-я])", r"\1 \2", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" .;:,")
    if not s:
        return ""
    if s[-1] not in ".!?":
        s += "."
    return s


def split_for_tts(text: str, max_len: int = 320) -> list[str]:
    if len(text) <= max_len:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    cur = ""
    for sent in sentences:
        if not sent:
            continue
        if len(sent) > max_len:
            while len(sent) > max_len:
                cut = sent.rfind(", ", 0, max_len)
                if cut < max_len // 2:
                    cut = sent.rfind(" ", 0, max_len)
                if cut <= 0:
                    cut = max_len
                chunks.append(sent[:cut].strip())
                sent = sent[cut:].lstrip(" ,")
            if sent:
                chunks.append(sent)
            continue
        if cur and len(cur) + 1 + len(sent) > max_len:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


def is_blank_transcript(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if re.fullmatch(r"[\[\(].{0,40}[\]\)]", t):
        return True
    return not re.search(r"\w", t, flags=re.UNICODE)


class SentenceStreamer:
    """Поток текста -> законченные предложения для TTS.

    Печатается весь текст; в TTS не попадают fenced-блоки и хвост
    few-shot-эха (Q:/A: после первого предложения).
    """

    ABBREV = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
        "etc", "e.g", "i.e", "fig", "approx", "no", "vol", "ch",
    }

    def __init__(self, max_buf: int = 240):
        self._buf: list[str] = []
        self._in_fence = False
        self._ticks = 0
        self._fence_had = False
        self._pending_punct = False
        self._line: list[str] = []
        self._emitted = False
        self._drop_rest = False
        self._max_buf = max_buf

    def feed(self, text: str) -> list[str]:
        out: list[str] = []
        for ch in text:
            if self._drop_rest:
                break
            if ch == "`":
                self._ticks += 1
                if self._ticks == 3:
                    self._ticks = 0
                    self._in_fence = not self._in_fence
                    if not self._in_fence:
                        if self._fence_had:
                            out.append("Code block omitted.")
                            self._emitted = True
                        self._fence_had = False
                continue
            if self._ticks:
                self._ticks = 0
            if self._in_fence:
                self._fence_had = True
                continue

            if ch == "\n":
                self._pending_punct = False
                self._flush(out)
                self._line = []
                continue

            if not self._line and ch.isspace():
                continue
            self._line.append(ch)
            line = "".join(self._line)
            if self._emitted and re.match(r"^(Q|A|User|Question):", line):
                self._drop_rest = True
                self._buf = []
                return out

            if ch in ".!?":
                self._buf.append(ch)
                self._pending_punct = True
                continue
            if self._pending_punct:
                if ch.isspace():
                    self._pending_punct = False
                    if not self._is_abbrev():
                        self._flush(out)
                    else:
                        self._buf.append(" ")
                    continue
                if ch in "\"'”’)]":
                    self._buf.append(ch)
                    continue
                self._pending_punct = False

            self._buf.append(ch)
            if len(self._buf) >= self._max_buf:
                self._flush_long(out)
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        self._flush(out)
        if self._in_fence and self._fence_had:
            out.append("Code block omitted.")
        self._in_fence = False
        self._fence_had = False
        return out

    def _is_abbrev(self) -> bool:
        text = "".join(self._buf).strip()
        if not text:
            return False
        # "1." / "2." в списке — не конец предложения (иначе TTS читает по слову)
        if re.search(r"\b\d+\.$", text):
            return True
        m = re.search(r"([A-Za-z][A-Za-z.]*)$", text)
        if not m:
            return False
        word = m.group(1).lower().rstrip(".")
        if word in self.ABBREV:
            return True
        return bool(re.fullmatch(r"(?:[a-z]\.)+[a-z]?", word + "."))

    def _flush(self, out: list[str]) -> None:
        text = "".join(self._buf).strip()
        self._buf = []
        self._pending_punct = False
        if not text or not re.search(r"[A-Za-zА-Яа-я]", text):
            return
        out.append(text)
        self._emitted = True

    def _flush_long(self, out: list[str]) -> None:
        s = "".join(self._buf)
        cut = max(s.rfind(" ", 0, self._max_buf), s.rfind(",", 0, self._max_buf))
        if cut <= 0:
            cut = self._max_buf
        head = s[:cut].strip()
        self._buf = list(s[cut:].lstrip(" ,"))
        if head and re.search(r"[A-Za-zА-Яа-я]", head):
            out.append(head)
            self._emitted = True


# ────────────────────────────── wav ──────────────────────────────

def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, OSError):
        return 0.0


def wav_params(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth()


# ────────────────────────────── assets ──────────────────────────────

KNOWN_ASR_SIZES = {"tiny": 77691713, "base": 147951465, "small": 487601967}


def model_complete(p: Path, name: str) -> bool:
    if not p.is_file() or p.stat().st_size == 0:
        return False
    expected = KNOWN_ASR_SIZES.get(name)
    if expected is not None:
        return p.stat().st_size == expected
    return not Path(str(p) + ".aria2").exists()


def complete_model(name: str) -> Path | None:
    p = MODELS_DIR / f"ggml-{name}.bin"
    return p if model_complete(p, name) else None


def pick_asr_model(explicit: str | None) -> tuple[str, Path]:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            p = MODELS_DIR / f"ggml-{explicit}.bin"
        if not p.is_file():
            raise VoiceError(f"ASR model not found: {explicit}")
        name = p.stem.replace("ggml-", "")
        if not model_complete(p, name):
            raise VoiceError(f"ASR model is incomplete/still downloading: {p}")
        return name, p
    for name in ("base", "tiny"):
        p = complete_model(name)
        if p is not None:
            return name, p
    raise VoiceError(
        f"no complete whisper base/tiny model in {MODELS_DIR} "
        "(small is opt-in: --asr-model small)"
    )


def engine_path(args: argparse.Namespace) -> Path:
    return Path(args.engine or os.environ.get("VOICE_ENGINE") or DEFAULT_ENGINE)


def engine_model() -> Path:
    return Path(os.environ.get("VOICE_MODEL", DEFAULT_MODEL))


def check_assets(args: argparse.Namespace) -> dict:
    problems: list[str] = []
    notes: list[str] = []

    for p in (ASR_SH, TTS_SH, WHISPER_DIR / "build" / "bin" / "whisper-cli"):
        if not p.is_file():
            problems.append(f"missing {p}")
    try:
        asr_name, asr_path = pick_asr_model(args.asr_model)
        notes.append(f"ASR model: {asr_name} ({asr_path.stat().st_size / 2**20:.0f} MiB)")
    except VoiceError as exc:
        problems.append(str(exc))
    voice = os.environ.get("TTS_VOICE", "en_US-lessac-medium")
    if not (MODELS_DIR / f"{voice}.onnx").is_file():
        problems.append(f"missing TTS voice: {MODELS_DIR / (voice + '.onnx')}")
    else:
        notes.append(f"TTS voice: {voice}")
    tts_mode = os.environ.get("VOICE_TTS", "inproc").lower()
    if tts_mode == "script":
        notes.append("TTS: tts.sh (script mode)")
    else:
        lib = AUDIO_DIR / "lib" / "piper_tts.py"
        deps_ok = (
            lib.is_file()
            and shutil.which("espeak-ng") is not None
            and importlib.util.find_spec("onnxruntime") is not None
            and importlib.util.find_spec("numpy") is not None
        )
        notes.append(
            "TTS: in-process Piper" if deps_ok
            else "TTS: in-process unavailable; tts.sh fallback"
        )

    eng = engine_path(args)
    if not eng.is_file():
        problems.append(f"missing engine binary: {eng}")
    elif not os.access(eng, os.X_OK):
        problems.append(f"engine is not executable: {eng}")
    else:
        notes.append(f"engine: {eng.name} (persistent chat protocol)")
    if not SAFE_RUN.is_file():
        problems.append(f"missing {SAFE_RUN}")
    model = engine_model()
    if not model.is_file():
        problems.append(f"missing engine model: {model}")
    else:
        notes.append(f"engine model: {model.name} ({model.stat().st_size / 2**30:.1f} GiB)")
    if not GUARD_LAUNCHER.is_file() and not args.no_guard:
        problems.append(f"missing guard launcher: {GUARD_LAUNCHER}")

    if not args.text and args.wav is None:
        for tool in ("termux-microphone-record", "ffmpeg"):
            if shutil.which(tool) is None:
                problems.append(f"missing {tool}")
    elif args.wav is not None and shutil.which("ffmpeg") is None:
        problems.append("missing ffmpeg (needed for non-wav input)")
    if not args.no_play:
        if shutil.which("ffplay") is None and shutil.which("termux-media-player") is None:
            problems.append("no player: install ffmpeg (ffplay) or termux-api")
        elif shutil.which("ffplay"):
            notes.append("player: ffplay (streaming PCM)")
        else:
            notes.append("player: termux-media-player")

    return {"problems": problems, "notes": notes}


def print_check(result: dict) -> None:
    for n in result["notes"]:
        print(f"  ok   {n}")
    for p in result["problems"]:
        print(f"  FAIL {p}")
    if not result["problems"]:
        print("all dependencies look good")


# ────────────────────────────── quality guard ──────────────────────────────

GUARD_PROMPT = "Question: What is the capital of France?\nAnswer:"
GUARD_CACHE = TMPDIR / "voice-chat-guard.json"
GUARD_VERSION = 2


def _guard_key() -> str:
    eng = DEFAULT_ENGINE
    try:
        st = eng.stat()
        bin_id = f"{eng}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        bin_id = str(eng)
    model = engine_model()
    try:
        st = model.stat()
        model_id = f"{model}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        model_id = str(model)
    return f"v{GUARD_VERSION}|{bin_id}|{model_id}"


def quality_guard(args: argparse.Namespace, console: Console) -> None:
    """Один probe движка за сборку; кэш, чтобы не платить 20 s каждый раз."""
    if args.no_guard:
        return
    key = _guard_key()
    try:
        cached = json.loads(GUARD_CACHE.read_text())
    except (OSError, ValueError):
        cached = {}
    if cached.get("key") == key:
        if cached.get("ok"):
            console.log("[guard] cached: engine answered Paris")
        else:
            console.log(
                f"[guard] cached ⚠ engine answer: {cached.get('answer', '?')[:120]!r}"
                " — build may be broken"
            )
        return

    console.log("[guard] probing engine (once per build)...")
    t0 = time.monotonic()
    env = os.environ.copy()
    env["TOKSHIT_RAW_PROMPT"] = "1"
    env["TOKSHIT_REASONING_BUDGET"] = "0"
    env.setdefault("TOKSHIT_TEMPERATURE", "0")
    try:
        rc, out, err = run_cmd(
            [str(GUARD_LAUNCHER), GUARD_PROMPT, "8"],
            timeout=600,
            env=env,
        )
    except VoiceError as exc:
        console.log(f"[guard] probe failed: {exc}")
        return
    wall = time.monotonic() - t0
    answer = strip_ansi(out).strip()
    ok = "paris" in answer.lower()
    if rc != 0 and not answer:
        console.log(f"[guard] probe failed rc={rc}: {err.strip()[-200:]}")
    elif ok:
        console.log(f"[guard] ok · Paris · {wall:.1f}s")
    else:
        console.log(
            f"[guard] ⚠ expected Paris, got {answer[:140]!r} ({wall:.1f}s) "
            "— perf build may be broken; continuing"
        )
    if args.verbose and answer:
        console.log(f"[guard] full answer: {answer!r}")
    try:
        GUARD_CACHE.write_text(
            json.dumps({"key": key, "ok": ok, "answer": answer, "wall": wall})
        )
    except OSError:
        pass


# ────────────────────────────── audio in ──────────────────────────────

_RECORDING = False


def stop_recorder(force: bool = False) -> None:
    global _RECORDING
    if not _RECORDING and not force:
        return
    _RECORDING = False
    try:
        run_cmd(["termux-microphone-record", "-q"], timeout=4)
    except VoiceError:
        pass


def record_wav(
    workdir: Path,
    *,
    seconds: float,
    max_seconds: float,
    hands_free: bool,
    console: Console,
) -> tuple[Path | None, float, float]:
    """Записать с микрофона -> (16 kHz WAV|None, длительность, секунды записи)."""
    rec = workdir / f"mic-{int(time.time())}.m4a"
    raw = workdir / f"mic-{int(time.time())}.wav"
    limit = int(seconds) if seconds > 0 else max(0, int(max_seconds))
    cmd = [
        "termux-microphone-record",
        "-f", str(rec),
        "-e", os.environ.get("VOICE_REC_ENCODER", "aac"),
        "-r", os.environ.get("VOICE_REC_RATE", "16000"),
        "-c", "1",
        "-b", os.environ.get("VOICE_REC_BITRATE", "48"),
        "-l", str(limit if limit > 0 else 0),
    ]
    console.log(f"🎤 запись {limit or max_seconds:.0f} с" if seconds > 0 else "🎤 запись…")
    t0 = time.monotonic()
    rc, _, err = run_cmd(cmd, timeout=10)
    if rc == 124:
        raise VoiceError(
            "termux-microphone-record завис (нет Termux:API или висит диалог "
            "разрешения микрофона)."
        )
    if rc != 0:
        raise VoiceError(
            "termux-microphone-record не запустился. Проверь Termux:API и "
            f"разрешение RECORD_AUDIO.\n{err.strip()}"
        )
    global _RECORDING
    _RECORDING = True

    if seconds > 0:
        remaining = seconds
        try:
            if not hands_free:
                console.log(f"   … говори, стоп через {seconds:.0f} с (или Enter)")
            while remaining > 0:
                time.sleep(0.1)
                remaining -= 0.1
                if not hands_free and sys.stdin.isatty():
                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.readline()
                        break
        except KeyboardInterrupt:
            stop_recorder()
            raise
    else:
        try:
            if sys.stdin.isatty():
                input(f"   … Enter — остановить (автостоп {max_seconds:.0f} с): ")
            else:
                time.sleep(max_seconds)
        except EOFError:
            pass
    stop_recorder()
    rec_s = time.monotonic() - t0

    for _ in range(20):
        if rec.is_file() and rec.stat().st_size > 0:
            break
        time.sleep(0.1)
    if not rec.is_file() or rec.stat().st_size == 0:
        raise VoiceError(
            "микрофон не создал файл. Проверь разрешение RECORD_AUDIO у "
            "Termux:API (Настройки Android → Приложения → Termux:API)."
        )

    af = (
        "silenceremove=start_periods=1:start_duration=0:start_threshold=-40dB:"
        "stop_periods=-1:stop_duration=0.35:stop_threshold=-40dB"
    )
    rc, _, err = run_cmd(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(rec), "-af", af, "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", str(raw),
        ],
        timeout=60,
    )
    if rc != 0 or not raw.is_file():
        rc, _, err = run_cmd(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(rec), "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le", str(raw),
            ],
            timeout=60,
        )
    if rc != 0 or not raw.is_file():
        raise VoiceError(f"ffmpeg не смог конвертировать запись:\n{err.strip()}")

    dur = wav_duration(raw)
    console.log(f"[mic] {dur:.1f} s · запись {rec_s:.1f} s")
    return (None if dur < 0.35 else raw), dur, rec_s


def transcribe(wav: Path, model_name: str, *, verbose: bool,
               console: Console) -> tuple[str, float, float]:
    dur = wav_duration(wav)
    t0 = time.monotonic()
    rc, out, err = run_cmd([str(ASR_SH), str(wav), model_name], timeout=600)
    dt = time.monotonic() - t0
    text = strip_ansi(out).strip()
    if rc != 0 and not text:
        raise VoiceError(f"ASR failed (rc={rc}):\n{err.strip() or out.strip()}")
    if verbose and err.strip():
        console.log(f"[asr] stderr: {err.strip()[:300]}")
    rate = dur / dt if dt > 0 else 0.0
    console.log(
        f"[asr] {model_name} · {dt * 1000:.0f} ms · {rate:.1f}× realtime · {dur:.2f} s audio"
    )
    return text, dt, dur


# ────────────────────────────── TTS + play ──────────────────────────────

@dataclass
class TurnStats:
    t_start: float = 0.0
    audio_s: float = 0.0
    asr_s: float = 0.0
    asr_model: str = ""
    engine_s: float = 0.0
    engine_ask_start: float = 0.0
    first_token_s: float | None = None
    first_tts_s: float | None = None
    first_audio_s: float | None = None
    first_audio_turn_s: float | None = None
    tts_s: float = 0.0
    tts_chunks: int = 0
    audio_out_s: float = 0.0
    play_s: float = 0.0
    total_s: float = 0.0
    chars: int = 0
    engine_metrics: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    blank: bool = False


class StreamingPlayer:
    """Один ffplay на ход, сырой PCM s16le в stdin — без щелей между чанками."""

    def __init__(self, console: Console, stats: TurnStats):
        self.console = console
        self.stats = stats
        mode = os.environ.get("VOICE_PLAYER", "auto")
        self.mode = mode
        self.proc: subprocess.Popen | None = None
        self.params: tuple[int, int, int] | None = None
        self.started = False
        self.failed = False
        if mode == "none":
            self.mode = "none"
        elif mode == "termux":
            self.mode = "termux" if shutil.which("termux-media-player") else "none"
        elif mode == "ffplay":
            self.mode = "ffplay" if shutil.which("ffplay") else "termux"
        else:
            if shutil.which("ffplay"):
                self.mode = "ffplay"
            elif shutil.which("termux-media-player"):
                self.mode = "termux"
            else:
                self.mode = "none"
        if self.mode == "termux" and not shutil.which("termux-media-player"):
            self.mode = "none"

    def start_clock(self) -> None:
        if not self.started:
            self.started = True
            self.stats.first_audio_s = time.monotonic()
            self.stats.first_audio_turn_s = self.stats.first_audio_s - self.stats.t_start
            extra = ""
            if self.stats.engine_ask_start:
                extra = (
                    f" · {self.stats.first_audio_s - self.stats.engine_ask_start:.2f}s "
                    "after engine start"
                )
            self.console.log(
                f"[play] first audio at {self.stats.first_audio_turn_s:.2f}s (turn){extra}",
                defer=True,
            )

    def _start_ffplay(self, params: tuple[int, int, int]) -> bool:
        rate, channels, width = params
        if width != 2 or channels not in (1, 2):
            self.console.log(f"[play] unsupported wav: {params}")
            return False
        layout = "mono" if channels == 1 else "stereo"
        try:
            p = subprocess.Popen(
                [
                    "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                    "-f", "s16le", "-ar", str(rate), "-ch_layout", layout,
                    "pipe:0",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                bufsize=0,
            )
        except OSError as exc:
            self.console.log(f"[play] ffplay failed: {exc}")
            return False
        _register(p)
        self.proc = p
        self.params = params
        return True

    def play(self, wav: Path, duration: float) -> None:
        if self.mode == "none":
            self.stats.audio_out_s += duration
            return
        if self.mode == "termux":
            if self.stats.first_audio_s is None:
                self.start_clock()
            try:
                rc, _, err = run_cmd(
                    ["termux-media-player", "play", str(wav)], timeout=15
                )
            except VoiceError as exc:
                self.stats.errors.append(f"termux-media-player: {exc}")
                return
            if rc != 0:
                self.stats.errors.append(f"termux-media-player: {err.strip()[-200:]}")
                return
            time.sleep(duration + 0.1)
            run_cmd(["termux-media-player", "stop"], timeout=10)
            self.stats.audio_out_s += duration
            return

        try:
            with wave.open(str(wav), "rb") as w:
                params = (w.getframerate(), w.getnchannels(), w.getsampwidth())
                frames = w.readframes(w.getnframes())
        except (wave.Error, OSError) as exc:
            self.stats.errors.append(f"wav read: {exc}")
            return
        if self.proc is None:
            if not self._start_ffplay(params):
                self.mode = "termux" if shutil.which("termux-media-player") else "none"
                self.play(wav, duration)
                return
            self.start_clock()
        elif self.proc.poll() is not None:
            self.stats.errors.append("ffplay exited before the turn ended")
            self.proc = None
            self.play(wav, duration)
            return
        elif params != self.params:
            self.stats.errors.append(f"wav params changed: {params} != {self.params}")
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(frames)
            self.proc.stdin.flush()
            self.stats.audio_out_s += duration
        except (BrokenPipeError, OSError) as exc:
            self.stats.errors.append(f"ffplay write: {exc}")
            self.proc = None

    def close(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=max(10.0, self.stats.audio_out_s + 5.0))
            except subprocess.TimeoutExpired:
                kill_proc(self.proc)
            _unregister(self.proc)
            self.proc = None
        if self.started:
            self.stats.play_s = time.monotonic() - (self.stats.first_audio_s or 0.0)


def tts_warmup(
    engine: "EngineSession",
    workdir: Path,
    console: Console,
    done: threading.Event,
) -> None:
    """Прогреть Piper до первого предложения: onnxruntime + страницы модели.
    Работает параллельно prefill, чтобы не добавлять задержку первому токену."""
    try:
        engine.wait_ready()
        out = workdir / "tts-warmup.wav"
        t0 = time.monotonic()
        rc, _, err = run_cmd(
            [str(TTS_SH), "Voice chat ready.", str(out)], timeout=120
        )
        dt = time.monotonic() - t0
        if rc == 0 and out.is_file():
            console.log(f"[tts] warmup {dt:.2f}s", defer=True)
        else:
            console.log(f"[tts] warmup failed rc={rc}: {err.strip()[-200:]}", defer=True)
    except Exception as exc:  # noqa: BLE001 - поток не должен падать
        console.log(f"[tts] warmup failed: {exc}", defer=True)
    finally:
        done.set()


class PiperTTS:
    """Piper в процессе voice_chat.py: onnxruntime-сессия живёт всю сессию.

    Использует ~/audio/lib/piper_tts.py и ту же ONNX-модель, ничего в
    ~/audio не меняет. Убирает запуск python/nice/ionice на каждый чанк —
    первый звук не ждёт диск движка. Fallback — ~/audio/tts.sh.
    """

    def __init__(self, console: Console):
        self.console = console
        self.ready_event = threading.Event()
        self.sess = None
        self.mod = None
        self.np = None
        self.id_map: dict = {}
        self.cfg: dict = {}
        self.sr = 22050
        self.scales = None
        self.input_names: set[str] = set()
        self.error: str | None = None
        self.load_s = 0.0

    def start(self) -> None:
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        t0 = time.monotonic()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import numpy as np
                import onnxruntime as ort

                lib = AUDIO_DIR / "lib" / "piper_tts.py"
                if not lib.is_file():
                    raise VoiceError(f"missing {lib}")
                spec = importlib.util.spec_from_file_location("piper_tts", lib)
                if spec is None or spec.loader is None:
                    raise VoiceError(f"cannot import {lib}")
                mod = importlib.util.module_from_spec(spec)
                sys.modules["piper_tts"] = mod
                spec.loader.exec_module(mod)

                voice = os.environ.get("TTS_VOICE", "en_US-lessac-medium")
                model = MODELS_DIR / f"{voice}.onnx"
                cfg_path = MODELS_DIR / f"{voice}.onnx.json"
                if not model.is_file() or not cfg_path.is_file():
                    raise VoiceError(f"missing TTS voice {voice}")
                cfg = json.loads(cfg_path.read_text())
                sess = ort.InferenceSession(
                    str(model), providers=["CPUExecutionProvider"]
                )

            inf = cfg.get("inference", {})
            ls = float(os.environ.get("TTS_LENGTH_SCALE", inf.get("length_scale", 1.0)))
            self.np = np
            self.mod = mod
            self.sess = sess
            self.cfg = cfg
            self.id_map = cfg["phoneme_id_map"]
            self.sr = cfg.get("audio", {}).get("sample_rate", 22050)
            self.scales = np.array(
                [inf.get("noise_scale", 0.667), ls, inf.get("noise_w", 0.8)],
                dtype=np.float32,
            )
            self.input_names = {i.name for i in sess.get_inputs()}
            self.load_s = time.monotonic() - t0
            self.console.log(
                f"[tts] in-process Piper ready in {self.load_s:.2f}s",
                defer=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.console.log(
                f"[tts] in-process Piper unavailable ({exc}); using tts.sh",
                defer=True,
            )
        finally:
            self.ready_event.set()

    def synth(self, text: str, out: Path) -> tuple[float, float]:
        if self.sess is None or self.mod is None or self.np is None:
            raise VoiceError(self.error or "Piper is not loaded")
        t0 = time.monotonic()
        voice = self.cfg.get("espeak", {}).get("voice", "en-us")
        ph = self.mod.phonemize(text, voice)
        ids = self.mod.to_ids(ph, self.id_map)
        if len(ids) <= 3:
            raise VoiceError(f"no phonemes for {text[:40]!r}")
        np = self.np
        feed = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": self.scales,
        }
        if "sid" in self.input_names:
            feed["sid"] = np.array([0], dtype=np.int64)
        feed = {k: v for k, v in feed.items() if k in self.input_names}
        audio = self.sess.run(None, feed)[0].reshape(-1)
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2")
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(pcm16.tobytes())
        return time.monotonic() - t0, len(pcm16) / self.sr


class SpeechPipeline:
    """sentence queue -> TTS worker -> audio queue -> single player worker."""

    def __init__(self, workdir: Path, console: Console, stats: TurnStats,
                 no_play: bool, warm_event: threading.Event | None = None,
                 piper: PiperTTS | None = None):
        self.workdir = workdir
        self.console = console
        self.stats = stats
        self.no_play = no_play
        self.warm_event = warm_event
        self.piper = piper
        self.text_q: queue.Queue = queue.Queue()
        self.audio_q: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.player = StreamingPlayer(console, stats) if not no_play else None
        self.tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._closed = False
        self._errors_lock = threading.Lock()

    def start(self) -> None:
        self.tts_thread.start()
        self.play_thread.start()

    def push(self, sentence: str) -> None:
        if self.stop_event.is_set() or self._closed:
            return
        if not sentence.strip():
            return
        self.text_q.put(sentence)
        if self.console.verbose:
            self.console.log(f"[stream] {sentence!r}", defer=True)

    def _error(self, msg: str) -> None:
        with self._errors_lock:
            self.stats.errors.append(msg)
        self.console.log(f"[tts] {msg}", defer=True)

    def _tts_loop(self) -> None:
        chunk_no = 0
        # Ждём загрузку Piper в процессе (или warmup tts.sh в script-режиме).
        if self.piper is not None:
            while not self.piper.ready_event.wait(0.2):
                if self.stop_event.is_set():
                    self.audio_q.put(None)
                    return
            use_piper = self.piper.sess is not None
            if not use_piper:
                self.console.log("[tts] Piper unavailable; using tts.sh", defer=True)
        elif self.warm_event is not None:
            while not self.warm_event.wait(0.2):
                if self.stop_event.is_set():
                    self.audio_q.put(None)
                    return
            use_piper = False
        else:
            use_piper = False
        try:
            while not self.stop_event.is_set():
                sentence = self.text_q.get()
                if sentence is None:
                    break
                spoken = clean_for_speech(sentence)
                if not spoken:
                    continue
                for part in split_for_tts(spoken):
                    if self.stop_event.is_set():
                        break
                    chunk_no += 1
                    out = self.workdir / f"tts-{chunk_no}.wav"
                    t0 = time.monotonic()
                    if use_piper:
                        try:
                            dt, dur = self.piper.synth(part, out)
                            rc, err = 0, ""
                        except Exception as exc:  # noqa: BLE001
                            rc, err, dt, dur = 1, str(exc), time.monotonic() - t0, 0.0
                    else:
                        try:
                            rc, _, err = run_cmd([str(TTS_SH), part, str(out)], timeout=300)
                        except VoiceError as exc:
                            self._error(f"chunk {chunk_no}: {exc}")
                            continue
                        dt = time.monotonic() - t0
                        dur = wav_duration(out) if rc == 0 else 0.0
                    if rc != 0 or not out.is_file():
                        self._error(f"chunk {chunk_no} failed rc={rc}: {err.strip()[-300:]}")
                        continue
                    self.stats.tts_s += dt
                    self.stats.tts_chunks += 1
                    if self.stats.first_tts_s is None:
                        self.stats.first_tts_s = time.monotonic()
                        self.console.log(
                            f"[tts] first chunk at "
                            f"{self.stats.first_tts_s - self.stats.t_start:.2f}s "
                            f"({dur:.2f}s audio)",
                            defer=True,
                        )
                    if self.console.verbose or self.no_play:
                        self.console.log(
                            f"[tts] chunk {chunk_no}: {dt * 1000:.0f} ms · "
                            f"{dur:.2f}s · {part[:70]!r}",
                            defer=True,
                        )
                    self.audio_q.put((out, dur))
        finally:
            self.audio_q.put(None)

    def _play_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                item = self.audio_q.get()
                if item is None:
                    break
                if self.player is not None:
                    self.player.play(*item)
                else:
                    self.stats.audio_out_s += item[1]
        finally:
            if self.player is not None:
                self.player.close()

    def finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.text_q.put(None)
        self.tts_thread.join(timeout=600)
        if self.tts_thread.is_alive():
            self.stop_event.set()
            self.audio_q.put(None)
        self.play_thread.join(timeout=600)
        if self.player is not None:
            self.player.close()

    def stop(self) -> None:
        self._closed = True
        self.stop_event.set()
        try:
            self.text_q.put_nowait(None)
            self.audio_q.put_nowait(None)
        except queue.Full:
            pass
        if self.player is not None:
            if self.player.proc is not None:
                kill_proc(self.player.proc)
            self.player.close()
        self.tts_thread.join(timeout=3)
        self.play_thread.join(timeout=3)


# ────────────────────────────── engine ──────────────────────────────

def parse_engine_stats(text: str) -> dict:
    st = {
        "prefill_tokens": 0, "prefill_s": 0.0,
        "gen_tokens": 0, "gen_s": 0.0, "tok_s": 0.0,
    }
    matches = list(ENGINE_RE.finditer(text))
    if matches:
        m = matches[-1]
        st["prefill_tokens"] = int(m.group(1))
        st["prefill_s"] = float(m.group(2))
        st["gen_tokens"] = int(m.group(3))
        st["gen_s"] = float(m.group(4))
        st["tok_s"] = float(m.group(5))
    return st


class EngineSession:
    """Один persistent chat-процесс на сессию (TOKSHIT_CHAT_PROTOCOL=1)."""

    def __init__(self, args: argparse.Namespace, console: Console):
        self.args = args
        self.console = console
        self.proc: subprocess.Popen | None = None
        self.started_at = 0.0
        self.loaded_at: float | None = None
        self.child_load_s = 0.0
        self.ready_at: float | None = None
        self.prefix_s = 0.0
        self.prefix_done = False
        self.fewshot = os.environ.get(
            "VOICE_FEWSHOT", "Q: 2 + 2?\\nA: 4."
        )
        self._expect_prefix = bool(self.fewshot) and not args.chat_template
        self.ready_event = threading.Event()
        self.loaded = False
        self.lock_wait = False
        self._lock_logged = False
        self._stderr: list[str] = []
        self._stderr_lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None
        self._status_stop = threading.Event()
        self._status_thread: threading.Thread | None = None
        self._outbuf = ""
        self.turn_no = 0

    @property
    def load_s(self) -> float:
        return (self.loaded_at - self.started_at) if self.loaded_at else 0.0

    @property
    def ready_s(self) -> float:
        return (self.ready_at - self.started_at) if self.ready_at else 0.0

    def _cmd(self) -> list[str]:
        temp = (
            os.environ.get("VOICE_TEMPERATURE")
            or os.environ.get("TOKSHIT_TEMPERATURE")
            or "0"
        )
        return [
            str(SAFE_RUN), str(engine_path(self.args)),
            "--temperature", temp, str(engine_model()),
        ]

    def start(self) -> None:
        env = os.environ.copy()
        env.pop("TOKSHIT_PROMPT_FILE", None)
        env["TOKSHIT_CHAT_PROTOCOL"] = "1"
        env["TOKSHIT_CHAT_MAX_NEW"] = str(self.args.max_new)
        env.setdefault("TOKSHIT_MIN_AVAIL_MB", "1200")
        env.setdefault("TOKSHIT_HOT", "0")
        env.setdefault("TOKSHIT_EXPERT_CACHE_MB", "512")
        env.setdefault("TOKSHIT_PREFAULT", "1")
        # A/B: 4 workers быстрее 5 на persistent chat (4.14 vs 3.9 tok/s).
        env["TOKSHIT_POOL_WORKERS"] = os.environ.get("VOICE_POOL_WORKERS", "4")
        env["TOKSHIT_INK_PREFILL_BATCH"] = os.environ.get("VOICE_PREFILL_BATCH", "16")
        env.setdefault("TOKSHIT_POOL_PIN", "1")
        env.setdefault("TOKSHIT_TEMPERATURE", env.get("VOICE_TEMPERATURE", "0"))

        if self.args.chat_template:
            # Opt-in: нативный шаблон Inkling (видимый reasoning — на любителя).
            env.pop("TOKSHIT_RAW_PROMPT", None)
            env.pop("TOKSHIT_CHAT_UNESCAPE", None)
            env.pop("TOKSHIT_STOP_AT", None)
            env.setdefault(
                "TOKSHIT_SYSTEM_PROMPT",
                "You are a helpful voice assistant. Answer in English, briefly "
                "and directly, in one or two short sentences. Do not describe "
                "your reasoning.",
            )
            env.setdefault(
                "TOKSHIT_THINK_EFFORT", os.environ.get("VOICE_THINK_EFFORT", "0.2")
            )
        else:
            # Проверенный голосовой путь: raw Q/A few-shot, без шаблона.
            env["TOKSHIT_RAW_PROMPT"] = "1"
            env["TOKSHIT_CHAT_UNESCAPE"] = "1"
            if self.fewshot:
                env["TOKSHIT_CHAT_PREFIX"] = self.fewshot
            stop_at = os.environ.get("VOICE_STOP_AT", "Q:")
            if stop_at:
                env["TOKSHIT_STOP_AT"] = stop_at
            # Второй маркер: конец сообщения/перевод строки. Ведущие \n
            # игнорируются движком, поэтому пустого ответа не будет.
            stop_at2 = os.environ.get("VOICE_STOP_AT2", r"\n")
            if stop_at2:
                env["TOKSHIT_STOP_AT2"] = stop_at2
        if os.environ.get("VOICE_REASONING_BUDGET"):
            env["TOKSHIT_REASONING_BUDGET"] = os.environ["VOICE_REASONING_BUDGET"]

        cmd = self._cmd()
        if self.console.verbose:
            self.console.log(f"[engine] exec: {' '.join(cmd)}")
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(TOKSHIT_ROOT),
                env=env,
                start_new_session=True,
                bufsize=0,
            )
        except OSError as exc:
            raise VoiceError(f"cannot start engine: {exc}")
        _register(self.proc)
        self.started_at = time.monotonic()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()
        self._status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self._status_thread.start()

    def _maybe_ready(self) -> None:
        if not self.loaded:
            return
        if self._expect_prefix and not self.prefix_done:
            return
        if self.ready_event.is_set():
            return
        self.ready_at = time.monotonic()
        self.ready_event.set()
        self._status_stop.set()
        self.console.clear_status()
        self.console.log(f"[engine] ready in {self.ready_s:.1f}s")

    def _stderr_loop(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            with self._stderr_lock:
                self._stderr.append(line)
            if not self._lock_logged and "another model run holds the lock" in line:
                self.lock_wait = True
                self._lock_logged = True
                self.console.clear_status()
                self.console.log("[engine] waiting for model lock (another tokshit)...")
            if not self.loaded and LOADED_RE.search(line):
                m = LOADED_RE.search(line)
                self.loaded = True
                self.loaded_at = time.monotonic()
                self.child_load_s = float(m.group(1)) if m else self.load_s
                self.console.clear_status()
                self.console.log(f"[engine] loaded in {self.child_load_s:.1f}s")
                self._maybe_ready()
            if not self.prefix_done:
                m = PREFIX_RE.search(line)
                if m:
                    self.prefix_done = True
                    self.prefix_s = float(m.group(2))
                    self.console.clear_status()
                    self.console.log(
                        f"[engine] prefix {m.group(1)} tok / {self.prefix_s:.1f}s"
                    )
                    self._maybe_ready()

    def _status_loop(self) -> None:
        last = 0.0
        tty = sys.stderr.isatty()
        while not self._status_stop.is_set() and not self.ready_event.is_set():
            now = time.monotonic()
            interval = 0.5 if tty else 2.0
            if now - last >= interval:
                if self.lock_wait:
                    phase = "waiting for model lock"
                elif self.loaded:
                    phase = "prefix"
                else:
                    phase = "loading model"
                self.console.status(
                    f"[engine] {phase}... {now - self.started_at:.1f}s"
                )
                last = now
            time.sleep(0.1)

    def stderr_len(self) -> int:
        with self._stderr_lock:
            return len(self._stderr)

    def stderr_since(self, index: int) -> list[str]:
        with self._stderr_lock:
            return self._stderr[index:]

    def stderr_tail(self, n: int = 10) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr[-n:])

    def wait_ready(self, timeout: float = 240.0) -> None:
        deadline = time.monotonic() + timeout
        while not self.ready_event.is_set():
            if self.proc is None or self.proc.poll() is not None:
                tail = self.stderr_tail()
                raise VoiceError(
                    f"engine exited during load (rc={self.proc.returncode if self.proc else '?'}):\n{tail}"
                )
            if time.monotonic() > deadline:
                raise VoiceError(f"engine did not load in {timeout:.0f}s")
            time.sleep(0.1)

    def _payload(self, text: str) -> str:
        # CHAT_UNESCAPE: литеральные \n в строке stdin -> реальные newline.
        text = " ".join(text.splitlines()).strip().replace("\\", " ")
        if self.args.chat_template:
            return text
        if not self.fewshot:
            if self.turn_no == 1:
                return f"Q: {text}\\nA:"
            return f"\\nQ: {text}\\nA:"
        return f"\\nQ: {text}\\nA:"

    def _wait_metric(self, start: int, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any("[prefill:" in line for line in self.stderr_since(start)):
                return
            time.sleep(0.02)

    def ask(self, text: str, on_text, stats: TurnStats) -> tuple[int, list[str]]:
        """Отправить реплику и читать stdout до RS. Возвращает (rc, stderr-срез)."""
        self.wait_ready()
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise VoiceError("engine is not running")
        self.turn_no += 1
        err_start = self.stderr_len()
        t0 = time.monotonic()
        stats.engine_ask_start = t0
        line = self._payload(text)
        if not line:
            return 0, []
        try:
            self.proc.stdin.write((line + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            stats.engine_s = time.monotonic() - t0
            raise VoiceError(f"engine stdin closed: {exc}")

        fd = self.proc.stdout.fileno()
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        first = False
        pending_ws = ""
        last_data = t0
        last_status = 0.0
        tty = sys.stderr.isatty()

        def handle(chunk: str) -> None:
            nonlocal first, pending_ws
            if not first:
                if not chunk.strip():
                    pending_ws += chunk
                    return
                chunk = pending_ws + chunk
                pending_ws = ""
                self.console.clear_status()
                first = True
                stats.first_token_s = time.monotonic() - t0
                self.console.log(f"[engine] first token {stats.first_token_s:.2f}s")
            if chunk:
                stats.chars += len(chunk)
                on_text(chunk)

        def consume() -> bool:
            """Разобрать накопленный stdout. True, если пришёл RS (конец хода)."""
            while RS in self._outbuf:
                chunk, self._outbuf = self._outbuf.split(RS, 1)
                handle(chunk)
                self.console.clear_status()
                if not first:
                    self.console.log("[engine] empty answer")
                if self._outbuf:
                    # по протоколу после RS ничего не должно быть
                    self.console.log(
                        f"[engine] protocol: trailing bytes {self._outbuf[:40]!r}"
                    )
                    self._outbuf = ""
                stats.engine_s = time.monotonic() - t0
                self._wait_metric(err_start)
                return True
            if self._outbuf:
                chunk = self._outbuf
                self._outbuf = ""
                handle(chunk)
            return False

        if consume():
            return 0, self.stderr_since(err_start)

        while True:
            try:
                r, _, _ = select.select([fd], [], [], 0.25)
            except InterruptedError:
                continue
            if r:
                data = os.read(fd, 4096)
                if not data:
                    self.console.clear_status()
                    stats.engine_s = time.monotonic() - t0
                    tail = self.stderr_tail(12)
                    raise VoiceError(
                        f"engine closed stdout mid-turn (rc={self.proc.poll()}):\n{tail}"
                    )
                last_data = time.monotonic()
                self._outbuf += dec.decode(data)
                if consume():
                    return 0, self.stderr_since(err_start)
            else:
                now = time.monotonic()
                if not first:
                    interval = 0.5 if tty else 2.0
                    if now - last_status >= interval:
                        self.console.status(f"[engine] prefill... {now - t0:.1f}s")
                        last_status = now
                if now - last_data > float(os.environ.get("VOICE_TURN_TIMEOUT", "900")):
                    stats.engine_s = time.monotonic() - t0
                    kill_proc(self.proc)
                    raise VoiceError("engine turn timed out")

    def stop(self) -> None:
        self._status_stop.set()
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except OSError:
                pass
            kill_proc(self.proc)
            _unregister(self.proc)
            self.proc = None
        for th in (self._status_thread, self._stderr_thread):
            if th is not None:
                th.join(timeout=2)


# ────────────────────────────── turn ──────────────────────────────

def turn_summary(stats: TurnStats, no_play: bool) -> None:
    def ms(x: float | None) -> str:
        return f"{x * 1000:.0f} ms" if x else "n/a"

    asr = f"asr {ms(stats.asr_s)} ({stats.asr_model})" if stats.asr_s else "asr n/a"
    parts = [
        asr,
        f"engine {ms(stats.engine_s)}",
        f"tts {ms(stats.tts_s)}",
        f"total {ms(stats.total_s)}",
    ]
    print("[summary] " + " · ".join(parts), file=sys.stderr, flush=True)
    if not no_play:
        print(
            f"[summary] play {ms(stats.play_s)} · {stats.audio_out_s:.1f} s audio",
            file=sys.stderr, flush=True,
        )
    m = stats.engine_metrics
    tok = (
        f"{m.get('gen_tokens', 0)} tok · {m.get('tok_s', 0):.2f} tok/s"
        if m.get("gen_tokens") else "tok/s n/a"
    )
    print(
        f"[stats]   {tok} · {stats.chars} chars · "
        f"{stats.tts_chunks} tts chunks · {stats.audio_out_s:.1f} s audio",
        file=sys.stderr, flush=True,
    )
    first = (
        f"first token {stats.first_token_s:.2f}s (engine)"
        if stats.first_token_s is not None else "first token n/a"
    )
    if stats.first_token_s is not None and stats.engine_ask_start:
        first += (
            f" / {stats.engine_ask_start + stats.first_token_s - stats.t_start:.2f}s (turn)"
        )
    if stats.first_tts_s is not None:
        first += f" · first tts {stats.first_tts_s - stats.t_start:.2f}s (turn)"
    if stats.first_audio_s is not None:
        first += f" · first audio {stats.first_audio_turn_s:.2f}s (turn)"
        if stats.engine_ask_start:
            first += f" / {stats.first_audio_s - stats.engine_ask_start:.2f}s (engine)"
    print(f"[latency] {first}", file=sys.stderr, flush=True)
    for err in stats.errors:
        print(f"[error]   {err}", file=sys.stderr, flush=True)


def one_turn(
    args: argparse.Namespace,
    engine: EngineSession,
    console: Console,
    model_name: str,
    workdir: Path,
    turn_no: int,
    *,
    warm_event: threading.Event | None = None,
    piper: PiperTTS | None = None,
    wav: Path | None = None,
    text: str | None = None,
) -> TurnStats:
    stats = TurnStats(t_start=time.monotonic(), asr_model=model_name)
    turn_dir = workdir / f"turn-{turn_no}"
    turn_dir.mkdir(parents=True, exist_ok=True)

    if text is None:
        assert wav is not None
        stats.audio_s = wav_duration(wav)
        transcript, asr_s, _ = transcribe(
            wav, model_name, verbose=args.verbose, console=console
        )
        stats.asr_s = asr_s
        if is_blank_transcript(transcript):
            console.log("[asr] тишина/не распознано — пропускаю")
            stats.blank = True
            stats.total_s = time.monotonic() - stats.t_start
            return stats
    else:
        transcript = text.strip()

    console.log(f"🗣  {transcript}")
    if text is not None:
        stats.asr_model = ""
    pipeline = SpeechPipeline(
        turn_dir, console, stats, args.no_play, warm_event, piper
    )
    streamer = SentenceStreamer()
    pipeline.start()
    rc = 0
    err_lines: list[str] = []
    completed = False
    try:
        def on_text(chunk: str) -> None:
            console.answer(chunk)
            for sentence in streamer.feed(chunk):
                pipeline.push(sentence)

        rc, err_lines = engine.ask(transcript, on_text, stats)
        for sentence in streamer.flush():
            pipeline.push(sentence)
        console.end_answer()
        pipeline.finish()
        completed = True
    finally:
        if not completed:
            pipeline.stop()
            console.end_answer()

    stats.engine_metrics = parse_engine_stats("\n".join(err_lines))
    stats.total_s = time.monotonic() - stats.t_start
    if rc != 0:
        stats.errors.append(f"engine rc={rc}")
    turn_summary(stats, args.no_play)
    if args.verbose:
        tail = engine.stderr_tail(8)
        if tail:
            console.log("[engine] stderr tail:\n" + tail)
    return stats


def session_summary(turns: list[TurnStats], load_s: float, prefix_s: float) -> None:
    real = [t for t in turns if not t.blank]
    if not real:
        return
    def avg(key: str) -> float:
        vals = [getattr(t, key) for t in real if getattr(t, key)]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n── session ──", file=sys.stderr)
    asr_model = next((t.asr_model for t in real if t.asr_model), "")
    startup = f"load {load_s:.1f}s"
    if prefix_s:
        startup += f" + prefix {prefix_s:.1f}s"
    print(
        f"turns {len(real)} · engine {startup} · ASR {asr_model or 'n/a'}",
        file=sys.stderr,
    )
    print(
        f"avg asr {avg('asr_s') * 1000:.0f} ms · engine {avg('engine_s') * 1000:.0f} ms"
        f" · tts {avg('tts_s') * 1000:.0f} ms · total {avg('total_s') * 1000:.0f} ms",
        file=sys.stderr,
    )
    fa = [t.first_audio_turn_s for t in real if t.first_audio_turn_s is not None]
    if fa:
        print(
            f"avg first audio {sum(fa) / len(fa):.2f} s (turn) · "
            f"avg tts chunks {avg('tts_chunks'):.1f}",
            file=sys.stderr,
        )


# ────────────────────────────── main ──────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Voice chat: mic -> whisper.cpp -> tokshit persistent chat -> Piper TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--text", action="append", default=[],
                    help="skip mic+ASR; repeat for multiple turns in one engine session")
    ap.add_argument("--wav", help="skip mic, transcribe this file")
    ap.add_argument("--seconds", type=float, default=0,
                    help="fixed recording length (0 = press Enter to stop)")
    ap.add_argument("--max-seconds", type=float, default=20,
                    help="hard cap for Enter-stopped recording (default 20)")
    ap.add_argument("--hands-free", action="store_true",
                    help="with --seconds: loop without keyboard")
    ap.add_argument("--max-new", type=int,
                    default=int(os.environ.get("VOICE_MAX_NEW", "64")),
                    help="max new tokens per turn (default 64)")
    ap.add_argument("--asr-model", default=os.environ.get("VOICE_ASR_MODEL"),
                    help="whisper model: small/base/tiny or a .bin path")
    ap.add_argument("--engine", default=os.environ.get("VOICE_ENGINE"),
                    help="toksjit binary for persistent chat (default build/tokshit)")
    ap.add_argument("--chat-template", action="store_true",
                    help="opt-in: native Inkling template instead of raw Q/A few-shot")
    ap.add_argument("--no-play", action="store_true",
                    help="synthesize but don't play; logs TTS chunks")
    ap.add_argument("--no-guard", action="store_true",
                    help="skip the per-build engine quality probe")
    ap.add_argument("--once", action="store_true", help="one mic turn, then exit")
    ap.add_argument("--keep", action="store_true", help="keep temp wav files")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--check", action="store_true", help="check dependencies and exit")
    ap.add_argument("--clean-text", metavar="TEXT",
                    help="print the TTS-cleaned version of TEXT and exit")
    args = ap.parse_args()

    if args.clean_text is not None:
        print(clean_for_speech(args.clean_text))
        return 0
    if any(not t.strip() for t in args.text):
        print("✗ --text is empty", file=sys.stderr)
        return 2
    if args.max_new < 1:
        print("✗ --max-new must be >= 1", file=sys.stderr)
        return 2
    if args.hands_free and args.seconds <= 0:
        print("✗ --hands-free requires --seconds N", file=sys.stderr)
        return 2

    deps = check_assets(args)
    if args.check:
        print_check(deps)
        return 2 if deps["problems"] else 0
    if deps["problems"]:
        for p in deps["problems"]:
            print(f"✗ {p}", file=sys.stderr)
        print("подсказка: scripts/voice_chat.py --check", file=sys.stderr)
        return 2

    console = Console(verbose=args.verbose)
    model_name, _ = pick_asr_model(args.asr_model)
    workdir = Path(tempfile.mkdtemp(prefix="voice-chat-"))
    cleaned = False

    def cleanup() -> None:
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        stop_recorder()
        shutdown_procs()
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"files kept in {workdir}", file=sys.stderr)

    engine: EngineSession | None = None
    turns: list[TurnStats] = []
    try:
        quality_guard(args, console)
        tts_mode = os.environ.get("VOICE_TTS", "inproc").lower()
        piper: PiperTTS | None = None
        warm_event = threading.Event()
        if tts_mode != "script":
            piper = PiperTTS(console)
            piper.start()
        engine = EngineSession(args, console)
        engine.start()
        if piper is None:
            threading.Thread(
                target=tts_warmup,
                args=(engine, workdir, console, warm_event),
                daemon=True,
            ).start()

        if args.text:
            for i, text in enumerate(args.text, 1):
                turns.append(one_turn(
                    args, engine, console, model_name, workdir, i,
                    warm_event=warm_event, piper=piper, text=text
                ))
        elif args.wav is not None:
            wav = Path(args.wav)
            if not wav.is_file():
                raise VoiceError(f"no such wav: {wav}")
            console.log(f"[wav] {wav.name} · {wav_duration(wav):.2f} s")
            turns.append(one_turn(
                args, engine, console, model_name, workdir, 1,
                warm_event=warm_event, piper=piper, wav=wav
            ))
        else:
            console.log("голосовой чат tokshit · q/Ctrl-C — выход")
            empty_streak = 0
            turn_no = 0
            while True:
                if args.hands_free:
                    if args.seconds <= 0:
                        raise VoiceError("--hands-free requires --seconds N")
                    time.sleep(0.4)
                elif sys.stdin.isatty():
                    try:
                        line = input("\n🎤 Enter — записать, q — выход: ").strip()
                    except EOFError:
                        break
                    if line.lower() in ("q", "quit", "exit", "/q", "/exit"):
                        break

                wav, dur, _ = record_wav(
                    workdir,
                    seconds=args.seconds,
                    max_seconds=args.max_seconds,
                    hands_free=args.hands_free,
                    console=console,
                )
                if wav is None:
                    empty_streak += 1
                    console.log("   тишина — попробуй ещё раз")
                    if args.once or empty_streak >= 2:
                        break
                    continue
                turn_no += 1
                turn = one_turn(
                    args, engine, console, model_name, workdir, turn_no,
                    warm_event=warm_event, piper=piper, wav=wav
                )
                turns.append(turn)
                if turn.blank:
                    empty_streak += 1
                    if empty_streak >= 2:
                        console.log("   два пустых хода подряд — выхожу")
                        break
                else:
                    empty_streak = 0
                if args.once:
                    break

        session_summary(turns, engine.child_load_s or engine.load_s, engine.prefix_s)
        return 0
    except KeyboardInterrupt:
        console.log("[stop] Ctrl-C")
        return 130
    except VoiceError as exc:
        console.log(f"✗ {exc}")
        return 1
    finally:
        if engine is not None:
            engine.stop()
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
