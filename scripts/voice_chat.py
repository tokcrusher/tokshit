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
  scripts/voice_chat.py                       # TUI, если stdin+stdout TTY
  scripts/voice_chat.py --no-tui              # построчный режим
  scripts/voice_chat.py --tui --text "..."    # TUI + готовый ход
  scripts/voice_chat.py --fast                # lossy top-4 turbo
  scripts/voice_chat.py --one-sentence        # ровно одно предложение
  scripts/voice_chat.py --ru                  # ASR/TTS по-русски
  scripts/voice_chat.py --lang ru             # то же через --lang
  scripts/voice_chat.py                       # запись разрешена уже во время WARMING
  scripts/voice_chat.py --seconds 5           # запись ровно 5 с
  scripts/voice_chat.py --hands-free --seconds 5
  scripts/voice_chat.py --wav in.wav          # без микрофона
  scripts/voice_chat.py --text "..."          # без ASR
  scripts/voice_chat.py --text "..." --text "..."   # два хода, один движок
  scripts/voice_chat.py --check               # зависимости
  scripts/voice_chat.py --no-guard ...        # пропустить probe движка

TUI: Enter/Space record, s stop, f fast, c clear, q quit, ? help,
PgUp/PgDn scroll. Кнопки: RECORD STOP FAST CLEAR QUIT (мышь, если терминал
её отдаёт; Termux часто не отдаёт — клавиши работают всегда).

Аудио — ~/audio/{asr.sh,tts.sh}; они только вызываются, не меняются.
"""

from __future__ import annotations

import argparse
import codecs
import curses
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
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
        text = strip_invisible(text)
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

    def user(self, text: str) -> None:
        """Пользовательская реплика: в TUI это сообщение в транскрипте."""
        if not text:
            return
        with self._lock:
            self._clear_status_locked()
            self._flush_locked()
            print(f"You: {text}", file=sys.stderr, flush=True)

    def log(self, msg: str, *, defer: bool = False) -> None:
        with self._lock:
            self._clear_status_locked()
            if defer and self._turn_open:
                self._deferred.append(msg)
                return
            self._flush_locked()
            print(msg, file=sys.stderr, flush=True)

    def stats(self, stats: "TurnStats", no_play: bool) -> None:
        turn_summary(stats, no_play, self)

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
    s = strip_invisible(text)
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


def normalize_sentence(text: str) -> str:
    """Для TTS-dedupe: lowercase, без пунктуации/zero-width, один пробел."""
    s = text.casefold()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


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

    def __init__(self, max_buf: int = 84, clause_min: int = 26):
        self._buf: list[str] = []
        self._in_fence = False
        self._ticks = 0
        self._fence_had = False
        self._pending_punct = False
        self._line: list[str] = []
        self._emitted = False
        self._drop_rest = False
        self._max_buf = max_buf
        self._clause_min = clause_min

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

            if ch in ",;:" and len(self._buf) >= self._clause_min:
                # Первый звук не должен ждать точку: после запятой/точки с
                # запятой/двоеточия кусок уже звучит как законченная клауза.
                self._buf.append(ch)
                self._flush(out)
                continue

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


MODEL_ALIASES = {
    "inkling": HOME / "Inkling-2-Mini-GGUF" / "Inkling-2-Mini-UD-Q2_K_XL.gguf",
    "inkling2": HOME / "Inkling-2-Mini-GGUF" / "Inkling-2-Mini-UD-Q2_K_XL.gguf",
    "inkling-mini": HOME / "Inkling-2-Mini-GGUF" / "Inkling-2-Mini-UD-Q2_K_XL.gguf",
    "flint": HOME / "models" / "flint-UD-Q3_K_XL.gguf",
    "dspark": HOME / "Inkling-2-Mini-GGUF" / "dspark-Inkling-2-Mini-Q8_0.gguf",
    "mmproj": HOME / "Inkling-2-Mini-GGUF" / "mmproj-F16.gguf",
}


def resolve_model_name(name: str) -> Path:
    p = Path(name)
    if "/" in name or p.suffix == ".gguf":
        return p
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    for d in (HOME / "Inkling-2-Mini-GGUF", HOME / "models", HOME):
        for cand in (d / name, d / f"{name}.gguf"):
            if cand.is_file():
                return cand
    return p


def engine_model() -> Path:
    raw = os.environ.get("VOICE_MODEL")
    return resolve_model_name(raw) if raw else DEFAULT_MODEL


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
                f"[guard] cached WARN engine answer: "
                f"{cached.get('answer', '?')[:120]!r} - build may be broken"
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
        console.log(f"[guard] ok | Paris | {wall:.1f}s")
    else:
        console.log(
            f"[guard] WARN expected Paris, got {answer[:140]!r} ({wall:.1f}s) "
            "- perf build may be broken; continuing"
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
    stop_event: threading.Event | None = None,
    on_tick: Callable[[float], None] | None = None,
) -> tuple[Path | None, float, float]:
    """Записать с микрофона -> (16 kHz WAV|None, длительность, секунды записи).

    TUI передаёт stop_event/on_tick: без input(), с ручной остановкой и
    тиками прошедшего времени. Без stop_event поведение старое (Enter/таймер).
    """
    rec = workdir / f"mic-{int(time.time())}.m4a"
    raw = workdir / f"mic-{int(time.time())}.wav"
    fixed = seconds > 0
    tui_driven = stop_event is not None
    if fixed:
        limit = int(seconds)
    elif tui_driven:
        limit = max(0, int(max_seconds))
    else:
        limit = 0
    cmd = [
        "termux-microphone-record",
        "-f", str(rec),
        "-e", os.environ.get("VOICE_REC_ENCODER", "aac"),
        "-r", os.environ.get("VOICE_REC_RATE", "16000"),
        "-c", "1",
        "-b", os.environ.get("VOICE_REC_BITRATE", "48"),
        "-l", str(limit),
    ]
    if fixed:
        console.log(f"[mic] recording {seconds:.0f}s (Enter/STOP to stop early)")
    elif tui_driven:
        console.log(f"[mic] recording... auto-stop {max_seconds:.0f}s")
    else:
        console.log("[mic] recording...")
    t0 = time.monotonic()
    rc, _, err = run_cmd(cmd, timeout=10)
    if rc == 124:
        raise VoiceError(
            "termux-microphone-record hung (no Termux:API or a mic permission "
            "dialog is blocking)."
        )
    if rc != 0:
        raise VoiceError(
            "termux-microphone-record failed to start. Check Termux:API and "
            f"the RECORD_AUDIO permission.\n{err.strip()}"
        )
    global _RECORDING
    _RECORDING = True

    try:
        if fixed:
            if not hands_free and not tui_driven:
                console.log(f"   ... speak; stops in {seconds:.0f}s (or Enter)")
        elif not tui_driven:
            if not hands_free and sys.stdin.isatty():
                input(
                    f"   ... Enter to stop (auto-stop {max_seconds:.0f}s): "
                )

        deadline = t0 + (seconds if fixed else max_seconds)
        last_tick = -1.0
        while True:
            now = time.monotonic()
            elapsed = now - t0
            if on_tick is not None and elapsed - last_tick >= 0.1:
                last_tick = elapsed
                on_tick(elapsed)
            if fixed and elapsed >= seconds:
                break
            if tui_driven:
                if stop_event is not None and stop_event.wait(0.05):
                    break
                if not fixed and elapsed >= max_seconds:
                    break
                continue
            # Старый линейный режим: фиксированное время или Enter/сон.
            if fixed:
                time.sleep(0.1)
                if not hands_free and sys.stdin.isatty():
                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.readline()
                        break
            else:
                # input() уже отработал; не-TTY просто ждёт автостоп.
                if not sys.stdin.isatty():
                    time.sleep(min(0.1, max(0.0, deadline - now)))
                    if now >= deadline:
                        break
                else:
                    break
    except KeyboardInterrupt:
        stop_recorder()
        raise
    finally:
        stop_recorder()
        if on_tick is not None:
            on_tick(time.monotonic() - t0)
    rec_s = time.monotonic() - t0

    for _ in range(20):
        if rec.is_file() and rec.stat().st_size > 0:
            break
        time.sleep(0.1)
    if not rec.is_file() or rec.stat().st_size == 0:
        raise VoiceError(
            "the microphone produced no file. Check the RECORD_AUDIO "
            "permission for Termux:API."
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
        raise VoiceError(f"ffmpeg could not convert the recording:\n{err.strip()}")

    dur = wav_duration(raw)
    if dur < 0.35:
        console.log(f"[mic] silence ({dur:.2f}s audio) - nothing to transcribe")
        return None, dur, rec_s
    console.log(f"[mic] {dur:.1f}s audio | recorded {rec_s:.1f}s")
    return raw, dur, rec_s


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
        f"[asr] {model_name} | {dt * 1000:.0f} ms | {rate:.1f}x realtime | "
        f"{dur:.2f}s audio"
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
                 piper: PiperTTS | None = None,
                 abort_event: threading.Event | None = None):
        self.workdir = workdir
        self.console = console
        self.stats = stats
        self.no_play = no_play
        self.warm_event = warm_event
        self.piper = piper
        self.abort_event = abort_event
        self.text_q: queue.Queue = queue.Queue()
        self.audio_q: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.player = StreamingPlayer(console, stats) if not no_play else None
        self.tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._closed = False
        self._errors_lock = threading.Lock()
        self._last_spoken_norm: str | None = None
        self._dup_run = 0
        self._tts_dup_stop = False

    def start(self) -> None:
        self.tts_thread.start()
        self.play_thread.start()

    def _aborted(self) -> bool:
        return self.abort_event is not None and self.abort_event.is_set()

    def push(self, sentence: str) -> None:
        if (
            self.stop_event.is_set() or self._closed
            or self._aborted() or self._tts_dup_stop
        ):
            return
        if not sentence.strip():
            return
        norm = normalize_sentence(sentence)
        if not norm:
            return
        if norm == self._last_spoken_norm:
            self._dup_run += 1
            if self.console.verbose:
                self.console.log(
                    f"[tts] duplicate sentence dropped: {sentence[:70]!r}",
                    defer=True,
                )
            if self._dup_run >= 2:
                self._tts_dup_stop = True
                self.console.log(
                    "[tts] duplicate guard: repeated sentence twice in a row; "
                    "TTS stopped for this turn",
                    defer=True,
                )
                try:
                    self.text_q.put_nowait(None)
                except queue.Full:
                    pass
            return
        self._last_spoken_norm = norm
        self._dup_run = 0
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
                if self.stop_event.is_set() or self._tts_dup_stop:
                    self.audio_q.put(None)
                    return
            use_piper = self.piper.sess is not None
            if not use_piper:
                self.console.log("[tts] Piper unavailable; using tts.sh", defer=True)
        elif self.warm_event is not None:
            while not self.warm_event.wait(0.2):
                if self.stop_event.is_set() or self._tts_dup_stop:
                    self.audio_q.put(None)
                    return
            use_piper = False
        else:
            use_piper = False
        try:
            while (
                not self.stop_event.is_set()
                and not self._aborted()
                and not self._tts_dup_stop
            ):
                sentence = self.text_q.get()
                if sentence is None:
                    break
                spoken = clean_for_speech(sentence)
                if not spoken:
                    continue
                for part in split_for_tts(spoken):
                    if (
                        self.stop_event.is_set() or self._aborted()
                        or self._tts_dup_stop
                    ):
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
            while not self.stop_event.is_set() and not self._aborted():
                item = self.audio_q.get()
                if item is None:
                    break
                if self._aborted():
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
        # No default few-shot: the model answers better without a prefix.
        # VOICE_FEWSHOT stays as an explicit power-user override only.
        self.fewshot = os.environ.get("VOICE_FEWSHOT", "")
        if not self.fewshot.strip():
            self.fewshot = ""
        # Minimal language anchor: without it the model drifts to French
        # for Paris and to English for short questions. Not a persona example.
        if not self.fewshot and not args.chat_template:
            self.fewshot = (
                "Q: Отвечай по-русски.\\nA: Хорошо."
                if getattr(args, "lang", "en") == "ru"
                else "Q: Answer in English.\\nA: Okay."
            )
        self._expect_prefix = bool(self.fewshot) and not args.chat_template
        self.ready_event = threading.Event()
        self.started_event = threading.Event()
        self.stopped_event = threading.Event()
        self.start_error: str | None = None
        self.loaded = False
        self.prefix_tokens = 0
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

    def fail(self, message: str) -> None:
        """Старт не состоится: разбудить queued-turn с внятной ошибкой."""
        self.start_error = message
        self.started_event.set()

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

        # VOICE_FAST=1: top-4 (EN) / top-6 (RU) experts, still 3 passes,
        # original dense weights. RU has no prefix and top-4 looped on the
        # second turn; top-6 stays readable.
        if os.environ.get("VOICE_FAST", "").strip() not in ("", "0"):
            lang = getattr(self.args, "lang", "en")
            env["TOKSHIT_DRAFT_TOPK"] = (
                os.environ.get("VOICE_TOPK")
                or os.environ.get("TOKSHIT_DRAFT_TOPK")
                or ("6" if lang == "ru" else "4")
            )
            env.pop("TOKSHIT_HOT_Q4", None)
            env.pop("TOKSHIT_INK_PASSES", None)  # keep the trained 3 passes
            env["TOKSHIT_POOL_WORKERS"] = os.environ.get(
                "VOICE_FAST_WORKERS", "6"
            )

        # Default: полный ответ до EOG/--max-new. Три маркера режут эхо
        # few-shot, не трогая обычные слова:
        #   "Q:"      — начало следующего вопроса;
        #   "\nQ"     — bare Q-строка после ответа;
        #   "\nA"     — bare A-строка после ответа.
        # Важно: НЕ "Q"/"A" без newline — иначе режутся слова.
        for env_name, voice_name, default in (
            ("TOKSHIT_STOP_AT", "VOICE_STOP_AT", "Q:"),
            ("TOKSHIT_STOP_AT2", "VOICE_STOP_AT2", r"\nQ"),
            ("TOKSHIT_STOP_AT3", "VOICE_STOP_AT3", r"\nA"),
        ):
            value = os.environ.get(voice_name, default)
            env.pop(env_name, None)
            if value:
                env[env_name] = value

        one_sentence = bool(
            getattr(self.args, "one_sentence", False)
            or os.environ.get("VOICE_ONE_SENTENCE", "").strip() not in ("", "0")
        )
        env.pop("TOKSHIT_STOP_SENTENCE", None)
        if one_sentence:
            env["TOKSHIT_STOP_SENTENCE"] = "1"

        if self.args.chat_template:
            # Opt-in: нативный шаблон Inkling (видимый reasoning — на любителя).
            env.pop("TOKSHIT_RAW_PROMPT", None)
            env.pop("TOKSHIT_CHAT_UNESCAPE", None)
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
        if os.environ.get("VOICE_REASONING_BUDGET"):
            env["TOKSHIT_REASONING_BUDGET"] = os.environ["VOICE_REASONING_BUDGET"]

        # Explicit model tuning wins over FAST/LOSSLESS presets.
        for attr, key in (
            ("experts", "TOKSHIT_DRAFT_TOPK"),
            ("passes", "TOKSHIT_INK_PASSES"),
            ("workers", "TOKSHIT_POOL_WORKERS"),
            ("cache_mb", "TOKSHIT_EXPERT_CACHE_MB"),
            ("kv_ctx", "TOKSHIT_KV_CTX"),
            ("layers", "GGUF_NLAYERS"),
            ("prefill_batch", "TOKSHIT_INK_PREFILL_BATCH"),
            ("hot", "TOKSHIT_HOT"),
        ):
            value = getattr(self.args, attr, None)
            if value is not None:
                env[key] = str(value)
        if getattr(self.args, "q4", False):
            env["TOKSHIT_HOT_Q4"] = "1"

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
            self.fail(f"cannot start engine: {exc}")
            raise VoiceError(f"cannot start engine: {exc}")
        self.started_event.set()
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
                    self.prefix_tokens = int(m.group(1))
                    self.prefix_s = float(m.group(2))
                    self.console.clear_status()
                    self.console.log(
                        f"[engine] prefix {self.prefix_tokens} tok / "
                        f"{self.prefix_s:.1f}s"
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
        """Дождаться engine.start()/load/prefix.

        Может вызываться до engine.start(): TUI ставит placeholder и даёт
        записать/распознать ход, пока идёт quality guard и загрузка модели.
        """
        deadline = time.monotonic() + timeout
        while not self.ready_event.is_set():
            if self.stopped_event.is_set():
                raise VoiceError("engine stopped before it became ready")
            if self.start_error:
                raise VoiceError(self.start_error)
            if self.started_event.is_set():
                if self.proc is None or self.proc.poll() is not None:
                    tail = self.stderr_tail()
                    rc = self.proc.returncode if self.proc else "?"
                    raise VoiceError(
                        f"engine exited during load (rc={rc}):\n{tail}"
                    )
            if time.monotonic() > deadline:
                if not self.started_event.is_set():
                    raise VoiceError(
                        f"engine did not start in {timeout:.0f}s "
                        "(quality guard/model lock?)"
                    )
                raise VoiceError(f"engine did not load in {timeout:.0f}s")
            time.sleep(0.1)

    def _payload(self, text: str) -> str:
        # CHAT_UNESCAPE: литеральные \n в строке stdin -> реальные newline.
        text = " ".join(text.splitlines()).strip().replace("\\", " ")
        if self.args.chat_template:
            return text
        # Language is anchored by the startup prefix; the payload adds a
        # compact quality instruction (detail + length), not a persona.
        lang = getattr(self.args, "lang", "en")
        quality = (
            " Ответь по-русски, подробно и по делу, 2-3 предложения."
            if lang == "ru"
            else " Answer in English, detailed and to the point, 2-3 sentences."
        )
        if text and text[-1] not in ".!?…":
            text += "."
        # Leading newline on every turn; tested engine protocol.
        return f"\\nQ: {text}{quality}\\nA:"

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

    def clear_context(self, timeout: float = 1.0) -> bool:
        """`/clear` движку: стереть KV/позицию, turn_no=0.

        Вызывать только когда ход не идёт. Ответ /clear идёт в stdout без RS,
        поэтому вычитываем до паузы, чтобы он не попал в следующий ask.
        """
        if not self.ready_event.is_set():
            return False
        if self.proc is None or self.proc.poll() is not None:
            return False
        if self._outbuf or self.proc.stdin is None or self.proc.stdout is None:
            return False
        try:
            self.proc.stdin.write(b"/clear\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        fd = self.proc.stdout.fileno()
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        data = ""
        quiet_at: float | None = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r, _, _ = select.select([fd], [], [], 0.05)
            except InterruptedError:
                continue
            if r:
                try:
                    raw = os.read(fd, 4096)
                except OSError:
                    break
                if not raw:
                    break
                data += dec.decode(raw)
                quiet_at = time.monotonic() + 0.12
            elif quiet_at is not None and time.monotonic() >= quiet_at:
                break
        self.turn_no = 0
        if self.console.verbose and data:
            self.console.log(f"[engine] /clear: {strip_ansi(data).strip()[:120]}")
        return True

    def query_stats(self, timeout: float = 0.8) -> dict:
        """Спросить движок `/stats` между ходами (TUI: cache/rss/ctx).

        Chat-протокол отвечает на /stats в stdout без RS. Читаем до короткой
        паузы; после этого буфер протокола снова пуст.
        """
        if not self.ready_event.is_set():
            return {}
        if self.proc is None or self.proc.poll() is not None:
            return {}
        if self._outbuf or self.proc.stdin is None or self.proc.stdout is None:
            return {}
        try:
            self.proc.stdin.write(b"/stats\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return {}
        fd = self.proc.stdout.fileno()
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        data = ""
        quiet_at: float | None = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r, _, _ = select.select([fd], [], [], 0.05)
            except InterruptedError:
                continue
            if r:
                try:
                    raw = os.read(fd, 8192)
                except OSError:
                    break
                if not raw:
                    break
                data += dec.decode(raw)
                quiet_at = time.monotonic() + 0.12
            elif quiet_at is not None and time.monotonic() >= quiet_at:
                break
        text = strip_ansi(data)
        if not text:
            return {}
        out: dict = {}
        # hit/miss counts печатаются даже когда rate = "—" (0 обращений).
        m = re.search(r"\(hit\s+(\d+)\s*/\s*miss\s+(\d+)\)", text)
        if m:
            hit = int(m.group(1))
            miss = int(m.group(2))
            out["cache_hit"] = hit
            out["cache_miss"] = miss
            out["cache_rate"] = (
                100.0 * hit / (hit + miss) if hit + miss else 0.0
            )
        else:
            m = re.search(r"кэш\s+([\d.]+)%", text)
            if m:
                out["cache_rate"] = float(m.group(1))
        m = re.search(r"RSS\s+(\d+)\s*MB", text)
        if m:
            out["rss_mb"] = int(m.group(1))
        m = re.search(r"MemAvailable\s+(\d+)\s*MB", text)
        if m:
            out["mem_avail_mb"] = int(m.group(1))
        m = re.search(r"контекст\s+(\d+)", text)
        if m:
            out["ctx_pos"] = int(m.group(1))
        m = re.search(r"prefill\s+([\d.]+)s", text)
        if m:
            out["prefill_s"] = float(m.group(1))
        m = re.search(r"последняя:\s*([\d.]+)\s*tok/s", text)
        if m:
            out["tok_s"] = float(m.group(1))
        m = re.search(r"сгенерировано\s+(\d+)\s+токенов", text)
        if m:
            out["total_tokens"] = int(m.group(1))
        m = re.search(r"(\d+)\s+ходов", text)
        if m:
            out["turns"] = int(m.group(1))
        if not out and self.console.verbose:
            self.console.log(f"[engine] /stats unparsed: {text[:200]!r}")
        return out

    def stop(self) -> None:
        self.stopped_event.set()
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

def turn_summary(stats: TurnStats, no_play: bool, console: Console) -> None:
    def ms(x: float | None) -> str:
        return f"{x * 1000:.0f} ms" if x else "n/a"

    asr = f"asr {ms(stats.asr_s)} ({stats.asr_model})" if stats.asr_s else "asr n/a"
    parts = [
        asr,
        f"engine {ms(stats.engine_s)}",
        f"tts {ms(stats.tts_s)}",
        f"total {ms(stats.total_s)}",
    ]
    console.log("[summary] " + " | ".join(parts))
    if not no_play:
        console.log(
            f"[summary] play {ms(stats.play_s)} | {stats.audio_out_s:.1f}s audio"
        )
    m = stats.engine_metrics
    tok = (
        f"{m.get('gen_tokens', 0)} tok | {m.get('tok_s', 0):.2f} tok/s"
        if m.get("gen_tokens") else "tok/s n/a"
    )
    console.log(
        f"[stats]   prefill {m.get('prefill_tokens', 0)} tok | "
        f"{m.get('prefill_s', 0):.2f}s | gen {tok} | {stats.chars} chars | "
        f"{stats.tts_chunks} tts chunks | {stats.audio_out_s:.1f}s audio"
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
        first += f" | first tts {stats.first_tts_s - stats.t_start:.2f}s (turn)"
    if stats.first_audio_s is not None:
        first += f" | first audio {stats.first_audio_turn_s:.2f}s (turn)"
        if stats.engine_ask_start:
            first += f" / {stats.first_audio_s - stats.engine_ask_start:.2f}s (engine)"
    console.log(f"[latency] {first}")
    for err in stats.errors:
        console.log(f"[error]   {err}")


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
    abort_event: threading.Event | None = None,
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
            console.log("[asr] silence / no speech recognized - skipping turn")
            stats.blank = True
            stats.total_s = time.monotonic() - stats.t_start
            return stats
    else:
        transcript = text.strip()

    console.user(transcript)
    if text is not None:
        stats.asr_model = ""
    # ASR уже сделан; движок может ещё грузиться/прогревать prefix.
    # turn worker висит здесь, а не шлёт Q: в полуготовый движок.
    engine.wait_ready()
    pipeline = SpeechPipeline(
        turn_dir, console, stats, args.no_play, warm_event, piper,
        abort_event=abort_event,
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
    console.stats(stats, args.no_play)
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


# ────────────────────────────── TUI ──────────────────────────────

INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")


def strip_invisible(text: str) -> str:
    return INVISIBLE_RE.sub("", text)


EMOJI_ASCII = {
    "\U0001f3a4": "mic",
    "\U0001f5e3": "user",
    "\u26a0": "WARN",
    "\u2717": "x",
    "\u2705": "ok",
    "\u2192": "->",
    "\u00b7": "|",
    "\u00d7": "x",
    "\u2026": "...",
    "\u2022": "*",
}


def sanitize_ui_text(text: str) -> str:
    s = strip_invisible(strip_ansi(text))
    for src, dst in EMOJI_ASCII.items():
        s = s.replace(src, dst)
    return s


def fmt_sec(x: float | None) -> str:
    if x is None or x <= 0:
        return "n/a"
    if x < 1.0:
        return f"{x * 1000:.0f}ms"
    return f"{x:.2f}s"


def fmt_mb(mb: float) -> str:
    if not mb or mb <= 0:
        return "n/a"
    if mb >= 1024:
        return f"{mb / 1024:.1f}G"
    return f"{mb:.0f}M"


def read_rss_mb(pid: int | None = None) -> float:
    path = "/proc/self/status" if pid is None else f"/proc/{pid}/status"
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def proc_tree_rss_mb(root_pid: int | None) -> float:
    if not root_pid:
        return 0.0
    total = 0.0
    seen: set[int] = set()
    stack = [root_pid]
    while stack and len(seen) < 64:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += read_rss_mb(pid)
        try:
            raw = Path(f"/proc/{pid}/task/{pid}/children").read_text()
        except OSError:
            continue
        for part in raw.split():
            try:
                stack.append(int(part))
            except ValueError:
                pass
    return total


def thermal_max_c() -> float | None:
    """Max CPU/SoC zone. Other zones (battery, charger, camera) can read
    90+ C on a cold phone and made the old gauge lie."""
    best_cpu: float | None = None
    best_any: float | None = None
    hints = ("cpu", "cpuss", "soc", "aoss")
    for temp_path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            raw = temp_path.read_text().strip()
            value = float(raw)
            ztype = (temp_path.parent / "type").read_text().strip().lower()
        except (OSError, ValueError):
            continue
        if abs(value) > 200:
            value /= 1000.0
        if not (-40.0 <= value <= 150.0):
            continue
        best_any = value if best_any is None else max(best_any, value)
        if any(h in ztype for h in hints):
            best_cpu = value if best_cpu is None else max(best_cpu, value)
    return best_cpu if best_cpu is not None else best_any


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip() not in ("", "0")


def apply_mode_env(fast: bool, lang: str = "en") -> None:
    """FAST/LOSSLESS переключают только env движка; перезапуск делает TUI.

    RU fast uses top-6: no-prefix top-4 looped on the second Russian turn.
    EN fast keeps top-4.
    """
    if fast:
        topk = os.environ.get("VOICE_TOPK") or ("6" if lang == "ru" else "4")
        os.environ["VOICE_FAST"] = "1"
        os.environ["TOKSHIT_DRAFT_TOPK"] = topk
        os.environ.pop("TOKSHIT_HOT_Q4", None)
        os.environ.pop("TOKSHIT_INK_PASSES", None)
        os.environ["TOKSHIT_POOL_WORKERS"] = "6"
        os.environ["VOICE_POOL_WORKERS"] = "6"
        os.environ["VOICE_FAST_WORKERS"] = "6"
    else:
        os.environ.pop("VOICE_FAST", None)
        os.environ.pop("TOKSHIT_DRAFT_TOPK", None)
        os.environ.pop("TOKSHIT_HOT_Q4", None)
        os.environ.pop("TOKSHIT_INK_PASSES", None)
        os.environ.pop("VOICE_FAST_WORKERS", None)
        os.environ["TOKSHIT_POOL_WORKERS"] = "4"
        os.environ["VOICE_POOL_WORKERS"] = "4"


def tts_voice_name() -> str:
    return os.environ.get("TTS_VOICE", "en_US-lessac-medium")


def apply_language(args: argparse.Namespace) -> None:
    """--ru/--lang/VOICE_LANG -> ASR_LANG, few-shot lang, TTS voice."""
    warnings: list[str] = []
    env_lang = os.environ.get("VOICE_LANG", "").strip().lower()
    lang = (getattr(args, "lang", None) or env_lang or "en").strip().lower()
    if getattr(args, "ru", False):
        lang = "ru"
    if lang not in ("en", "ru"):
        warnings.append(f"[lang] unknown language {lang!r}; using en")
        lang = "en"
    args.lang = lang
    args.ru = lang == "ru"
    os.environ["ASR_LANG"] = lang
    if lang == "ru" and not os.environ.get("TTS_VOICE"):
        ru_voice = MODELS_DIR / "ru_RU-irina-medium.onnx"
        if ru_voice.is_file():
            os.environ["TTS_VOICE"] = "ru_RU-irina-medium"
        else:
            warnings.append(
                f"[tts] Russian voice missing: {ru_voice}; "
                f"keeping {tts_voice_name()}"
            )
    args.lang_warning = "; ".join(warnings)


def cfg_label(args: argparse.Namespace) -> str:
    parts: list[str] = []
    for attr, label in (
        ("experts", "top"),
        ("passes", "passes"),
        ("workers", "w"),
        ("cache_mb", "cache"),
        ("kv_ctx", "kv"),
        ("layers", "layers"),
        ("prefill_batch", "pf"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            parts.append(f"{label} {value}")
    if getattr(args, "hot", None) is not None:
        parts.append(f"hot {args.hot}")
    if getattr(args, "q4", False):
        parts.append("q4")
    return " ".join(parts) if parts else "auto"


def stop_label(args: argparse.Namespace) -> str:
    stop = os.environ.get("VOICE_STOP_AT", "Q:")
    parts = [stop if stop else "off"]
    for name, default in (
        ("VOICE_STOP_AT2", r"\nQ"),
        ("VOICE_STOP_AT3", r"\nA"),
    ):
        value = os.environ.get(name, default)
        if value:
            shown = value.replace("\n", "\\n")
            parts.append(shown)
    one = bool(
        getattr(args, "one_sentence", False)
        or env_truthy("VOICE_ONE_SENTENCE")
    )
    if one:
        parts.append("one-sentence")
    return " / ".join(parts)


def wrap_ui_line(text: str, width: int) -> list[str]:
    if width < 1:
        return [text]
    s = text.replace("\t", "    ").replace("\r", "")
    if not s:
        return [""]
    out: list[str] = []
    cur = ""
    for word in s.split(" "):
        if not word:
            if cur:
                cur += " "
            continue
        while len(word) > width:
            if cur:
                out.append(cur)
                cur = ""
            out.append(word[:width])
            word = word[width:]
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            out.append(cur)
            cur = word
    if cur or not out:
        out.append(cur)
    return out


@dataclass
class UiMessage:
    role: str
    text: str = ""
    done: bool = True


class UiState:
    """Состояние TUI. Меняется из воркеров под lock, curses рисует main."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.messages: list[UiMessage] = []
        self.log_lines: deque[str] = deque(maxlen=80)
        self.status = ""
        self.notice = ""
        self.notice_until = 0.0
        self.phase = "LOADING"
        self.ready = False
        self.busy = False
        self.recording = False
        self.rec_elapsed = 0.0
        self.restarting = False
        self.fast = False
        self.load_s = 0.0
        self.turn_no = 0
        self.scroll = 0
        self.last_stats: dict = {}
        self.engine_stats: dict = {}
        self.abort_event: threading.Event | None = None
        self._stream_idx: int | None = None

    def snapshot(self) -> dict:
        with self.lock:
            msgs = [
                (m.role, m.text, m.done) for m in self.messages[-120:]
            ]
            return {
                "messages": msgs,
                "logs": list(self.log_lines),
                "status": self.status,
                "notice": self.notice,
                "notice_until": self.notice_until,
                "phase": self.phase,
                "ready": self.ready,
                "busy": self.busy,
                "recording": self.recording,
                "rec_elapsed": self.rec_elapsed,
                "restarting": self.restarting,
                "fast": self.fast,
                "load_s": self.load_s,
                "turn_no": self.turn_no,
                "scroll": self.scroll,
                "last_stats": dict(self.last_stats),
                "engine_stats": dict(self.engine_stats),
            }

    def user(self, text: str) -> None:
        if not text:
            return
        clean = sanitize_ui_text(text).strip()
        with self.lock:
            if self._stream_idx is not None:
                self.messages[self._stream_idx].done = True
                self._stream_idx = None
            self.messages.append(UiMessage("user", clean))
            self.scroll = 0
            self.status = ""
            self.notice = ""

    def answer(self, text: str) -> None:
        text = strip_invisible(text)
        if not text:
            return
        with self.lock:
            if self._stream_idx is None:
                self.messages.append(UiMessage("assistant", "", False))
                self._stream_idx = len(self.messages) - 1
            self.messages[self._stream_idx].text += text

    def end_answer(self) -> None:
        with self.lock:
            if self._stream_idx is not None:
                self.messages[self._stream_idx].done = True
                self._stream_idx = None

    def log(self, msg: str, *, defer: bool = False) -> None:
        del defer
        clean = sanitize_ui_text(str(msg))
        with self.lock:
            for line in clean.splitlines() or [""]:
                self.log_lines.append(line)

    def set_status(self, msg: str) -> None:
        with self.lock:
            self.status = sanitize_ui_text(msg)

    def clear_status(self) -> None:
        with self.lock:
            self.status = ""

    def notice_msg(self, msg: str, ttl: float = 5.0) -> None:
        with self.lock:
            self.notice = sanitize_ui_text(msg)
            self.notice_until = time.monotonic() + ttl if ttl > 0 else 0.0

    def clear_notice(self) -> None:
        with self.lock:
            self.notice = ""
            self.notice_until = 0.0

    def record_tick(self, elapsed: float) -> None:
        with self.lock:
            self.rec_elapsed = elapsed

    def set_stats(self, stats: "TurnStats", no_play: bool) -> None:
        m = stats.engine_metrics or {}
        first_turn = None
        if stats.first_token_s is not None and stats.engine_ask_start:
            first_turn = stats.engine_ask_start + stats.first_token_s - stats.t_start
        first_tts_turn = (
            stats.first_tts_s - stats.t_start
            if stats.first_tts_s is not None else None
        )
        with self.lock:
            self.last_stats = {
                "audio_s": stats.audio_s,
                "asr_s": stats.asr_s,
                "asr_model": stats.asr_model,
                "engine_s": stats.engine_s,
                "first_token_s": stats.first_token_s,
                "first_token_turn_s": first_turn,
                "first_tts_turn_s": first_tts_turn,
                "first_audio_turn_s": stats.first_audio_turn_s,
                "tts_s": stats.tts_s,
                "tts_chunks": stats.tts_chunks,
                "audio_out_s": stats.audio_out_s,
                "play_s": stats.play_s,
                "total_s": stats.total_s,
                "chars": stats.chars,
                "prefill_tokens": int(m.get("prefill_tokens", 0) or 0),
                "prefill_s": float(m.get("prefill_s", 0.0) or 0.0),
                "gen_tokens": int(m.get("gen_tokens", 0) or 0),
                "gen_s": float(m.get("gen_s", 0.0) or 0.0),
                "tok_s": float(m.get("tok_s", 0.0) or 0.0),
                "no_play": no_play,
                "blank": stats.blank,
                "errors": list(stats.errors),
            }

    def set_engine_stats(self, stats: dict) -> None:
        if not stats:
            return
        with self.lock:
            for key, value in stats.items():
                if value is not None:
                    self.engine_stats[key] = value

    def set_scroll(self, value: int) -> None:
        with self.lock:
            self.scroll = max(0, value)

    def clear(self) -> None:
        with self.lock:
            self.messages.clear()
            self.log_lines.clear()
            self.scroll = 0
            self.status = ""
            self.notice = ""
            self.notice_until = 0.0
            self._stream_idx = None


class TuiConsole(Console):
    """Console для воркеров: никакого curses/stderr, только UiState."""

    def __init__(self, state: UiState, verbose: bool = False):
        self.state = state
        self.verbose = verbose

    def answer(self, text: str) -> None:
        self.state.answer(text)

    def end_answer(self) -> None:
        self.state.end_answer()

    def user(self, text: str) -> None:
        self.state.user(text)

    def log(self, msg: str, *, defer: bool = False) -> None:
        self.state.log(msg, defer=defer)

    def status(self, msg: str) -> None:
        self.state.set_status(msg)

    def clear_status(self) -> None:
        self.state.clear_status()

    def stats(self, stats: "TurnStats", no_play: bool) -> None:
        self.state.set_stats(stats, no_play)


class EngineHolder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.engine: EngineSession | None = None

    def get(self) -> EngineSession | None:
        with self.lock:
            return self.engine

    def set(self, engine: EngineSession | None) -> None:
        with self.lock:
            self.engine = engine

    def swap(self, engine: EngineSession | None) -> EngineSession | None:
        with self.lock:
            old = self.engine
            self.engine = engine
            return old


class TuiApp:
    BUTTONS = [
        ("record", "[ RECORD ]"),
        ("stop", "[ STOP ]"),
        ("fast", "[ FAST ]"),
        ("clear", "[ CLEAR ]"),
        ("quit", "[ QUIT ]"),
    ]

    HELP_LINES = [
        "KEYS",
        "  Enter / Space   start recording (also while engine warms up)",
        "  s               stop recording / stop audio",
        "  f               toggle FAST / LOSSLESS (engine restart)",
        "  c               clear transcript + engine KV context",
        "  q / Ctrl-C      quit",
        "  PgUp / PgDn     scroll transcript",
        "  ?               this help",
        "",
        "BUTTONS",
        "  Tap or click the button row if the terminal reports mouse events.",
        "  Termux may not report taps; the keys above always work.",
    ]

    def __init__(
        self,
        args: argparse.Namespace,
        model_name: str,
        workdir: Path,
        console: TuiConsole,
        state: UiState,
    ):
        self.args = args
        self.model_name = model_name
        self.workdir = workdir
        self.console = console
        self.state = state
        self.holder = EngineHolder()
        self.events: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.warm_event = threading.Event()
        self.piper: PiperTTS | None = None
        self.fast = bool(
            getattr(args, "fast", False) or env_truthy("VOICE_FAST")
        )
        self.state.fast = self.fast
        if getattr(args, "lang_warning", ""):
            self.state.log(args.lang_warning)
        self.running = True
        self.help = False
        self.color = False
        self.turn_no = 0
        self.rec_stop_event: threading.Event | None = None
        self.abort_event: threading.Event | None = None
        self.initial_tasks: list[tuple[str, str]] = [
            ("text", t) for t in args.text
        ]
        if args.wav is not None:
            self.initial_tasks.append(("wav", args.wav))
        self.had_initial = bool(self.initial_tasks)
        self.initial_done = not self.had_initial
        self._buttons: list[tuple[int, int, int, str]] = []
        self._page = 10
        self._last_click = (0, 0, 0.0)
        self._sys: dict = {}
        self._sys_at = 0.0
        self._auto_at = 0.0
        self.started_at = time.monotonic()
        self.startup_thread: threading.Thread | None = None
        self.restart_thread: threading.Thread | None = None

    # ----- lifecycle -------------------------------------------------

    def run(self) -> None:
        try:
            curses.wrapper(self._run)
        except curses.error as exc:
            raise VoiceError(f"curses TUI failed: {exc}")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.running = False
        self.stop_flag.set()
        if self.rec_stop_event is not None:
            self.rec_stop_event.set()
        if self.abort_event is not None:
            self.abort_event.set()
        engine = self.holder.get()
        if engine is not None:
            engine.stop()
        shutdown_procs()
        # Стартовый/restart-поток мог пройти проверку stop_flag и запустить
        # движок уже после первого kill. Ждём его и гасим ещё раз.
        current = threading.current_thread()
        for th in (self.startup_thread, self.restart_thread):
            if th is not None and th is not current:
                th.join(timeout=5.0)
        engine = self.holder.get()
        if engine is not None:
            engine.stop()
        shutdown_procs()

    def _init_curses(self, stdscr) -> None:
        stdscr.keypad(True)
        stdscr.timeout(100)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        try:
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_CYAN, -1)
                curses.init_pair(2, curses.COLOR_GREEN, -1)
                curses.init_pair(3, curses.COLOR_YELLOW, -1)
                curses.init_pair(4, curses.COLOR_RED, -1)
                self.color = True
        except curses.error:
            self.color = False
        try:
            curses.mouseinterval(0)
        except curses.error:
            pass
        try:
            mask = curses.ALL_MOUSE_EVENTS
            mask |= getattr(curses, "REPORT_MOUSE_POSITION", 0)
            curses.mousemask(mask)
        except curses.error:
            pass

    def _run(self, stdscr) -> None:
        self._init_curses(stdscr)
        self.startup_thread = threading.Thread(
            target=self._startup, daemon=True
        )
        self.startup_thread.start()
        while self.running:
            try:
                self._drain_events()
                self._sync_engine()
                self._maybe_initial()
                self._maybe_hands_free()
                self._expire_notice()
                self._draw(stdscr)
                key = stdscr.get_wch()
            except curses.error:
                continue
            except KeyboardInterrupt:
                self._quit()
                break
            self._handle_key(key)

    def _startup(self) -> None:
        # Placeholder EngineSession публикуется ДО quality guard/load: TUI
        # сразу разрешает запись, ASR делает ход, а ask ждёт started/ready.
        try:
            engine = EngineSession(self.args, self.console)
            tts_mode = os.environ.get("VOICE_TTS", "inproc").lower()
            if tts_mode != "script":
                # Объект нужен queued-turn'у до старта; загрузка ниже.
                self.piper = PiperTTS(self.console)
        except Exception as exc:  # noqa: BLE001
            self.console.log(f"ERROR: {exc}")
            with self.state.lock:
                self.state.phase = "ERROR"
                self.state.ready = False
            self.events.put(("startup_error", str(exc)))
            return
        self.holder.set(engine)
        if self.stop_flag.is_set():
            engine.stop()
            return
        try:
            quality_guard(self.args, self.console)
            if self.stop_flag.is_set():
                engine.stop()
                return
            apply_mode_env(self.fast, getattr(self.args, "lang", "en"))
            if self.piper is not None:
                self.piper.start()
            if self.stop_flag.is_set():
                engine.stop()
                return
            with self.state.lock:
                self.state.phase = "LOADING"
            self.console.log("[engine] starting persistent chat")
            engine.start()
            if self.stop_flag.is_set():
                engine.stop()
                return
            if self.piper is None:
                threading.Thread(
                    target=tts_warmup,
                    args=(engine, self.workdir, self.console, self.warm_event),
                    daemon=True,
                ).start()
            self.events.put(("startup_ok", None))
        except Exception as exc:  # noqa: BLE001 - поток не должен падать
            self.console.log(f"ERROR: {exc}")
            engine.fail(str(exc))
            engine.stop()
            with self.state.lock:
                self.state.phase = "ERROR"
                self.state.ready = False
            self.events.put(("startup_error", str(exc)))

    # ----- state/events ----------------------------------------------

    def _sync_engine(self) -> None:
        engine = self.holder.get()
        if engine is None:
            return
        proc = engine.proc
        if proc is not None and proc.poll() is not None:
            with self.state.lock:
                if self.state.phase != "ERROR":
                    self.state.phase = "ERROR"
                    self.state.ready = False
                    self.console.log(
                        f"ERROR: engine exited (rc={proc.returncode})"
                    )
                    self.state.notice_msg("Engine exited - see log", ttl=0)
            return
        if engine.ready_event.is_set():
            with self.state.lock:
                self.state.ready = True
                self.state.phase = "READY"
                self.state.load_s = engine.child_load_s or engine.load_s
        else:
            with self.state.lock:
                self.state.ready = False
                if not self.state.restarting and self.state.phase != "ERROR":
                    self.state.phase = "LOADING"

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "startup_error":
                self.state.notice_msg(f"Engine failed: {payload}", ttl=0)
            elif kind == "restart_ok":
                with self.state.lock:
                    self.state.restarting = False
                    self.state.phase = "LOADING"
                self.console.log("[engine] restart complete; loading")
            elif kind == "restart_error":
                with self.state.lock:
                    self.state.restarting = False
                    self.state.phase = "ERROR"
                    self.state.ready = False
                self.console.log(f"ERROR: engine restart failed: {payload}")
                self.state.notice_msg("Engine restart failed - see log", ttl=0)
            elif kind == "record_done":
                self._handle_record_done(payload)
            elif kind == "record_error":
                with self.state.lock:
                    self.state.recording = False
                    self.state.rec_elapsed = 0.0
                    self.state.status = ""
                self.rec_stop_event = None
                self.console.log(f"ERROR: microphone: {payload}")
                self.state.notice_msg("Microphone error - see log", ttl=6)
                if self.args.hands_free:
                    self._auto_at = time.monotonic() + 5.0
            elif kind == "turn_done":
                stats = payload
                with self.state.lock:
                    self.state.busy = False
                    self.state.abort_event = None
                    self.state.scroll = 0
                self.abort_event = None
                if getattr(stats, "blank", False):
                    self.state.notice_msg(
                        "Silence / no speech recognized - try again", ttl=5
                    )
                if self.args.once:
                    self._quit()
                elif self.args.hands_free:
                    self._auto_at = time.monotonic() + 0.4
            elif kind == "turn_error":
                with self.state.lock:
                    self.state.busy = False
                    self.state.abort_event = None
                self.abort_event = None
                self.console.log(f"ERROR: turn failed: {payload}")
                self.state.notice_msg("Turn failed - see log", ttl=8)
                if self.args.hands_free:
                    self._auto_at = time.monotonic() + 5.0
                if self.args.once:
                    self._quit()
            elif kind == "startup_ok":
                pass

    def _expire_notice(self) -> None:
        with self.state.lock:
            if self.state.notice and self.state.notice_until:
                if time.monotonic() >= self.state.notice_until:
                    self.state.notice = ""
                    self.state.notice_until = 0.0

    def _maybe_initial(self) -> None:
        with self.state.lock:
            ready = self.state.ready
            busy = self.state.busy
            recording = self.state.recording
            restarting = self.state.restarting
        if self.initial_done or not ready or busy or recording or restarting:
            return
        if not self.initial_tasks:
            self.initial_done = True
            if self.args.once and self.had_initial:
                self._quit()
            return
        kind, value = self.initial_tasks.pop(0)
        if kind == "text":
            self._start_turn(text=value)
        else:
            self._start_turn(wav=Path(value))

    def _maybe_hands_free(self) -> None:
        if not self.args.hands_free:
            return
        with self.state.lock:
            phase = self.state.phase
            busy = self.state.busy
            recording = self.state.recording
        if phase == "ERROR" or busy or recording or not self.initial_done:
            return
        if time.monotonic() < self._auto_at:
            return
        self._auto_at = 0.0
        self._start_recording()

    # ----- actions ---------------------------------------------------

    def _start_recording(self) -> None:
        with self.state.lock:
            ready = self.state.ready
            busy = self.state.busy
            recording = self.state.recording
            phase = self.state.phase
        if busy or recording:
            self.state.notice_msg("Finish the current turn before recording", ttl=3)
            return
        if phase == "ERROR":
            self.state.notice_msg("Engine failed to load - see log", ttl=0)
            return
        ev = threading.Event()
        self.rec_stop_event = ev
        with self.state.lock:
            self.state.recording = True
            self.state.rec_elapsed = 0.0
            self.state.notice = ""
            self.state.notice_until = 0.0
            self.state.status = (
                "WARMING - speak now" if not ready else "SPEAK NOW"
            )

        def work() -> None:
            try:
                wav, dur, rec_s = record_wav(
                    self.workdir,
                    seconds=self.args.seconds,
                    max_seconds=self.args.max_seconds,
                    hands_free=self.args.hands_free,
                    console=self.console,
                    stop_event=ev,
                    on_tick=self.state.record_tick,
                )
                self.events.put(("record_done", (wav, dur, rec_s)))
            except Exception as exc:  # noqa: BLE001
                self.events.put(("record_error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _handle_record_done(self, payload) -> None:
        wav, _dur, _rec_s = payload
        with self.state.lock:
            self.state.recording = False
            self.state.rec_elapsed = 0.0
            self.state.status = ""
        self.rec_stop_event = None
        if wav is None:
            self.state.notice_msg(
                "Silence detected - nothing transcribed. Try again.", ttl=6
            )
            self.console.log("[mic] silence - turn skipped")
            if self.args.once:
                self._quit()
            elif self.args.hands_free:
                self._auto_at = time.monotonic() + 1.0
            return
        self._start_turn(wav=wav)

    def _wait_for_engine(self) -> "EngineSession":
        while True:
            engine = self.holder.get()
            if engine is not None:
                if engine.start_error:
                    raise VoiceError(engine.start_error)
                return engine
            with self.state.lock:
                if self.state.phase == "ERROR":
                    raise VoiceError("engine failed to start")
            if self.stop_flag.is_set():
                raise VoiceError("stopped before the engine became ready")
            time.sleep(0.05)

    def _start_turn(
        self, *, text: str | None = None, wav: Path | None = None
    ) -> None:
        with self.state.lock:
            phase = self.state.phase
            busy = self.state.busy
            recording = self.state.recording
        if busy or recording:
            self.state.notice_msg("Finish the current turn before recording", ttl=3)
            return
        if phase == "ERROR":
            self.state.notice_msg("Engine failed to load - see log", ttl=0)
            return
        abort = threading.Event()
        self.abort_event = abort
        self.turn_no += 1
        turn_no = self.turn_no
        with self.state.lock:
            self.state.turn_no = turn_no
            self.state.busy = True
            self.state.abort_event = abort
            self.state.scroll = 0
            self.state.status = ""
        self._auto_at = 0.0

        def work() -> None:
            try:
                engine = self._wait_for_engine()
                stats = one_turn(
                    self.args,
                    engine,
                    self.console,
                    self.model_name,
                    self.workdir,
                    turn_no,
                    warm_event=self.warm_event,
                    piper=self.piper,
                    wav=wav,
                    text=text,
                    abort_event=abort,
                )
                try:
                    eng_stats = engine.query_stats()
                    if eng_stats:
                        self.state.set_engine_stats(eng_stats)
                except Exception as exc:  # noqa: BLE001
                    self.console.log(f"[stats] engine /stats failed: {exc}")
                self.events.put(("turn_done", stats))
            except Exception as exc:  # noqa: BLE001
                self.console.log(f"ERROR: {exc}")
                self.events.put(("turn_error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _stop(self) -> None:
        if self.rec_stop_event is not None:
            self.rec_stop_event.set()
            self.state.notice_msg("Stopping recording...", ttl=2)
            return
        if self.abort_event is not None:
            self.abort_event.set()
            self.state.notice_msg("Stopped audio; engine finishes the text", ttl=4)
            return
        self.state.notice_msg("Nothing to stop", ttl=2)

    def _toggle_fast(self) -> None:
        with self.state.lock:
            if self.state.restarting:
                return
            if self.state.recording or self.state.busy:
                self.state.notice_msg("Finish the current turn before switching mode", ttl=4)
                return
            if not self.state.ready:
                self.state.notice_msg("Engine is still loading", ttl=3)
                return
            new_fast = not self.state.fast
            self.state.fast = new_fast
            self.state.restarting = True
            self.state.ready = False
            self.state.phase = "LOADING"
            self.state.status = ""
            self.fast = new_fast
        mode = "FAST" if new_fast else "LOSSLESS"
        self.console.log(f"[engine] switching to {mode}; restarting...")
        self.restart_thread = threading.Thread(
            target=self._restart_engine, args=(new_fast,), daemon=True
        )
        self.restart_thread.start()

    def _restart_engine(self, fast: bool) -> None:
        try:
            old = self.holder.swap(None)
            if old is not None:
                old.stop()
            if self.stop_flag.is_set():
                return
            apply_mode_env(fast, getattr(self.args, "lang", "en"))
            engine = EngineSession(self.args, self.console)
            self.holder.set(engine)
            engine.start()
            if self.stop_flag.is_set():
                engine.stop()
                return
            self.events.put(("restart_ok", None))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("restart_error", str(exc)))

    def _clear(self) -> None:
        with self.state.lock:
            busy = self.state.busy
            recording = self.state.recording
            ready = self.state.ready
            phase = self.state.phase
        if busy or recording:
            self.state.notice_msg(
                "Cannot clear context during a turn or recording", ttl=4
            )
            return
        engine = self.holder.get()
        ok = True
        if ready and phase != "ERROR" and engine is not None:
            ok = engine.clear_context()
        self.state.clear()
        if ok:
            self.turn_no = 0
            self.state.turn_no = 0
            self.state.notice_msg("Transcript + engine context cleared", ttl=3)
        else:
            self.console.log("ERROR: engine /clear failed")
            self.state.notice_msg(
                "Transcript cleared; engine /clear failed - see log", ttl=5
            )

    def _quit(self) -> None:
        self.running = False
        self.stop_flag.set()
        if self.rec_stop_event is not None:
            self.rec_stop_event.set()
        if self.abort_event is not None:
            self.abort_event.set()

    # ----- input -----------------------------------------------------

    def _handle_key(self, key) -> None:
        if self.help:
            if key in ("?", "\x1b"):
                self.help = False
            elif key in ("q", "Q", "\x03"):
                self._quit()
            return
        if isinstance(key, int):
            if key == curses.KEY_MOUSE:
                self._handle_mouse()
            elif key in (curses.KEY_PPAGE, curses.KEY_UP):
                self._scroll(self._page)
            elif key in (curses.KEY_NPAGE, curses.KEY_DOWN):
                self._scroll(-self._page)
            elif key == curses.KEY_HOME:
                self._scroll(10 ** 6)
            elif key == curses.KEY_END:
                self._scroll(-10 ** 6)
            return
        if key in ("\x03",):
            self._quit()
        elif key in ("q", "Q"):
            self._quit()
        elif key in ("\n", "\r", " "):
            self._start_recording()
        elif key in ("s", "S"):
            self._stop()
        elif key in ("f", "F"):
            self._toggle_fast()
        elif key in ("c", "C"):
            self._clear()
        elif key == "?":
            self.help = True

    def _handle_mouse(self) -> None:
        try:
            _, mx, my, _, bstate = curses.getmouse()
        except curses.error:
            return
        if self.help:
            self.help = False
            return
        mask = 0
        for name in (
            "BUTTON1_CLICKED", "BUTTON1_PRESSED", "BUTTON1_RELEASED",
            "BUTTON1_DOUBLE_CLICKED",
        ):
            mask |= getattr(curses, name, 0)
        if not (bstate & mask):
            return
        now = time.monotonic()
        lx, ly, lt = self._last_click
        if now - lt < 0.30 and (mx, my) == (lx, ly):
            return
        self._last_click = (mx, my, now)
        for by, x0, x1, action in self._buttons:
            if my == by and x0 <= mx <= x1:
                if action == "record":
                    self._start_recording()
                elif action == "stop":
                    self._stop()
                elif action == "fast":
                    self._toggle_fast()
                elif action == "clear":
                    self._clear()
                elif action == "quit":
                    self._quit()
                return

    def _scroll(self, delta: int) -> None:
        with self.state.lock:
            self.state.scroll = max(0, self.state.scroll + delta)

    # ----- drawing ---------------------------------------------------

    def _attr(self, pair: int = 0, bold: bool = False) -> int:
        attr = curses.A_BOLD if bold else 0
        if self.color and pair:
            attr |= curses.color_pair(pair)
        return attr

    def _put(self, scr, y: int, x: int, text: str, attr: int = 0) -> None:
        try:
            h, w = scr.getmaxyx()
        except curses.error:
            return
        if y < 0 or y >= h or x < 0 or x >= w or not text:
            return
        maxlen = w - x
        if y == h - 1:
            maxlen -= 1
        if maxlen <= 0:
            return
        try:
            scr.addstr(y, x, text[:maxlen], attr)
        except curses.error:
            pass

    def _line(self, scr, y: int, text: str, attr: int = 0) -> None:
        try:
            _, w = scr.getmaxyx()
        except curses.error:
            return
        self._put(scr, y, 0, text[:w].ljust(w), attr)

    def _separator(self, scr, y: int, label: str, right: str = "") -> None:
        try:
            _, w = scr.getmaxyx()
        except curses.error:
            return
        self._line(scr, y, "-" * w, curses.A_DIM)
        self._put(scr, y, 0, f"[{label}]", self._attr(bold=True))
        if right:
            self._put(scr, y, max(0, w - len(right) - 1), right, curses.A_DIM)

    def _update_sys(self) -> None:
        now = time.monotonic()
        if now - self._sys_at < 0.7:
            return
        self._sys_at = now
        engine = self.holder.get()
        pid = engine.proc.pid if engine is not None and engine.proc is not None else None
        self._sys = {
            "thermal": thermal_max_c(),
            "self_mb": read_rss_mb(),
            "engine_mb": proc_tree_rss_mb(pid),
        }

    def _engine_state(self, snap: dict) -> str:
        if snap["phase"] == "ERROR":
            return "ERROR"
        if not snap["ready"]:
            return "WARMING"
        if snap["busy"]:
            return "BUSY"
        return "READY"

    def _metric_lines(self, snap: dict) -> list[str]:
        m = snap["last_stats"] or {}
        e = snap["engine_stats"] or {}
        rec = snap["recording"]
        mic = f"REC {snap['rec_elapsed']:.1f}s" if rec else "idle"
        if m:
            asr_s = float(m.get("asr_s") or 0.0)
            audio_s = float(m.get("audio_s") or 0.0)
            model = m.get("asr_model") or ""
            if asr_s:
                rate = audio_s / asr_s if asr_s else 0.0
                asr = (
                    f"ASR {model} {asr_s * 1000:.0f}ms {rate:.1f}x | "
                    f"audio {audio_s:.2f}s"
                )
            else:
                asr = f"ASR text | audio {audio_s:.2f}s"
            first = m.get("first_token_turn_s")
            if first is None:
                first = m.get("first_token_s")
            first_s = fmt_sec(first) if first else "n/a"
            pf_t = int(m.get("prefill_tokens") or 0)
            pf_s = float(m.get("prefill_s") or 0.0)
            gen_t = int(m.get("gen_tokens") or 0)
            tok_s = float(m.get("tok_s") or 0.0)
            tts_chunks = int(m.get("tts_chunks") or 0)
            tts_s = float(m.get("tts_s") or 0.0)
            audio_out = float(m.get("audio_out_s") or 0.0)
            play_s = float(m.get("play_s") or 0.0)
            total = float(m.get("total_s") or 0.0)
            engine_s = float(m.get("engine_s") or 0.0)
            chars = int(m.get("chars") or 0)
            gen = (
                f"gen {gen_t}t {tok_s:.1f}t/s" if gen_t else "gen n/a"
            )
            line1 = f"MIC {mic} | {asr}"
            line2 = (
                f"GEN pf {pf_t}t/{fmt_sec(pf_s)} | 1st {first_s} | {gen}"
            )
            line3 = (
                f"TTS {tts_chunks}ch/{fmt_sec(tts_s)} | audio {audio_out:.1f}s"
                f" | play {play_s:.1f}s | chars {chars}"
            )
            total_part = (
                f"TOTAL {fmt_sec(total)} | engine {fmt_sec(engine_s)}"
            )
        else:
            line1 = f"MIC {mic} | ASR n/a | audio n/a"
            line2 = "GEN pf n/a | 1st n/a | gen n/a"
            line3 = "TTS n/a | audio n/a | play n/a"
            total_part = "TOTAL n/a | engine n/a"
        rss_self = self._sys.get("self_mb") or 0.0
        rss_engine = (
            float(e.get("rss_mb") or 0.0)
            or float(self._sys.get("engine_mb") or 0.0)
        )
        thermal = self._sys.get("thermal")
        thermal_s = f"{thermal:.1f}C" if thermal is not None else "n/a"
        line4 = (
            f"{total_part} | rss {fmt_mb(rss_self)}/{fmt_mb(rss_engine)}"
            f" | therm {thermal_s} | turn {snap['turn_no']}"
        )
        hit = e.get("cache_hit")
        miss = e.get("cache_miss")
        if hit is not None and miss is not None:
            try:
                rate = float(e.get("cache_rate") or 0.0)
                cache = f"{rate:.1f}% hit{int(hit)} miss{int(miss)}"
            except (TypeError, ValueError):
                cache = "n/a"
        else:
            cache = "n/a"
        ctx = e.get("ctx_pos")
        ctx_s = str(ctx) if ctx is not None else "n/a"
        gen_total = e.get("total_tokens")
        gen_s = str(gen_total) if gen_total is not None else "-"
        line5 = f"CACHE {cache} | ctx {ctx_s} | session gen {gen_s}"
        line_lang = (
            f"LANG {getattr(self.args, 'lang', 'en')} | TTS {tts_voice_name()}"
        )
        line_cfg = (
            f"CFG max_new {self.args.max_new} | {cfg_label(self.args)}"
            f" | stop {stop_label(self.args)}"
        )
        return [
            line1, line2, line3, line4, line5, line_lang, line_cfg
        ]

    def _draw(self, scr) -> None:
        snap = self.state.snapshot()
        self._update_sys()
        try:
            h, w = scr.getmaxyx()
        except curses.error:
            return
        if h < 8 or w < 24:
            try:
                scr.erase()
                self._put(
                    scr, 0, 0,
                    f"terminal too small: {w}x{h} (need at least 24x8)",
                    curses.A_BOLD,
                )
                scr.refresh()
            except curses.error:
                pass
            return
        try:
            scr.erase()
        except curses.error:
            return
        header_h = 3 if h >= 18 else 2
        button_rows = self._build_button_rows(w)
        b_h = len(button_rows)
        if h >= 24:
            desired_d = 10
        elif h >= 20:
            desired_d = 7
        elif h >= 16:
            desired_d = 5
        else:
            desired_d = 2
        d_h = min(desired_d, max(1, h - header_h - b_h - 4 - 2))
        t_h = h - header_h - b_h - d_h - 4
        while t_h < 1 and d_h > 1:
            d_h -= 1
            t_h += 1
        t_h = max(1, t_h)

        engine = self.holder.get()
        estate = self._engine_state(snap)
        mode = "FAST" if snap["fast"] else "LOSSLESS"
        load_s = snap["load_s"]
        if not snap["ready"]:
            base = (
                engine.started_at
                if engine is not None and engine.started_at
                else self.started_at
            )
            load_s = time.monotonic() - base
        workers = "6" if snap["fast"] else "4"
        batch = os.environ.get("VOICE_PREFILL_BATCH", "16")
        thermal = self._sys.get("thermal")
        thermal_s = f"{thermal:.1f}C" if thermal is not None else "n/a"
        e = snap["engine_stats"] or {}
        rss_self = self._sys.get("self_mb") or 0.0
        rss_engine = (
            float(e.get("rss_mb") or 0.0)
            or float(self._sys.get("engine_mb") or 0.0)
        )
        ctx = e.get("ctx_pos")
        ctx_s = str(ctx) if ctx is not None else "n/a"
        mic = (
            f"REC {snap['rec_elapsed']:.1f}s"
            if snap["recording"] else "idle"
        )

        line0 = (
            f"TOKSHIT VOICE | {estate} | {mode} | turn {snap['turn_no']}"
            f" | max_new {self.args.max_new}"
        )
        self._line(scr, 0, line0, curses.A_REVERSE | curses.A_BOLD)
        line1 = (
            f"lang {getattr(self.args, 'lang', 'en')} | "
            f"load {fmt_sec(load_s)} | workers {workers} | batch {batch}"
            f" | therm {thermal_s}"
        )
        self._line(scr, 1, line1, self._attr(bold=True))
        if header_h >= 3:
            line2 = (
                f"rss {fmt_mb(rss_self)}/{fmt_mb(rss_engine)} | ctx {ctx_s}"
                f" | mic {mic} | turn {snap['turn_no']}"
            )
            self._line(scr, 2, line2, self._attr(bold=True))

        scroll = snap["scroll"]
        right = f"scroll {scroll} " if scroll else ""
        self._separator(scr, header_h, "TRANSCRIPT", right)
        top = header_h + 1
        self._draw_transcript(scr, top, t_h, w, snap)

        det_head = top + t_h
        self._separator(scr, det_head, "DETAILS")
        det_top = det_head + 1
        self._draw_details(scr, det_top, d_h, w, snap)

        status_y = det_top + d_h
        self._draw_status(scr, status_y, w, snap)

        btn_y = status_y + 1
        self._draw_buttons(scr, btn_y, w, snap, button_rows)

        hint_y = btn_y + b_h
        hints = (
            "Enter/Space record | s stop | f fast | c clear | q quit"
            " | ? help | PgUp/PgDn scroll"
        )
        self._line(scr, hint_y, hints, curses.A_DIM)
        if self.help:
            self._draw_help(scr)
        try:
            scr.refresh()
        except curses.error:
            pass

    def _draw_transcript(self, scr, top: int, height: int, w: int,
                         snap: dict) -> None:
        try:
            for yy in range(top, top + height):
                self._put(scr, yy, 0, " " * w)
        except curses.error:
            pass
        lines: list[tuple[str, str, str]] = []
        for role, text, _done in snap["messages"][-80:]:
            label = "You: " if role == "user" else "Tokshit: "
            body = sanitize_ui_text(str(text)) if text else ""
            first = True
            for raw in (body.split("\n") if body else [""]):
                chunks = wrap_ui_line(raw, max(1, w - len(label)))
                for chunk in chunks:
                    if first:
                        lines.append((role, label, chunk))
                        first = False
                    else:
                        lines.append((role, " " * len(label), chunk))
            lines.append(("", "", ""))
        if not lines:
            self._put(
                scr, top, 1,
                "Press Enter/Space or tap RECORD to speak.",
                self._attr(bold=True),
            )
            return
        max_scroll = max(0, len(lines) - height)
        scroll = min(snap["scroll"], max_scroll)
        if scroll != snap["scroll"]:
            self.state.set_scroll(scroll)
        end = len(lines) - scroll
        start = max(0, end - height)
        visible = lines[start:end]
        self._page = max(3, height - 2)
        for i, (role, label, content) in enumerate(visible[:height]):
            y = top + i
            if role == "user":
                attr = self._attr(1, bold=True)
            elif role == "assistant":
                attr = self._attr(2, bold=True)
            else:
                attr = 0
            self._put(scr, y, 0, label, attr)
            self._put(scr, y, len(label), content, 0)

    def _draw_details(self, scr, top: int, height: int, w: int,
                      snap: dict) -> None:
        metrics = self._metric_lines(snap)
        logs = [
            sanitize_ui_text(x) for x in snap["logs"][-3:]
        ]
        if height >= 6:
            n_logs = min(3, len(logs))
            budget = max(0, height - n_logs)
            if budget >= len(metrics):
                out = metrics[:]
            else:
                order = [0, 1, 6, 5, 2, 3, 4]
                out = [metrics[i] for i in order[:budget]]
            out.extend(f"LOG {line}" for line in logs[-n_logs:])
        else:
            order = [1, 6, 5, 0, 2, 3, 4]
            out = [metrics[i] for i in order[:height]]
        for i in range(height):
            y = top + i
            text = out[i] if i < len(out) else ""
            self._put(scr, y, 0, text[: max(0, w - 1)])

    def _draw_status(self, scr, y: int, w: int, snap: dict) -> None:
        if snap["notice"]:
            text = snap["notice"]
            attr = self._attr(3, bold=True)
        elif snap["recording"]:
            lead = "WARMING - speak now" if not snap["ready"] else "SPEAK NOW"
            text = (
                f"{lead}  {snap['rec_elapsed']:.1f}s  |  "
                f"auto-stop {self.args.max_seconds:.0f}s  |  STOP / s"
            )
            attr = curses.A_REVERSE | self._attr(3, bold=True)
        elif snap["busy"]:
            if not snap["ready"]:
                text = "WARMING - turn queued; waiting for engine..."
            else:
                text = "BUSY: streaming answer and speech..."
            attr = self._attr(3, bold=True)
        elif not snap["ready"]:
            extra = snap["status"] or "engine starting"
            text = f"WARMING - speak now | {extra}"
            attr = self._attr(3, bold=True)
        elif snap["restarting"]:
            text = "Restarting engine..."
            attr = self._attr(3, bold=True)
        elif snap["status"]:
            text = snap["status"]
            attr = self._attr(bold=True)
        else:
            text = "READY - press Enter/Space or tap RECORD"
            attr = self._attr(2, bold=True)
        self._put(scr, y, 0, text[: max(0, w - 1)], attr)

    def _build_button_rows(self, w: int) -> list[list[tuple[str, str, int]]]:
        rows: list[list[tuple[str, str, int]]] = []
        row: list[tuple[str, str, int]] = []
        used = 0
        for action, label in self.BUTTONS:
            need = len(label) + (1 if row else 0)
            if row and used + need > w:
                rows.append(row)
                row = []
                used = 0
                need = len(label)
            x = used + (1 if row else 0)
            row.append((action, label, x))
            used = x + len(label)
        if row:
            rows.append(row)
        return rows

    def _draw_buttons(self, scr, top: int, w: int, snap: dict,
                      rows: list[list[tuple[str, str, int]]]) -> None:
        self._buttons = []
        for ri, row in enumerate(rows):
            y = top + ri
            self._put(scr, y, 0, " " * w)
            for action, label, x in row:
                enabled = True
                if action == "record":
                    enabled = (
                        snap["phase"] != "ERROR" and not snap["busy"]
                        and not snap["recording"]
                    )
                elif action == "stop":
                    enabled = snap["recording"] or snap["busy"]
                elif action == "fast":
                    enabled = (
                        snap["ready"] and not snap["busy"]
                        and not snap["recording"] and not snap["restarting"]
                    )
                attr = curses.A_REVERSE
                if enabled:
                    attr |= curses.A_BOLD
                else:
                    attr |= curses.A_DIM
                self._put(scr, y, x, label, attr)
                self._buttons.append((y, x, x + len(label) - 1, action))

    def _draw_help(self, scr) -> None:
        try:
            h, w = scr.getmaxyx()
        except curses.error:
            return
        body = self.HELP_LINES
        box_w = min(max(len(x) for x in body) + 4, w - 2)
        box_h = min(len(body) + 2, h - 2)
        if box_w < 20 or box_h < 5:
            return
        y0 = max(0, (h - box_h) // 2)
        x0 = max(0, (w - box_w) // 2)
        for yy in range(y0, y0 + box_h):
            self._put(scr, yy, x0, " " * box_w)
        self._put(scr, y0, x0, "+" + "-" * (box_w - 2) + "+", curses.A_BOLD)
        self._put(
            scr, y0 + box_h - 1, x0,
            "+" + "-" * (box_w - 2) + "+", curses.A_BOLD,
        )
        for i, line in enumerate(body[: box_h - 2]):
            self._put(scr, y0 + 1 + i, x0, "|", curses.A_BOLD)
            self._put(scr, y0 + 1 + i, x0 + 2, line[: box_w - 4])
            self._put(scr, y0 + 1 + i, x0 + box_w - 1, "|", curses.A_BOLD)
        self._put(scr, y0, x0 + 2, " HELP ", curses.A_REVERSE | curses.A_BOLD)


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
    ap.add_argument("--models", action="store_true",
                    help="list available .gguf models and exit")
    ap.add_argument("--model",
                    help="model name/path: inkling, flint, dspark, or a .gguf path")
    ap.add_argument("--experts", type=int,
                    help="experts per token (TOKSHIT_DRAFT_TOPK); default 8 / fast 4-6")
    ap.add_argument("--passes", type=int,
                    help="Inkling2 core passes 1..3; default 3 (2 is LOSSY)")
    ap.add_argument("--workers", type=int,
                    help="compute pool workers; default 4")
    ap.add_argument("--cache-mb", type=int,
                    help="expert cache MB; default 512")
    ap.add_argument("--kv-ctx", type=int,
                    help="KV positions; default engine value (2048)")
    ap.add_argument("--layers", type=int,
                    help="limit model layers (debug/tuning)")
    ap.add_argument("--hot", type=int, choices=(0, 1),
                    help="1 = copy dense weights to RAM, 0 = mmap; default 0")
    ap.add_argument("--q4", action="store_true",
                    help="LOSSY: Q4_0 dense weights")
    ap.add_argument("--prefill-batch", type=int,
                    help="prefill batch size; default 16")
    ap.add_argument("--settings", action="store_true",
                    help="print model/runtime settings and exit")
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
    tui_group = ap.add_mutually_exclusive_group()
    tui_group.add_argument("--tui", action="store_true",
                           help="force the curses TUI (requires a TTY)")
    tui_group.add_argument("--no-tui", action="store_true",
                           help="force line mode")
    ap.add_argument("--fast", action="store_true",
                    help="FAST mode: top-4 draft experts, 6 workers (lossy)")
    ap.add_argument("--one-sentence", action="store_true",
                    help="stop the answer after one sentence (TOKSHIT_STOP_SENTENCE=1)")
    ap.add_argument("--ru", action="store_true",
                    help="Russian mode: ASR_LANG=ru, ru TTS voice")
    ap.add_argument("--lang", choices=("en", "ru"), default=None,
                    help="language: en or ru (default: VOICE_LANG or en)")
    args = ap.parse_args()

    if args.models:
        subprocess.run([str(engine_path(args)), "--models"], check=False)
        return 0
    if args.model:
        os.environ["VOICE_MODEL"] = str(resolve_model_name(args.model))
    if args.settings:
        def show(name: str, value: object) -> None:
            print(f"{name:16s} {value}")
        show("model", engine_model())
        show("fast", "yes" if (args.fast or os.environ.get("VOICE_FAST", "0") not in ("", "0")) else "no")
        show("lang", args.lang or os.environ.get("VOICE_LANG") or "en")
        show("experts", args.experts if args.experts is not None else "auto (8 / fast 4 EN, 6 RU)")
        show("passes", args.passes if args.passes is not None else 3)
        show("workers", args.workers if args.workers is not None else 4)
        show("cache_mb", args.cache_mb if args.cache_mb is not None else 512)
        show("kv_ctx", args.kv_ctx if args.kv_ctx is not None else "engine default (2048)")
        show("layers", args.layers if args.layers is not None else "all")
        show("hot", args.hot if args.hot is not None else 0)
        show("q4", "yes (LOSSY)" if args.q4 else "no")
        show("prefill_batch", args.prefill_batch if args.prefill_batch is not None else 16)
        show("max_new", args.max_new)
        show("temperature", os.environ.get("VOICE_TEMPERATURE", "0"))
        return 0

    if args.clean_text is not None:
        print(clean_for_speech(args.clean_text))
        return 0
    if any(not t.strip() for t in args.text):
        print("ERROR: --text is empty", file=sys.stderr)
        return 2
    if args.max_new < 1:
        print("ERROR: --max-new must be >= 1", file=sys.stderr)
        return 2
    if args.hands_free and args.seconds <= 0:
        print("ERROR: --hands-free requires --seconds N", file=sys.stderr)
        return 2

    args.fast = bool(args.fast or env_truthy("VOICE_FAST"))
    args.one_sentence = bool(
        args.one_sentence or env_truthy("VOICE_ONE_SENTENCE")
    )
    apply_language(args)

    deps = check_assets(args)
    if args.check:
        if args.lang_warning:
            print(f"  WARN {args.lang_warning}")
        print_check(deps)
        return 2 if deps["problems"] else 0
    if deps["problems"]:
        for p in deps["problems"]:
            print(f"ERROR: {p}", file=sys.stderr)
        print("hint: scripts/voice_chat.py --check", file=sys.stderr)
        return 2

    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    use_tui = is_tty and not args.no_tui
    if use_tui and not args.tui and (args.text or args.wav is not None):
        use_tui = False
    if args.tui and not is_tty:
        print("TUI needs stdin+stdout on a TTY; using line mode", file=sys.stderr)
        use_tui = False

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

    if use_tui:
        if args.wav is not None and not Path(args.wav).is_file():
            print(f"ERROR: no such wav: {args.wav}", file=sys.stderr)
            cleanup()
            return 2
        state = UiState()
        tui_console = TuiConsole(state, verbose=args.verbose)
        app = TuiApp(args, model_name, workdir, tui_console, state)
        try:
            app.run()
        except KeyboardInterrupt:
            return 130
        except VoiceError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        finally:
            cleanup()
        return 0

    console = Console(verbose=args.verbose)
    if getattr(args, "lang_warning", ""):
        console.log(args.lang_warning)
    engine: EngineSession | None = None
    turns: list[TurnStats] = []
    try:
        quality_guard(args, console)
        apply_mode_env(args.fast, getattr(args, "lang", "en"))
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
            console.log(f"[wav] {wav.name} | {wav_duration(wav):.2f}s")
            turns.append(one_turn(
                args, engine, console, model_name, workdir, 1,
                warm_event=warm_event, piper=piper, wav=wav
            ))
        else:
            console.log("voice chat tokshit | Enter to record | q quits")
            empty_streak = 0
            turn_no = 0
            while True:
                if args.hands_free:
                    if args.seconds <= 0:
                        raise VoiceError("--hands-free requires --seconds N")
                    time.sleep(0.4)
                elif sys.stdin.isatty():
                    try:
                        line = input("\n[mic] Enter to record, q to quit: ").strip()
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
        console.log(f"ERROR: {exc}")
        return 1
    finally:
        if engine is not None:
            engine.stop()
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
