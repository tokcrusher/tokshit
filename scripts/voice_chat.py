#!/data/data/com.termux/files/usr/bin/env python
"""voice_chat.py — голосовой чат с движком tokshit.

mic -> whisper.cpp ASR -> tokshit (Inkling-2-Mini) -> Piper TTS -> speaker

Режимы:
  scripts/voice_chat.py                 # интерактивно: Enter начать/остановить запись
  scripts/voice_chat.py --seconds 5     # запись ровно 5 с, без второго Enter
  scripts/voice_chat.py --hands-free --seconds 5   # цикл без клавиатуры
  scripts/voice_chat.py --wav in.wav    # без микрофона: ASR -> движок -> TTS -> play
  scripts/voice_chat.py --text "..."    # без ASR: движок -> TTS -> play
  scripts/voice_chat.py --check         # проверить зависимости и выйти

Аудио-часть — это ~/audio/{asr.sh,tts.sh}; мы их только вызываем, не меняем.
Движок — scripts/run_inkling2.sh (lossless). Быстрый LOSSY-профиль:
  VOICE_ENGINE=$HOME/tokshit/scripts/run_inkling2_fast.sh scripts/voice_chat.py

Почему raw prompt: чат-шаблон Inkling-2-Mini на Q2_K_XL часто вываливает
рассуждения в текст ("The user asks..."). Короткий English few-shot
("Q: ... / A:") даёт прямой ответ и не улетает в другой язык.
Отключается флагом --chat-template.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path

HOME = Path(os.environ.get("HOME", "/data/data/com.termux/files/home"))
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", HOME / "audio"))
TOKSHIT_ROOT = Path(os.environ.get("TOKSHIT_ROOT", HOME / "tokshit"))
ASR_SH = AUDIO_DIR / "asr.sh"
TTS_SH = AUDIO_DIR / "tts.sh"
MODELS_DIR = AUDIO_DIR / "models"
WHISPER_DIR = AUDIO_DIR / "whisper.cpp"
DEFAULT_ENGINE = TOKSHIT_ROOT / "scripts" / "run_inkling2.sh"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")
ENGINE_RE = re.compile(
    r"\[prefill:\s*(\d+)\s+tok.*?([\d.]+)s\s*\|\s*генерация:\s*(\d+)\s+tok"
    r".*?([\d.]+)s\s*=\s*([\d.]+)\s+tok/s\]"
)
LOADED_RE = re.compile(r"loaded in\s+([\d.]+)s")
GUARD_RE = re.compile(
    r"peak RSS=(\d+)MB,\s*min MemAvailable=(\d+)MB"
)
LOCK_RE = re.compile(r"another model run holds the lock")


class VoiceError(RuntimeError):
    pass


# ────────────────────────────── text helpers ──────────────────────────────

def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def strip_echoed_prompt(text: str) -> str:
    """Few-shot модель иногда продолжает шаблон: ответ, потом снова Q:.
    Для речи режем на первой такой строке; печатаем всё как есть."""
    lines = text.splitlines()
    kept: list[str] = []
    for i, line in enumerate(lines):
        if i > 0 and re.match(r"^\s*(Q|Question|User|A)\s*:", line):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def clean_for_speech(text: str) -> str:
    """Markdown/code -> текст, который не стыдно отдать Piper."""
    s = text
    # незакрытый ```-блок: всё после него — код
    parts = s.split("```")
    out = []
    for i, part in enumerate(parts):
        out.append(part if i % 2 == 0 else " Code block omitted. ")
    s = "".join(out)
    # inline code: оставляем содержимое, но без обратных кавычек
    s = re.sub(r"`([^`]*)`", r"\1", s)
    # html/комментарии/спецтокены
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r"<\|[^|>]*\|>", " ", s)
    s = re.sub(r"</?[A-Za-z][^>]*>", " ", s)
    # ссылки и картинки
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"https?://\S+", "link", s)
    s = html.unescape(s)
    # таблицы: разделители выкидываем, пайпы -> запятая
    s = re.sub(r"(?m)^\s*\|?[\s:|-]+\|?\s*$", " ", s)
    s = s.replace("|", ", ")
    # заголовки, цитаты, маркеры списков
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
    s = re.sub(r"(?m)^\s{0,3}>\s?", "", s)
    s = re.sub(r"(?m)^\s*[-*+•]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+[.)]\s+", "", s)
    # акценты/подчёркивания/тильды
    s = re.sub(r"[*_~]{1,3}", "", s)
    # прочие одиночные звёздочки и решётки
    s = s.replace("*", " ").replace("#", " ")
    # пробелы и переносы
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", ". ", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    # схлопываем следы разметки: "..", ": .", ",,", ".,"
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
    """Режем длинный текст по предложениям, чтобы VITS не давился."""
    if len(text) <= max_len:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    cur = ""
    for sent in sentences:
        if not sent:
            continue
        if len(sent) > max_len:
            # очень длинное предложение — режем по запятым/пробелам
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
    if re.fullmatch(r"[\[\(].{0,40}[\]\)]", t):  # [BLANK_AUDIO], (silence)
        return True
    return not re.search(r"\w", t, flags=re.UNICODE)


# ────────────────────────────── process helpers ──────────────────────────────

def run_cmd(
    cmd: list[str],
    *,
    timeout: float | None,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    stdin_data: str | None = None,
) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return 124, out, err + f"\n[timeout after {timeout:.0f}s]"
    except FileNotFoundError:
        raise VoiceError(f"command not found: {cmd[0]}")


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, OSError):
        return 0.0


def wave_params(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth()


def concat_wavs(parts: list[Path], out: Path) -> Path:
    if len(parts) == 1:
        shutil.copyfile(parts[0], out)
        return out
    params = wave_params(parts[0])
    frames = []
    for p in parts:
        if wave_params(p) != params:
            raise VoiceError(
                f"TTS chunks have different formats: {p} {wave_params(p)} != {params}"
            )
        with wave.open(str(p), "rb") as w:
            frames.append(w.readframes(w.getnframes()))
    rate, channels, width = params
    with wave.open(str(out), "wb") as w:
        w.setframerate(rate)
        w.setnchannels(channels)
        w.setsampwidth(width)
        for f in frames:
            w.writeframes(f)
    return out


# ────────────────────────────── assets ──────────────────────────────

KNOWN_ASR_SIZES = {
    "tiny": 77691713,
    "base": 147951465,
    "small": 487601967,
}


def model_complete(p: Path, name: str) -> bool:
    """Файл докачан? Известный размер надёжнее .aria2-сайдкара."""
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
    for name in ("small", "base", "tiny"):
        p = complete_model(name)
        if p is not None:
            return name, p
    raise VoiceError(
        f"no complete whisper model in {MODELS_DIR} "
        "(need ggml-small.bin, ggml-base.bin or ggml-tiny.bin)"
    )


def check_assets(args: argparse.Namespace) -> dict:
    problems: list[str] = []
    notes: list[str] = []

    if not ASR_SH.is_file():
        problems.append(f"missing {ASR_SH}")
    if not TTS_SH.is_file():
        problems.append(f"missing {TTS_SH}")
    if not (WHISPER_DIR / "build" / "bin" / "whisper-cli").is_file():
        problems.append(f"missing {WHISPER_DIR}/build/bin/whisper-cli")
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

    engine = Path(args.engine_script)
    if not engine.is_file():
        problems.append(f"missing engine launcher: {engine}")
    if not (TOKSHIT_ROOT / "build" / "tokshit").is_file():
        problems.append(f"missing binary: {TOKSHIT_ROOT}/build/tokshit")
    elif engine.is_file():
        notes.append(f"engine: {engine.name}")
    model = HOME / "Inkling-2-Mini-GGUF" / "Inkling-2-Mini-UD-Q2_K_XL.gguf"
    if not model.is_file():
        problems.append(f"missing engine model: {model}")
    else:
        notes.append(f"engine model: {model.name} ({model.stat().st_size / 2**30:.1f} GiB)")

    if args.wav is None and args.text is None:
        for tool in ("termux-microphone-record", "ffmpeg"):
            if shutil.which(tool) is None:
                problems.append(f"missing {tool}")
    if not args.no_play:
        if shutil.which("ffplay") is None and shutil.which("termux-media-player") is None:
            problems.append("no player: install ffmpeg (ffplay) or termux-api")
        if shutil.which("ffplay"):
            notes.append("player: ffplay")
        elif shutil.which("termux-media-player"):
            notes.append("player: termux-media-player")

    return {"problems": problems, "notes": notes}


def print_check(result: dict) -> None:
    for n in result["notes"]:
        print(f"  ok   {n}")
    for p in result["problems"]:
        print(f"  FAIL {p}")
    if not result["problems"]:
        print("all dependencies look good")


# ────────────────────────────── audio ──────────────────────────────

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
) -> Path | None:
    """Записать с микрофона и вернуть 16 kHz mono WAV (или None)."""
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
    print(f"🎤 запись {limit or max_seconds:.0f} с" if seconds > 0 else "🎤 запись…")
    rc, out, err = run_cmd(cmd, timeout=10)
    if rc != 0 and rc != 124:
        raise VoiceError(
            "termux-microphone-record не запустился. Проверь Termux:API и "
            f"разрешение RECORD_AUDIO.\n{err.strip()}"
        )
    if rc == 124:
        raise VoiceError(
            "termux-microphone-record завис (обычно нет Termux:API или "
            "висит диалог разрешения микрофона)."
        )
    global _RECORDING
    _RECORDING = True

    if seconds > 0:
        remaining = seconds
        try:
            if not hands_free:
                print(f"   … пиши/говори, стоп через {seconds:.0f} с (или Enter)")
            while remaining > 0:
                time.sleep(0.1)
                remaining -= 0.1
                if not hands_free and sys.stdin.isatty():
                    # не блокируемся: Enter просто обрывает запись раньше
                    import select
                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.readline()
                        break
        except KeyboardInterrupt:
            stop_recorder()
            raise
    else:
        try:
            if sys.stdin.isatty():
                input(
                    f"   … Enter — остановить (автостоп {max_seconds:.0f} с): "
                )
            else:
                time.sleep(max_seconds)
        except EOFError:
            pass
    stop_recorder()

    # Android MediaRecorder дописывает файл не мгновенно
    for _ in range(20):
        if rec.is_file() and rec.stat().st_size > 0:
            break
        time.sleep(0.1)
    if not rec.is_file() or rec.stat().st_size == 0:
        raise VoiceError(
            "микрофон не создал файл. Проверь разрешение RECORD_AUDIO у "
            "Termux:API (Настройки Android → Приложения → Termux:API)."
        )

    # 16 kHz mono + обрезка тишины по краям
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
        # без VAD-фильтра, лишь бы конвертировать
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
    print(f"   записано {dur:.1f} с")
    if dur < 0.35:
        return None
    return raw


def transcribe(wav: Path, model_name: str, *, verbose: bool) -> tuple[str, float, float]:
    """Вернуть (текст, секунды ASR, длительность аудио)."""
    dur = wav_duration(wav)
    t0 = time.monotonic()
    rc, out, err = run_cmd(
        [str(ASR_SH), str(wav), model_name],
        timeout=600,
    )
    dt = time.monotonic() - t0
    text = strip_ansi(out).strip()
    # asr.sh всегда выходит 1 из-за EXIT-трапа (`[ -n "" ]`), но stdout валиден
    if rc != 0 and not text:
        raise VoiceError(f"ASR failed (rc={rc}):\n{err.strip() or out.strip()}")
    if verbose and err.strip():
        print(f"   asr stderr: {err.strip()[:400]}")
    return text, dt, dur


def speak(text: str, workdir: Path, *, verbose: bool) -> tuple[Path | None, float]:
    """Синтезировать очищенный текст. Возвращает (wav, секунды)."""
    spoken = clean_for_speech(strip_echoed_prompt(text))
    if not spoken:
        return None, 0.0
    chunks = split_for_tts(spoken)
    t0 = time.monotonic()
    parts: list[Path] = []
    for i, chunk in enumerate(chunks):
        out = workdir / f"tts-{i}.wav"
        rc, so, se = run_cmd([str(TTS_SH), chunk, str(out)], timeout=300)
        if rc != 0 or not out.is_file():
            raise VoiceError(f"TTS failed (rc={rc}):\n{se.strip()[-600:]}")
        if verbose:
            print(f"   tts[{i}]: {so.strip()}")
        parts.append(out)
    dt = time.monotonic() - t0
    final = concat_wavs(parts, workdir / "answer.wav")
    return final, dt


def play_wav(path: Path, *, seconds: float, enabled: bool, verbose: bool) -> float:
    if not enabled:
        return 0.0
    player = os.environ.get("VOICE_PLAYER", "auto")
    t0 = time.monotonic()
    if player in ("auto", "ffplay") and shutil.which("ffplay"):
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
        rc, _, err = run_cmd(cmd, timeout=seconds + 30)
        if rc == 0:
            if verbose:
                print(f"   played via ffplay ({time.monotonic() - t0:.1f}s)")
            return time.monotonic() - t0
        if player == "ffplay":
            raise VoiceError(f"ffplay failed: {err.strip()[-400:]}")
        print(f"   ffplay не смог ({err.strip()[:120]}), пробую termux-media-player")
    if player in ("auto", "termux") and shutil.which("termux-media-player"):
        rc, _, err = run_cmd(
            ["termux-media-player", "play", str(path)], timeout=15
        )
        if rc != 0:
            raise VoiceError(f"termux-media-player failed: {err.strip()[-400:]}")
        time.sleep(seconds + 0.15)  # play асинхронный
        run_cmd(["termux-media-player", "stop"], timeout=10)
        return time.monotonic() - t0
    if player == "none":
        return 0.0
    raise VoiceError("нет плеера: нужен ffplay или termux-media-player")


# ────────────────────────────── engine ──────────────────────────────

def parse_engine_stats(stderr: str) -> dict:
    st: dict = {
        "prefill_tokens": 0,
        "prefill_s": 0.0,
        "gen_tokens": 0,
        "gen_s": 0.0,
        "tok_s": 0.0,
        "load_s": 0.0,
        "peak_rss_mb": 0,
        "min_avail_mb": 0,
        "lock_wait": False,
    }
    m = ENGINE_RE.search(stderr)
    if m:
        st["prefill_tokens"] = int(m.group(1))
        st["prefill_s"] = float(m.group(2))
        st["gen_tokens"] = int(m.group(3))
        st["gen_s"] = float(m.group(4))
        st["tok_s"] = float(m.group(5))
    m = LOADED_RE.search(stderr)
    if m:
        st["load_s"] = float(m.group(1))
    m = GUARD_RE.search(stderr)
    if m:
        st["peak_rss_mb"] = int(m.group(1))
        st["min_avail_mb"] = int(m.group(2))
    st["lock_wait"] = bool(LOCK_RE.search(stderr))
    return st


def ask_engine(
    text: str,
    engine: Path,
    *,
    max_new: int,
    raw_prompt: bool,
    verbose: bool,
) -> tuple[str, float, dict]:
    env = os.environ.copy()
    prompt = text
    if raw_prompt:
        tmpl = os.environ.get(
            "VOICE_PROMPT_TEMPLATE",
            "Answer in English with one short sentence.\n"
            "Q: What is the capital of Germany?\n"
            "A: Berlin.\n"
            "Q: {text}\n"
            "A:",
        )
        if "{text}" not in tmpl:
            raise VoiceError("VOICE_PROMPT_TEMPLATE must contain {text}")
        prompt = tmpl.format(text=text)
        env["TOKSHIT_RAW_PROMPT"] = "1"
        env.setdefault("TOKSHIT_REASONING_BUDGET", "0")
    if os.environ.get("VOICE_TEMPERATURE"):
        env["TOKSHIT_TEMPERATURE"] = os.environ["VOICE_TEMPERATURE"]

    t0 = time.monotonic()
    rc, out, err = run_cmd(
        [str(engine), prompt, str(max_new)],
        timeout=1800,
        env=env,
        cwd=TOKSHIT_ROOT,
    )
    wall = time.monotonic() - t0
    answer = strip_ansi(out).strip()
    stats = parse_engine_stats(err)

    if rc != 0:
        tail = "\n".join(err.strip().splitlines()[-12:])
        raise VoiceError(f"engine failed (rc={rc}):\n{tail}")
    if not answer:
        tail = "\n".join(err.strip().splitlines()[-8:])
        raise VoiceError(f"engine returned empty stdout:\n{tail}")
    if verbose:
        print("   --- engine stderr ---")
        for line in err.strip().splitlines()[-10:]:
            print(f"   {line}")
    return answer, wall, stats


# ────────────────────────────── turn loop ──────────────────────────────

@dataclass
class Turn:
    audio_s: float = 0.0
    asr_s: float = 0.0
    engine_wall_s: float = 0.0
    engine_stats: dict | None = None
    tts_s: float = 0.0
    play_s: float = 0.0
    total_s: float = 0.0


def fmt_turn(turn: Turn) -> None:
    st = turn.engine_stats or {}
    parts = []
    if turn.audio_s:
        rtf = turn.audio_s / turn.asr_s if turn.asr_s > 0 else 0.0
        parts.append(f"asr {turn.asr_s:.2f}s ({rtf:.1f}x realtime)")
    parts.append(f"engine {turn.engine_wall_s:.1f}s")
    if st.get("tok_s"):
        parts.append(
            f"{st['gen_tokens']} tok @ {st['tok_s']:.2f} tok/s"
            f" (prefill {st['prefill_s']:.1f}s, load {st['load_s']:.1f}s)"
        )
    if st.get("lock_wait"):
        parts.append("waited for model lock")
    if turn.tts_s:
        parts.append(f"tts {turn.tts_s:.2f}s")
    if turn.play_s:
        parts.append(f"play {turn.play_s:.2f}s")
    parts.append(f"total {turn.total_s:.1f}s")
    print("⏱  " + " · ".join(parts))


def one_turn(args: argparse.Namespace, workdir: Path, model_name: str,
             wav: Path | None, text: str | None) -> Turn:
    t_start = time.monotonic()
    turn = Turn()

    if text is None:
        assert wav is not None
        turn.audio_s = wav_duration(wav)
        transcript, asr_s, _ = transcribe(wav, model_name, verbose=args.verbose)
        turn.asr_s = asr_s
        if is_blank_transcript(transcript):
            print("🗣  (тишина/не распознано — пропускаю)")
            turn.total_s = time.monotonic() - t_start
            return turn
    else:
        transcript = text.strip()
    print(f"🗣  {transcript}")

    answer, wall, stats = ask_engine(
        transcript,
        Path(args.engine_script),
        max_new=args.max_new,
        raw_prompt=not args.chat_template,
        verbose=args.verbose,
    )
    turn.engine_wall_s = wall
    turn.engine_stats = stats
    print(f"🤖 {answer}")
    if args.no_play:
        print(f"(spoken) {clean_for_speech(strip_echoed_prompt(answer))}")

    wav_out, tts_s = speak(answer, workdir, verbose=args.verbose)
    turn.tts_s = tts_s
    if wav_out is not None:
        dur = wav_duration(wav_out)
        print(f"🔊 {dur:.1f} s речи")
        turn.play_s = play_wav(
            wav_out,
            seconds=dur,
            enabled=not args.no_play,
            verbose=args.verbose,
        )
    turn.total_s = time.monotonic() - t_start
    fmt_turn(turn)
    return turn


def session_summary(turns: list[Turn]) -> None:
    real = [t for t in turns if t.total_s > 0]
    if not real:
        return
    print("\n── сессия ──")
    print(f"ходов: {len(real)}")
    for label, key in (
        ("asr", "asr_s"),
        ("engine", "engine_wall_s"),
        ("tts", "tts_s"),
        ("play", "play_s"),
        ("total", "total_s"),
    ):
        vals = [getattr(t, key) for t in real if getattr(t, key)]
        if vals:
            print(f"{label:7s}: avg {sum(vals) / len(vals):6.1f} s  "
                  f"min {min(vals):5.1f}  max {max(vals):5.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Voice chat: mic -> whisper.cpp -> tokshit -> Piper TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--text", help="skip mic+ASR, feed this text to the engine")
    ap.add_argument("--wav", help="skip mic, transcribe this file")
    ap.add_argument("--seconds", type=float, default=0,
                    help="fixed recording length (0 = press Enter to stop)")
    ap.add_argument("--max-seconds", type=float, default=20,
                    help="hard cap for Enter-stopped recording (default 20)")
    ap.add_argument("--hands-free", action="store_true",
                    help="with --seconds: loop without keyboard")
    ap.add_argument("--max-new", type=int,
                    default=int(os.environ.get("VOICE_MAX_NEW", "32")),
                    help="max new tokens (default 32)")
    ap.add_argument("--asr-model", default=os.environ.get("VOICE_ASR_MODEL"),
                    help="whisper model name: small/base/tiny or a .bin path")
    ap.add_argument("--engine-script",
                    default=os.environ.get("VOICE_ENGINE", str(DEFAULT_ENGINE)),
                    help="engine launcher (default scripts/run_inkling2.sh)")
    ap.add_argument("--chat-template", action="store_true",
                    help="use the engine chat template instead of raw Q/A")
    ap.add_argument("--no-play", action="store_true", help="synthesize but don't play")
    ap.add_argument("--once", action="store_true", help="one turn, then exit")
    ap.add_argument("--keep", action="store_true", help="keep temp wav files")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="check dependencies and exit")
    ap.add_argument("--clean-text", metavar="TEXT",
                    help="print the TTS-cleaned version of TEXT and exit")
    args = ap.parse_args()

    if args.clean_text is not None:
        print(clean_for_speech(args.clean_text))
        return 0
    if args.text is not None and not args.text.strip():
        print("✗ --text is empty", file=sys.stderr)
        return 2
    if args.max_new < 1:
        print("✗ --max-new must be >= 1", file=sys.stderr)
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

    model_name, _ = pick_asr_model(args.asr_model)
    workdir = Path(tempfile.mkdtemp(prefix="voice-chat-"))
    turns: list[Turn] = []
    cleaned = False

    def cleanup(*_a):
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"files kept in {workdir}")
        stop_recorder()

    # Ctrl-C: гасим запись и выходим без трейсбека
    def on_signal(signum, _frame):
        cleanup()
        print("\nпока.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, on_signal)
    try:
        # ── один прогон без микрофона ──
        if args.text is not None or args.wav is not None:
            wav = Path(args.wav) if args.wav else None
            if wav is not None and not wav.is_file():
                raise VoiceError(f"no such wav: {wav}")
            turns.append(one_turn(args, workdir, model_name, wav, args.text))
            session_summary(turns)
            return 0

        # ── интерактивный цикл ──
        print("голосовой чат tokshit · q/Ctrl-C — выход")
        empty_streak = 0
        while True:
            if args.hands_free:
                if args.seconds <= 0:
                    raise VoiceError("--hands-free requires --seconds N")
                time.sleep(0.4)  # не записывать хвост собственного TTS
            elif sys.stdin.isatty():
                try:
                    line = input("\n🎤 Enter — записать, q — выход: ").strip()
                except EOFError:
                    break
                if line.lower() in ("q", "quit", "exit", "/q", "/exit"):
                    break

            try:
                wav = record_wav(
                    workdir,
                    seconds=args.seconds,
                    max_seconds=args.max_seconds,
                    hands_free=args.hands_free,
                )
                if wav is None:
                    empty_streak += 1
                    print("   тишина — попробуй ещё раз")
                    if args.once or empty_streak >= 2:
                        if empty_streak >= 2:
                            print("   два пустых хода подряд — выхожу")
                        break
                    continue
            except VoiceError as exc:
                print(f"✗ {exc}", file=sys.stderr)
                return 1

            try:
                turn = one_turn(args, workdir, model_name, wav, None)
                turns.append(turn)
                if turn.engine_wall_s == 0:  # ASR не распознал речь
                    empty_streak += 1
                    if empty_streak >= 2:
                        print("   два пустых хода подряд — выхожу")
                        break
                else:
                    empty_streak = 0
            except VoiceError as exc:
                print(f"✗ {exc}", file=sys.stderr)
                if args.once:
                    return 1
            if args.once:
                break
        session_summary(turns)
        return 0
    finally:
        cleanup()
        signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VoiceError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        sys.exit(1)
