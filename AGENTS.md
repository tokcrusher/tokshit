# AGENTS.md — tokshit (Inkling-2-Mini worktree)

## Repo / branch

- Worktree: `/data/data/com.termux/files/home/tokshit`
- Branch: `main` (feature branch deleted; all work lives in main)
- Remote: `git@github.com:tokcrusher/tokshit.git`
- Binary: `build/tokshit`
- Tests: `build/tokshit_tests` (163 cases / 8338 assertions at last run;
  `tests/` is gitignored, so test edits are local only)

## Build / test

```sh
cd /data/data/com.termux/files/home/tokshit
cmake --build build -j8
./build/tokshit_tests
bash scripts/check_links.sh
```

## Run

Model selection is explicit:

```sh
./build/tokshit --models              # list .gguf files and exit
./build/tokshit --model inkling "hi" 16
./build/tokshit --model flint  "hi" 16
scripts/chat --models                 # same list, voice wrapper
scripts/chat --model inkling          # TUI on Inkling
```

Without `--model`/`TOKSHIT_MODEL` the default is the Inkling GGUF if it
exists, otherwise Flint. The old "silently run Flint" fallback is gone.

Direct env form (advanced):

```sh
env TOKSHIT_MIN_AVAIL_MB=1200 TOKSHIT_HOT=0 TOKSHIT_EXPERT_CACHE_MB=512 \
  TOKSHIT_PREFAULT=1 TOKSHIT_KV_CTX=512 TOKSHIT_POOL_WORKERS=4 \
  TOKSHIT_POOL_PIN=1 scripts/safe_run.sh build/tokshit --temperature 0 \
  --model inkling "hi" 16
```

Always one model process at a time; `scripts/safe_run.sh` holds the lock and
kills the process if `MemAvailable` drops below `TOKSHIT_MIN_AVAIL_MB`.

## Model paths

- Target: `~/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf`
- DSpark draft: `~/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf`
- mmproj: `~/Inkling-2-Mini-GGUF/mmproj-F16.gguf`

Architecture: Inkling2, 24 physical blocks, 48 KV slots, 3 loop passes,
hidden 2304, 128 experts top-8, vocab 200058/201024.

## Current state (2026-11-23)

- `fb=0`: scalar `dot_fallback` removed for Q5K/Bf16; per-quant `fbq:` in
  `TOKSHIT_TOKEN_DEBUG=1`.
- Attention: NEON f16 dot/axpy + relative-bias table bounded by `pos`;
  8–10× faster at long context. Global layers [5,11,17,23] must see the full
  context; only the 20 local layers use window 1024. Never clip global.
- Pool: main thread on X4, workers pinned to the more homogeneous A720 cores
  (skip the fastest core for workers). This was a real win: 2.9 → ~4.4 burst.
  `TOKSHIT_POOL_CPUS` can pin an explicit mask.
- Dense F32/BF16: NEON `dot_f32_neon` / `dot_bf16_neon`.
- DSpark: implemented, greedy, lossless on checked prompts; batched verify
  remains slower than plain (acceptance 47.6 %, ~1.1–1.3 vs 1.6–1.8 tok/s).
- **Lossless profile** (`scripts/run_inkling2.sh`): ~4.3–4.5 tok/s burst,
  ~3.4–3.5 sustained (16 tokens). `w=2801.5 MB/token`; model fits
  (RSS ~6.8–8.5 GB, `MemAvailable` ~8–9.5 GB).
- **LOSSY coherent profile** (`scripts/run_inkling2_fast.sh`):
  top-4 experts + **3 core passes** + original dense weights + 6 workers.
  Measured ~6.2–6.7 tok/s; text stays readable. This replaces the old
  top-2/2-pass/Q4 profile: that reached 13.3 tok/s but looped
  ("Capital capital capital..."). `--passes 2` is the real quality cliff;
  dropping experts alone to 4/8 is tolerable for voice.
- Batched prefill for Inkling2 is wired (`TOKSHIT_INK_PREFILL_BATCH`,
  default 16, `TOKSHIT_INK_PREFILL_ONLY_LAST`): fixed a real bug where
  `moe_batch()` ran inside the per-token loop (B× the MoE work). Measured
  same-prompt, temperature 0: sequential 3.24 tok/s (61 tok in 18.8 s);
  B=16 4.45 tok/s (13.7 s), B=64 4.92 tok/s (12.4 s); 197-token prompt:
  B=16 5.00, B=64 4.97. Greedy ids were identical for B=0/16/32/64 on both
  prompts. Logits are not bit-exact (max|dlogit|≈0.36 at tap 22), so keep
  the token-identity check when changing kernels. Prefill is compute-bound
  at the same 3.06 Gweight/token; batching does not reduce MACs.
- **Voice** (`scripts/chat`, curses TUI over `scripts/voice_chat.py`):
  one persistent `build/tokshit` chat process per session (no 9–11 s reload
  per turn), `TOKSHIT_CHAT_PROTOCOL=1` (stdout turn framing with `0x1e`),
  streamed tokens, clause/sentence-level streaming TTS. There is no startup
  prefix: `VOICE_FEWSHOT` defaults empty and `TOKSHIT_CHAT_PREFIX` is not
  set. Startup is load-only, ready ~7–10 s; the TUI accepts recording
  during `WARMING` and queues the turn.
  `scripts/chat` adds `--tui`; `--text`, `--wav`, `--check`, `--no-play`,
  `--seconds`, `--verbose`, `--fast`, `--ru`, `--lang` and `--one-sentence`
  pass through.
  Per-turn language directive: EN payload
  `\nQ: Answer in English. <text>\nA:`, RU payload
  `\nQ: Отвечай по-русски. <text>\nA:`. The transcript keeps the original
  ASR/typed text. `--ru` / `--lang ru` / `VOICE_LANG=ru` set `ASR_LANG=ru`
  and `TTS_VOICE=ru_RU-irina-medium` (installed, 63 MB, 22 kHz).
  TUI: header (phase/mode/lang/load/thermal/RSS/ctx/mic), transcript,
  details (mic/ASR/first token/decode/TTS/audio/total/RSS/expert
  hit-miss), buttons `[ RECORD ]`, `[ STOP ]`, `[ FAST ]`, `[ CLEAR ]`,
  `[ QUIT ]`, keys Enter/Space, `s`, `f`, `c`, `q`, PgUp/PgDn. `c` sends
  `/clear`, drains output, clears KV+transcript+turn; busy/recording shows
  a notice. TTS duplicate guard drops identical consecutive sentences and
  stops TTS for the turn after two duplicates; transcript stays intact.
  Stops are `Q:`, `\nQ`, `\nA`; `TOKSHIT_STOP_SENTENCE` is unset by
  default. `--one-sentence` / `VOICE_ONE_SENTENCE=1` is opt-in.
  Keep `TOKSHIT_POOL_WORKERS=4` for chat decode (A/B: W=4 4.1–4.2,
  W=5 3.8–3.9, W=6 3.8–3.9); prefill can use 6 workers but decode is the
  long pole.
- **Speed/quality tiers** (16-token decode, cool phone ~37–40 °C):
  | mode | tok/s | quality |
  |---|---:|---|
  | lossless top-8 / 3 passes | **5.5** | reference |
  | EN fast top-4 / **3 passes** (`scripts/chat --fast`) | 6.2–6.7 | readable; LOSSY 4/8 experts |
  | RU fast top-6 / **3 passes** (`scripts/chat --ru --fast`) | not separately benched | readable; LOSSY 6/8; top-4 loops on Russian turn 2 |
  | top-8 or top-4/top-6 / **2 passes** | 6.4–9.2 | loops on many prompts |
  | top-2 / 2 passes + Q4 dense | 13.3 | garbage |
  The 2-pass modes are the quality cliff, not the expert count. Never
  default them for voice/code. Lossless voice uses top-8/3 passes; the
  coherent fast voice tier uses 3 passes, original dense weights and 6
  workers: top-4 for English, top-6 for Russian. Both fast variants are
  LOSSY and must never be called lossless.
- **Voice latency** (persistent, no startup prefix, base ASR, streaming TTS):
  startup is load-only, ready ~7–10 s; Russian fast measured 6.8–9.1 s.
  The first turn is load-bound; recording during `WARMING` queues it. These
  numbers are thermal-sensitive: compare at a similar phone temperature,
  use the TUI `therm` field, and do not treat single runs as A/B medians.
  All older warmup-based latency rows are obsolete.
- Audio: `~/audio` has whisper.cpp ASR + Piper TTS; `scripts/chat` (TUI
  wrapper around `scripts/voice_chat.py`) and `docs/VOICE_CHAT.md` are the
  entry point and docs. Russian mode uses whisper base with `ASR_LANG=ru`
  and the installed `ru_RU-irina-medium` Piper voice (63 MB, 22 kHz). The
  Termux:API APK is downloaded to
  `~/storage/downloads/termux-api-0.53.0.apk` (checksum verified) and the
  installer was opened; the user must tap Install and grant Microphone.
  `termux-api` CLI package is installed; the APK was missing, which is why
  `termux-microphone-record` hung. Mic check:
  `termux-microphone-record -f "$TMPDIR/mic-test.m4a" -l 5`, then wait for
  the recorder to finalize the file before ASR.

## Goal

The current lossy 10.5 profile is NOT the goal. The order is **15 tok/s
lossless**: keep top-8 experts, 3 passes, Q2_K_XL weights and greedy output.

**Physics check (measured, 2026-11-23):** 15 tok/s at 2801.5 MB/token needs
**42 GB/s**. `bwbench` on this phone reaches 32–42 GB/s with 8 threads (peak
41.9), and less with 5 decode threads. So 15 tok/s lossless is not reachable
without a quality loss. Realistic lossless ceiling is ~10–13 tok/s if
everything becomes clean streaming; the current wall is expert GEMV
(instruction-bound): 3.06 Gweight/token at 16–18 Gweight/s on 5 threads,
~170 ms of ~255 ms per token, while DRAM is only ~5 GB/s busy. Dense is
~75 ms (1.78 GB, 23 GB/s), already near its limit. `kbench` hot numbers:
Iq1S 34.4, Iq2S 22.3, Iq2Xxs 17.2 (2-row, was 14.5), Nvfp4 34.7–38.2
(was 14.0), Q2_0 33.7, Q4K 35.9, Q8_0 22.5 (i8mm 26.6), Q6K 7.8
single-thread (4-row exists, parity 0, not enabled; 5 MB/token).

Remaining non-lossy levers, in order:

1. Repack+SMMLA for dense Q8_0/Q4K and (if it fits) IQ2_XXS; expectation
   ~5–8 ms/token, not a path to 15.
2. i8mm for dense Q6K/Q4K/Q8_0 (Q8_0 +15–18 %, not default).
3. LTO / inlining of `dot_row` in the pool tasks.
4. Previous-token/next-layer expert prefetch was neutral; do not spend more.
5. Keep the lossy 10 tok/s profile clearly marked; never call it lossless.
6. Audio: `~/audio` + `scripts/chat` (docs: `docs/VOICE_CHAT.md`); do not
   use llama.cpp.
7. Hundreds of prefill tok/s is also compute-bound (same 3.06 Gweight/token);
   batching is exact in ids but does not reduce MACs. Measured prefill is
   3.2 → 4.5–5.0 tok/s after the MoE B× fix. Even the fastest existing
   kernels (Iq1S/Q2_0 ~34 Gweight/s) cap prefill at ~11 tok/s; hundreds would
   need ~20× the expert GEMV throughput. The practical target is the same
   kernel ceiling as decode.
8. Audio ASR: use whisper **base** (74M, ~3× realtime, 283 MB) as default;
   small (244M) runs at 0.65× realtime on this phone and starves the LLM.
   TTS Piper lessac-medium ≈2.5 s audio per wall second; round-trip WER 0 %.

## Rules

- Do not use Nvidia/CUDA. CPU/NEON/mobile only.
- Do not enable AHB hot-pool experiments: a previous attempt rebooted the
  phone. `TOKSHIT_HOT_Q4`/`TOKSHIT_INK_HOT` remain opt-in.
- Always run `tokshit_tests` after code changes; commit with a clear message;
  keep `main` green and push.
- Do not claim a speed win without an A/B measurement (same prompt, same
  thermal state, median of ≥3 runs).
