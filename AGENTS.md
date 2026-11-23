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

Use `scripts/run_inkling2.sh` (robust launcher) or:

```sh
env TOKSHIT_MIN_AVAIL_MB=1200 TOKSHIT_HOT=0 TOKSHIT_EXPERT_CACHE_MB=512 \
  TOKSHIT_PREFAULT=1 TOKSHIT_KV_CTX=512 TOKSHIT_POOL_WORKERS=4 \
  TOKSHIT_POOL_PIN=1 scripts/safe_run.sh build/tokshit --temperature 0 \
  /data/data/com.termux/files/home/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf "hi" 16
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
- **LOSSY 10 tok/s profile** (`scripts/run_inkling2_fast.sh`):
  top-2 experts + 2 core passes + Q4_0 dense + 6 workers. Measured
  **10.04–10.56 tok/s** burst/short and **9.2–9.4 tok/s** on a 24-token
  coding prompt. Quality is visibly degraded; never present as lossless.
- `--passes 2` alone ~2.5; top2+Q4+passes2 is the 9–10.5 configuration.
  Batched prefill B=32 was slower; keep sequential.
- Audio: a subagent is downloading tiny ASR/TTS models (whisper.cpp small/base
  + sherpa-onnx/Piper) into `~/audio/` for future voice integration.

## Goal

10 tok/s has been reached with the **LOSSY** profile. Do not stop here if the
user wants a lossless 10: remaining non-lossy levers, in order:

1. Fix/benchmark 4-row IQ2_XXS; optimize IQ1_S/Q2_0/NVFP4 table kernels.
2. i8mm for dense Q6K/Q4K/Q8_0 (measure cold; `TOKSHIT_I8MM=1` was slower in
   one noisy run).
3. LTO / inlining of `dot_row` in the pool tasks.
4. Previous-token/next-layer expert prefetch (`TOKSHIT_INK_PREFETCH_PREV`)
   was inconclusive; re-test with the new A720 pinning.
5. Keep the lossy 10 tok/s profile clearly marked; never call it lossless.

## Rules

- Do not use Nvidia/CUDA. CPU/NEON/mobile only.
- Do not enable AHB hot-pool experiments: a previous attempt rebooted the
  phone. `TOKSHIT_HOT_Q4`/`TOKSHIT_INK_HOT` remain opt-in.
- Always run `tokshit_tests` after code changes; commit with a clear message;
  keep `main` green and push.
- Do not claim a speed win without an A/B measurement (same prompt, same
  thermal state, median of ≥3 runs).
