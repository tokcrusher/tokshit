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
- Pool: 4 workers, pinning by max frequency, no 1 ms sleep in the main barrier.
- Dense F32/BF16: NEON `dot_f32_neon` / `dot_bf16_neon`.
- DSpark: implemented, greedy, lossless on checked prompts; batched verify
  remains slower than plain (acceptance 47.6 %, ~1.1–1.3 vs 1.6–1.8 tok/s).
- Measured: 4-token burst 2.7–4.0 tok/s (median ~2.9), 16-token sustained
  ~2.35 tok/s; `w=2801.5 MB/token`; model fits (RSS ~6.8 GB,
  `MemAvailable` ~8.2 GB), `majf` tens per token.
- `--passes 2` / `--topk-experts 2` are LOSSY; varloop 1.97 → 2.52;
  top2/passes2 ~3.9–5.2. Batched prefill B=32 was slower; keep sequential.

## Goal

Reach 10 tok/s on Inkling-2-Mini without stopping at the first plateau.
Remaining code levers, in order:

1. Prefetch experts for the **next layer** using the previous token's routing
   (`TOKSHIT_INK_PREFETCH_PREV`) to raise memory-level parallelism.
2. Fix/benchmark 4-row IQ2_XXS; optimize IQ1_S/Q2_0/NVFP4 table kernels.
3. i8mm for dense Q6K/Q4K/Q8_0 (measure; `TOKSHIT_I8MM=1` was slower in one
   noisy run, re-test cold).
4. LTO / inlining of `dot_row` in the pool tasks.
5. Only then lossy top2/passes2; keep them clearly marked.

## Rules

- Do not use Nvidia/CUDA. CPU/NEON/mobile only.
- Do not enable AHB hot-pool experiments: a previous attempt rebooted the
  phone. `TOKSHIT_HOT_Q4`/`TOKSHIT_INK_HOT` remain opt-in.
- Always run `tokshit_tests` after code changes; commit with a clear message;
  keep `main` green and push.
- Do not claim a speed win without an A/B measurement (same prompt, same
  thermal state, median of ≥3 runs).
