# Performance

Target: decode ≥ 15 tok/s, prefill ≥ 50 tok/s for Flint-40B-A2.5B, top-6,
lossless, on the target phone. The target is not reached. Current numbers,
method and limits: [PERFORMANCE.ru.md](PERFORMANCE.ru.md). A second model,
Inkling-2-Mini, is measured separately below.

## Current numbers

| mode | tok/s |
|---|---:|
| normal top-6, 8 tokens, 47 °C | **2.40** |
| normal top-6, 16 tokens, 60 °C | 0.96 |
| `--turbo` (top-2 + Q4_0, **LOSSY**) | 3.52 |
| prefill union, T=224 | 4.53 |
| prefill union, T=128 | 6.01 |
| prefill-bulk, T=224 | 2.32 |
| speculation, full verify (lossless) | 0.93–1.0 |

### Inkling-2-Mini

The second target is Inkling-2-Mini UD-Q2_K_XL (9.14 GB): 24 physical
blocks, 48 KV slots, 3 loop passes, 128 experts top-8, hidden 2304. Plain
greedy is noisy: roughly 1.6–3.0 tok/s depending on thermals and cache
(1.61–1.72 in the 2026-11-22 runs). The experimental DSpark draft
(`--dspark`) is greedy-only and lossless on tested prompts; sequential
verify measured 0.72 tok/s and batched verify 0.52 tok/s against
1.61–1.72 tok/s plain (16 tokens, K=3, acceptance 47.6 %), so DSpark is not
a speed win. Details: [PERF_SESSION3.md](PERF_SESSION3.md),
[DSPARK.md](DSPARK.md).

New diagnostic (2026-11-23): `TOKSHIT_TOKEN_DEBUG=1` now prints `fb=`, the
number of `dot_fallback` scalar calls per token. One normal-mode token gave
`fb=34944` and `w=2801.5 MB`. That many scalar calls is direct evidence that
the fallback is a major compute cost. Effective `10.6 GMAC/s` is ~1–2 % of
the SDOT peak, not a hardware limit; if the fallback is removed, 10 tok/s
is plausible. This is a **diagnosis pending measurement**, not a benchmark.
The 5.5 GB/token budget is not the bottleneck; compute is. `--passes 2` and
`--topk-experts 2` remain **LOSSY**: the varloop experiment measured
1.97 → 2.52 tok/s, top-2/2-pass ~3.9–5.2 tok/s, and quality degrades.

## Flint token budget

- dense weights after Q8_0: 1.74 GB;
- experts on a full miss: 0.377 GB;
- total: 2.12 GB per token.

15 tok/s needs 31.8 GB/s sustained. The device sustains 4.2–12.6 GB/s
depending on temperature. At 60 °C the dense stream drops from 12.6 to
4.2 GB/s.

## Limits

The dense weights are read on every token. At 12.6 GB/s that is 138 ms,
below 7 tok/s before experts are counted. Experts add 0.377 GB and
200–700 ms on a hot phone. Prefill is bound by UFS and by the expert region
of all layers.

Next levers: Q6_K/Q5_K kernels instead of Q8_0 conversion, sidecar expert
layout, i8mm GEMM for prefill, external cooling. Without one of them the
15/50 target stays out of reach.

## Method

- Device: Snapdragon 8s Gen 4, 8 cores, 15.2 GB RAM, UFS, Android/Termux.
- Model: Flint-40B-A2.5B, 18.49 GB, 49 layers, 256 experts, top-6.
- Always `scripts/safe_run.sh`, one model process, `termux-wake-lock`.
- Compare configs interleaved with `scripts/bench_matrix.sh`.
- A run without temperature, wake lock and token count is invalid.

## Tests

`build/tokshit_tests`: 186 cases, 9951 assertions. NEON kernels are checked
against the scalar path. Documentation links are checked by
`scripts/check_links.sh`.
