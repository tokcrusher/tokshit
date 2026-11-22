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
(2.1–2.4 in the session-3 runs). The experimental DSpark draft
(`--dspark`) is greedy-only and lossless on tested prompts, but sequential
verify measured 1.23 tok/s against 1.65 tok/s plain on a coding prompt
(24 tokens, K=3). A batched verify path exists and is the default, but it
has no measured tok/s yet, so DSpark is not a speed win. Details:
[PERF_SESSION3.md](PERF_SESSION3.md), [DSPARK.md](DSPARK.md).

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
