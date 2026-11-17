# performance (en)

Measured numbers and engineering conclusions. All runs are on one phone
(8 cores: 1× Cortex-X4 + 7× Cortex-A720, 15.2 GB RAM, Android/Termux) with
`flint-UD-Q3_K_XL.gguf` (18.49 GB). Numbers are noisy: the phone throttles and
background load moves results by tens of percent.

Full Russian version with more detail:
[PERFORMANCE.ru.md](PERFORMANCE.ru.md). Current session baseline and plan:
[PERF_SESSION2.md](PERF_SESSION2.md).

## summary

| question | answer |
|---|---|
| normal top-6 decode, 8 tokens, 47 °C | **2.40 tok/s** |
| normal top-6 decode, 16 tokens, 60 °C | 0.96 tok/s |
| normal decode, session range | 1.0–2.4 tok/s (thermal) |
| earlier long runs | 2.0–2.7 tok/s (historical) |
| no cache, mmap (historical) | 1.24–1.37 tok/s |
| 2–3 GB cache (historical) | ~2.05 tok/s, no gain |
| `--fast` (historical) | 4.52 tok/s, but **lossy top-2** |
| `--turbo` (LOSSY, top-2 + Q4_0) | 3.52 tok/s |
| speculation, full verify | 0.93–1.00 tok/s, 50–78% acceptance |
| batched prefill, T=224 | 4.53 tok/s (`TOKSHIT_PREFILL_BATCH=32`) |
| prefill-bulk, T=224 | 2.32 tok/s (worse than union) |
| AHB | 5.2 GB/s on the hot stream; removes ZRAM/swap |

## model

49 blocks (2 dense + 47 MoE), 256 experts, top-6, hidden 2048, expert ffn 512,
shared expert 1024. Hot weights: 391 tensors, 1551 MB (1740 MB after the
default Q8_0 conversion). The 18.49 GB model does not fit in 15.2 GB RAM.

## methodology

- Always `scripts/safe_run.sh`: it checks `MemAvailable` once per second and
  kills the process below `TOKSHIT_MIN_AVAIL_MB` (default 1200).
- 12–24 new tokens per run, `--temperature 0`.
- Compare configs interleaved in one session (`scripts/ab.sh`); single runs
  drift with temperature and throttling.
- "Cold" means right after load with no expert warm-up. "Warm page cache" means
  the file has been read and fits in page cache as far as RAM allows.
- `--fast` and `--spec-*` are **lossy**: they change the model output (top-2
  instead of top-6) and are not comparable to normal decode.
- Microbenchmarks (`kbench`, `bwbench`, `ahbtest`) do **not** predict full
  decode tok/s.

## commands

```sh
# normal decode, 512 MB cache
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  --expert-stats ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# no cache (mmap)
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 0 \
  --expert-stats ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# lossy preset
scripts/safe_run.sh build/tokshit --fast \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# speculation, full verify
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  --spec --spec-k 3 ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# speculation, verify top-2 (lossy)
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  --spec --spec-k 3 --spec-verify-topk 2 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# phase and I/O breakdown
GGUF_TIME=1 scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  --expert-stats ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# microbenchmarks
build/kbench ~/models/flint-UD-Q3_K_XL.gguf 20 blk.2.attn_q.weight
build/bwbench ~/models/flint-UD-Q3_K_XL.gguf 2048 3
scripts/safe_run.sh build/ahbtest --mb 1024 --blocks 256
GGUF_TIME=1 scripts/safe_run.sh build/tokshit --bench-mm \
  ~/models/flint-UD-Q3_K_XL.gguf
```

## decode results

| mode | cache | I/O | tok/s | hit rate | status |
|---|---:|---|---:|---:|---|
| top-6 | 512 MB | `O_DIRECT` | 1.1–1.9 | — | current session |
| top-6 | 0 | mmap | 1.24–1.37 | — | historical |
| top-6 | 512 MB | `O_DIRECT` | 2.0–2.7 | ~32% | historical long runs |
| top-6 | 2048 MB | `O_DIRECT` | ~2.05 | ~40% | historical |
| top-6 | 3072 MB | `O_DIRECT` | ~2.05 | ~40% | historical |
| top-2 | 0 | buffered | 4.52 | — | **lossy** (`--fast`), historical |

The current session baseline is 1.1–1.9 tok/s on the normal top-6 path with a
512 MB cache. The 2.0–2.7 range is the best long-run result from earlier
sessions and is kept as a historical reference; it is not the current
baseline. Thermal state, background load and code revision differ, so do not
mix the two ranges.

In the historical long runs, the 512 MB cache was a real win over mmap. 2–3 GB
adds ~8 points of hit rate but no tok/s: dense weights and reclaim eat the
gain, and 3–4 GB on a 15 GB phone easily ends in OOM. The 4.52 tok/s result
is top-2 of 6, a different mode.

## speculation results

| mode | verify | tok/s | accepted | status |
|---|---|---:|---:|---|
| `--spec --spec-k 3` | full | 0.93–1.00 | 50–78 % | lossless, greedy parity confirmed |
| `--spec --spec-k 3 --spec-verify-topk 2` | top-2 | ~1.5 | ~50 % | **lossy** |
| `--spec-fast` | top-2 | 2.09 | 83 % | **lossy** |

Full verify is lossless: acceptance is 50–78% on prompts up to ~500 tokens,
and the ids match plain greedy byte for byte. The earlier zero-acceptance
rows do not reproduce (see [SPEC_KV.md](SPEC_KV.md)); `expert_hit/miss` is
now counted in the batched path as well.

## microbenchmarks

| what | tool | result |
|---|---|---|
| Q8_0 matvec, individual segments | `--bench-mm` / `kbench` | 25–35 GB/s |
| hot region from page cache | `bwbench` | 39.5 GB/s |
| 328 KB blocks: AHB | `ahbtest` | 25.7 GB/s |
| 328 KB blocks: malloc | `ahbtest` | 21.7 GB/s |
| full 1.55 GB hot stream, `rare` | full matvec | 5.21 GB/s |
| full 1.55 GB hot stream, `often` | full matvec | 3.10 GB/s |
| raw sequential flash read | `dd iflag=direct` | ~1.4 GB/s |

The key contradiction: segments run at 25–35 GB/s, but the stream over 391 AHB
buffers runs at 5.21 GB/s. The second number is what decode sees.

## token budget

Dense weights read on every token:

```
attention q/k/v/o          849 MB
shared expert              267 MB
dense ffn + router         124 MB
output.weight              312 MB
----------------------------------
total                     ~1551 MB
```

Experts on a full miss:

```
15.77 GB of all experts / 256 × 6 = 0.370 GB/token
```

At a 32% hit rate this drops to ~0.252 GB; at 40% to ~0.222 GB. With a 512 MB
cache the token needs `1.551 + 0.252 ≈ 1.80 GB`. For the historical 2.35 tok/s
run that is ~0.43 s per token and ~4.2 GB/s aggregate. The current session's
1.1–1.9 tok/s is below that historical range and gives roughly 2.0–3.4 GB/s
aggregate; the difference is not yet attributed to a single cause.

## where the time goes

### dense weights and AHB

Dense weights are 1.55 GB per token, the largest item. Segments hit 25–35 GB/s,
but the full stream over 391 buffers gives 5.21 GB/s (`rare`). At 5.21 GB/s the
dense part alone costs ~0.30 s/token, a ceiling of ~3.3 tok/s before any expert
read. The current 1.1–1.9 tok/s is below that ceiling: expert I/O, attention
and scheduling add the rest.

Why 391 buffers: gralloc rejects one 1.55 GB buffer (`rc=-129`). Splitting by
tensor avoids the rejection but costs TLB/contiguity. `often` returns an
uncached mapping here (3.10 GB/s), so `rare` is the default. `TOKSHIT_AHB=0`
moves everything to malloc: memory is freed, but ZRAM starts swapping and
latency grows.

Requantizing dense tensors without a NEON kernel to `Q8_0`
(`TOKSHIT_HOT_CONV=1`, default) raises matvec from 5.47 to 9.15 GB/s and full
decode from 1.90 to 2.28 tok/s; hot weights grow from 1626 to 1740 MB. This is
a historical A/B on one code revision: the current baseline is lower, so use
the table for the conversion effect, not absolute speed. See
[QUANTIZATION.md](QUANTIZATION.md) for the table and caveats.

### experts and flash

Raw flash read is ~1.4 GB/s. At a 32% hit rate the runtime reads ~0.25 GB,
about 180 ms. The expert matvec is fault-latency bound: at cache 0, 12 passes
take 7–9 s, i.e. 0.6–0.75 s/token. That is the historical 1.24–1.37 tok/s
result.

`O_DIRECT` does not make the drive faster; it keeps experts from evicting the
dense weights from page cache. In the historical long runs a 512 MB cache gave
2.0–2.7 tok/s instead of 1.3; the current session gives 1.1–1.9.

### pool

The pool is not the main bottleneck: on short runs (4–8 layers) it delivers
12–24 tok/s and 25+ GB/s. In a full run `GGUF_TIME=1` shows `load` and
`downwait` — waiting for experts, not barrier time. Large `downwait` means the
drive is behind; large `load` means cache misses.

### speculation

A verify batch of `b` rows widens the expert union by about `b`. Dense weights
are reused, experts are not: every token has its own top-6. Short accepted runs
do not compensate. The acceptance and cache-counter bugs are still open.

## why 5 tok/s does not happen

5 tok/s means 200 ms/token:

| item | at 5 tok/s |
|---|---|
| dense 1.551 GB | needs ≥ 7.8 GB/s just for this |
| experts 0.25 GB | another ~180 ms at 1.4 GB/s |
| total | already > 380 ms |

Even with free experts, 200 ms needs a ~7.8 GB/s dense stream. The full AHB
stream is 5.21 GB/s. The historical `--fast` run (top-2, warm page cache)
reaches 4.52 tok/s, i.e. ~7.0 GB/s effective, but it is lossy.

The current full top-6 path is limited by the dense stream and expert reads.
Reaching 5 tok/s needs a faster dense stream (contiguous memory, fewer
buffers, another pinning method), a higher expert-cache hit rate, or a lossy
mode. The 5 tok/s target is in progress; concrete steps are in
[ROADMAP.md](ROADMAP.md).

## what to try next

First reproduce the current 1.1–1.9 baseline and the historical 2.0–2.7
configuration interleaved in one session. Without that, no kernel or cache
change can be credited.

1. **Fix speculation metrics and acceptance.** Count `expert_hit/miss` in the
   batch path, then explain the 0% full-verify acceptance.
2. **Speed up the dense stream.** Compare 391 AHB buffers, one large malloc,
   `mlock`, and `TOKSHIT_HOT=0`. Find where 25–35 GB/s becomes 5.21.
3. **Try `TOKSHIT_HOT=0` with AHB only for the expert cache.** Page cache may
   beat 391 dma-buf mappings for dense weights.
4. **Issue down-expert reads earlier.** Gate/up reads happen before swiglu and
   down; if the drive idles, queue all 24 reads up front.
5. **Reduce dense traffic without losing quality.** For example, quantise
   `output.weight` more aggressively; that changes output and needs the parity
   gate.
6. **Do not chase kernel microbenchmarks.** 25–35 GB/s on segments is already
   there; full decode is limited by the memory stream and the drive.
7. **Compare interleaved.** Use `scripts/ab.sh` and `scripts/sweep.sh`.

## measurement rules

- Do not set a 3–4 GB cache "just in case": reclaim and OOM.
- Do not enable `--prefetch`: ~2x slower on this phone.
- Do not enable `--fadv-dontneed` without measuring.
- Do not use `TOKSHIT_AHB_USAGE=often`: slower.
- Do not run two model processes: 18.49 GB and a shared AHB budget.
- Compare `--fast` only with other lossy modes: it is top-2.
- Quote normal-decode numbers only from full top-6 runs, not from `--fast`
  or microbenchmarks.

## open bugs and issues

1. The batched path breaks on contexts longer than the sliding window (513):
   prefilling a ~1040-token prompt yields blown-up logits and garbage decode.
   Short prompts are correct. Main open speculation bug.
2. gralloc rejects one 1.55 GB AHB buffer (`rc=-129`); 391 buffers are a
   workaround, not a fix.
3. Current top-6 short runs are 1.1–1.9 tok/s, below the historical 2.0–2.7
   long-run range. The cause is not isolated; treat it as an open performance
   issue until an interleaved A/B clears it.
