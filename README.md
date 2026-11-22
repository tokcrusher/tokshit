# tokshit

C++20 inference engine for ARM64/Snapdragon. It runs MoE models larger than
the device's RAM. Two targets: Flint-40B-A2.5B (GGUF, 18.5 GB) and
Inkling-2-Mini (UD-Q2_K_XL, 9.14 GB), both on a 15.2 GB phone.

The engine keeps a fixed hot path for one device. Each model family has its
own backend: NEON dot-product kernels, AHB (dma-buf) for dense weights,
`O_DIRECT` for experts, a per-layer LRU expert cache with a routing-frequency
profile, and batched prefill. No runtime CPU dispatch, no cross-platform
compromise layer.

Russian entry point with full details: [README.ru.md](README.ru.md).

## Features

- MoE larger than RAM: 256 experts per layer, top-6, streamed from flash.
- Inkling-2-Mini path: 24 physical blocks, 48 KV slots, 3 loop passes,
  128 experts top-8, hidden 2304; mmap'd into page cache, no expert streaming.
- Dense weights pinned in AHB, outside ZRAM and LMK reclaim.
- NEON dot-product kernels for Q8_0, Q4_0 and IQ formats.
- Optional i8mm and two-row IQ kernels; both experimental and off by default
  at runtime.
- Speculative decode with reduced-router draft, batched verify and KV
  rollback. Full verify keeps greedy output; measured slower than plain
  decode on this device.
- Experimental DSpark draft for Inkling-2-Mini (`--dspark FILE`): 402 MB
  `dflash` GGUF, block size 8. Greedy-only, lossless on tested prompts,
  currently slower than plain decode and off by default.
- Routing-frequency profile persisted between runs.
- Batched prefill: weights are read once per chunk, experts by routing union.
- TUI with prefill progress, status line, `/stats`, `/temp`, `/topk`,
  `/system`, `/save`.
- `safe_run.sh` watchdog stops the process before Android kills it.

## Performance

Measured on the target device: 8 cores (1× Cortex-X4 + 7× Cortex-A720),
15.2 GB RAM, UFS. The phone throttles, so single runs are noisy. Use
`scripts/bench_matrix.sh` for interleaved A/B.

| mode | tok/s |
|---|---:|
| decode top-6, 8 tokens, 47 °C, 1024 MB cache, 8 io | **2.40** |
| decode top-6, 16 tokens, 60 °C | 0.96 |
| decode top-6, historical long runs | 1.0–2.7 |
| `--turbo` (top-2 + Q4_0, **LOSSY**) | 3.52 |
| prefill union, T=224, batch 32 | 4.53 |
| prefill union, T=128 | 6.01 |
| prefill-bulk, T=224 | 2.32 |
| speculation, full verify (lossless) | 0.93–1.0 |

At 60 °C the dense stream drops from 12.6 to 4.2 GB/s. Run with
`termux-wake-lock`, watch the temperature, let the phone cool between runs.

Inkling-2-Mini is a second target. Plain greedy there is noisy too: roughly
1.6–3.0 tok/s depending on thermals and cache (2.1–2.4 in the session-3
runs). Experimental DSpark: 1.53 sequential vs 1.13–1.23 batched vs 1.65 plain on
a 16-token coding prompt (warm, K=3, acceptance 47.6 %); still not a win on
this phone. **LOSSY** knobs `--passes 2` and `--topk-experts 2` are measured
up to ~5.2 tok/s in short runs and degrade the output. Details:
[docs/PERF_SESSION3.md](docs/PERF_SESSION3.md).

Flint token budget: about 1.74 GB of dense weights after Q8_0 conversion
plus about 0.38 GB of experts on a full miss. Details:
[docs/PERFORMANCE.ru.md](docs/PERFORMANCE.ru.md).

Inkling-2-Mini prefaults all pages by default; `--prefault=0` switches to
demand paging and `--kv-ctx 512` saves about 300 MB. If the loader prints the
RSS WARNING, the 9.14 GB model is not resident and page cache will thrash:
free memory, use `--drop-siblings`, or lower `--kv-ctx`.

## Models

### Flint-40B-A2.5B

| parameter | value |
|---|---|
| blocks | 49: 2 dense + 47 MoE |
| experts | 256 per layer, top-6 |
| hidden / expert ffn | 2048 / 512 |
| shared expert | 1024 |
| attention | 32 q heads, 4 kv, head_dim 128, SWA 513 |
| vocab / context | 150144 / 262144 |
| quants | experts IQ3_S/XXS and IQ4_XS; attention Q8_0; router BF16 |

### Inkling-2-Mini

| parameter | value |
|---|---|
| quant / size | UD-Q2_K_XL, 9.14 GB |
| parameters | 26.4B total, 3.4B active |
| blocks | 24 physical, 48 KV slots |
| loop | 3 passes over core slots 6..17 |
| experts | 128 per layer, top-8 |
| hidden | 2304 |
| vocab | 200058 used, 201024 padded |

## Quick start

```sh
git clone https://github.com/tokcrusher/tokshit.git
cd tokshit
./install.sh

# one answer
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  ~/models/flint-UD-Q3_K_XL.gguf "hello" 32

# interactive chat
scripts/safe_run.sh build/tokshit --cache-mb 512 \
  ~/models/flint-UD-Q3_K_XL.gguf

# Inkling-2-Mini
scripts/safe_run.sh build/tokshit --temperature 0 \
  ~/models/Inkling-2-Mini-UD-Q2_K_XL.gguf "hello" 24

# Inkling-2-Mini with the experimental DSpark draft (greedy only)
scripts/safe_run.sh build/tokshit --temperature 0 \
  --dspark ~/models/dspark-Inkling-2-Mini-Q8_0.gguf --dspark-k 3 \
  ~/models/Inkling-2-Mini-UD-Q2_K_XL.gguf "hello" 24

# all flags
build/tokshit --help
```

Without a prompt the binary opens the chat. Run one model process at a time.
`safe_run.sh` checks `MemAvailable` once per second and stops the process
below `TOKSHIT_MIN_AVAIL_MB` (default 1200).

Manual build:

```sh
cmake --preset release
cmake --build build --parallel 2
```

Requires CMake 3.20+, Ninja and a C++20 compiler. On aarch64 the NEON files
are compiled separately with `+dotprod+fp16+lse` and `+i8mm`; the rest of
the runtime is not built with those ISA extensions. The i8mm code is compiled
by default (`TOKSHIT_I8MM` CMake option) and enabled at runtime with
`TOKSHIT_I8MM=1`.

## Flags

```sh
build/tokshit [flags] <model.gguf> "<prompt>" [max_new]
```

| flag | effect |
|---|---|
| `--cache-mb N` | expert cache budget; 512 MB is the working point |
| `--io-threads N` | background readers |
| `--temperature/--top-k/--top-p` | sampling |
| `--turbo` | top-2 + Q4_0 dense, faster and **LOSSY** |
| `--passes N` | Inkling-2-Mini core passes, 1..3; fewer is faster and **LOSSY** |
| `--topk-experts K` | experts per token; 2 = turbo router, **LOSSY** |
| `--kv-ctx N` | Inkling-2-Mini KV positions (default 2048); 512 saves ~300 MB |
| `--prefault=0` | demand paging instead of touching every model page |
| `--spec --spec-k N` | speculative decode, lossless with full verify |
| `--dspark FILE` | experimental DSpark draft for Inkling-2-Mini; greedy only |
| `--dspark-k K` | DSpark candidates per round, 1..7 (default 3) |
| `--dspark-batch 0\|1` | DSpark batched verify (default 1); 0 = sequential |
| `--expert-stats` | cache hit/miss summary |

Full flag and environment reference: [docs/CLI.md](docs/CLI.md). The topic
docs are written in Russian; the tables and command lines are language-neutral.

## Docs

- [docs/README.md](docs/README.md) — reading order.
- [docs/CLI.md](docs/CLI.md) — flags, env vars, presets, TUI.
- [docs/ARCHITECTURE.ru.md](docs/ARCHITECTURE.ru.md) — modules and data flow.
- [docs/RUNTIME.md](docs/RUNTIME.md) — load, forward, MoE, cache, AHB, pool.
- [docs/PERFORMANCE.ru.md](docs/PERFORMANCE.ru.md) — measurements and method.
- [docs/PERF_SESSION3.md](docs/PERF_SESSION3.md) — Inkling-2-Mini and the DSpark attempt.
- [docs/DSPARK.md](docs/DSPARK.md) — DSpark draft design, tap mapping, measurements.
- [docs/SPEC_KV.md](docs/SPEC_KV.md) — KV cache, batched matvec, speculation.

## License

Custom WTF License 1.2, see [LICENSE](LICENSE).
