# tokshit

C++20 inference engine for ARM64/Snapdragon. It runs MoE models larger than
the device's RAM. Target: Flint-40B-A2.5B (GGUF, 18.5 GB) on a 15.2 GB phone.

The engine keeps one fixed hot path for one device and one model family:
NEON dot-product kernels, AHB (dma-buf) for dense weights, `O_DIRECT` for
experts, a per-layer LRU expert cache with a routing-frequency profile, and
batched prefill. No runtime CPU dispatch, no cross-platform compromise layer.

Russian entry point with full details: [README.ru.md](README.ru.md).

## Features

- MoE larger than RAM: 256 experts per layer, top-6, streamed from flash.
- Dense weights pinned in AHB, outside ZRAM and LMK reclaim.
- NEON dot-product kernels for Q8_0, Q4_0 and IQ formats.
- Optional i8mm and two-row IQ kernels; both experimental and off by default
  at runtime.
- Speculative decode with reduced-router draft, batched verify and KV
  rollback. Full verify keeps greedy output; measured slower than plain
  decode on this device.
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

Token budget: about 1.74 GB of dense weights after Q8_0 conversion plus
about 0.38 GB of experts on a full miss. Details:
[docs/PERFORMANCE.ru.md](docs/PERFORMANCE.ru.md).

## Model

| parameter | value |
|---|---|
| blocks | 49: 2 dense + 47 MoE |
| experts | 256 per layer, top-6 |
| hidden / expert ffn | 2048 / 512 |
| shared expert | 1024 |
| attention | 32 q heads, 4 kv, head_dim 128, SWA 513 |
| vocab / context | 150144 / 262144 |
| quants | experts IQ3_S/XXS and IQ4_XS; attention Q8_0; router BF16 |

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
| `--spec --spec-k N` | speculative decode, lossless with full verify |
| `--expert-stats` | cache hit/miss summary |

Full flag and environment reference: [docs/CLI.md](docs/CLI.md). The topic
docs are written in Russian; the tables and command lines are language-neutral.

## Docs

- [docs/README.md](docs/README.md) — reading order.
- [docs/CLI.md](docs/CLI.md) — flags, env vars, presets, TUI.
- [docs/ARCHITECTURE.ru.md](docs/ARCHITECTURE.ru.md) — modules and data flow.
- [docs/RUNTIME.md](docs/RUNTIME.md) — load, forward, MoE, cache, AHB, pool.
- [docs/PERFORMANCE.ru.md](docs/PERFORMANCE.ru.md) — measurements and method.
- [docs/SPEC_KV.md](docs/SPEC_KV.md) — KV cache, batched matvec, speculation.

## License

Custom WTF License 1.2, see [LICENSE](LICENSE).
