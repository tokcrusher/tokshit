# tokshit

A specialized C++20 inference engine for ARM64/Snapdragon: it runs MoE
models that are larger than the device's RAM. Target combination:
Flint-40B-A2.5B (GGUF, 18.5 GB) on a phone with 15.2 GB of RAM.

Universal engines pay a cross-platform tax: dynamic dispatch,
general-purpose allocators and compromises for x86 and CUDA. `tokshit`
removes that layer: a fixed hot path, NEON dot-product and i8mm kernels,
AHB (dma-buf) storage for dense weights, `O_DIRECT` expert streaming, a
per-layer LRU cache with a routing frequency profile, and a batched GEMM
prefill pipeline.

Русская версия: [README.ru.md](README.ru.md).

## Features

- **MoE larger than RAM.** Experts (256 per layer, top-6) are streamed from
  flash into a cache; dense weights are pinned in non-swappable AHB memory —
  no ZRAM, no latency spikes.
- **NEON/i8mm kernels.** Dot-product kernels for Q8_0/Q4_0/IQ formats, an
  SMMLA path for Q8_0 (`TOKSHIT_I8MM=1`), batched matvecs and a batched
  prefill GEMM pipeline.
- **Speculative decode.** A draft pass with a reduced router, batched
  verification, lossless greedy mode; 50–78% draft acceptance in short
  rounds.
- **Routing frequency profile.** The profile is saved between runs and
  warms the expert cache before the first token.
- **Batched prefill.** Weights are read once per token chunk, experts by
  the chunk's routing union.
- **TUI.** Live decode indicator (speed, cache hit rate, free memory),
  prefill progress, a `/stats` dashboard, chat commands `/help`, `/temp`,
  `/topk`, `/system`, `/save` and more.
- **Memory guard.** `safe_run.sh` watches `MemAvailable` so the OOM killer
  never reboots the phone.

## Performance

Measured on the target device (8 cores: 1× Cortex-X4 + 7× Cortex-A720,
15.2 GB RAM, UFS ~1.4 GB/s on working chunks, 40–55 °C). The phone
throttles; configs are compared interleaved with `scripts/ab.sh`, single
runs are noisy.

| mode | tok/s |
|---|---:|
| top-6 decode, 512 MB cache, `O_DIRECT` (this session) | 1.1–1.9 |
| top-6 decode, 512 MB cache (earlier long runs, historical) | 2.0–2.7 |
| batched prefill (`TOKSHIT_PREFILL_BATCH=32`) | 4–5.3 |
| `--fast` (top-2, **LOSSY**) | ~4.5 |
| speculation, full verify (lossless) | 0.93–1.0, 50–78% acceptance |

The 2.0–2.7 row is the best long-run result from earlier sessions. It is a
historical reference, not the current baseline: spot runs in the current
session land at 1.1–1.9 tok/s with the same normal top-6 path.

Microbenchmarks: Q8_0 matvec on individual segments 25–35 GB/s; the full
1.74 GB hot stream over AHB 5.2 GB/s; raw sequential flash read ~1.4 GB/s.
The gap between the segment and full-stream numbers is the decode
bottleneck.

Token budget: dense weights ~1.74 GB (requantized to Q8_0), experts ~0.37 GB
on a full miss (6 of 256 × 3 projections × 47 layers). Methodology, phase
breakdowns and detailed runs: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
Current session baseline and plan: [docs/PERF_SESSION2.md](docs/PERF_SESSION2.md).

## Model

| parameter | value |
|---|---|
| blocks | 49: 2 dense + 47 MoE |
| experts | 256 per layer, top-6 |
| hidden / expert ffn | 2048 / 512 |
| shared expert | 1024 |
| attention | 32 q heads, 4 kv heads, head_dim 128, SWA 513 |
| vocab / context | 150144 / 262144 |
| quants | experts IQ3_S/XXS and IQ4_XS; attention Q8_0; router BF16 |

## Quick start

```sh
git clone git@github.com:tokcrusher/tokshit.git
cd tokshit

# toolchain, build, install to $PREFIX/bin
./install.sh

# the model is not included: put the GGUF somewhere and pass the path
# one answer
scripts/safe_run.sh build/tokshit --temperature 0 --cache-mb 512 \
  ~/models/flint-UD-Q3_K_XL.gguf "hello" 32

# interactive chat
scripts/safe_run.sh build/tokshit --cache-mb 512 \
  ~/models/flint-UD-Q3_K_XL.gguf

# full flag reference
build/tokshit --help
```

`./install.sh` checks the toolchain, installs missing packages, builds
Release/Ninja and installs the binary into `$PREFIX/bin` (Termux) or
`/usr/local/bin` (Linux).

`scripts/safe_run.sh` checks `MemAvailable` once per second and kills the
process below `TOKSHIT_MIN_AVAIL_MB` (default 1200). Run the model only
through it, and only one process at a time.

### Manual build

```sh
cmake --preset release
cmake --build build --parallel 2
```

You need CMake 3.20+, Ninja and a C++20 compiler. On aarch64 the NEON files
are compiled separately: the dot-product kernel with
`-march=armv8.2-a+dotprod+fp16+lse`, the i8mm kernel with
`+dotprod+i8mm+fp16+lse`; the flags do not leak into scalar objects. i8mm
variants are controlled by the `TOKSHIT_I8MM` option (ON by default).

## Usage

```sh
build/tokshit [flags] <model.gguf> "<prompt>" [max_new]
```

Without a prompt it starts the interactive chat. Key flags:

| flag | what it does |
|---|---|
| `--cache-mb N` | expert cache budget (512 MB is the working point) |
| `--io-threads N` | background flash readers |
| `--temperature/--top-k/--top-p` | sampling |
| `--fast` | top-2 instead of top-6, faster and **LOSSY** |
| `--spec --spec-k N` | speculative decode (lossless with full verify) |
| `--expert-stats` | expert cache hit/miss at the end of the run |

Presets:

```sh
# fast LOSSY run: top-2 of 6, page cache
scripts/safe_run.sh build/tokshit --fast --temperature 0 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 24

# speculation with lossless verification
scripts/safe_run.sh build/tokshit --spec --spec-k 3 --temperature 0 \
  ~/models/flint-UD-Q3_K_XL.gguf "hi" 24
```

The full flag, environment and chat-command reference is in
[docs/CLI.md](docs/CLI.md).

## Memory

One process-wide AHB budget (`TOKSHIT_AHB_MAX_MB`, 5120 MiB by default) is
shared by hot weights and the expert cache. 512 MB of cache is the working
point; 2 GB adds hit rate but the memory goes to reclaim. gralloc rejects
one buffer for all hot weights, so they live in 391 per-tensor buffers.

Low-memory run:

```sh
scripts/safe_run.sh build/tokshit --hot=0 --cache-mb 256 --io-threads 1 \
  --temperature 0 ~/models/flint-UD-Q3_K_XL.gguf "hi" 8
```

## Layout

```text
include/tokshit/   public headers
src/               runtime
src/compute/       scalar, NEON and batched kernels
src/tools/         qstat, kbench, vcheck, bwbench, pbench, ahbtest
scripts/           safe_run.sh, device_lock.sh, ab.sh, sweep.sh, deploy.sh
docs/              thematic documentation
install.sh         Termux/Linux installer
CMakePresets.json  release/debug presets
LICENSE            custom WTF 1.2
```

## Docs

Full index and reading order: [docs/README.md](docs/README.md).

| document | topic |
|---|---|
| [docs/CLI.md](docs/CLI.md) | flags, environment, presets, TUI, `safe_run` |
| [docs/ARCHITECTURE.en.md](docs/ARCHITECTURE.en.md) | modules and data flow |
| [docs/RUNTIME.md](docs/RUNTIME.md) | load, forward, MoE, cache, AHB, pool (RU) |
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md) | block formats and dequant (RU) |
| [docs/KERNELS.md](docs/KERNELS.md) | NEON dotprod, i8mm, dispatch (RU) |
| [docs/GGUF.md](docs/GGUF.md) | parsing, metadata, mmap/pread (RU) |
| [docs/TOKENIZER.md](docs/TOKENIZER.md) | BPE and sampling (RU) |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | measurements and methodology |
| [docs/PERF_SESSION2.md](docs/PERF_SESSION2.md) | current session baseline, limits and plan (RU) |
| [docs/SPEC_KV.md](docs/SPEC_KV.md) | KV, batched matvec, speculation |
| [docs/ROADMAP.md](docs/ROADMAP.md) | stages, risks, ideas (RU) |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | build and development rules (RU) |

## License

Custom WTF License 1.2, see [LICENSE](LICENSE). You can use, modify and
redistribute the code, including commercially. When redistributing the
software or a substantial part of it, include the license text. No warranty
and no support obligations.
