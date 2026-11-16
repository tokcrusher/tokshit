# architecture (en)

A map of modules and data flow. The end-to-end token path, cache, I/O and AHB
are in [RUNTIME.md](RUNTIME.md); weight formats are in
[QUANTIZATION.md](QUANTIZATION.md); kernels are in [KERNELS.md](KERNELS.md);
GGUF parsing is in [GGUF.md](GGUF.md); measurements are in
[PERFORMANCE.md](PERFORMANCE.md).

## model configuration

The numbers below come from `flint-UD-Q3_K_XL.gguf` (779 tensors, 18.49 GB)
and from the loader code.

| parameter | value |
|---|---|
| blocks | 49: 2 dense + 47 MoE |
| experts | 256 per MoE layer, top-6 |
| hidden | 2048 |
| expert ffn | 512 |
| shared expert | 1024 |
| attention | 32 q heads, 4 kv, head_dim 128 |
| vocab | 150144 |
| rope theta | 10000 |
| sliding window | 513, per-layer 0/1 pattern |
| hot weights | 391 tensors, 1551 MB |
| expert traffic | ~0.37 GB/token on a full miss (6/256 × 3 projections) |

"47 layers" in older notes means 47 MoE layers; there are 49 blocks in total.

## modules

- `gguf.cpp` — parses the header, metadata and tensor table. The file is
  mmap'd once and all weights are spans into that mapping.
- `mmap.cpp` — RAII around `mmap` plus best-effort `madvise`/`mlock`. Android
  can reject these, so calls are not guarantees.
- `quant.cpp`, `dequant.cpp` — block layouts and dequant. `dequant_into` is for
  cold paths and reference checks; the hot path computes over quants directly.
- `compute/kernels_scalar.cpp` — scalar reference.
- `compute/kernels_neon_dotprod.cpp` — NEON dotprod, compiled as a separate TU
  with `-march=armv8.2-a+dotprod+fp16+lse`.
- `compute/kernels_dispatch.cpp` — runtime kernel selection.
- `compute/kernels_batch.cpp`, `attn_batch.cpp` — batched matvec and attention
  for the speculative path.
- `model.cpp` — load, forward, MoE, hot weights, cache, metrics.
- `model_batch.cpp` — batched forward (`token_logits_batch`).
- `expert_io.cpp` — background expert reads, `O_DIRECT`, deduplication.
- `ahb.cpp` — AHardwareBuffer: hot weights and the expert-cache slot pool.
- `pool.cpp` — persistent worker pool with a spin barrier.
- `sampling.cpp` — temperature/top-k/top-p, greedy, UTF-8 wrapper.
- `reasoning.cpp` — thinking-token budget and the forced `</think/>`.
- `tokenizer.cpp` — BPE.
- `spec_loop.cpp` — speculative loop.
- `main.cpp` — CLI: flags, presets, modes.
- `src/tools/` — qstat, kbench, vcheck, bwbench, pbench, ahbtest.

## load

`Model::load`:

1. opens the GGUF and mmaps the file;
2. reads the `flint.*` keys (block_count, expert_count, hidden, ffn, attention
   patterns);
3. builds `Layer` structs with `Meta` offsets into the mapping;
4. computes budgets: total `TOKSHIT_EXPERT_CACHE_MB`, per layer
   `total / n_moe`;
5. creates `ResidentCache`, `FreqState`, `Workspace`;
6. initialises the AHB pool for the expert cache (unless the budget is 0);
7. creates `ExpertIo` with background readers.

Nothing heavy is copied except the norm vectors. `collect_hot()` separately
copies the dense weights: attention q/k/v/o, dense ffn, shared expert, router,
`output.weight` — 391 tensors, 1551 MB. After that `Meta.base` points either
into the mmap or into hot storage; `meta_bytes()` picks based on the `hot` flag.

`TOKSHIT_HOT=0` keeps dense weights in the mmap. By default the model's page
cache is dropped after the copy (`POSIX_FADV_DONTNEED`) so the memory goes to
the expert cache.

During the copy, `TOKSHIT_HOT_CONV=1` (default) requantizes dense tensors
without a NEON kernel (`Q5_K`, `Q6_K`, `BF16`) to `Q8_0`.

## one token

`token_logits(token, kv, pos, out)`:

1. dequant the embedding row straight into `ws.x`;
2. build the rope table into `ws.rope_table` without allocating;
3. call `layer()` for each of the 49 blocks;
4. final rmsnorm and matvec into `out` (vocab 150144).

The batched path (`token_logits_batch`) does the same for several rows at once.

## layer

`layer()` does attention first:

- rmsnorm;
- qkv as one `run_segs`;
- per-head q/k norm;
- rope when `rope_on_[layer]`;
- append k/v to KV;
- scores + softmax with the attention sink;
- o matvec.

Then the FFN:

- dense layer (first 2): gate/up, swiglu, down;
- MoE layer: router `gate_inp` (BF16, 256 logits), top-6, sigmoid weights
  normalised, then three projections per expert (gate/up/down) and the shared
  expert.

Experts are loaded through `resident_expert()` / `take_expert()`. Gate and up
run as one batched `run_segs`, the activation is quantised once per expert,
down runs after swiglu. The shared expert is a plain dense MLP of width 1024.

## matvec and segments

Everything is a `Seg`: a byte span of weights, a quant type, the input, the
pre-quantised activation and the output. `run_segs` splits segments into row
tasks (16 or 32 rows) and sends them to the pool. The calling thread takes
tasks too.

The activation is quantised once per matvec input (`quantize_into`) and reused
by all segments. That is why there is no allocation per segment, and why
`layer()` / `token_logits()` do not allocate.

## expert cache

`ResidentCache` is per layer: every layer has its own map and intrusive LRU
list, so eviction is O(1). The key is `(proj << 32) | expert`, where `proj` is
0/1/2 for gate/up/down. The per-layer budget is `total_budget / n_moe`.

- budget 0 means "direct": the span points into the mmap and nothing is copied;
- otherwise `ExpertIo::load` checks the cache first, then reads;
- `pin_hot_experts(k)` after prefill adds the hottest experts via batched
  sequential `pread` (up to 64 MB per read);
- `TOKSHIT_PREFETCH` enables prediction from the previous token; on this phone
  it is about 2x slower.

## expert I/O

`ExpertIo` keeps an `O_DIRECT` descriptor (if it opens and passes a check
against buffered reads), aligned bounce buffers and a pool of readers.

- `load()` — demand: cache first, then read on the calling thread. This gives
  the drive more queue depth than a couple of background readers.
- `demand_async()` — order a read without waiting; `take_expert()` waits for it
  and takes the block from the cache.
- `inflight_` prevents reading the same block twice: a second request waits for
  the read already in flight.
- `prefetch()` — a separate prediction queue; `new_epoch()` on a new token
  drops stale predictions.
- the read window is aligned to `st_blksize`; the unaligned tail is copied out
  of a bounce buffer.
- a failed read does not break the token: `resident_expert` falls back to the
  mmap.

## AHB

Why: under pressure Android swaps anonymous pages into ZRAM, and every dense
weight read then pays for decompression. `AHardwareBuffer` is dma-buf memory:
it is not swapped and not reclaimed like anon memory.

Access: `dlopen("libandroid.so")`, not `-landroid`. A `-landroid` binary
carries a `LIBANDROID vN` version requirement that Termux's unversioned stub
cannot satisfy, so the linker fails before `main`. `dlopen` skips both the link
and the version check.

Budget:

- one per process, **5120 MiB** by default (`TOKSHIT_AHB_MAX_MB`);
- shared by hot weights and the expert-cache pool;
- `reserve_ahb` refuses if less than 1 GiB of `MemAvailable` would remain;
- hot weights use **one buffer per tensor**, 391 buffers, 1551 MB. gralloc
  rejects a single 1.55 GB buffer with `rc=-129`;
- the expert pool uses 128 MB slabs and size classes matching the gate/up/down
  slices. It is lazy: slabs appear only after the router starts choosing
  experts;
- the pool gets `3/4` of the cache budget as its own limit; the rest is
  headroom. Everything still counts against the global 5120 MiB;
- the cache's malloc fallback is capped at **256 MiB** so "AHB + as much malloc
  as fits" cannot cause another OOM;
- `TOKSHIT_AHB=0` disables AHB completely: hot weights and cache use malloc;
- `TOKSHIT_AHB_USAGE=rare|often`, default `rare`. On this device `often`
  produces an uncached mapping: the hot-weight stream runs at 3.10 vs
  5.21 GB/s.

AHB is not an accelerator by itself: 328 KB blocks read at 25.7 GB/s vs
21.7 GB/s for malloc, while the full 1.55 GB stream over 391 buffers runs at
only ~5.2 GB/s. The benefit is no ZRAM/swap and stable latency.

## pool

`TaskPool` is a singleton with persistent threads and a busy-wait barrier.
`TOKSHIT_POOL_WORKERS` sets the worker count (default `nproc - 1`); the main
thread always participates. During decode `set_active(true)` keeps idle
workers spinning instead of sleeping. `TOKSHIT_POOL_PIN` pins workers to cores;
`TOKSHIT_POOL_RT` tries `SCHED_FIFO` (usually denied on Android).

## kernels

- the scalar reference is the source of truth for tests;
- NEON dotprod is the hot path for `Q8_0`, `Q4_0`, `IQ2_S`, `IQ3_S`,
  `IQ3_XXS`, `IQ4_XS`;
- `Q4_K`, `Q5_K`, `Q6_K`, `BF16` use the fallback; dense weights are
  requantized to `Q8_0` at load (`TOKSHIT_HOT_CONV`);
- dispatch selects NEON via `HWCAP_ASIMDDP`, with no virtual calls;
- batched kernels (`dot_row_multi`, `run_bsegs`) reuse a weight row for several
  activations;
- a new NEON kernel is accepted only together with a scalar comparison.

Full tables, layouts and the rule for adding a kernel are in
[KERNELS.md](KERNELS.md).

## speculative path

`spec_loop.cpp` uses a separate f16 KV cache (`Kv::alloc`, capacity
`TOKSHIT_SPEC_CTX`, default 2048; 2k context ≈ 195 MB).

1. Prefill runs in chunks of `maxb` (8) rows.
2. Draft: the first token comes from the full logits, the next `K-1` use a
   reduced router (`TOKSHIT_DRAFT_TOPK`, default 2). This is lossy.
3. Verify: a batch over the candidates, full router or `SPEC_VERIFY_TOPK`.
4. The accepted prefix is kept, KV is rolled back via `truncate_all(pos)`.
5. Greedy output should stay lossless under full verify.

Known issues:

- **the batched path breaks on contexts longer than the sliding window
  (513).** Prefilling a ~1040-token prompt yields blown-up logits and
  garbage decode; short prompts (up to ~500 tokens) are correct. Until this
  is fixed, long speculative runs are invalid;
- `c_expert_hit/miss` counters are now updated in the batched path
  (`model_batch.cpp`) with the same semantics as the single-token path;
- acceptance under full verify is measured: 50–78% on prompts up to ~500
  tokens (`--spec --spec-k 3` runs at 0.93–1.00 tok/s), and the ids match
  plain greedy byte for byte; `--spec-verify-topk 2` gives ~50% and ~1.5
  tok/s, `--spec-fast` gives 83% and 2.09 tok/s (both **lossy**). All of
  these are slower than normal top-6 with a 512 MB cache (2.0–2.7 tok/s)
  on short prompts.

## errors

No exceptions. `Result<T>` is a variant of value or `Error`, `TOKSHIT_TRY` is
the equivalent of `?`. The hot path never constructs an error; only the loader
and tokenizer do.

## tests

doctest, one file per area: quants, dequant, GGUF, mmap, cache, expert I/O,
pool, sampling, tokenizer, KV, attention, batched kernels. The baseline is
Kernel and layer code is checked against the scalar reference in `compute/kernels_scalar.cpp`.

## invariants

- no allocation in `layer()` or `token_logits()`;
- `rope_on_` and `swa_` have exactly `n_layers` entries;
- `n_dense <= n_layers`;
- every `Seg` has `rows * rb <= bytes.size()`;
- cache budget 0 is legal and means "do not cache";
- a hot tensor never crosses an AHB buffer boundary;
- `reserve_ahb` never takes the last 1 GiB of `MemAvailable`.
