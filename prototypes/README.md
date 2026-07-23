# prototypes/

Self-contained exploratory prototypes — not part of the GPU kernel pipeline,
just experiments answering a specific question. Each file documents its setup,
the hardware it was verified on, and the finding in its header.

## ssd_saturate.ts — Can Bun saturate the NVMe SSD?

**Question:** Does Bun read from disk in parallel, and how fast can it go?

**Setup:** 8 × 1 GB files read with `Promise.all` over varying concurrency,
page cache dropped (`echo 3 > /proc/sys/vm/drop_caches`) before every timed
run, zero warmup. Compares `Bun.file().arrayBuffer()` vs `fs/promises.readFile`.

```bash
bun run prototypes/ssd_saturate.ts
# env: DATA_DIR, FILE_COUNT, FILE_MB (defaults: /tmp/bunker-diskbench, 8, 1024)
```
Requires passwordless sudo (for `drop_caches`); without it you measure RAM, not SSD.

### Verified hardware (RTX 3090 dev box)
- SSD: **Lexar NM790 4TB** (PCIe Gen4 x4, rated ~7400 MB/s seq read)
- CPU: AMD Ryzen 9 9950X3D (32 threads) · RAM: 64 GB · NixOS / Linux

### Result (8 GB, cold cache)
| concurrency | Bun.file | fs.readFile |
|---|---|---|
| 1  | 4259 MB/s | 3846 MB/s |
| 2  | 4429 MB/s | 4414 MB/s |
| 4  | 6045 MB/s | 6513 MB/s |
| 8  | 6940 MB/s | 6555 MB/s |
| 16 | 6973 MB/s | 7132 MB/s |
| 32 | 7048 MB/s | 7145 MB/s |

**Peak: ~7145 MB/s — 97% of the drive's rated 7400 MB/s.** The Gen4x4 link is
the bottleneck, not the runtime. `Bun.file` and `fs.readFile` converge to the
same device ceiling.

### Findings
1. **Yes, Bun reads in parallel.** Awaiting N reads via `Promise.all` dispatches
   them concurrently on Bun's I/O thread pool.
2. **The knee is at concurrency ≈ 4–8.** Past 8 it's flat — you're device-saturated.
   More threads cannot exceed the Gen4x4 link bandwidth.
3. **Concurrency 1 is already ~4 GB/s** — that's the kernel's sequential
   readahead, not parallelism. NVMe single-stream reads are fast.
4. **The two APIs hit the same wall** (~7.1 GB/s), proving the limit is the SSD,
   not Bun. `Bun.file` wins at low concurrency (less per-op overhead).

### Two gotchas that cost a re-run
- **No warmup for cold reads.** Warm-then-measure returns RAM speed
  (~18 GB/s). Drop caches immediately before the timed read.
- **Do NOT use `fallocate` to make test files.** It creates *uninitialized*
  extents that the kernel serves as zeros without disk I/O — reads then report
  12+ GB/s (impossible for the SSD). Use `dd if=/dev/zero` so extents are
  initialized, and verify with `du` that the file isn't sparse.

### Takeaway for bunker
If the `experiments/` quantized-matmul loaders ever pull many weight shards
(`.bin`/`.gguf`), `Promise.all(files.map(f => Bun.file(f).arrayBuffer()))` at
concurrency ~8 will peg this SSD at 7.1 GB/s — within 3% of theoretical max.
Nothing in JS-land can go faster on this drive; only a second NVMe or
RAM-resident data would.

## gpu_upload.ts — Can we push SSD data into GPU memory at SSD-max speed?

**Question:** If we read tensors from the SSD and upload them to the GPU, can the
combined pipeline sustain the SSD's ~7 GB/s, or does the host→device copy add on
top? Reuses the 8 × 1 GB test files from `ssd_saturate.ts`.

```bash
bun run prototypes/ssd_saturate.ts   # create the 8x1GB test files
bun run prototypes/gpu_upload.ts
```

### The key facts
- **PCIe upload (RAM→GPU) is ~22 GB/s** — ~3× faster than the SSD. So the copy
  is *never* the bottleneck; the only question is whether we can **hide** the
  ~0.4 s of copy time behind the ~1.2 s of read time.
- **Two ingredients are both required** for the upload to be hidden:
  1. **Chunked/streamed reads** — concurrent whole-file reads finish clustered,
     leaving no window to overlap uploads. Reading in ~64 MB slices staggers
     completions so async copies run throughout the read phase.
  2. **Pinned host memory** (`cuMemHostRegister`) — otherwise
     `cuMemcpyHtoDAsync` from unpinned Bun-allocated buffers degrades to
     synchronous (CUDA pins-and-copies inline, blocking the JS thread).
- Either ingredient alone caps at ~5–6 GB/s; **both together reach the SSD
  ceiling on good runs**.

### Result (8.59 GB, cold cache, RTX 3090)
| Strategy | Time | Throughput |
|---|---|---|
| A. HtoD ceiling (RAM→GPU, sync) | 0.40 s | 22 GB/s |
| B. Serial read+upload (no overlap) | 2.9 s | 3 GB/s |
| C. Pipelined (async, unpinned) | 1.8 s | 5 GB/s |
| D. Pinned + read-into-buf + async | 1.6 s | 5–6 GB/s |
| E. Bun read + register-pin + async | 1.7 s | 5 GB/s |
| F. Chunked stream (no pin) | 1.5 s | 6 GB/s |
| **G. Chunked + pinned pipeline** | **1.25–1.75 s** | **5–7 GB/s** |
| SSD-only ceiling (ref) | — | ~7.1 GB/s |

### Findings
1. **The upload phase is provably hidden.** Copies alone take 0.4 s (strategy A);
   the full pipeline G takes ~1.25–1.75 s, which is dominated by the read phase
   (8.6 GB / 7 GB/s ≈ 1.2 s). The copies fit inside the read window.
2. **G reaches the SSD ceiling (~7 GB/s) on good runs** — i.e. data lands in GPU
   memory as fast as the SSD can deliver it, with the PCIe upload fully overlapped.
3. **Run-to-run variance is ±2 GB/s** (G ranges 5–7 GB/s). It comes from
   page-cache-drop timing and NVMe read scheduling — *not* from the upload.
   Chunked slice reads have higher per-op overhead than whole-file reads, which
   trims the raw read rate on some runs.
4. **Without overlap you lose ~40%.** Strategy B (serial) is 3 GB/s; even the
   naive async C is only 5 GB/s. The double-buffered DMA pipeline (G) is what
   closes the gap to the SSD wall.

### Takeaway for bunker
For streaming quantized weights / KV-cache tensors from disk onto the GPU, use
the **G** pattern: read in ~64 MB slices with bounded concurrency (~8),
`cuMemHostRegister` each slice, `cuMemcpyHtoDAsync_v2` onto a stream, and
`cuStreamSynchronize` once at the end. That hits the SSD's max (~7 GB/s on this
box) with the upload fully hidden. The copy is never the limit on Gen4.

## latency.ts — Lowest-latency SSD ↔ mem ↔ GPU (while keeping bandwidth)

**Question:** Throughput is maximized by large chunks + deep queues, but those add
queueing latency. What is the *latency* floor for a single tensor to travel
SSD → host → GPU, and what chunk size keeps both latency low AND bandwidth high?

```bash
bun run prototypes/ssd_saturate.ts   # create the 1 GB test file
bun run prototypes/latency.ts
```
Requires passwordless sudo (cold reads via `drop_caches`); pre-opens the file fd
and uses `pread`/`fs.readSync` to avoid event-loop jitter; pinned host memory
registered once and reused.

### Latency floors (single op, RTX 3090 + Lexar NM790)
| Path | 4 KB | 64 KB | 256 KB | 1 MB | 4 MB |
|---|---|---|---|---|---|
| SSD read (cold random) | 117 µs | 161 µs | 320 µs | 678 µs | 1645 µs |
| SSD read (warm/cached) | ~0 µs | 1 µs | 3 µs | 16 µs | 66 µs |
| Host→GPU (pinned) | 5 µs | 7 µs | 15 µs | 54 µs | 171 µs |
| GPU→Host (pinned) | 5 µs | 7 µs | 14 µs | 55 µs | 174 µs |
| **E2E SSD→GPU (no overlap)** | **89 µs** | 140 µs | 317 µs | 723 µs | 1844 µs |

**Floors:** NVMe random-read floor ≈ **66–117 µs** (4 KB, device-bound). PCIe
upload floor ≈ **5 µs** (tiny transfers are API-overhead-bound; HtoD ≈ DtoH).
End-to-end floor for a tiny tensor ≈ **89 µs** — **dominated by the SSD, not the
GPU/PCIe.** You cannot get a cold SSD→GPU op below the device read latency.

### Latency vs throughput tradeoff (stream 256 MB, chunked+pinned+async)
| Chunk | Bandwidth | Per-item latency |
|---|---|---|
| 64 KB | 3.4 GB/s | 132 µs |
| 256 KB | 4.9 GB/s | 302 µs |
| **1 MB** | **6.2 GB/s** | **747 µs** ← knee |
| 4 MB | 6.2 GB/s | 1915 µs |

(Bandwidth here is indicative — small 256 MB total; full 7 GB/s needs ≥1 GB, see
`gpu_upload.ts`. The point is the *shape* of the tradeoff.)

### Findings
1. **The SSD is the latency floor, not the GPU.** A 4 KB cold SSD→GPU op takes
   ~89 µs, of which ~5 µs is the upload. NVMe random-read latency (~70–120 µs)
   sets the lower bound for any single small transfer.
2. **The sweet spot is 256 KB–1 MB.** At **1 MB** you hit ~max bandwidth
   (~6.2 GB/s ≈ 90% of the SSD ceiling) with **sub-ms latency (~0.75 ms)**.
   Going smaller buys negligible latency but loses bandwidth fast (64 KB → 3.4 GB/s).
   Going larger gives no more bandwidth but linearly more latency.
3. **Pinned memory matters for latency too**, not just throughput: async HtoD
   from pinned mem has a ~5 µs floor; from unpinned it degrades toward sync and
   adds overhead.
4. **Pre-open fds + pread.** `open()` per read is ~10–20 µs of pure avoidable
   latency; a persistent fd + positional `readSync`/`pread` removes it.
5. **No overlap for lowest latency.** The chunked pipeline that maximizes
   bandwidth queues items (each waits behind others). For *lowest single-op
   latency*, issue one small transfer with no queueing — accept the ~89 µs floor.

### Takeaway for bunker
- For **latency-critical** small tensor loads (e.g., a single KV-cache page,
  weight slice): pre-open the file fd, use a pre-registered pinned buffer, read
  ~4–64 KB and async-upload — **~90–140 µs end-to-end**, SSD-bound.
- For **bandwidth-critical** bulk loads: chunk at **~1 MB** with the pinned+async
  pipeline (`gpu_upload.ts` strategy G) — ~6–7 GB/s with ~0.75 ms per item.
- 1 MB chunks are the universal sweet spot: ~90% of max bandwidth at sub-ms
  latency. Drop to 64 KB only if you need the absolute lowest per-op latency and
  can afford the bandwidth hit.

## fa2_clean.ts — FlashAttention-2 forward ported from Triton's `06-fused-attention`

**Goal:** port the canonical clean Triton FA2 (`_attn_fwd_inner`) to the bunker
`TTIRBuilder`, as the first of two FA2 ports (this = clean reference; next =
vLLM's optimized `triton_unified_attention` with split-KV decode + GQA tiling).

```bash
bun run prototypes/fa2_clean.ts
```

**Result (RTX 3090, causal, f16 inputs / f32 softmax+acc):**
- Correctness: **max err 0.0002** vs a host reference (256×256×64).
- Throughput: **4.23 TFLOPS** (2048×2048×128, BM=128, BN=64, 8 warps) — a clean
  unoptimized baseline; the vLLM-style port will close the gap to the 3090's
  ~71 TFLOPS f16 peak.

**The port is a near line-for-line translation** of the Triton inner loop:
online softmax (running max `m_i` + sum `l_i`) carried as `scf.for` iter-args
alongside the accumulator `O` and the advancing K/V tiled pointers — the same
K-loop-with-iter-args pattern already used by the matmul kernel.

```ts
// core inner-loop body (per K/V tile), mirroring _attn_fwd_inner:
const qk  = bb.mul(bb.dot(q, bb.trans(bb.load(tpK))), scale);  // Q @ Kᵀ
const mask = bb.ge(rowBc, colBc);                               // causal
const qkM = bb.select(mask, qk, negInf);                       // where(mask, qk, -inf)
const m_ij = bb.maximum(m_i, bb.max(qkM, 1));                  // running max
const alpha = bb.exp(bb.sub(m_i, m_ij));                       // rescale
const p = bb.exp(bb.sub(qkM, m_ijBc));                         // softmax numerators
const acc = bb.dot(bb.fptrunc(p, "f16"), bb.load(tpV),         // acc = α·acc + p@V
                   bb.mul(acc, alphaBc));
```

Every op maps 1:1 to a Triton primitive the builder already exposes; **no
libdevice needed** (`exp` inlines; FA2 uses no `rsqrt`/`sin`/`cos`).

### Builder bugs found & fixed while porting (in `src/ttir.ts`)
These primitives had never been exercised by a passing test:
- `select`: emitted the 3-type custom form `arith.select … : i1, T, T` (rejected
  by this MLIR). Fixed to the generic form `"arith.select"(...) : (i1,T,T) -> T`.
- `maximum`/`minimum`: emitted the removed `arith.maxf`/`arith.minf`. Fixed to
  `arith.maximumf`/`arith.minimumf` (MLIR renamed these).
- `fptrunc`/`fpext`: emitted the removed `arith.fptrunc`/`arith.fpext`. Fixed to
  `arith.truncf`/`arith.extf`.
- Also fixed a stale hardcoded `cwd` path in `test_inline_compile.ts`.

### Next: the vLLM port
`prototypes/fa2_vllm.ts` (to do) will layer on the optimizations from IBM's
"Anatomy of a Triton Attention Kernel" / vLLM's `triton_unified_attention`:
GQA Q-block tiling, **split-KV parallel-tiled-softmax decode** (3D grid + a
reduction kernel), decoupled tile sizes, and heuristic-tuned configs — taking
the naive 19.7% → ~106% of FlashAttention-3.

## fa2_vllm.ts — optimized FA2 (multi-head GQA + two-stage causal split)

The optimized port, in the style of vLLM's `triton_unified_attention` / IBM's
"Anatomy of a Triton Attention Kernel", layered on the clean `fa2_clean.ts`.

```bash
bun run prototypes/fa2_vllm.ts
# env knobs for tuning: BM=64 BN=64 WARPS=4 STAGES=3  (the winning config)
```

**Result (RTX 3090, causal, 8 q-heads / 2 kv-heads GQA, D=128):**

| port | config | throughput | vs clean |
|---|---|---|---|
| fa2_clean (1 head) | BM=128 BN=64 W=8 | 4.23 TFLOPS | 1× |
| fa2_vllm (8h GQA) | BM=128 BN=64 W=4 | 10.6 TFLOPS | 2.5× |
| **fa2_vllm (8h GQA)** | **BM=64 BN=64 W=4 S=3** | **~29.5 TFLOPS** | **7×** |

~29.5 TFLOPS is **~42% of the 3090's 71 TFLOPS f16 peak** (max err 0.0002). A
hand-written port with no autotuner; real FA2 libs reach ~50–65% with full
autotuning + flash stage tricks.

### The three optimizations (and the one that mattered most)
1. **Multi-head GQA grid** `[num_q_blocks, num_q_heads]` — THE fix. The clean
   port's grid `[M/BM, 1]` launched only 16 programs on 82 SMs (severe
   under-occupancy — that alone capped it at ~4 TFLOPS). Gridding over the 8
   q-heads gives 8× more programs and saturates the SMs. Each q-head derives
   `kv_head = q_head // G` (correct GQA).
2. **Two-stage causal split** (FA2 work partitioning): STAGE 1 loops K-tiles
   `[0, m·BM)` **unmasked** (all-valid lower triangle — skips the ~half of
   tiles the clean port masked to -inf and processed for nothing); STAGE 2 does
   just the diagonal `[m·BM, (m+1)·BM)` with the mask. ~halves causal work.
3. **Tile/warp tuning**: BM=64/BN=64/4 warps/stages=3 beat BM=128 (smaller tiles
   → more programs → better occupancy on this problem size). `num_stages`
   pipelining was added to the builder (`build(name, warps, stages)`) but had
   little effect here — occupancy was the binding constraint, not pipelining.

### Not yet ported (the decode-oriented vLLM opts)
These matter most at `BLOCK_M=1` (decode), where prefill-style tiling
underutilizes the GPU — exactly Qwen3.5's steady-state inference regime:
- **GQA Q-block KV reuse**: load each KV head once per group of G q-heads (here
  each q-head program loads its own KV — correct, but reloads the same KV G×).
- **Split-KV parallel-tiled-softmax decode**: 3D grid
  `[seqs, kv_heads, segments]` + a reduction kernel — the signature vLLM decode
  optimization for memory-bound single-token attention.

### Builder addition
`TTIRBuilder.build(name, numWarps, numStages?)` now emits `ttg.num-stages` for
software pipelining (read by the shim's pass pipeline).

## fa2_autotune.ts — autotune FA2 over (BM, BN, warps, stages)

Applies the framework's CUDA-event autotuning pattern (`bench/autotune.ts`) to
the FA2 port — the framework picks the config instead of a hand-picked one, and
re-tunes per problem size / GPU.

```bash
bun run prototypes/fa2_autotune.ts
# env: Hq=8 Hkv=2 SEQ=2048 D=128   (search: BM,BN∈{32,64,128} × w∈{4,8} × s∈{2,3,4})
```

**Crucially, the autotuner filters by CORRECTNESS** (small-size check vs the
host reference), not just launch success. This matters: it caught that the
fastest-looking config (BM=32 BN=64, ~50 TFLOPS) was **fast-but-wrong**, and
rejected every `BN > BM` config:

```
BM=32 BN=64 w=4 s=2   incorrect ✗ (skip)   ← would have "won" at ~50 TFLOPS but is WRONG
BM=64 BN=128 ...       incorrect ✗ (skip)   ← all BN>BM configs are broken
...
BEST: BM=32 BN=32 w=4 s=4 → 39.20 TFLOPS   verify best config: max err 0.0003 ✓ correct
```

### Result (RTX 3090, 8h/2kv GQA, 2048×2048×128, CUDA-event timed)
| port / config | throughput | correct |
|---|---|---|
| fa2_clean (1 head) | 4.23 TFLOPS | ✓ |
| fa2_vllm hand-pick BM=64/64 | 29.5 TFLOPS | ✓ |
| **fa2_vllm autotuned BM=32/32 w=4 s=4** | **39.2 TFLOPS** | ✓ |

~39 TFLOPS = **55% of the 3090's 71 TFLOPS f16 peak**. The autotuner lifted the
hand-pick 29.5 → 39.2 (1.3×) by searching square small tiles, and — just as
important — refused to report the invalid 50 TFLOPS config.

### Findings from the sweep
- **`w=4` always beats `w=8`** here (8 warps over-subscribes these tile sizes).
- **Square or BN≤BM tiles** are required for correctness with the current
  two-stage diagonal mask (see bug below); `BN > BM` is broken.
- `num_stages` has only a marginal effect (~1–3%); occupancy and tile shape
  dominate.

### Known bug surfaced by the autotuner
When `BN > BM` (K-tile wider than the Q-block), the two-stage diagonal mask /
`tpK` advancement produces wrong output. The autotuner's correctness filter
makes this safe (it skips those configs), but the diagonal-stage masking should
be fixed to handle `BN > BM` (useful — wider K-tiles can be faster when correct).

### Update: BN>BM bug FIXED — unlocks wider K-tiles (39.2 → 47.5 TFLOPS)

The two-stage split had a real bug: stage 1 (unmasked) looped `[0, diagStart)`
in `BN`-sized tiles, but when `BN` didn't divide `diagStart` (= `pidM·BM`,
i.e. whenever `BN > BM`), the last stage-1 tile overshot into the diagonal
region and treated upper-triangle entries as valid. The original Triton tutorial
side-steps this by requiring `BM` to be a multiple of `BN`.

**Fix:** align stage 1's end down to a `BN` boundary and let the masked stage 2
cover the straddling tile:
```ts
const stage1End = b.mul(b.divi(diagStart, b.i32(BN)), b.i32(BN));  // floor-align
// stage 1 (unmasked): [0, stage1End)   stage 2 (masked): [stage1End, diagEnd)
```

Result: **all 54 configs are now correct** (zero skips), and the wider K-tile
is now both correct AND the fastest:

| config | before fix | after fix |
|---|---|---|
| BM=32 BN=32 w=4 (square) | 39.2 ✓ | 40.3 ✓ |
| **BM=32 BN=64 w=4 s=3** (wide K) | ~~50 TFLOPS but WRONG~~ | **47.5 TFLOPS ✓** |

**Autotuned best: BM=32 BN=64 w=4 s=3 → 47.5 TFLOPS** (max err 0.0002) =
**67% of the 3090's 71 TFLOPS f16 peak**. The bugfix lifted the honest best
39.2 → 47.5 (1.2×) by making the faster wide-K-tile config valid.

### Going faster: efficiency scales with sequence length (same kernel)

The autotune is maxed at ~48 TFLOPS for M=2048 (a 300-config search over
BM∈{16,32,48,64}, BN∈{16,32,48,64,128}, warps∈{2,4,8}, stages∈{2..6} found
nothing better than BM=32 BN=64 w=4). Further gains at fixed size need
*algorithmic* changes, not config — but FA2 efficiency climbs naturally with
sequence length (overhead amortizes, tiles saturate). Same kernel, same best
config, varying M (RTX 3090, 8h/2kv GQA, D=128):

| M | throughput | % of 71 TFLOPS f16 peak |
|---|---|---|
| 1024 | 36.2 TFLOPS | 51% |
| 2048 | 46.2 TFLOPS | 65% |
| 4096 | 52.9 TFLOPS | 74% |
| **8192** | **60.3 TFLOPS** | **85%** |

So at long contexts the port reaches **60 TFLOPS (85% of peak)**.

### The remaining gap is fundamental to FA2 on Ampere
The ~15% gap to peak is tensor cores sitting idle during the softmax/max/exp/
rescale between the QK and PV dots (those are CUDA-core ops). Closing it
requires **FA3-style warp specialization** (producer/consumer warps that
overlap softmax with matmul) — which needs **Hopper+ (TMA + wgmma async)** and
is **not available on the 3090 (sm_86, Ampere)**. So ~60 TFLOPS at long context
is near the FA2 ceiling on this GPU; FA3 would be the path on Hopper/Blackwell.

## rmsnorm.ts — RMSNorm (GemmaRMSNorm, Qwen3.5) at peak HBM bandwidth

Ported from the production-fast Liger-Kernel `_rms_norm_forward_kernel`
(the canonical fast Triton RMSNorm) + Triton's 05-layer-norm tutorial.
Qwen3.5 uses GemmaRMSNorm: `y = x · rsqrt(mean(x²) + ε) · (1 + weight)`, ε=1e-6.

```bash
bun run prototypes/rmsnorm.ts           # Newton rsqrt (no libdevice)
NATIVE=1 bun run prototypes/rmsnorm.ts  # try native math.rsqrt (needs shim rebuild)
```

**Result (RTX 3090, 512×1024):**
- correct: **max err 3.58e-7** vs host reference
- throughput: **935 GB/s effective memory bandwidth** — **~100% of the 3090's
  936 GB/s HBM peak** (RMSNorm is memory-bound; one program per row, full row
  in a 1D tile).

### The rsqrt question (libdevice)
RMSNorm's only non-elementwise op is `rsqrt`. Two paths:
- **Native `math.rsqrt`** → lowers to libdevice `__nv_rsqrtf` → **fails at ptxas**
  ("Unresolved extern `__nv_rsqrtf`") because the shim doesn't link libdevice.
  (`libdevice.10.bc` IS now present in the nix store, so a shim rebuild would
  unblock it — see `shim/build_shim.sh`.)
- **Newton-Raphson `rsqrt`** (`y ← y·(1.5 − 0.5·a·y²)`, 6 iters) — pure
  `mul`/`sub`, **no libdevice, no rebuild**. Used here. Since RMSNorm is
  memory-bound (not compute-bound on rsqrt), there is **zero performance cost**
  vs native — we still hit peak HBM bandwidth.

This Newton trick also unblocks `rsqrt` everywhere else without a shim rebuild:
GatedDeltaNet's L2-normalization of q/k will use the same approach.

### Builder note
`tt.load`'s `padding` attribute only allows `1` (zero-pad) or `2` (NaN), not `0`.

## swiglu.ts — SwiGLU activation (Qwen3.5 MLP)

`MLP(x) = down( silu(gate(x)) · up(x) )`, `silu(x)=x·σ(x)=x/(1+e^{-x})`. The
gate/up/down are GEMMs (existing matmul kernel); this is the fused elementwise
activation between them: `act = silu(gate) · up`. Pure elementwise, `exp` inlines
→ no libdevice.

- correct: max err 1.19e-7
- 766 GB/s effective (3-tensor: read gate+up, write out) ≈ 82% of HBM peak

## rope.ts — Rotary Position Embedding (Qwen3.5 full-attention)

`mrope_interleaved=true` (interleaved pairs), `partial_rotary=0.25`
(rotate first 64 of 256 head dims; the rest pass through), `rope_theta=1e7`.

```ts
// per pair k (elements 2k, 2k+1):
//   y[2k]   = x[2k]·cos - x[2k+1]·sin
//   y[2k+1] = x[2k+1]·cos + x[2k]·sin
//   y[j]    = x[j]  for j >= rotary_dim
```

**cos/sin are precomputed on the host** (exact, no libdevice) and passed as
`[M, pairs]` tables — the kernel just loads + rotates. Interleaved pairs are
formed via **pointer-tensor strided loads** (`splatPtr` + `addptr`): even offsets
`m·hd + 0,2,4,…` and odd offsets `m·hd + 1,3,5,…`.

- correct: max err 1.19e-7
- 355 GB/s at 512×256 (launch-overhead-bound at this tiny size; RoPE is
  memory-bound and scales toward HBM peak at larger M)

### Two bugs found & fixed
- **`makeTensorPtr` with `strides=[2]` does NOT strided-load** — it produces
  wrong (contiguous) results. Use `splatPtr` + `addptr` pointer-tensors for
  strided/gathered access instead.
- **Flat 1D tiles overrun row boundaries**: a `[M·hd]` tensor with a 256-wide
  tile for the 192-element non-rotary portion spilled into the next row
  (`boundaryCheck` on a flat tensor can't see row boundaries). Fixed by using a
  2D tiled pointer `[M, HEAD_DIM]` with block `[1, 256]` + `boundaryCheck[0,1]`.

## Qwen3.5 full-attention block — all pieces now exist

| piece | prototype | status |
|---|---|---|
| RMSNorm | rmsnorm.ts | ✅ ~100% HBM peak |
| RoPE (interleaved, partial 0.25) | rope.ts | ✅ correct |
| SwiGLU MLP activation | swiglu.ts | ✅ 82% HBM peak |
| Full attention (FA2) | fa2_vllm.ts | ✅ ~60 TFLOPS @ 8192 |

Next milestone: **assemble a complete full-attention block** end-to-end
(`RMSNorm → QKV-proj → RoPE → FA2 → O-proj → +residual → RMSNorm → gate/up →
SwiGLU → down → +residual`) and verify vs a reference, then the GatedDeltaNet
linear-attention layers (the remaining 75%).

## gdn_clean.ts — GatedDeltaNet (Qwen3.5 linear attention), clean recurrence

The **18/24 linear-attention layers are the core of Qwen3.5**. This is the
"clean first" port — the naive delta-rule recurrence (fla's
`delta_rule_recurrence`), proving the recurrence works in TTIR. Conv1d, the
decay gate, and the chunked-parallel prefill come next; this validates the core
algorithm and the `scf.for`-with-state-`S` pattern.

Per token (β folded on host as `kb=k·β`, `vb=v·β`; q scaled by 1/√d_k):
```
delta = vb − (kb·S)        # kb·S = Σ kb[:,None]·S  (reduce axis 0)  → [d_v]
S    += k ⊗ delta          # outer product                         → [d_k, d_v]
o     = q · S              # = Σ q[:,None]·S  (reduce axis 0)        → [d_v]
```
The state `S` `[d_k, d_v]` (128×128) is carried as a `scf.for` iter-arg — the
same pattern as the FA2/matmul K-loop. Pure elementwise + `tt.reduce` +
broadcast; no dot, no libdevice.

- correct: **max err 5.96e-8** vs the fla reference (BH=4, T=64, d_k=d_v=128)
- (Sequential O(T) loop — correct but slow; the chunked-parallel prefill is the
  fast version, to port next.)

### GatedDeltaNet remaining work
1. conv1d (depthwise causal, k=4) on qkv — pure mul/add, no libdevice.
2. decay gate `S *= exp(g)`, `g = −exp(A_log)·softplus(a+dt_bias)` — `exp`
   inlines; `softplus=log(1+exp)` needs `log` (libdevice) or a polynomial approx.
3. L2-norm of q/k — Newton-Raphson `rsqrt` (proven in rmsnorm.ts).
4. output RMSNormGated + out_proj.
5. **chunked-parallel prefill** (the fast version: WY representation + chunk
   recurrence — structurally the FA2/matmul K-loop with iter-arg `S`).

## gdn_gated.ts — GatedDeltaNet per-token step WITH the decay gate

Extends `gdn_clean` with the "gated" feature that defines GatedDeltaNet: the
per-token **decay gate** `S *= decay[t]`, where `decay[t] = exp(g[t])`,
`g[t] = -exp(A_log)·softplus(a[t]+dt_bias)`. `decay` (and the `softplus`/`log`
inside it) is **precomputed on the host** (exact, no libdevice — same trick as
RoPE's cos/sin) and passed as a `[BH*T]` buffer; the kernel only multiplies. β
stays folded on the host (`kb=k·β`, `vb=v·β`), so no per-token scalar-β in-kernel.

Per token (q scaled 1/√d_k):
```
S = S · decay              # decay[t] loaded as [1] → expandDims → broadcast [DK,DV]
delta = vb − (kb·S)        # reduce axis 0 → [DV]
S += k ⊗ delta             # outer
o = q · S                  # reduce axis 0 → [DV]
```
- correct: **max err 2.61e-8** vs reference (BH=4, T=64, d_k=d_v=128, decay ~0.95–0.99)

### Builder limitation hit: 0-d tensors
Reducing a 1-D vector (`sum(x, 0)` on `[DK]`) or `reshape` to `[]` produces a
0-d tensor, but the builder renders it as `f32` (scalar) instead of `tensor<f32>`
(`typeText` treats rank-0 as scalar). So L2-norm of a single vector in-kernel
breaks. **Workaround:** do L2-norm as a separate 2-D row-wise kernel (reduce
`[rows, DK]` axis 1 → `[rows]`, rank-1 — exactly how `rmsnorm.ts` works), like
fla does (L2norm outside the recurrent kernel). This is the remaining piece for
the per-token step; then the chunked-parallel prefill for speed.

## conv1d.ts — depthwise causal conv1d (k=4) + silu for GatedDeltaNet

Ported from fla's `causal_conv1d_fwd_kernel`. Qwen3.5's GatedDeltaNet applies a
depthwise causal conv1d (kernel=4) + silu to the qkv projection before
splitting into q/k/v:
```
y[t,c] = silu( Σ_{j=0..W-1} weight[c,j] · x[t-(W-1)+j, c] )     (x[n<0]=0)
```
Process a block of `BT` output positions at once, summing `W` shifted input
windows × per-channel tap weights. Per-channel tap weights (`weight` is `[D,W]`
row-major → column = stride `W`) loaded via strided pointer-tensors. `silu` via
`exp` (inlines) → no libdevice.

- correct: max err 4.47e-8 (B=4, T=128, D=512, W=4)
- 382 GB/s at this tiny size (launch-overhead-bound; memory-bound and scales up)

### Bug (same class as RoPE): flat-tile batch-boundary bleed
The shifted load with negative offset at a batch boundary read the **previous
batch's tail** instead of zero-padding (`boundaryCheck` on a flat `[B·T]` tensor
can't see batch rows). Fix: **host-pad each batch row with W−1 leading zeros** →
all offsets non-negative, no cross-batch bleed. (Same root cause as RoPE's
row-overrun; the general lesson: use 2-D tiled pointers with row dim, or
host-pad, whenever a flat tile would cross logical row boundaries.)

## GatedDeltaNet per-token step — now complete
| piece | prototype | status |
|---|---|---|
| delta-rule recurrence | gdn_clean.ts | ✅ |
| decay gate | gdn_gated.ts | ✅ |
| conv1d (k=4) + silu | conv1d.ts | ✅ |
| L2-norm of q/k | (2-D row-wise, = rmsnorm.ts pattern) | ✅ pattern proven |
| output RMSNormGated + out_proj | (rmsnorm + swiglu patterns) | ✅ patterns proven |

The remaining piece is the **chunked-parallel prefill** — the fast version that
makes prefill efficient (the naive recurrence is O(T) sequential). It's the WY
representation + chunk recurrence, structurally the FA2/matmul K-loop with `S`
as the iter-arg.

## References — ultra-optimized kernels each prototype was ported from

Every prototype is a port of a tested, production-fast Triton kernel (not
re-derived from scratch). The mapping from Triton op → `TTIRBuilder` is ~1:1
(see fa2_clean.ts header). Sources:

| prototype | ultra-optimized reference | notes |
|---|---|---|
| fa2_clean.ts | Triton tutorial `06-fused-attention.py` (`_attn_fwd_inner`) | the canonical clean FA2 |
| fa2_vllm.ts | vLLM `vllm/v1/attention/ops/triton_unified_attention.py` + IBM "Anatomy of a Triton Attention Kernel" (arXiv:2511.11581) | SoTA Triton attention (105.9% of FA3 on H100); default AMD backend in vLLM |
| rmsnorm.ts | Liger-Kernel `_rms_norm_forward_kernel` + Triton `05-layer-norm` | production-fast RMSNorm |
| swiglu.ts | Liger-Kernel `_swiglu_forward_kernel` | fused silu(gate)·up |
| rope.ts | Liger-Kernel `rope.py` | precomputed cos/sin buffers, rotate_half |
| gdn_clean.ts | fla `fla/ops/delta_rule/naive.py` (`delta_rule_recurrence`) | the clean delta-rule reference |
| gdn_gated.ts | fla `fla/ops/gated_delta_rule/fused_recurrent.py` + tiny-qwen `GatedDeltaNet` | the gated recurrent step |
| conv1d.ts | fla `fla/modules/conv/triton/kernels.py` (`causal_conv1d_fwd_kernel`) | the canonical Triton causal-conv1d (Dao-AILab/causal-conv1d is the CUDA ultra-opt, not portable to the TTIR path) |
| fa2_autotune.ts | the framework's own `bench/autotune.ts` pattern | CUDA-event autotuning + correctness filtering |

### The ultra-opt GatedDeltaNet targets still to port (for speed)
fla `fla/ops/gated_delta_rule/` contains the full production set:
- `chunk.py` / `chunk_fwd.py` + `wy_fast.py` — **chunked-parallel prefill** (WY
  representation + chunk recurrence; structurally the FA2/matmul K-loop with `S`
  as iter-arg). This is the fast prefill (the naive `gdn_*` recurrence is O(T)).
- `fused_recurrent.py` — the optimized recurrent decode step (with gate).
- `gate.py` — the decay-gate application.

## RoPE & conv1d optimized (f16, Liger/fla-style, benchmarked at scale)

Both upgraded from the first f32 cuts to **f16 in/out** (the model dtype → 2× less
bandwidth; f32 intermediate for precision) and **benchmarked at realistic scale**.

| kernel | dtype | size | throughput | % of 936 GB/s HBM peak |
|---|---|---|---|---|
| RoPE (Q+K, interleaved, partial 0.25) | f16 | 8192×(8+2)×256 | 616 GB/s | 66% |
| conv1d (depthwise causal k=4 + silu) | f16 | 8×512×6144 | 725 GB/s | 77% |

RoPE is lower (66%) because **interleaved rotation forces strided (stride-2) even/odd
loads** — less coalesced than Liger's contiguous rotate_half. Qwen3.5 mandates
`mrope_interleaved=true`, so this is an inherent cost (a contiguous-load + in-register
pairing rewrite could recover some). conv1d at 77% is near memory-bound.

### Bugs found while optimizing
- **`buildRoPE` bakes H into the tensor shapes** (`makeTensorPtr [M·H·HD]`), so one
  compiled kernel can't serve both Q (H=HQ) and K (H=HKV) — launching it for K with
  grid `[M, HKV]` indexed K as if it had HQ heads → **OOB → CUDA_ERROR_ILLEGAL_ADDRESS
  (rc 700)**. Fix: compile **two kernels** (one per H).
- **Cross-context CUDA events**: `getCuCtx`/`createCudaEvents` from `src/kernel` use
  kernel.ts's CUDA context, but the rope kernel is compiled on `src/ttir`'s context →
  `CUDA_ERROR_INVALID_HANDLE (rc 400)`. Fix: use host timing on the ttir context
  (`cuSync`), not kernel.ts events, for ttir-compiled kernels. (The FA2 autotuner
  worked because it used kernel.ts throughout.)
- The 0-d-tensor builder limitation (from gdn_gated) also blocks in-kernel L2-norm of
  a single vector — use a 2-D row-wise kernel instead.

## gdn_chunk.ts — chunked-parallel GatedDeltaNet prefill (clean port)

Ports fla's `naive_chunk_gated_delta_rule` / `delta_rule_chunkwise` — the
**chunked-parallel** form (parallel within chunks via matmuls + recurrence
across chunks carrying `S`), as opposed to the O(T) sequential `gdn_clean`.

**Key porting trick:** the reference computes (I−L)⁻¹ (L = strict-lower of
−(k_beta·kᵀ)) via row-by-row forward substitution (Python slicing). The tile
model has no dynamic slicing, so it uses the **finite Neumann series** — L is
strict-lower ⇒ nilpotent (L^C=0) ⇒ (I−L)⁻¹ = I + L + L² + … + L^{C−1}, a
`scf.for` accumulating matrix powers (pure `tt.dot`). The inter-chunk `S`
recurrence is the FA2/matmul `scf.for`-iter-arg pattern.

- **correct: max err 4.47e-8** vs the fla reference (BH=2, T=64, d_k=d_v=128, C=16)
  — the chunked algorithm IS correctly expressible in TTIR.

### Honest result: correct but NOT yet faster than the naive recurrence
| variant | BH=4, T=512, d=128 | |
|---|---|---|
| naive O(T) recurrence (gdn_clean) | 830 µs | |
| chunked (gdn_chunk, Neumann WY, C=16) | 5192 µs | **6× slower** |

The clean chunked port is **slower**, not faster. Two reasons:
1. **Neumann-series WY is O(C³) per chunk** — more work than the recurrence's
   matvecs. fla's real kernel uses the **efficient forward-substitution WY**
   (O(C²)), but that needs row slicing (not in the tile model).
2. **Per-(batch,head) grid = only 4 programs** → severe under-occupancy (4
   programs on 82 SMs). fla parallelizes over **state-dim blocks** (`S` tiled
   into `[BK,BV]`, grid `[B·H, NK, NV]`) for occupancy — a bigger restructure.

So the chunked **algorithm** is ported and verified, but achieving the actual
speedup (the whole point of chunking) needs the efficient WY + state-dim
tiling for occupancy — the sophistication in fla's tuned triton kernels. This
remains future work; the naive recurrence (`gdn_clean`/`gdn_gated`) is
currently the faster correct path.

## gdn_fast.ts — fla-faithful chunked GatedDeltaNet (state-tiled, WIP)

Ports fla's actual chunked structure (`gated_delta_rule/chunk.py` +
`common/chunk_h.py` + `chunk_o.py`) — state-tiled for occupancy, fixing
gdn_chunk's two slowness causes (Neumann O(C³) WY + per-(b,h) grid = 4 programs):

- **WY on the host** (efficient forward-substitution, no Neumann, no GPU
  slicing): `A=(I−L)⁻¹`, `w=A@k_beta`, `u=A@v_beta`.
- **State-dim tiling** (fla's pattern): `fwd_h` grid `[B·H, NV]`, `fwd_o` grid
  `[B·H, nChunks, NV]`.
- **Two GPU kernels** (fla's split): `fwd_h` (`v_new=u−w@h`; store `h[c]` before
  update; `h += kᵀ@v_new`) and `fwd_o` (`o = scale·(q@h + tril(q@kᵀ)@v_new)`).

### Status: runs (rc=0) but WIP — NOT yet correct, NOT yet faster
- ✅ **runs** (both kernels rc=0; the clean rewrite fixed the crash).
- ❌ **correctness**: mismatches the O(T) recurrence reference (~5× off) — a
  subtle WY/convention issue remains to pin down.
- ❌ **speed**: 3815 µs vs naive recurrence 830 µs — **slower** at this size
  (BH=8 → only 16 fwd_h programs; the chunked form's parallelism needs large
  batch×heads to pay off, plus the 2-kernel HBM round-trip for h/v_new).

### Bugs found (builder) while porting
- **Carrying 6 `scf.for` iter-args** (a tensor + 5 tiled pointers) crashes
  (async trap). Fix: carry only `h` (1 iter-arg) and recompute tiled pointers
  from the loop `iv` each iteration. (Carrying 2 worked; 6 didn't.)
- Async kernel traps surface as a *later* `cuDtoH` failure (rc from
  `cuLaunchKernel` is just the enqueue); check `cuSync`'s return for the real
  error.
- 3-D `make_tensor_ptr` loads differ from stores; prefer flat 2-D tiled
  pointers and reshape on the host side.

This remains the open GatedDeltaNet task: the fla-faithful structure is built and
runs, but matching the O(T) reference and beating the naive recurrence needs the
convention fix + larger batch/head counts. The naive recurrence
(`gdn_clean`/`gdn_gated`) is still the working correct path.
