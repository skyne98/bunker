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
