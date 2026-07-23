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
