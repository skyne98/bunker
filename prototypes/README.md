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
