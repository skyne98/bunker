// prototypes/latency.ts
//
// Prototype: lowest-latency path SSD ↔ host mem ↔ GPU, while still sustaining
// high bandwidth. The companion to ssd_saturate.ts / gpu_upload.ts, which
// maximized THROUGHPUT. Here we minimize LATENCY — the wall time for a SINGLE
// small tensor to make the trip — and then map the latency/throughput tradeoff
// so we can pick a sweet spot that keeps both low.
//
// Why latency and throughput fight each other:
//   - Throughput wants LARGE chunks + deep async queues (hides per-op overhead,
//     saturates the SSD/DMA) — but every queued item waits behind the others.
//   - Latency wants SMALL chunks, pre-opened fds, pinned memory, and few/no
//     per-op syncs — but small transfers waste bandwidth on per-op overhead.
//
// Techniques applied here to cut latency:
//   - Pre-open the file fd once (open() per read is ~10-20 µs of avoidable cost).
//   - pread via fs.readSync (synchronous; no event-loop scheduling jitter).
//   - Pinned host memory (cuMemHostRegister once, reused) — async HtoD from
//     pinned mem is non-blocking AND lower-overhead than sync/unpinned.
//   - Distinct random offsets for cold reads (each is a real device miss).
//
// Verified hardware (RTX 3090 dev box):
//   SSD Lexar NM790 4TB (Gen4x4) · Ryzen 9 9950X3D · RTX 3090 (24 GB)
//
// Requires passwordless sudo for `echo 3 > /proc/sys/vm/drop_caches` (cold reads).
// Reuses the 1 GB test file from prototypes/ssd_saturate.ts.
//
// Usage:
//   bun run prototypes/ssd_saturate.ts   # create /tmp/bunker-diskbench/big_0.bin
//   bun run prototypes/latency.ts

import { dlopen, ptr as ffiPtr } from "bun:ffi";
import { execSync } from "child_process";
import { openSync, readSync, closeSync } from "node:fs";

const DATA = process.env.DATA_FILE ?? "/tmp/bunker-diskbench/big_0.bin";
const FILESIZE = Number(Bun.file(DATA).size);
const drop = () => execSync("sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches'");

const CU = dlopen("/run/opengl-driver/lib/libcuda.so", {
  cuInit:               { args: ["u32"], returns: "i32" },
  cuDeviceGet:          { args: ["ptr", "i32"], returns: "i32" },
  cuCtxCreate_v2:       { args: ["ptr", "u32", "i64"], returns: "i32" },
  cuMemAlloc_v2:        { args: ["ptr", "i64"], returns: "i32" },
  cuMemFree_v2:         { args: ["i64"], returns: "i32" },
  cuStreamCreate:       { args: ["ptr", "u32"], returns: "i32" },
  cuStreamSynchronize:  { args: ["i64"], returns: "i32" },
  cuCtxSynchronize:     { args: [], returns: "i32" },
  cuMemcpyHtoD_v2:      { args: ["i64", "ptr", "i64"], returns: "i32" },
  cuMemcpyHtoDAsync_v2: { args: ["i64", "ptr", "i64", "i64"], returns: "i32" },
  cuMemcpyDtoH_v2:      { args: ["ptr", "i64", "i64"], returns: "i32" },
  cuMemcpyDtoHAsync_v2: { args: ["ptr", "i64", "i64", "i64"], returns: "i32" },
  cuMemHostRegister:    { args: ["ptr", "u64", "u32"], returns: "i32" },
  cuMemHostUnregister:  { args: ["ptr"], returns: "i32" },
}).symbols;
function rc(n: string, r: number) { if (r !== 0) throw new Error(`${n} rc=${r}`); }

rc("cuInit", CU.cuInit(0));
const dev = new Int32Array(1); rc("cuDeviceGet", CU.cuDeviceGet(dev, 0));
const ctx = Buffer.alloc(8); rc("cuCtxCreate", CU.cuCtxCreate_v2(ctx, 0, BigInt(dev[0])));
const sbuf = Buffer.alloc(8); rc("cuStreamCreate", CU.cuStreamCreate(sbuf, 0));
const STREAM = sbuf.readBigUInt64LE(0);
const dbuf = Buffer.alloc(8); rc("cuMemAlloc", CU.cuMemAlloc_v2(dbuf, BigInt(FILESIZE)));
const DEV = dbuf.readBigUInt64LE(0);

// a pinned host buffer per size, registered once and reused
function makePinned(size: number) {
  const b = Buffer.allocUnsafeSlow(size);
  rc("cuMemHostRegister", CU.cuMemHostRegister(ffiPtr(b), BigInt(size), 0));
  return b;
}

// stats helper
function stats(us: number[]) {
  const s = [...us].sort((a, b) => a - b);
  const pct = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { min: s[0], p50: pct(0.5), p99: pct(0.99) };
}
function rnd(n: number) { return Math.floor(Math.random() * n); }

const SIZES = [4 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024];
const K = (n: number) => n >= 1048576 ? `${n / 1048576}MB` : `${n / 1024}KB`;

// ─── 1. SSD read latency (cold random vs warm cached) ──────────────────
// cold = distinct random offsets (device miss); warm = same offset (page cache).
{
  const fd = openSync(DATA, "r");
  console.log("\n1. SSD read latency (pread into pinned buf):");
  console.log("   size     cold-min  cold-p50  cold-p99   warm-p50");
  for (const S of SIZES) {
    const buf = makePinned(S);
    const blocks = Math.floor(FILESIZE / S);
    const N = 100;
    // cold: drop cache once, read N distinct offsets
    drop();
    const cold: number[] = [];
    const seen = new Set<number>();
    for (let i = 0; i < N; i++) {
      let bi: number;
      do { bi = rnd(blocks); } while (seen.has(bi));
      seen.add(bi);
      const t0 = performance.now();
      readSync(fd, buf, 0, S, bi * S);
      cold.push((performance.now() - t0) * 1000);
    }
    // warm: same offset repeatedly (cached)
    const warm: number[] = [];
    for (let i = 0; i < N; i++) {
      const t0 = performance.now();
      readSync(fd, buf, 0, S, 0);
      warm.push((performance.now() - t0) * 1000);
    }
    const c = stats(cold), w = stats(warm);
    console.log(`   ${K(S).padEnd(6)}  ${c.min.toFixed(0).padStart(6)}µs  ${c.p50.toFixed(0).padStart(6)}µs  ${c.p99.toFixed(0).padStart(6)}µs   ${w.p50.toFixed(0).padStart(5)}µs`);
    CU.cuMemHostUnregister(ffiPtr(buf));
  }
  closeSync(fd);
}

// ─── 2. Host → GPU latency (sync vs async-pinned, per-op sync) ─────────
{
  console.log("\n2. Host→GPU latency (pinned, single op, per-op sync):");
  console.log("   size     sync-p50   async-p50  (asyncEnq+streamSync)");
  for (const S of SIZES) {
    const buf = makePinned(S);
    const N = 200;
    const sync: number[] = [];
    for (let i = 0; i < N; i++) {
      const t0 = performance.now();
      CU.cuMemcpyHtoD_v2(DEV, ffiPtr(buf), BigInt(S));
      sync.push((performance.now() - t0) * 1000);
    }
    const asyn: number[] = [];
    for (let i = 0; i < N; i++) {
      const t0 = performance.now();
      CU.cuMemcpyHtoDAsync_v2(DEV, ffiPtr(buf), BigInt(S), STREAM);
      CU.cuStreamSynchronize(STREAM);
      asyn.push((performance.now() - t0) * 1000);
    }
    const s = stats(sync), a = stats(asyn);
    console.log(`   ${K(S).padEnd(6)}  ${s.p50.toFixed(1).padStart(7)}µs  ${a.p50.toFixed(1).padStart(8)}µs`);
    CU.cuMemHostUnregister(ffiPtr(buf));
  }
}

// ─── 3. GPU → Host latency (pinned, async) ────────────────────────────
{
  console.log("\n3. GPU→Host latency (pinned, async + sync):");
  console.log("   size     async-p50");
  for (const S of SIZES) {
    const buf = makePinned(S);
    const N = 200;
    const asyn: number[] = [];
    for (let i = 0; i < N; i++) {
      const t0 = performance.now();
      CU.cuMemcpyDtoHAsync_v2(ffiPtr(buf), DEV, BigInt(S), STREAM);
      CU.cuStreamSynchronize(STREAM);
      asyn.push((performance.now() - t0) * 1000);
    }
    const a = stats(asyn);
    console.log(`   ${K(S).padEnd(6)}  ${a.p50.toFixed(1).padStart(7)}µs`);
    CU.cuMemHostUnregister(ffiPtr(buf));
  }
}

// ─── 4. End-to-end single-op: SSD(cold) → host → GPU ───────────────────
// One tensor, no overlap — the pure one-shot latency of the full path.
{
  const fd = openSync(DATA, "r");
  console.log("\n4. End-to-end single-op SSD→GPU latency (no overlap):");
  console.log("   size     e2e-p50   (read + asyncUpload + sync)");
  for (const S of SIZES) {
    const buf = makePinned(S);
    const blocks = Math.floor(FILESIZE / S);
    const N = 100;
    drop();
    const e2e: number[] = [];
    const seen = new Set<number>();
    for (let i = 0; i < N; i++) {
      let bi: number;
      do { bi = rnd(blocks); } while (seen.has(bi));
      seen.add(bi);
      const t0 = performance.now();
      readSync(fd, buf, 0, S, bi * S);                       // SSD → mem
      CU.cuMemcpyHtoDAsync_v2(DEV, ffiPtr(buf), BigInt(S), STREAM); // mem → GPU
      CU.cuStreamSynchronize(STREAM);
      e2e.push((performance.now() - t0) * 1000);
    }
    const e = stats(e2e);
    console.log(`   ${K(S).padEnd(6)}  ${e.p50.toFixed(0).padStart(6)}µs`);
    CU.cuMemHostUnregister(ffiPtr(buf));
  }
  closeSync(fd);
}

// ─── 5. Latency vs throughput tradeoff (the sweet spot) ───────────────
// Stream a BOUNDED total (256 MB) through the chunked+pinned+async pipeline
// at a few chunk sizes. Bounded chunk count keeps it fast (one pinned buffer
// per chunk, capped). Per-item latency comes from an isolated cold probe.
// Bandwidth here is indicative (small total); for full-bandwidth numbers see
// gpu_upload.ts. The point is the SHAPE of the latency/throughput tradeoff.
{
  console.log("\n5. Latency vs throughput (stream 256 MB, chunked+pinned+async):");
  console.log("   chunk    bandwidth   per-item e2e latency");
  const TOTAL = 256 * 1024 * 1024;
  for (const CHUNK of [64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]) {
    const nChunks = Math.floor(TOTAL / CHUNK);
    // pinned pool, one buffer per chunk (bounded by TOTAL/CHUNK)
    const bufs = Array.from({ length: nChunks }, () => {
      const b = Buffer.allocUnsafeSlow(CHUNK);
      rc("pin", CU.cuMemHostRegister(ffiPtr(b), BigInt(CHUNK), 0));
      return b;
    });
    drop();
    let next = 0;
    const CONC = 8;
    const worker = async () => {
      while (true) {
        const idx = next++;
        if (idx >= nChunks) break;
        const ab = await Bun.file(DATA).slice(idx * CHUNK, idx * CHUNK + CHUNK).arrayBuffer();
        bufs[idx].set(new Uint8Array(ab));            // copy into pinned buf
        CU.cuMemcpyHtoDAsync_v2(DEV + BigInt(idx * CHUNK), ffiPtr(bufs[idx]), BigInt(CHUNK), STREAM);
      }
    };
    const t0 = performance.now();
    await Promise.all(Array.from({ length: CONC }, () => worker()));
    CU.cuStreamSynchronize(STREAM);
    const dt = (performance.now() - t0) / 1000;
    const bw = TOTAL / 1e9 / dt;
    // isolated cold single-op e2e latency for this chunk size
    drop();
    const probe = bufs[0];
    const samples: number[] = [];
    const seen = new Set<number>();
    const fd = openSync(DATA, "r");
    for (let i = 0; i < 30; i++) {
      let bi: number;
      do { bi = rnd(Math.floor(FILESIZE / CHUNK)); } while (seen.has(bi));
      seen.add(bi);
      const a = performance.now();
      readSync(fd, probe, 0, CHUNK, bi * CHUNK);
      CU.cuMemcpyHtoDAsync_v2(DEV, ffiPtr(probe), BigInt(CHUNK), STREAM);
      CU.cuStreamSynchronize(STREAM);
      samples.push((performance.now() - a) * 1000);
    }
    closeSync(fd);
    const lat = stats(samples).p50;
    console.log(`   ${K(CHUNK).padEnd(6)}  ${bw.toFixed(1).padStart(5)} GB/s   ${lat.toFixed(0).padStart(6)}µs`);
    for (const b of bufs) CU.cuMemHostUnregister(ffiPtr(b));
  }
}

rc("free", CU.cuMemFree_v2(DEV));
