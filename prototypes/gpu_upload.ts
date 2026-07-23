// prototypes/gpu_upload.ts
//
// Prototype: can we push SSD data into GPU memory at the SSD's maximum rate?
//
// The SSD delivers ~7 GB/s (see prototypes/ssd_saturate.ts). PCIe Gen4 x16
// host→device copies deliver ~20-25 GB/s — faster than the SSD. So in PRINCIPLE
// the SSD should remain the bottleneck if we overlap the two. The question is
// whether a Bun program can actually achieve that overlap:
//
//   read file i (async, I/O pool)  ‖  cuMemcpyHtoDAsync(file i-1) (GPU DMA)
//
// We compare several strategies against the SSD-only ceiling. The
// theoretically-optimal pipeline needs BOTH ingredients:
//   1. chunked/streamed reads  — so completions stagger and uploads can run
//      throughout the read phase (concurrent whole-file reads finish clustered,
//      leaving no window to overlap).
//   2. PINNED host memory      — cuMemHostRegister, else cuMemcpyHtoDAsync from
//      unpinned Bun buffers degrades to synchronous (blocks the JS thread).
//
//   A. HtoD ceiling      — RAM → GPU in a tight loop (no disk). The PCIe wall.
//   B. Serial            — per file: await read, then sync HtoD. No overlap.
//   C. Pipelined (async) — concurrent reads; enqueue async HtoD as each lands.
//   D. Pinned + node:fs read into pinned buf + async upload.
//   E. Bun read + register-pin + async upload.
//   F. Chunked stream pipeline (no pin).
//   G. Chunked + pinned pipeline (optimal overlap).
//
// Verified hardware (RTX 3090 dev box):
//   SSD Lexar NM790 4TB (Gen4x4, ~7400 MB/s) · Ryzen 9 9950X3D · RTX 3090 (24 GB)
//
// Requires passwordless sudo for `echo 3 > /proc/sys/vm/drop_caches` (cold reads).
// Reuses test files at /tmp/bunker-diskbench (build via prototypes/ssd_saturate.ts).
//
// Usage:
//   bun run prototypes/ssd_saturate.ts   # first, to create the 8x1GB test files
//   bun run prototypes/gpu_upload.ts

import { dlopen, ptr as ffiPtr } from "bun:ffi";
import { execSync } from "child_process";
import { open as fsOpen } from "node:fs/promises";

const DATA_DIR = process.env.DATA_DIR ?? "/tmp/bunker-diskbench";
const FILES = Array.from({ length: Number(process.env.FILE_COUNT ?? 8) },
  (_, i) => `${DATA_DIR}/big_${i}.bin`);

const CU = dlopen("/run/opengl-driver/lib/libcuda.so", {
  cuInit:               { args: ["u32"], returns: "i32" },
  cuDeviceGet:          { args: ["ptr", "i32"], returns: "i32" },
  cuCtxCreate_v2:       { args: ["ptr", "u32", "i64"], returns: "i32" },
  cuMemAlloc_v2:        { args: ["ptr", "i64"], returns: "i32" },
  cuMemFree_v2:         { args: ["i64"], returns: "i32" },
  cuStreamCreate:       { args: ["ptr", "u32"], returns: "i32" },
  cuStreamSynchronize:  { args: ["i64"], returns: "i32" },
  cuCtxSynchronize:     { args: [], returns: "i32" },
  cuMemcpyHtoD_v2:      { args: ["i64", "ptr", "i64"], returns: "i32" },        // sync
  cuMemcpyHtoDAsync_v2: { args: ["i64", "ptr", "i64", "i64"], returns: "i32" },  // async + stream
  cuMemHostRegister:     { args: ["ptr", "u64", "u32"], returns: "i32" },   // pin host memory
  cuMemHostUnregister:   { args: ["ptr"], returns: "i32" },
}).symbols;

function rc(name: string, r: number) { if (r !== 0) throw new Error(`${name} failed rc=${r}`); }

// --- init CUDA context + a stream ---
rc("cuInit", CU.cuInit(0));
const dev = new Int32Array(1); rc("cuDeviceGet", CU.cuDeviceGet(dev, 0));
const ctx = Buffer.alloc(8); rc("cuCtxCreate", CU.cuCtxCreate_v2(ctx, 0, BigInt(dev[0])));
const streamBuf = Buffer.alloc(8); rc("cuStreamCreate", CU.cuStreamCreate(streamBuf, 0));
const STREAM = streamBuf.readBigUInt64LE(0);

const sizes = await Promise.all(FILES.map(f => Bun.file(f).size));
const TOTAL = sizes.reduce((a, b) => a + b, 0);
console.log(`files=${FILES.length} total=${(TOTAL/1e9).toFixed(2)} GB  GPU=RTX 3090`);

// one big device buffer; each file uploaded to successive offsets
const devBuf = Buffer.alloc(8);
rc("cuMemAlloc", CU.cuMemAlloc_v2(devBuf, BigInt(TOTAL)));
const DEV = devBuf.readBigUInt64LE(0);

function dropCaches() { execSync("sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches'"); }
function fmt(dt_s: number) { return (TOTAL / 1e9 / dt_s).toFixed(0) + " GB/s"; }

// ─── A. HtoD ceiling: RAM → GPU, no disk ──────────────────────────────
{
  const host = Buffer.alloc(Number(sizes[0]));           // 1 GB zero buffer in RAM
  const hostPtr = ffiPtr(host);
  const iters = FILES.length;                             // upload iters × 1GB = TOTAL
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    const off = BigInt(i * Number(sizes[0]));
    rc("HtoD", CU.cuMemcpyHtoD_v2(DEV + off, hostPtr, BigInt(sizes[0])));
  }
  rc("sync", CU.cuCtxSynchronize());
  const dt = (performance.now() - t0) / 1000;
  console.log(`A. HtoD ceiling (RAM→GPU, sync) : ${dt.toFixed(3)}s  ${fmt(dt)}`);
}

// ─── B. Serial: per file await read → sync HtoD (no overlap) ──────────
{
  dropCaches();
  const t0 = performance.now();
  for (let i = 0; i < FILES.length; i++) {
    const ab = await Bun.file(FILES[i]).arrayBuffer();
    const buf = Buffer.from(ab);
    const off = BigInt(i * Number(sizes[i]));
    rc("HtoD", CU.cuMemcpyHtoD_v2(DEV + off, ffiPtr(buf), BigInt(sizes[i])));
  }
  rc("sync", CU.cuCtxSynchronize());
  const dt = (performance.now() - t0) / 1000;
  console.log(`B. Serial read+upload (no overlap): ${dt.toFixed(3)}s  ${fmt(dt)}`);
}

// ─── C. Pipelined: concurrent reads ‖ async HtoD, one sync at end ─────
{
  dropCaches();
  const t0 = performance.now();
  // concurrent reads; as each resolves, enqueue an ASYNC copy (returns fast),
  // letting the remaining reads continue on the I/O pool while the GPU DMAs.
  await Promise.all(FILES.map(async (f, i) => {
    const ab = await Bun.file(f).arrayBuffer();
    const buf = Buffer.from(ab);
    const off = BigInt(i * Number(sizes[i]));
    rc("HtoDAsync", CU.cuMemcpyHtoDAsync_v2(DEV + off, ffiPtr(buf), BigInt(sizes[i]), STREAM));
  }));
  rc("streamSync", CU.cuStreamSynchronize(STREAM));
  const dt = (performance.now() - t0) / 1000;
  console.log(`C. Pipelined read ‖ async upload : ${dt.toFixed(3)}s  ${fmt(dt)}`);
}

// ─── D. Pipelined + PINNED host memory (true async overlap) ───────────
// cuMemcpyHtoDAsync from UNPINNED memory degrades to synchronous (CUDA pins
// and copies inline, blocking the JS thread). cuMemHostRegister page-locks
// the buffer so async copies are genuinely non-blocking, letting the SSD
// reads run fully concurrently with the GPU DMA. We read each file directly
// into its pinned buffer (node:fs FileHandle.read supports a dest Buffer).
{
  const bufs = FILES.map(() => Buffer.allocUnsafe(Number(sizes[0])));
  const registered: Buffer[] = [];
  let ok = true;
  for (const b of bufs) {
    const r = CU.cuMemHostRegister(ffiPtr(b), BigInt(b.byteLength), 0);
    if (r !== 0) { console.log(`D. cuMemHostRegister failed rc=${r}; skipping`); ok = false; break; }
    registered.push(b);
  }
  if (ok) {
    dropCaches();
    const t0 = performance.now();
    await Promise.all(FILES.map(async (f, i) => {
      const fh = await fsOpen(f, "r");
      await fh.read(bufs[i], 0, Number(sizes[i]), 0);   // read directly into pinned buf
      await fh.close();
      const off = BigInt(i * Number(sizes[i]));
      rc("HtoDAsync", CU.cuMemcpyHtoDAsync_v2(DEV + off, ffiPtr(bufs[i]), BigInt(sizes[i]), STREAM));
    }));
    rc("streamSync", CU.cuStreamSynchronize(STREAM));
    const dt = (performance.now() - t0) / 1000;
    console.log(`D. Pinned read ‖ async upload  : ${dt.toFixed(3)}s  ${fmt(dt)}`);
    for (const b of registered) rc("unregister", CU.cuMemHostUnregister(ffiPtr(b)));
  }
}

// ─── E. Bun.file read + register-pin + async upload (best of both) ───
// Read with Bun's fast native arrayBuffer() (the 7 GB/s path), then pin the
// resulting buffer with cuMemHostRegister so the async HtoD is genuinely
// non-blocking. Buffers stay alive (pinned) until the final stream sync.
{
  const live: Buffer[] = [];
  let ok = true;
  dropCaches();
  const t0 = performance.now();
  await Promise.all(FILES.map(async (f, i) => {
    const ab = await Bun.file(f).arrayBuffer();
    const buf = Buffer.from(ab);
    const r = CU.cuMemHostRegister(ffiPtr(buf), BigInt(buf.byteLength), 0);
    if (r !== 0) { if (ok) { console.log(`E. cuMemHostRegister failed rc=${r}`); ok = false; } return; }
    live.push(buf);
    const off = BigInt(i * Number(sizes[i]));
    rc("HtoDAsync", CU.cuMemcpyHtoDAsync_v2(DEV + off, ffiPtr(buf), BigInt(sizes[i]), STREAM));
  }));
  rc("streamSync", CU.cuStreamSynchronize(STREAM));
  const dt = (performance.now() - t0) / 1000;
  if (ok) {
    console.log(`E. Bun read + pin + async upload: ${dt.toFixed(3)}s  ${fmt(dt)}`);
    for (const b of live) rc("unregister", CU.cuMemHostUnregister(ffiPtr(b)));
  }
}

// ─── F. Chunked streaming pipeline (definitive overlap) ─────────────
// The problem with C/D/E: 8 equal-size concurrent reads finish clustered, so
// async copies only enqueue AFTER the read phase — no overlap. Fix: read each
// file in small slices (Bun.file.slice) with bounded concurrency and enqueue
// an async HtoD per slice as it lands. Slices complete staggered, so uploads
// run continuously throughout the read phase — hiding the 0.39s copy budget
// behind the ~1.2s read budget.
{
  const CHUNK = 64 * 1024 * 1024;          // 64 MB slices
  // build (fileIdx, offset, length, devOff) slice descriptors
  const slices: { f: number; off: number; len: number; devOff: bigint }[] = [];
  for (let i = 0; i < FILES.length; i++) {
    const sz = Number(sizes[i]);
    for (let o = 0; o < sz; o += CHUNK) {
      const len = Math.min(CHUNK, sz - o);
      slices.push({ f: i, off: o, len, devOff: BigInt(i * sz + o) });
    }
  }
  const live: Buffer[] = [];
  const CONC = 8;
  let next = 0;
  const worker = async () => {
    while (true) {
      const idx = next++; if (idx >= slices.length) break;
      const s = slices[idx];
      const ab = await Bun.file(FILES[s.f]).slice(s.off, s.off + s.len).arrayBuffer();
      const buf = Buffer.from(ab);
      live.push(buf);
      rc("HtoDAsync", CU.cuMemcpyHtoDAsync_v2(DEV + s.devOff, ffiPtr(buf), BigInt(s.len), STREAM));
    }
  };
  dropCaches();
  const t0 = performance.now();
  await Promise.all(Array.from({ length: CONC }, () => worker()));
  rc("streamSync", CU.cuStreamSynchronize(STREAM));
  const dt = (performance.now() - t0) / 1000;
  console.log(`F. Chunked stream pipeline     : ${dt.toFixed(3)}s  ${fmt(dt)}  (${slices.length} x ${CHUNK/1048576}MB slices, concur=${CONC})`);
}

// ─── G. Chunked streaming + pinned slices (optimal overlap) ──────────
// Combines F's staggered slice completion with D's pinned-memory async copies:
// read each slice, cuMemHostRegister it (truly non-blocking async HtoD), keep
// it alive until the final sync, then unregister. This is the textbook
// double-buffered DMA pipeline.
{
  const CHUNK = 64 * 1024 * 1024;
  const slices: { f: number; off: number; len: number; devOff: bigint }[] = [];
  for (let i = 0; i < FILES.length; i++) {
    const sz = Number(sizes[i]);
    for (let o = 0; o < sz; o += CHUNK) {
      const len = Math.min(CHUNK, sz - o);
      slices.push({ f: i, off: o, len, devOff: BigInt(i * sz + o) });
    }
  }
  const live: Buffer[] = [];
  const CONC = 8;
  let next = 0;
  const worker = async () => {
    while (true) {
      const idx = next++; if (idx >= slices.length) break;
      const s = slices[idx];
      const ab = await Bun.file(FILES[s.f]).slice(s.off, s.off + s.len).arrayBuffer();
      const buf = Buffer.from(ab);
      rc("pin", CU.cuMemHostRegister(ffiPtr(buf), BigInt(buf.byteLength), 0));
      live.push(buf);
      rc("HtoDAsync", CU.cuMemcpyHtoDAsync_v2(DEV + s.devOff, ffiPtr(buf), BigInt(s.len), STREAM));
    }
  };
  dropCaches();
  const t0 = performance.now();
  await Promise.all(Array.from({ length: CONC }, () => worker()));
  rc("streamSync", CU.cuStreamSynchronize(STREAM));
  const dt = (performance.now() - t0) / 1000;
  console.log(`G. Chunked + pinned pipeline    : ${dt.toFixed(3)}s  ${fmt(dt)}  (${slices.length} slices, concur=${CONC})`);
  for (const b of live) { CU.cuMemHostUnregister(ffiPtr(b)); }
}

rc("free", CU.cuMemFree_v2(DEV));
console.log(`\nSSD-only ceiling (ref)         : ~7100 MB/s  (~7 GB/s)`);
