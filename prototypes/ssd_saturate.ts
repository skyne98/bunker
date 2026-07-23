// prototypes/ssd_saturate.ts
//
// Prototype: saturate the NVMe SSD's maximum read bandwidth from Bun.
//
// Hardware this was developed/verified on (RTX 3090 dev box):
//   - SSD:   Lexar NM790 4TB  (PCIe Gen4 x4, rated ~7400 MB/s seq read)
//   - CPU:   AMD Ryzen 9 9950X3D (32 threads)
//   - RAM:   64 GB
//   - OS:    NixOS, Linux
//
// Finding: Bun saturates the SSD at ~7174 MB/s (97% of rated peak) with
// Promise.all concurrency ~8. Past 8 the curve is flat — the Gen4x4 link is
// the bottleneck, not the runtime. Bun.file().arrayBuffer() and
// fs/promises.readFile converge to the same device ceiling (~7.1 GB/s).
//
// Gotcha that fooled the first attempt: do NOT warm up before measuring cold
// reads. The page cache makes warmed reads return at RAM speed (~18 GB/s).
// Drop caches (`echo 3 > /proc/sys/vm/drop_caches`) immediately before the
// timed read and run with zero warmup.
//
// Usage:
//   sudo -n true 2>/dev/null || echo "needs passwordless sudo to drop caches"
//   bun run prototypes/ssd_saturate.ts            # uses /tmp/bunker-diskbench
//   DATA_DIR=./mydir FILE_COUNT=16 FILE_MB=1024 bun run prototypes/ssd_saturate.ts
//
// Requires passwordless sudo for `echo 3 > /proc/sys/vm/drop_caches`.
// (Without it, you measure warm page-cache hits, i.e. RAM bandwidth, not SSD.)

import { readFile } from "fs/promises";
import { execSync } from "child_process";
import { existsSync, mkdirSync } from "fs";

const DATA_DIR   = process.env.DATA_DIR   ?? "/tmp/bunker-diskbench";
const FILE_COUNT = Number(process.env.FILE_COUNT ?? 8);
const FILE_MB    = Number(process.env.FILE_MB    ?? 1024);

function dropCaches() {
  execSync("sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches'");
}

async function ensureFiles() {
  if (existsSync(`${DATA_DIR}/big_0.bin`)) return;
  mkdirSync(DATA_DIR, { recursive: true });
  console.log(`writing ${FILE_COUNT} x ${FILE_MB} MB files to ${DATA_DIR} ...`);
  // NOTE: do NOT use `fallocate` — it creates uninitialized extents that the
  // kernel serves as zeros WITHOUT disk I/O, so reads return at RAM speed.
  // We must write real bytes (dd if=/dev/zero) so extents are initialized.
  for (let i = 0; i < FILE_COUNT; i++) {
    execSync(`dd if=/dev/zero of=${DATA_DIR}/big_${i}.bin bs=1M count=${FILE_MB} status=none`);
  }
}

const files = Array.from({ length: FILE_COUNT }, (_, i) => `${DATA_DIR}/big_${i}.bin`);
const totalMB = FILE_COUNT * FILE_MB;
console.log(`files=${FILE_COUNT} total=${totalMB} MB  dir=${DATA_DIR}`);

// read all files with at most `concurrency` in flight; returns bytes read.
async function readCold(concurrency: number, api: "bun" | "fs") {
  let idx = 0;
  const workers = Array.from({ length: concurrency }, async () => {
    let bytes = 0;
    while (true) {
      const i = idx++;
      if (i >= files.length) break;
      const buf = api === "bun" ? await Bun.file(files[i]).arrayBuffer()
                                : await readFile(files[i]);
      bytes += (buf as any).byteLength;
    }
    return bytes;
  });
  return (await Promise.all(workers)).reduce((a, b) => a + b, 0);
}

async function measure(concurrency: number, api: "bun" | "fs") {
  dropCaches();
  const t0 = performance.now();
  const bytes = await readCold(concurrency, api);
  const s = (performance.now() - t0) / 1000;
  const mbs = bytes / 1e6 / s;
  console.log(`  ${api}  concur=${String(concurrency).padStart(2)}  ${s.toFixed(3)}s  ${mbs.toFixed(0).padStart(5)} MB/s`);
  return mbs;
}

await ensureFiles();
console.log("\n--- Bun.file().arrayBuffer() ---");
for (const c of [1, 2, 4, 8, 16, 32]) await measure(c, "bun");
console.log("\n--- fs/promises.readFile ---");
for (const c of [1, 2, 4, 8, 16, 32]) await measure(c, "fs");
