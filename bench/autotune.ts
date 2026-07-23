// autotune.ts — CUDA-event-based autotuner for INT8 TC matmul
import { dlopen, ptr } from "bun:ffi";
import { int8MatmulTTIR, compileTTIR } from "../src/kernel.ts";

// Search space for INT8 TC matmul
const SEARCH = [
  { BM: 16, BN: 16, BK: 256, numWarps: 4 },
  { BM: 16, BN: 16, BK: 1024, numWarps: 4 },
  { BM: 32, BN: 16, BK: 256, numWarps: 4 },
  { BM: 32, BN: 16, BK: 1024, numWarps: 4 },
  { BM: 32, BN: 32, BK: 32, numWarps: 4 },
  { BM: 32, BN: 32, BK: 256, numWarps: 4 },
  { BM: 32, BN: 32, BK: 512, numWarps: 8 },
  { BM: 32, BN: 32, BK: 1024, numWarps: 4 },
  { BM: 32, BN: 32, BK: 512, numWarps: 4 },
  { BM: 64, BN: 32, BK: 1024, numWarps: 8 },
  { BM: 64, BN: 64, BK: 256, numWarps: 8 },
  { BM: 64, BN: 64, BK: 512, numWarps: 8 },
  { BM: 128, BN: 128, BK: 64, numWarps: 8 },
];

interface TCConfig {
  BM: number; BN: number; BK: number; numWarps: number;
}

function compileAndBench(cfg: TCConfig, cs: any, ev: any): number | null {
  const ttir = int8MatmulTTIR(cfg);
  let ptx, shmem: number;
  try {
    const result = compileTTIR(ttir, cfg.numWarps);
    ptx = result.ptx; shmem = result.shmem;
  } catch { return null; }

  const mma = (ptx.match(/mma\.sync/g) || []).length;
  const entry = ptx.match(/\.visible\s+\.entry\s+(\w+)/)?.[1];
  if (!entry) return null;

  const mod = Buffer.alloc(8); cs.cuModuleLoadData(mod, ptr(Buffer.from(ptx + "\0")));
  const fn = Buffer.alloc(8);
  cs.cuModuleGetFunction(fn, Number(mod.readBigUInt64LE(0)), ptr(Buffer.from(entry + "\0")));
  const fh = Number(fn.readBigUInt64LE(0));

  const np = (ptx.match(/\.param/g) || []).length;
  const tcArgs = Buffer.alloc(np * 8);
  const pp = Number(ptr(tcArgs));
  const kp = Buffer.alloc((np + 1) * 8);
  for (let i = 0; i < np; i++) kp.writeBigUInt64LE(BigInt(pp + i * 8), i * 8);
  kp.writeBigUInt64LE(0n, np * 8);

  const SZ = BigInt(512 * 1024 * 1024);
  const dA = Buffer.alloc(8); cs.cuMemAlloc_v2(dA, SZ);
  const dB = Buffer.alloc(8); cs.cuMemAlloc_v2(dB, SZ);
  const dC = Buffer.alloc(8); cs.cuMemAlloc_v2(dC, SZ);

  tcArgs.writeBigUInt64LE(dA.readBigUInt64LE(0), 0);
  tcArgs.writeBigUInt64LE(dB.readBigUInt64LE(0), 8);
  tcArgs.writeBigUInt64LE(dC.readBigUInt64LE(0), 16);
  for (let i = 3; i < np; i++) tcArgs.writeBigUInt64LE(0n, i * 8);

  const gx = Math.ceil(1024 / cfg.BM), gy = Math.ceil(1024 / cfg.BN);
  const bt = cfg.numWarps * 32;

  // Warmup
  for (let i = 0; i < 3; i++) cs.cuLaunchKernel(fh, gx, gy, 1, bt, 1, 1, shmem, 0n, ptr(kp), null);
  cs.cuCtxSynchronize();

  // CUDA event timing
  const eStart = Number(ev.start.readBigUInt64LE(0));
  const eStop = Number(ev.stop.readBigUInt64LE(0));
  const times: number[] = [];
  for (let i = 0; i < 10; i++) {
    cs.cuEventRecord(eStart, 0);
    cs.cuLaunchKernel(fh, gx, gy, 1, bt, 1, 1, shmem, 0n, ptr(kp), null);
    cs.cuEventRecord(eStop, 0);
    cs.cuEventSynchronize(eStop);
    const elapsed = new Float32Array(1);
    cs.cuEventElapsedTime(elapsed, eStart, eStop);
    times.push(elapsed[0] * 1000);
  }

  cs.cuMemFree_v2(Number(dA.readBigUInt64LE(0)));
  cs.cuMemFree_v2(Number(dB.readBigUInt64LE(0)));
  cs.cuMemFree_v2(Number(dC.readBigUInt64LE(0)));

  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  const flops = 2 * 1024 * 1024 * 1024 / (avg / 1e6) / 1e12;
  console.log(`  ${cfg.BM}x${cfg.BN}x${cfg.BK} w${cfg.numWarps}  mma:${mma}  shmem:${shmem}  ${avg.toFixed(1)} µs  ${flops.toFixed(2)} TFLOPS`);
  return avg;
}

const cs = dlopen("/run/opengl-driver/lib/libcuda.so", {
  cuInit:{args:["u32"],returns:"i32"},cuDeviceGet:{args:["ptr","i32"],returns:"i32"},
  cuCtxCreate_v2:{args:["ptr","u32","i64"],returns:"i32"},
  cuEventCreate:{args:["ptr","u32"],returns:"i32"},
  cuEventRecord:{args:["i64","i64"],returns:"i32"},
  cuEventSynchronize:{args:["i64"],returns:"i32"},
  cuEventElapsedTime:{args:["ptr","i64","i64"],returns:"i32"},
  cuModuleLoadData:{args:["ptr","ptr"],returns:"i32"},
  cuModuleGetFunction:{args:["ptr","i64","ptr"],returns:"i32"},
  cuLaunchKernel:{args:["i64","u32","u32","u32","u32","u32","u32","u32","ptr","ptr","ptr"],returns:"i32"},
  cuMemAlloc_v2:{args:["ptr","i64"],returns:"i32"},
  cuMemFree_v2:{args:["i64"],returns:"i32"},cuCtxSynchronize:{args:[],returns:"i32"},
}).symbols;
cs.cuInit(0);const dev=new Int32Array(1);cs.cuDeviceGet(dev,0);
cs.cuCtxCreate_v2(Buffer.alloc(8),0,BigInt(dev[0]));

const evStart = Buffer.alloc(8); cs.cuEventCreate(evStart, 0);
const evStop = Buffer.alloc(8); cs.cuEventCreate(evStop, 0);

let best: { cfg: TCConfig; time: number } | null = null;
for (const cfg of SEARCH) {
  const t = compileAndBench(cfg, cs, { start: evStart, stop: evStop });
  if (t !== null && (best === null || t < best.time)) best = { cfg, time: t };
}

if (best) {
  console.log(`\nBest: ${best.cfg.BM}x${best.cfg.BN}x${best.cfg.BK} w${best.cfg.numWarps} = ${best.time.toFixed(1)} µs`);
}
