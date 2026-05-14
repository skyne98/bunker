// test_dequant.ts — Test & benchmark on-device Q4_K dequant kernel
import { DequantKernel } from "./kernel.ts";

const M = 128, BLOCK_SIZE = 32, BLOCKS_PER_ROW = 32;
const NUM_BLOCKS = M * BLOCKS_PER_ROW;
const BLOCK_BYTES = 20;
const INPUT_SIZE = NUM_BLOCKS * BLOCK_BYTES;
const OUTPUT_SIZE = M * BLOCK_SIZE * BLOCKS_PER_ROW;

// Generate random Q4_K data
const hInput = Buffer.alloc(INPUT_SIZE);
const hRef = new Int8Array(OUTPUT_SIZE);
for (let b = 0; b < NUM_BLOCKS; b++) {
  const base = b * BLOCK_BYTES;
  for (let j = 0; j < 16; j++) {
    const lo = (Math.random() * 16) | 0;
    const hi = (Math.random() * 16) | 0;
    hInput[base + j] = lo | (hi << 4);
    hRef[b * 32 + j * 2] = lo - 8;
    hRef[b * 32 + j * 2 + 1] = hi - 8;
  }
  hInput[base + 16] = 0;
  hInput[base + 18] = 0;
}

console.log(`Blocks: ${NUM_BLOCKS}, Input: ${INPUT_SIZE}B, Output: ${OUTPUT_SIZE}B`);

import { dlopen, ptr } from "bun:ffi";
const CUDA = "/run/opengl-driver/lib/libcuda.so";
const cs = dlopen(CUDA, {
  cuInit:{args:["u32"],returns:"i32"},cuDeviceGet:{args:["ptr","i32"],returns:"i32"},
  cuCtxCreate_v2:{args:["ptr","u32","i64"],returns:"i32"},
  cuMemAlloc_v2:{args:["ptr","i64"],returns:"i32"},
  cuMemcpyHtoD_v2:{args:["i64","ptr","i64"],returns:"i32"},
  cuMemcpyDtoH_v2:{args:["ptr","i64","i64"],returns:"i32"},
  cuMemFree_v2:{args:["i64"],returns:"i32"},
}).symbols;
cs.cuInit(0);const dev=new Int32Array(1);cs.cuDeviceGet(dev,0);
cs.cuCtxCreate_v2(Buffer.alloc(8),0,BigInt(dev[0]));

const dIn = Buffer.alloc(8); cs.cuMemAlloc_v2(dIn, BigInt(INPUT_SIZE));
const dOut = Buffer.alloc(8); cs.cuMemAlloc_v2(dOut, BigInt(OUTPUT_SIZE));
cs.cuMemcpyHtoD_v2(Number(dIn.readBigUInt64LE(0)), hInput, BigInt(INPUT_SIZE));

const k = new DequantKernel();

// Warmup
k.run(dIn.readBigUInt64LE(0), dOut.readBigUInt64LE(0), NUM_BLOCKS);

// Benchmark
const ITERS = 100;
const times: number[] = [];
for (let i = 0; i < ITERS; i++) {
  const t0 = performance.now();
  k.run(dIn.readBigUInt64LE(0), dOut.readBigUInt64LE(0), NUM_BLOCKS);
  times.push(performance.now() - t0);
}
const avg = times.reduce((a, b) => a + b, 0) / times.length;
const bw = (OUTPUT_SIZE / (avg / 1000)) / 1e9;
console.log(`Dequant kernel: ${(avg * 1000).toFixed(1)} µs avg × ${ITERS} iters`);
console.log(`Output bandwidth: ${bw.toFixed(2)} GB/s`);

// Verify on last run
const hOutput = Buffer.alloc(OUTPUT_SIZE);
cs.cuMemcpyDtoH_v2(hOutput, Number(dOut.readBigUInt64LE(0)), BigInt(OUTPUT_SIZE));
const outI8 = new Int8Array(hOutput.buffer);
let errors = 0;
for (let i = 0; i < OUTPUT_SIZE; i++) {
  if (outI8[i] !== hRef[i]) {
    errors++;
    if (errors <= 5) console.log(`  Mismatch at [${i}]: got ${outI8[i]}, expected ${hRef[i]}`);
  }
}
console.log(`Errors: ${errors} / ${OUTPUT_SIZE}`);
if (errors === 0) console.log("✓ All correct!");
else console.log("✗ ERRORS DETECTED");

cs.cuMemFree_v2(Number(dIn.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dOut.readBigUInt64LE(0)));
