// test_q4k_pipeline.ts — Full Q4_K×Q8_1 pipeline (single CUDA context)
import { ptr } from "bun:ffi";
import { DequantKernel, int8MatmulTTIR, compileTTIR, getCuCtx } from "./kernel.ts";

const cs = getCuCtx();
const M = 1024, N = 1024, K = 1024;
const BLOCK_SIZE = 32, BLOCKS_PER_ROW = K / BLOCK_SIZE;
const NUM_BLOCKS = M * BLOCKS_PER_ROW;
const Q4K_BLOCK_BYTES = 20;
const INPUT_SIZE = NUM_BLOCKS * Q4K_BLOCK_BYTES;
const DQ_SIZE = M * K;
const B_SIZE = K * N;
const C_SIZE = M * N;

// Generate Q4_K + reference
const hA_q4k = Buffer.alloc(INPUT_SIZE);
const hA_ref = new Int8Array(DQ_SIZE);
const hB = new Int8Array(B_SIZE);
const hC = new Int32Array(C_SIZE);

for (let r = 0; r < M; r++) {
  for (let kb = 0; kb < BLOCKS_PER_ROW; kb++) {
    const blockIdx = r * BLOCKS_PER_ROW + kb;
    const base = blockIdx * Q4K_BLOCK_BYTES;
    for (let j = 0; j < 16; j++) {
      const lo = (Math.random() * 16) | 0;
      const hi = (Math.random() * 16) | 0;
      hA_q4k[base + j] = lo | (hi << 4);
      hA_ref[r * K + kb * BLOCK_SIZE + j * 2] = lo - 8;
      hA_ref[r * K + kb * BLOCK_SIZE + j * 2 + 1] = hi - 8;
    }
    hA_q4k[base + 16] = 0;
    hA_q4k[base + 18] = 0;
  }
}
for (let i = 0; i < B_SIZE; i++) hB[i] = (Math.random() * 256 - 128) | 0;

// GPU allocs
const SZ = BigInt(512 * 1024 * 1024);
const dA_q4k = Buffer.alloc(8); cs.cuMemAlloc_v2(dA_q4k, SZ);
const dB = Buffer.alloc(8); cs.cuMemAlloc_v2(dB, SZ);
const dA_dq = Buffer.alloc(8); cs.cuMemAlloc_v2(dA_dq, SZ);
const dC = Buffer.alloc(8); cs.cuMemAlloc_v2(dC, SZ);
cs.cuMemcpyHtoD_v2(Number(dA_q4k.readBigUInt64LE(0)), hA_q4k, BigInt(hA_q4k.length));
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)), Buffer.from(hB.buffer), BigInt(hB.byteLength));

// Compile TC kernel
const cfg = { BM: 32, BN: 32, BK: 1024, numWarps: 4 };
const { ptx, shmem } = compileTTIR(int8MatmulTTIR(cfg), cfg.numWarps);
console.log(`TC kernel: ${(ptx.match(/mma\.sync/g)||[]).length} mma.sync, shmem=${shmem}`);

const entry = ptx.match(/\.visible\s+\.entry\s+(\w+)/)[1];
const tcMod = Buffer.alloc(8); cs.cuModuleLoadData(tcMod, ptr(Buffer.from(ptx + "\0")));
const tcFn = Buffer.alloc(8);
cs.cuModuleGetFunction(tcFn, Number(tcMod.readBigUInt64LE(0)), ptr(Buffer.from(entry + "\0")));
const fhTC = Number(tcFn.readBigUInt64LE(0));

// Load dequant kernel
const deq = new DequantKernel();
deq.load();

// Build persistent TC kernel params
const np = (ptx.match(/\.param/g) || []).length;
const tcArgs = Buffer.alloc(np * 8);
const tcPp = Number(ptr(tcArgs));
const tcKp = Buffer.alloc((np + 1) * 8);
for (let i = 0; i < np; i++) tcKp.writeBigUInt64LE(BigInt(tcPp + i * 8), i * 8);
tcKp.writeBigUInt64LE(0n, np * 8);

// Set addresses: A=dA_dq, B=dB, C=dC
tcArgs.writeBigUInt64LE(dA_dq.readBigUInt64LE(0), 0);
tcArgs.writeBigUInt64LE(dB.readBigUInt64LE(0), 8);
tcArgs.writeBigUInt64LE(dC.readBigUInt64LE(0), 16);
for (let i = 3; i < np; i++) tcArgs.writeBigUInt64LE(0n, i * 8);

// Verify dequant first
deq.launch(dA_q4k.readBigUInt64LE(0), dA_dq.readBigUInt64LE(0), NUM_BLOCKS);
cs.cuCtxSynchronize();
const hDQ = new Int8Array(DQ_SIZE);
cs.cuMemcpyDtoH_v2(hDQ, Number(dA_dq.readBigUInt64LE(0)), BigInt(DQ_SIZE));
let dqErr = 0;
for (let i = 0; i < DQ_SIZE; i++) if (hDQ[i] !== hA_ref[i]) dqErr++;
console.log(`Dequant: ${dqErr} errors / ${DQ_SIZE}${dqErr === 0 ? ' ✓' : ''}`);

// Pipeline benchmark
const gx = Math.ceil(N / 32), gy = Math.ceil(M / 32);
const FLOP = 2n * BigInt(M) * BigInt(N) * BigInt(K);
const ITERS = 30;

deq.launch(dA_q4k.readBigUInt64LE(0), dA_dq.readBigUInt64LE(0), NUM_BLOCKS);
cs.cuLaunchKernel(fhTC, gx, gy, 1, cfg.numWarps * 32, 1, 1, shmem, 0n, ptr(tcKp), null);
cs.cuCtxSynchronize();

const times: number[] = [];
for (let i = 0; i < ITERS; i++) {
  const t0 = performance.now();
  deq.launch(dA_q4k.readBigUInt64LE(0), dA_dq.readBigUInt64LE(0), NUM_BLOCKS);
  cs.cuLaunchKernel(fhTC, gx, gy, 1, cfg.numWarps * 32, 1, 1, shmem, 0n, ptr(tcKp), null);
  cs.cuCtxSynchronize();
  times.push(performance.now() - t0);
}
const avg = times.reduce((a, b) => a + b, 0) / times.length;
const tops = Number(FLOP) / (avg / 1000) / 1e12;
console.log(`\nQ4_K×Q8_1  ${M}×${K}×${N}: ${(avg*1000).toFixed(0)} µs → ${tops.toFixed(2)} TFLOPS`);

// Verify
cs.cuMemcpyDtoH_v2(hC, Number(dC.readBigUInt64LE(0)), BigInt(C_SIZE * 4));

const refC = new Int32Array(C_SIZE);
for (let i = 0; i < M; i++) {
  for (let j = 0; j < N; j++) {
    let sum = 0;
    for (let k = 0; k < K; k++) sum += hA_ref[i * K + k] * hB[k * N + j];
    refC[i * N + j] = sum;
  }
}
let maxErr = 0, errCount = 0;
for (let i = 0; i < C_SIZE; i++) {
  const err = Math.abs(hC[i] - refC[i]);
  if (err > maxErr) maxErr = err;
  if (err !== 0) errCount++;
}
console.log(`Max error: ${maxErr}, errors: ${errCount}/${C_SIZE}`);
if (maxErr === 0) console.log("✓ Pipeline correct!");
else console.log("✗ ERRORS");
