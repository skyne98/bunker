// test_fused_ttir.ts — Test fused TTIR kernel end-to-end
import { ptr } from "bun:ffi";
import { fusedDequantMatmulTTIR, compileTTIR, getCuCtx, createCudaEvents, gpuTimeUs } from "./kernel.ts";

const cs = getCuCtx();
const ev = createCudaEvents();
const eStart = Number(ev.start.readBigUInt64LE(0));
const eStop = Number(ev.stop.readBigUInt64LE(0));
const el = new Float32Array(1);

const M = 1024, N = 1024, K = 1024;
const BLOCK_SIZE = 32, BLOCKS_PER_ROW = K / BLOCK_SIZE;
const NUM_BLOCKS = M * BLOCKS_PER_ROW;
const Q4K_BLOCK_BYTES = 20;
const INPUT_SIZE = NUM_BLOCKS * Q4K_BLOCK_BYTES;
const B_SIZE = K * N;
const C_SIZE = M * N;

// Generate Q4_K data + reference
const hQ = Buffer.alloc(INPUT_SIZE);
const hA_ref = new Int8Array(M * K);
const hB = new Int8Array(B_SIZE);
const hC = new Int32Array(C_SIZE);

for (let r = 0; r < M; r++) {
  for (let kb = 0; kb < BLOCKS_PER_ROW; kb++) {
    const blockIdx = r * BLOCKS_PER_ROW + kb;
    const base = blockIdx * Q4K_BLOCK_BYTES;
    for (let j = 0; j < 16; j++) {
      const lo = (Math.random() * 16) | 0;
      const hi = (Math.random() * 16) | 0;
      hQ[base + j] = lo | (hi << 4);
      hA_ref[r * K + kb * BLOCK_SIZE + j * 2] = lo - 8;
      hA_ref[r * K + kb * BLOCK_SIZE + j * 2 + 1] = hi - 8;
    }
  }
}
for (let i = 0; i < B_SIZE; i++) hB[i] = (Math.random() * 256 - 128) | 0;

// Compile fused kernel
const ttir = fusedDequantMatmulTTIR({ BM: 32, BN: 32, BK: 32, numWarps: 4 });
console.log(`TTIR: ${ttir.length} bytes`);
const { ptx, shmem } = compileTTIR(ttir, 4);
console.log(`PTX: ${(ptx.match(/mma\.sync/g)||[]).length} mma.sync, shmem=${shmem}`);

// Load
const mod = Buffer.alloc(8); cs.cuModuleLoadData(mod, ptr(Buffer.from(ptx + "\0")));
const fn = Buffer.alloc(8); cs.cuModuleGetFunction(fn, Number(mod.readBigUInt64LE(0)), ptr(Buffer.from("fused\0")));
const fh = Number(fn.readBigUInt64LE(0));
if (fh === 0) { console.log("Fn not found"); process.exit(1); }

// Allocate
const SZ = BigInt(512 * 1024 * 1024);
const dQ = Buffer.alloc(8); cs.cuMemAlloc_v2(dQ, SZ);
const dB = Buffer.alloc(8); cs.cuMemAlloc_v2(dB, SZ);
const dC = Buffer.alloc(8); cs.cuMemAlloc_v2(dC, SZ);
cs.cuMemcpyHtoD_v2(Number(dQ.readBigUInt64LE(0)), hQ, BigInt(hQ.length));
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)), Buffer.from(hB.buffer), BigInt(hB.byteLength));

// Kernel params: 5 params (Q, B, C, scratch, unused)
const np = (ptx.match(/\.param/g) || []).length;
const kVals: Buffer[] = [];
kVals.push(Buffer.alloc(8)); kVals[0].writeBigUInt64LE(dQ.readBigUInt64LE(0), 0);
kVals.push(Buffer.alloc(8)); kVals[1].writeBigUInt64LE(dB.readBigUInt64LE(0), 0);
kVals.push(Buffer.alloc(8)); kVals[2].writeBigUInt64LE(dC.readBigUInt64LE(0), 0);
while (kVals.length < np) kVals.push(Buffer.alloc(8));

const kp = Buffer.alloc((np + 1) * 8);
for (let i = 0; i < np; i++) kp.writeBigUInt64LE(BigInt(Number(ptr(kVals[i]))), i * 8);
kp.writeBigUInt64LE(0n, np * 8);

const gx = Math.ceil(N / 32), gy = Math.ceil(M / 32);
console.log(`Grid: ${gx}x${gy}, Block: 128, shmem: ${shmem}`);

// Warmup
cs.cuLaunchKernel(fh, gx, gy, 1, 128, 1, 1, shmem, 0n, ptr(kp), null);
cs.cuCtxSynchronize();

// Benchmark
const times: number[] = [];
for (let i = 0; i < 30; i++) {
  cs.cuEventRecord(eStart, 0);
  cs.cuLaunchKernel(fh, gx, gy, 1, 128, 1, 1, shmem, 0n, ptr(kp), null);
  cs.cuEventRecord(eStop, 0);
  cs.cuEventSynchronize(eStop);
  cs.cuEventElapsedTime(el, eStart, eStop);
  times.push(el[0] * 1000);
}
const avg = times.reduce((a, b) => a + b, 0) / times.length;
const tops = 2 * M * N * K / (avg / 1e6) / 1e12;
console.log(`\nFused TTIR kernel: ${avg.toFixed(1)} µs → ${tops.toFixed(2)} TFLOPS`);

// Verify
cs.cuMemcpyDtoH_v2(hC, Number(dC.readBigUInt64LE(0)), BigInt(C_SIZE * 4));
const refC = new Int32Array(C_SIZE);
for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
  let s = 0; for (let k = 0; k < K; k++) s += hA_ref[i * K + k] * hB[k * N + j];
  refC[i * N + j] = s;
}
let max = 0, cnt = 0;
for (let i = 0; i < C_SIZE; i++) {
  const d = Math.abs(hC[i] - refC[i]);
  if (d > max) max = d;
  if (d !== 0) cnt++;
}
console.log(`Errors: ${cnt}/${C_SIZE}, max=${max}`);
if (max === 0) console.log("✓ CORRECT!"); else console.log("✗ ERRORS");
