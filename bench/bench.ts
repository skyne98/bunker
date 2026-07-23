// Benchmark Q4_K × Q8_1 matmul — measure TOPS on RTX 3090
import { dlopen, ptr as ffiPtr, CString } from "bun:ffi";
import { kernel, ptr, f32, i32, i8, i16, struct, array, sizeof } from "../src/dsl";

// ─── Struct definitions ─────────────────────────────────────────────
const Q8_1 = struct({ q: array(i8, 32), d: i16, s: i16 }, "Q8_1");
const Q4_K = struct({ q: array(i8, 16), d: i16, dmin: i16 }, "Q4_K");

// ─── Quantized matmul kernel ────────────────────────────────────────
const qmm = kernel((A: ptr<Q8_1>, B: ptr<Q4_K>, C: ptr<f32>, M: i32, N: i32, K: i32) => {
  const row = blockIdx.y * 16 + threadIdx.y;
  const col = blockIdx.x * 16 + threadIdx.x;
  if (row >= M || col >= N) return;

  const nb = K / 32;
  let sum = 0;

  for (let b = 0; b < nb; b++) {
    const aBlk = A[row * nb + b];
    const bBlk = B[col * nb + b];

    const d8 = half_to_float(aBlk.d);
    const d4 = half_to_float(bBlk.d);
    const dm = half_to_float(bBlk.dmin);

    let s1 = 0, s2 = 0;
    for (let i = 0; i < 16; i++) {
      const q4 = bBlk.q[i];
      const lo = q4 & 15;  const hi = (q4 >> 4) & 15;
      const q8lo = aBlk.q[i * 2];  const q8hi = aBlk.q[i * 2 + 1];
      s1 += lo * q8lo + hi * q8hi;
      s2 += q8lo + q8hi;
    }
    sum += d4 * d8 * s1 + dm * d8 * s2;
  }

  C[row * N + col] = sum;
});

// ─── Packing helpers ────────────────────────────────────────────────
function encodeHalf(v: number): number {
  if (v === 0) return 0;
  const u = new DataView(new ArrayBuffer(4));
  u.setFloat32(0, v, true);
  const bits = u.getUint32(0, true);
  const s = (bits >> 16) & 0x8000, e = (bits >> 23) & 0xff, m = bits & 0x7fffff;
  const ne = e - 127 + 15;
  if (ne >= 31) return s | 0x7c00;
  if (ne <= 0) return s;
  return s | (ne << 10) | (m >> 13);
}

function packQ8_1(data: Float32Array): Uint8Array {
  const nb = Math.ceil(data.length / 32);
  const buf = new Uint8Array(nb * 36);
  const dv = new DataView(buf.buffer);
  for (let b = 0; b < nb; b++) {
    const off = b * 32;
    const block = data.slice(off, Math.min(off + 32, data.length));
    let amax = 0; for (const v of block) { const a = Math.abs(v); if (a > amax) amax = a; }
    const d = amax / 127;
    let sum = 0;
    for (let i = 0; i < 32; i++) {
      const q = i < block.length ? Math.round(block[i] / (d || 1)) : 0;
      buf[b * 36 + i] = Math.max(-128, Math.min(127, q));
      sum += q;
    }
    dv.setUint16(b * 36 + 32, encodeHalf(d), true);
    dv.setUint16(b * 36 + 34, encodeHalf(sum * d), true);
  }
  return buf;
}

function packQ4_K_col(data: Float32Array, K: number, N: number): Uint8Array {
  const nb = Math.ceil(K / 32);
  const buf = new Uint8Array(N * nb * 20);
  const dv = new DataView(buf.buffer);
  for (let c = 0; c < N; c++) {
    for (let b = 0; b < nb; b++) {
      const off = (c * nb + b) * 20;
      const block: number[] = [];
      for (let k = b * 32; k < Math.min((b + 1) * 32, K); k++) block.push(data[k * N + c]);
      let min = Infinity, max = -Infinity;
      for (const v of block) { if (v < min) min = v; if (v > max) max = v; }
      const d = (max - min) / 15; const dmin = min;
      for (let i = 0; i < 16; i++) {
        const lo = i * 2 < block.length ? Math.round((block[i * 2] - dmin) / (d || 1)) : 0;
        const hi = i * 2 + 1 < block.length ? Math.round((block[i * 2 + 1] - dmin) / (d || 1)) : 0;
        buf[off + i] = Math.max(0, Math.min(15, lo)) | (Math.max(0, Math.min(15, hi)) << 4);
      }
      dv.setUint16(off + 16, encodeHalf(d), true);
      dv.setUint16(off + 18, encodeHalf(dmin), true);
    }
  }
  return buf;
}

// ─── LLVM → PTX → GPU runtime ──────────────────────────────────────
const LLVM = "/nix/store/6r234y6pkbyyr8pk1wh7nfsmnzdxyswx-llvm-19.1.7-lib/lib/libLLVM-19.so";
const CUDA = "/run/opengl-driver/lib/libcuda.so";

const ls = dlopen(LLVM, {
  LLVMContextCreate:{args:[],returns:"pointer"},LLVMParseIRInContext:{args:["pointer","pointer","pointer","pointer"],returns:"i32"},
  LLVMCreateMemoryBufferWithMemoryRange:{args:["pointer","i64","pointer","i32"],returns:"pointer"},
  LLVMGetTargetFromTriple:{args:["pointer","pointer","pointer"],returns:"i32"},
  LLVMCreateTargetMachine:{args:["pointer","pointer","pointer","pointer","i32","i32","i32"],returns:"pointer"},
  LLVMTargetMachineEmitToMemoryBuffer:{args:["pointer","pointer","i32","pointer","pointer"],returns:"i32"},
  LLVMGetBufferSize:{args:["pointer"],returns:"i64"},LLVMGetBufferStart:{args:["pointer"],returns:"pointer"},
  LLVMInitializeNVPTXTargetInfo:{args:[],returns:"void"},LLVMInitializeNVPTXTarget:{args:[],returns:"void"},
  LLVMInitializeNVPTXTargetMC:{args:[],returns:"void"},LLVMInitializeNVPTXAsmPrinter:{args:[],returns:"void"},
  LLVMDisposeMessage:{args:["pointer"],returns:"void"},
}).symbols;
ls.LLVMInitializeNVPTXTargetInfo();ls.LLVMInitializeNVPTXTarget();ls.LLVMInitializeNVPTXTargetMC();ls.LLVMInitializeNVPTXAsmPrinter();

function llvmIRtoPTX(src: string): string {
  const ctx = ls.LLVMContextCreate();
  const ib = Buffer.from(src + "\0");
  const mb = ls.LLVMCreateMemoryBufferWithMemoryRange(ffiPtr(ib), BigInt(ib.length - 1), ffiPtr(Buffer.from("k.ll\0")), 1);
  const ma = new BigUint64Array(1);
  const errPtr = new BigUint64Array(1);
  if (ls.LLVMParseIRInContext(ctx, mb, ffiPtr(ma), ffiPtr(errPtr))) {
    const errMsg = new CString(Number(errPtr[0])).toString().split('\0')[0];
    console.log("LLVM Error:", errMsg);
    ls.LLVMDisposeMessage(Number(errPtr[0]));
    throw Error("LLVM IR parse failed");
  }
  const mod = Number(ma[0]);
  const tp = Buffer.from("nvptx64-nvidia-cuda\0"); const ta = new BigUint64Array(1);
  ls.LLVMGetTargetFromTriple(ffiPtr(tp), ffiPtr(ta), ffiPtr(new BigUint64Array(1)));
  const tm = ls.LLVMCreateTargetMachine(Number(ta[0]), ffiPtr(tp), ffiPtr(Buffer.from("sm_86\0")), ffiPtr(Buffer.from("\0")), 2, 0, 0);
  const pa = new BigUint64Array(1);
  if (ls.LLVMTargetMachineEmitToMemoryBuffer(tm, mod, 0, ffiPtr(new BigUint64Array(1)), ffiPtr(pa))) throw Error("PTX failed");
  return new CString(ls.LLVMGetBufferStart(Number(pa[0])), 0, Number(ls.LLVMGetBufferSize(Number(pa[0])))).toString();
}

const cs = dlopen(CUDA, {
  cuInit:{args:["u32"],returns:"i32"},cuDeviceGet:{args:["pointer","i32"],returns:"i32"},
  cuDeviceGetName:{args:["pointer","i32","i32"],returns:"i32"},cuCtxCreate_v2:{args:["pointer","u32","i32"],returns:"i32"},
  cuModuleLoadData:{args:["pointer","pointer"],returns:"i32"},cuModuleGetFunction:{args:["pointer","pointer","pointer"],returns:"i32"},
  cuMemAlloc_v2:{args:["pointer","i64"],returns:"i32"},
  cuMemcpyHtoD_v2:{args:["i64","pointer","i64"],returns:"i32"},
  cuMemcpyDtoH_v2:{args:["pointer","i64","i64"],returns:"i32"},
  cuLaunchKernel:{args:["pointer","u32","u32","u32","u32","u32","u32","u32","pointer","pointer","pointer"],returns:"i32"},
  cuMemFree_v2:{args:["i64"],returns:"i32"},cuCtxSynchronize:{args:[],returns:"i32"},
}).symbols;

// ─── Benchmark ──────────────────────────────────────────────────────
const M = 1024, N = 1024, K = 1024;  // 1K × 1K matmul
const SIZE = M * N * K;
const OPS = 2 * M * N * K;  // multiply + add per element

console.log(`Q4_K × Q8_1 matmul ${M}x${K} * ${K}x${N}`);
console.log(`Operations: ${(OPS / 1e9).toFixed(1)} GFLOPs`);

// Generate random data and pack
const hA = new Float32Array(M * K); for (let i = 0; i < M * K; i++) hA[i] = Math.random() * 2 - 1;
const hB = new Float32Array(K * N); for (let i = 0; i < K * N; i++) hB[i] = Math.random() * 2 - 1;

console.log("Packing data...");
const packedA = packQ8_1(hA);
const packedB = packQ4_K_col(hB, K, N);

// Compile kernel to PTX
console.log("Compiling kernel...");
const ptx = llvmIRtoPTX(qmm.source);

// Init GPU
cs.cuInit(0);
const dev = new Int32Array(1); cs.cuDeviceGet(dev, 0);
const nb = new Uint8Array(256); cs.cuDeviceGetName(nb, 256, dev[0]);
const ctxBuf = Buffer.alloc(8); cs.cuCtxCreate_v2(ctxBuf, 0, dev[0]);
console.log(`GPU: ${new CString(ffiPtr(nb))}`);

// Allocate device memory
const dA = Buffer.alloc(8); cs.cuMemAlloc_v2(dA, BigInt(packedA.length));
const dB = Buffer.alloc(8); cs.cuMemAlloc_v2(dB, BigInt(packedB.length));
const dC = Buffer.alloc(8); cs.cuMemAlloc_v2(dC, BigInt(M * N * 4));

const da = Number(dA.readBigUInt64LE(0));
const db = Number(dB.readBigUInt64LE(0));
const dc = Number(dC.readBigUInt64LE(0));

cs.cuMemcpyHtoD_v2(da, packedA, BigInt(packedA.length));
cs.cuMemcpyHtoD_v2(db, packedB, BigInt(packedB.length));

// Load module
const mod = Buffer.alloc(8);
if (cs.cuModuleLoadData(mod, Buffer.from(ptx)) !== 0) throw Error("PTX load failed");
const fn = Buffer.alloc(8);
cs.cuModuleGetFunction(fn, Number(mod.readBigUInt64LE(0)), Buffer.from("kernel\0"));

// Build kernel params
const pb = Buffer.alloc(6 * 8);
pb.writeBigUInt64LE(BigInt(da), 0);
pb.writeBigUInt64LE(BigInt(db), 8);
pb.writeBigUInt64LE(BigInt(dc), 16);
pb.writeInt32LE(M, 24); // pad 28-31 zeros (slot 3)
pb.writeInt32LE(N, 32); // pad 36-39 zeros (slot 4)
pb.writeInt32LE(K, 40); // pad 44-47 zeros (slot 5)

const kp = Buffer.alloc(7 * 8);
const pp = ffiPtr(pb);
for (let i = 0; i < 6; i++) kp.writeBigUInt64LE(BigInt(pp + i * 8), i * 8);
kp.writeBigUInt64LE(0n, 48);

const gridX = Math.ceil(N / 16), gridY = Math.ceil(M / 16);

// Warmup
cs.cuLaunchKernel(Number(fn.readBigUInt64LE(0)), gridX, gridY, 1, 16, 16, 1, 0, null, kp, null);
cs.cuCtxSynchronize();

// Benchmark
const ITERS = 10;
const t0 = performance.now();
for (let i = 0; i < ITERS; i++) {
  cs.cuLaunchKernel(Number(fn.readBigUInt64LE(0)), gridX, gridY, 1, 16, 16, 1, 0, null, kp, null);
}
cs.cuCtxSynchronize();
const t1 = performance.now();

const avgMs = (t1 - t0) / ITERS;
const tops = OPS / (avgMs / 1000) / 1e12;

console.log(`\nResults:`);
console.log(`  Grid: ${gridX}×${gridY} blocks, 16×16 threads/block`);
console.log(`  Avg time: ${avgMs.toFixed(2)} ms`);
console.log(`  Performance: ${tops.toFixed(3)} TOPS`);
console.log(`  Bandwidth: ${(OPS * 4 / (avgMs / 1000) / 1e9).toFixed(1)} GB/s (est.)`);

// Verify correctness (small slice)
const hC = new Float32Array(M * N);
cs.cuMemcpyDtoH_v2(hC, dc, BigInt(M * N * 4));
let err = 0;
for (let i = 0; i < Math.min(100, M * N); i++) {
  let ref = 0;
  const r = Math.floor(i / N), c = i % N;
  for (let k = 0; k < K; k++) ref += hA[r * K + k] * hB[k * N + c];
  if (Math.abs(hC[i] - ref) / Math.max(Math.abs(ref), 0.001) > 0.2) err++;
}
console.log(`  Errors (tol=20%): ${err}/100`);
console.log(`  Performance: ${tops.toFixed(3)} TOPS`);

cs.cuMemFree_v2(da); cs.cuMemFree_v2(db); cs.cuMemFree_v2(dc);
console.log("\nDone");
