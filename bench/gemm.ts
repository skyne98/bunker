import { dlopen, ptr as ffiPtr, CString } from "bun:ffi";
import { kernel, ptr, f32, i32, i8 } from "../src/dsl";

// ─── v1: naive INT8 ──────────────────────────────────────────────
const v1 = kernel((A: ptr<i8>, B: ptr<i8>, C: ptr<i32>, M: i32, N: i32, K: i32) => {
  const row = blockIdx.y * 16 + threadIdx.y;
  const col = blockIdx.x * 16 + threadIdx.x;
  if (row >= M || col >= N) return;
  let sum: i32 = 0;
  for (let kk = 0; kk < K; kk++) sum += A[row * K + kk] * B[kk * N + col];
  C[row * N + col] = sum;
});

// ─── v2: 2-wide tile ─────────────────────────────────────────────
const v2 = kernel((A: ptr<i8>, B: ptr<i8>, C: ptr<i32>, M: i32, N: i32, K: i32) => {
  const row = blockIdx.y * 16 * 2 + threadIdx.y;
  const col = blockIdx.x * 16 + threadIdx.x;
  if (row >= M || col >= N) return;
  let s0: i32 = 0, s1: i32 = 0;
  for (let kk = 0; kk < K; kk++) {
    const b = B[kk * N + col];
    s0 += A[(row + 0) * K + kk] * b;
    s1 += A[(row + 1) * K + kk] * b;
  }
  C[(row + 0) * N + col] = s0;
  C[(row + 1) * N + col] = s1;
});



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
  LLVMDisposeMessage:{args:["pointer"],returns:"pointer"},
}).symbols;
ls.LLVMInitializeNVPTXTargetInfo();ls.LLVMInitializeNVPTXTarget();ls.LLVMInitializeNVPTXTargetMC();ls.LLVMInitializeNVPTXAsmPrinter();
function llvmIRtoPTX(s: string): string {
  const ctx = ls.LLVMContextCreate(); const ib = Buffer.from(s + "\0");
  const mb = ls.LLVMCreateMemoryBufferWithMemoryRange(ffiPtr(ib), BigInt(ib.length - 1), ffiPtr(Buffer.from("k.ll\0")), 1);
  const ma = new BigUint64Array(1); const ep = new BigUint64Array(1);
  if (ls.LLVMParseIRInContext(ctx, mb, ffiPtr(ma), ffiPtr(ep))) throw Error("IR fail");
  const mod = Number(ma[0]); const tp = Buffer.from("nvptx64-nvidia-cuda\0"); const ta = new BigUint64Array(1);
  ls.LLVMGetTargetFromTriple(ffiPtr(tp), ffiPtr(ta), ffiPtr(new BigUint64Array(1)));
  const tm = ls.LLVMCreateTargetMachine(Number(ta[0]), ffiPtr(tp), ffiPtr(Buffer.from("sm_86\0")), ffiPtr(Buffer.from("\0")), 2, 0, 0);
  const pa = new BigUint64Array(1);
  if (ls.LLVMTargetMachineEmitToMemoryBuffer(tm, mod, 0, ffiPtr(new BigUint64Array(1)), ffiPtr(pa))) throw Error("PTX fail");
  return new CString(ls.LLVMGetBufferStart(Number(pa[0])), 0, Number(ls.LLVMGetBufferSize(Number(pa[0])))).toString();
}

const cs = dlopen(CUDA, {
  cuInit:{args:["u32"],returns:"i32"},cuDeviceGet:{args:["pointer","i32"],returns:"i32"},
  cuDeviceGetName:{args:["pointer","i32","i32"],returns:"i32"},cuCtxCreate_v2:{args:["pointer","u32","i32"],returns:"i32"},
  cuModuleLoadData:{args:["pointer","pointer"],returns:"i32"},cuModuleGetFunction:{args:["pointer","pointer","pointer"],returns:"i32"},
  cuModuleUnload:{args:["pointer"],returns:"i32"},
  cuMemAlloc_v2:{args:["pointer","i64"],returns:"i32"},
  cuMemcpyHtoD_v2:{args:["i64","pointer","i64"],returns:"i32"},
  cuMemcpyDtoH_v2:{args:["pointer","i64","i64"],returns:"i32"},
  cuLaunchKernel:{args:["pointer","u32","u32","u32","u32","u32","u32","u32","pointer","pointer","pointer"],returns:"i32"},
  cuMemFree_v2:{args:["i64"],returns:"i32"},cuCtxSynchronize:{args:[],returns:"i32"},
}).symbols;

const M = 1024, N = 1024, K = 1024;
const OPS = 2n * BigInt(M) * BigInt(N) * BigInt(K);
const hA = new Int8Array(M * K); for (let i = 0; i < M * K; i++) hA[i] = (Math.random() * 256 - 128) | 0;
const hB = new Int8Array(K * N); for (let i = 0; i < K * N; i++) hB[i] = (Math.random() * 256 - 128) | 0;

cs.cuInit(0); const dev = new Int32Array(1); cs.cuDeviceGet(dev, 0);
const nb = new Uint8Array(256); cs.cuDeviceGetName(nb, 256, dev[0]);
cs.cuCtxCreate_v2(Buffer.alloc(8), 0, dev[0]);
console.log(`GPU: ${new CString(ffiPtr(nb))}`);
console.log(`INT8 ${M}x${K} * ${K}x${N} = ${(Number(OPS) / 1e9).toFixed(0)} OPs\n`);

const dA = Buffer.alloc(8); cs.cuMemAlloc_v2(dA, BigInt(hA.length));
const dB = Buffer.alloc(8); cs.cuMemAlloc_v2(dB, BigInt(hB.length));
const dC = Buffer.alloc(8); cs.cuMemAlloc_v2(dC, BigInt(M * N * 4));
cs.cuMemcpyHtoD_v2(Number(dA.readBigUInt64LE(0)), Buffer.from(hA.buffer), BigInt(hA.length));
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)), Buffer.from(hB.buffer), BigInt(hB.length));

const pb = Buffer.alloc(6 * 8);
pb.writeBigUInt64LE(BigInt(Number(dA.readBigUInt64LE(0))), 0);
pb.writeBigUInt64LE(BigInt(Number(dB.readBigUInt64LE(0))), 8);
pb.writeBigUInt64LE(BigInt(Number(dC.readBigUInt64LE(0))), 16);
pb.writeInt32LE(M, 24); pb.writeInt32LE(N, 32); pb.writeInt32LE(K, 40);
const kp = Buffer.alloc(7 * 8); const pp = ffiPtr(pb);
for (let i = 0; i < 6; i++) kp.writeBigUInt64LE(BigInt(pp + i * 8), i * 8);
kp.writeBigUInt64LE(0n, 48);
const gX = Math.ceil(N / 16), gY = Math.ceil(M / 16);

function bench(k: any, label: string, tileRows: number, tileCols: number = 1) {
  const src = llvmIRtoPTX(k.source);
  const mod = Buffer.alloc(8);
  if (cs.cuModuleLoadData(mod, Buffer.from(src)) !== 0) { console.log(`  ${label}: load FAIL`); return; }
  const fn = Buffer.alloc(8); cs.cuModuleGetFunction(fn, Number(mod.readBigUInt64LE(0)), Buffer.from("kernel\0"));
  const gY2 = Math.ceil(M / (16 * tileRows));
  const gX2 = Math.ceil(N / (16 * tileCols));
  cs.cuLaunchKernel(Number(fn.readBigUInt64LE(0)), gX2, gY2, 1, 16, 16, 1, 0, null, kp, null);
  cs.cuCtxSynchronize();
  const T = 10; const t0 = performance.now();
  for (let i = 0; i < T; i++) cs.cuLaunchKernel(Number(fn.readBigUInt64LE(0)), gX, gY2, 1, 16, 16, 1, 0, null, kp, null);
  cs.cuCtxSynchronize();
  const ms = (performance.now() - t0) / T;
  const tops = Number(OPS) / (ms / 1000) / 1e12;
  console.log(`  ${label}: ${ms.toFixed(2)}ms = ${tops.toFixed(3)} TOPS`);
  cs.cuModuleUnload(mod);
}

bench(v1, "v1 naive", 1);
bench(v2, "v2 tile2", 2);


const hC = new Int32Array(M * N);
cs.cuMemcpyDtoH_v2(hC, Number(dC.readBigUInt64LE(0)), BigInt(M * N * 4));
let err = 0;
for (let i = 0; i < 100; i++) {
  const r = Math.floor(i / N), c = i % N; let ref = 0;
  for (let kk = 0; kk < K; kk++) ref = (ref + hA[r * K + kk] * hB[kk * N + c]) | 0;
  if (hC[i] !== ref) err++;
}
console.log(`  Errors (from last kernel): ${err}/100`);

cs.cuMemFree_v2(Number(dA.readBigUInt64LE(0))); cs.cuMemFree_v2(Number(dB.readBigUInt64LE(0))); cs.cuMemFree_v2(Number(dC.readBigUInt64LE(0)));
