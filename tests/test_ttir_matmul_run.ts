// test_ttir_matmul_run.ts — full FP16 matmul via fluent TTIRBuilder: compile + run + verify.
import { dlopen, ptr as ffiPtr, CString } from "bun:ffi";
import { TTIRBuilder, compileTTIR } from "../src/ttir";
import { f32to16 } from "../src/kernel";

const M = 512, N = 512, K = 512;
const BM = 64, BN = 64, BK = 512;

const b = new TTIRBuilder();
const A = b.param("A", { ptr: "f16" });
const Bp = b.param("B", { ptr: "f16" });
const C = b.param("C", { ptr: "f32" });

const pidM = b.programId(0);
const pidN = b.programId(1);
const offM = b.mul(pidM, b.i32(BM));
const offN = b.mul(pidN, b.i32(BN));

const tpA = b.makeTensorPtr(A, [M, K], [K, 1], [offM, b.i32(0)], [BM, BK], "f16", [1, 0]);
const tpB = b.makeTensorPtr(Bp, [K, N], [N, 1], [b.i32(0), offN], [BK, BN], "f16", [1, 0]);
const tpC = b.makeTensorPtr(C, [M, N], [N, 1], [offM, offN], [BM, BN], "f32", [1, 0]);

const acc = b.zeros([BM, BN], "f32");
const out = b.dot(b.load(tpA), b.load(tpB), acc);
b.store(tpC, out, { boundaryCheck: [0, 1] });

const ttir = b.build("matmul", 4);
const { ptx, shmem } = compileTTIR(ttir, 4);
console.log(`PTX: ${ptx.length} bytes, shmem=${shmem}`);

// ── GPU launch ──
const cs = dlopen("/run/opengl-driver/lib/libcuda.so", {
  cuInit: { args: ["u32"], returns: "i32" },
  cuDeviceGet: { args: ["ptr", "i32"], returns: "i32" },
  cuCtxCreate_v2: { args: ["ptr", "u32", "i64"], returns: "i32" },
  cuModuleLoadData: { args: ["ptr", "ptr"], returns: "i32" },
  cuModuleGetFunction: { args: ["ptr", "i64", "ptr"], returns: "i32" },
  cuMemAlloc_v2: { args: ["ptr", "i64"], returns: "i32" },
  cuMemcpyHtoD_v2: { args: ["i64", "ptr", "i64"], returns: "i32" },
  cuMemcpyDtoH_v2: { args: ["ptr", "i64", "i64"], returns: "i32" },
  cuLaunchKernel: { args: ["i64", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "ptr", "ptr", "ptr"], returns: "i32" },
  cuCtxSynchronize: { args: [], returns: "i32" },
  cuMemFree_v2: { args: ["i64"], returns: "i32" },
}).symbols;
cs.cuInit(0);
const dev = new Int32Array(1); cs.cuDeviceGet(dev, 0);
cs.cuCtxCreate_v2(Buffer.alloc(8), 0, BigInt(dev[0]));

// Init A, B with small random values (f16-range safe)
const hA32 = new Float32Array(M * K);
const hB32 = new Float32Array(K * N);
for (let i = 0; i < M * K; i++) hA32[i] = (Math.random() * 2 - 1) * 0.5;
for (let i = 0; i < K * N; i++) hB32[i] = (Math.random() * 2 - 1) * 0.5;
const hA = f32to16(hA32);
const hB = f32to16(hB32);
const hC = new Float32Array(M * N);

const szA = BigInt(hA.byteLength), szB = BigInt(hB.byteLength), szC = BigInt(hC.byteLength);
const dA = Buffer.alloc(8), dB = Buffer.alloc(8), dC = Buffer.alloc(8);
cs.cuMemAlloc_v2(dA, szA); cs.cuMemAlloc_v2(dB, szB); cs.cuMemAlloc_v2(dC, szC);
cs.cuMemcpyHtoD_v2(Number(dA.readBigUInt64LE(0)), Buffer.from(hA.buffer), szA);
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)), Buffer.from(hB.buffer), szB);

const mod = Buffer.alloc(8);
cs.cuModuleLoadData(mod, ffiPtr(Buffer.from(ptx)));
const fn = Buffer.alloc(8);
cs.cuModuleGetFunction(fn, Number(mod.readBigUInt64LE(0)), ffiPtr(Buffer.from("matmul\0")));

// 3 ptr params (A,B,C) — shim adds extra slots
const pb = Buffer.alloc(6 * 8);
pb.writeBigUInt64LE(dA.readBigUInt64LE(0), 0);
pb.writeBigUInt64LE(dB.readBigUInt64LE(0), 8);
pb.writeBigUInt64LE(dC.readBigUInt64LE(0), 16);
const pp = Number(ffiPtr(pb));
const kp = Buffer.alloc(7 * 8);
for (let i = 0; i < 6; i++) kp.writeBigUInt64LE(BigInt(pp + i * 8), i * 8);
kp.writeBigUInt64LE(0n, 48);

const gM = Math.ceil(M / BM), gN = Math.ceil(N / BN);
const threads = 4 * 32;
console.log(`Launch grid=${gM}x${gN} block=${threads}...`);
cs.cuLaunchKernel(Number(fn.readBigUInt64LE(0)), gM, gN, 1, threads, 1, 1, shmem, 0n, ffiPtr(kp), null);
cs.cuCtxSynchronize();
cs.cuMemcpyDtoH_v2(ffiPtr(Buffer.from(hC.buffer)), Number(dC.readBigUInt64LE(0)), szC);

// Reference (f16-rounded inputs)
let maxErr = 0, ok = true;
for (let r = 0; r < 32 && ok; r++) {
  for (let c = 0; c < 32; c++) {
    let ref = 0;
    for (let k = 0; k < K; k++) ref += hA32[r * K + k] * hB32[k * N + c];
    const diff = Math.abs(hC[r * N + c] - ref);
    if (diff > 1.0) { ok = false; console.log(`mismatch [${r},${c}]: got ${hC[r*N+c]} ref ${ref}`); break; }
    if (diff > maxErr) maxErr = diff;
  }
}
console.log(ok ? `✓ matmul correct (max err ${maxErr.toFixed(3)})` : "✗ FAILED");

cs.cuMemFree_v2(Number(dA.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dB.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dC.readBigUInt64LE(0)));
