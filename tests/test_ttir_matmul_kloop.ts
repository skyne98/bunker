// test_ttir_matmul_kloop.ts — FP16 matmul with a K-loop (scf.for + tt.advance +
// iter-arg accumulator). Uses the shared cu() helpers + compileAndLoad.
import { ffiPtr } from "bun:ffi";
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";
import { f32to16 } from "../src/kernel";

const M = 512, N = 512, K = 512;
const BM = 64, BN = 64, BK = 64;

const b = new TTIRBuilder();
const A = b.param("A", { ptr: "f16" });
const Bp = b.param("B", { ptr: "f16" });
const C = b.param("C", { ptr: "f32" });
const pidM = b.programId(0), pidN = b.programId(1);
const offM = b.mul(pidM, b.i32(BM)), offN = b.mul(pidN, b.i32(BN));

const tpA0 = b.makeTensorPtr(A, [M, K], [K, 1], [offM, b.i32(0)], [BM, BK], "f16", [1, 0]);
const tpB0 = b.makeTensorPtr(Bp, [K, N], [N, 1], [b.i32(0), offN], [BK, BN], "f16", [1, 0]);
const tpC = b.makeTensorPtr(C, [M, N], [N, 1], [offM, offN], [BM, BN], "f32", [1, 0]);

const acc0 = b.zeros([BM, BN], "f32");
const [acc] = b.forIter(b.index(0), b.index(K), b.index(BK), [acc0, tpA0, tpB0], (bb, k, [a, tpA, tpB]) => {
  const next = bb.dot(bb.load(tpA), bb.load(tpB), a);
  const nextTpA = bb.advance(tpA, [bb.i32(0), bb.i32(BK)]);
  const nextTpB = bb.advance(tpB, [bb.i32(BK), bb.i32(0)]);
  return [next, nextTpA, nextTpB];
});
b.store(tpC, acc, { boundaryCheck: [0, 1] });

const ttir = b.build("matmul_kloop", 4);
const k = compileAndLoad(ttir, "matmul_kloop", 4);
console.log(`K-loop matmul loaded (shmem=${k.shmem})`);

const hA32 = new Float32Array(M * K), hB32 = new Float32Array(K * N);
for (let i = 0; i < M * K; i++) hA32[i] = (Math.random() * 2 - 1) * 0.5;
for (let i = 0; i < K * N; i++) hB32[i] = (Math.random() * 2 - 1) * 0.5;
const hA = f32to16(hA32), hB = f32to16(hB32), hC = new Float32Array(M * N);

const dA = cuAlloc(hA.byteLength), dB = cuAlloc(hB.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer); cuHtoD(dB, hB.buffer);

const gM = Math.ceil(M / BM), gN = Math.ceil(N / BN);
// args: A, B, C (3 ptr params). cuLaunch pads trailing slots for the shim.
const lr = cuLaunch(k, [gM, gN, 1], [128, 1, 1], [dA, dB, dC]);
cuSync();
console.log(`launch rc=${lr}`);
cuDtoH(hC.buffer, dC);

let maxErr = 0, ok = true;
for (let r = 0; r < 32 && ok; r++) for (let c = 0; c < 32; c++) {
  let ref = 0; for (let kk = 0; kk < K; kk++) ref += hA32[r * K + kk] * hB32[kk * N + c];
  const d = Math.abs(hC[r * N + c] - ref);
  if (d > 1) { ok = false; console.log(`mismatch [${r},${c}]: got ${hC[r*N+c]} ref ${ref}`); break; }
  if (d > maxErr) maxErr = d;
}
console.log(ok ? `✓ K-loop matmul (scf.for + tt.advance) correct (max err ${maxErr.toFixed(3)})` : "✗ FAILED");
cuFree(dA); cuFree(dB); cuFree(dC);
void ffiPtr;
