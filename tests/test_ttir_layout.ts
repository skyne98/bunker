// test_ttir_layout.ts — verify layout ops: trans (transpose) + reshape + broadcast.
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const M = 512, N = 512;
const BM = 64, BN = 64;

// Transpose: C = A^T. Load A[offM:offM+BM, offN:offN+BN], transpose, store to C[offN, offM].
const b = new TTIRBuilder();
const A = b.param("A", { ptr: "f32" });
const C = b.param("C", { ptr: "f32" });
const pidM = b.programId(0), pidN = b.programId(1);
const offM = b.mul(pidM, b.i32(BM)), offN = b.mul(pidN, b.i32(BN));
const tpA = b.makeTensorPtr(A, [M, N], [N, 1], [offM, offN], [BM, BN], "f32", [1, 0]);
const tile = b.load(tpA);
const t = b.trans(tile);
const tpC = b.makeTensorPtr(C, [N, M], [M, 1], [offN, offM], [BN, BM], "f32", [1, 0]);
b.store(tpC, t, { boundaryCheck: [0, 1] });

const ttir = b.build("transpose", 4);
console.log("=== TTIR ===");
console.log(ttir);
const k = compileAndLoad(ttir, "transpose", 4);
console.log(`loaded (shmem=${k.shmem})`);

const hA = new Float32Array(M * N);
const hC = new Float32Array(N * M);
for (let i = 0; i < M * N; i++) hA[i] = Math.random();
const dA = cuAlloc(hA.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer);
cuLaunch(k, [Math.ceil(M / BM), Math.ceil(N / BN), 1], [128, 1, 1], [dA, dC]);
cuSync();
cuDtoH(hC.buffer, dC);

let maxErr = 0, ok = true;
for (let r = 0; r < 64 && ok; r++) for (let c = 0; c < 64; c++) {
  const d = Math.abs(hC[c * M + r] - hA[r * N + c]);   // C[c,r] == A[r,c]
  if (d > 1e-5) { ok = false; console.log(`mismatch [${r},${c}]`); break; }
  if (d > maxErr) maxErr = d;
}
console.log(ok ? `✓ transpose (tt.trans) correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");
cuFree(dA); cuFree(dC);
