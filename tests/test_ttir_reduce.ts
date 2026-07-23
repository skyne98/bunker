// test_ttir_reduce.ts — verify tt.reduce (row-sum) end-to-end on the 3090.
// Exercises the reduction op, a key feature for softmax/layernorm/etc.
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const M = 512, N = 256;
const BM = 64, BN = 256;   // one block per BM rows, full BN cols

const b = new TTIRBuilder();
const A = b.param("A", { ptr: "f32" });
const C = b.param("C", { ptr: "f32" });   // row sums, M elements
const pidM = b.programId(0);
const offM = b.mul(pidM, b.i32(BM));

// Tiled pointer to A[offM:offM+BM, 0:BN]
const tpA = b.makeTensorPtr(A, [M, N], [N, 1], [offM, b.i32(0)], [BM, BN], "f32", [1, 0]);
const tile = b.load(tpA);
// Row-sum: reduce along axis 1 (the BN dim) → tensor<BMxf32>
const rowSums = b.sum(tile, 1);

// Store BM row-sums to C[offM:offM+BM]
const tpC = b.makeTensorPtr(C, [M], [1], [offM], [BM], "f32", [0]);
b.store(tpC, rowSums);

const ttir = b.build("rowsum", 4);
console.log("=== TTIR ===");
console.log(ttir);

const k = compileAndLoad(ttir, "rowsum", 4);
console.log(`loaded (shmem=${k.shmem})`);

const hA = new Float32Array(M * N);
const hC = new Float32Array(M);
for (let i = 0; i < M * N; i++) hA[i] = (Math.random() * 2 - 1);
const dA = cuAlloc(hA.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer);
cuLaunch(k, [Math.ceil(M / BM), 1, 1], [128, 1, 1], [dA, dC]);
cuSync();
cuDtoH(hC.buffer, dC);

let maxErr = 0, ok = true;
for (let r = 0; r < M; r++) {
  let ref = 0; for (let c = 0; c < N; c++) ref += hA[r * N + c];
  const d = Math.abs(hC[r] - ref);
  if (d > 0.01) { ok = false; console.log(`mismatch row ${r}: got ${hC[r]} ref ${ref}`); if (r > 4) break; }
  if (d > maxErr) maxErr = d;
}
console.log(ok ? `✓ row-sum reduce correct (max err ${maxErr.toFixed(4)})` : "✗ FAILED");
cuFree(dA); cuFree(dC);
