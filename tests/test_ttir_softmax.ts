// test_ttir_softmax.ts — online-block softmax: max → sub → exp → sum → div.
// A flagship kernel exercising tt.reduce (max+sum), math.exp, elementwise,
// and tiled pointers — all in the portable TTIR layer.
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const M = 512, N = 256;
const BM = 64, BN = 256;   // one block per BM rows, full BN cols per tile

const b = new TTIRBuilder();
const A = b.param("A", { ptr: "f32" });
const C = b.param("C", { ptr: "f32" });
const pidM = b.programId(0);
const offM = b.mul(pidM, b.i32(BM));

const tpA = b.makeTensorPtr(A, [M, N], [N, 1], [offM, b.i32(0)], [BM, BN], "f32", [1, 0]);
const tpC = b.makeTensorPtr(C, [M, N], [N, 1], [offM, b.i32(0)], [BM, BN], "f32", [1, 0]);

const x = b.load(tpA);
// row-max (reduce axis 1) → tensor<BMxf32>, splat back to tile for broadcast.
const rowMax = b.max(x, 1);                 // tensor<BMxf32>
const maxBc = b.broadcast(b.expandDims(rowMax, 1), [BM, BN]);  // tensor<BMxBNxf32>
const shifted = b.sub(x, maxBc);             // x - max
const expX = b.exp(shifted);                 // exp(x - max)
const denom = b.sum(expX, 1);                // tensor<BMxf32>
const denomBc = b.broadcast(b.expandDims(denom, 1), [BM, BN]);
const out = b.divf(expX, denomBc);           // softmax
b.store(tpC, out, { boundaryCheck: [0, 1] });

const ttir = b.build("softmax", 4);
const k = compileAndLoad(ttir, "softmax", 4);
console.log(`softmax loaded (shmem=${k.shmem})`);

const hA = new Float32Array(M * N);
const hC = new Float32Array(M * N);
for (let i = 0; i < M * N; i++) hA[i] = (Math.random() * 2 - 1) * 3;
const dA = cuAlloc(hA.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer);
cuLaunch(k, [Math.ceil(M / BM), 1, 1], [128, 1, 1], [dA, dC]);
cuSync();
cuDtoH(hC.buffer, dC);

let maxErr = 0, ok = true;
for (let r = 0; r < M && ok; r++) {
  let mx = -Infinity; for (let c = 0; c < N; c++) mx = Math.max(mx, hA[r * N + c]);
  let s = 0; for (let c = 0; c < N; c++) s += Math.exp(hA[r * N + c] - mx);
  for (let c = 0; c < N; c++) {
    const ref = Math.exp(hA[r * N + c] - mx) / s;
    const d = Math.abs(hC[r * N + c] - ref);
    if (d > 1e-4) { ok = false; console.log(`mismatch [${r},${c}]: got ${hC[r*N+c]} ref ${ref}`); if (r > 2) break; }
    if (d > maxErr) maxErr = d;
  }
}
console.log(ok ? `✓ softmax correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");
cuFree(dA); cuFree(dC);
