// test_kernel_ttir.ts — arrow-function AST→TTIR lift (pleasant authoring surface).
// Write a plain TS arrow function; kernel_ttir parses it and lowers to TTIR.
import { ffiPtr } from "bun:ffi";
import { kernel_ttir, ptr, i32, f32, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const N = 4096;
const BLOCK = 256;

// ── Vector add as a plain arrow function ──
const add = kernel_ttir((A: ptr<f32>, B: ptr<f32>, C: ptr<f32>, n: i32) => {
  const pid = programId(0);
  const offs = arange(0, BLOCK) + pid * BLOCK;
  const mask = offs < n;
  const a = load(A + offs, { mask });
  const b = load(B + offs, { mask });
  store(C + offs, a + b, { mask });
});

console.log("=== Generated TTIR ===");
console.log(add.ttir);
const k = compileAndLoad(add.ttir, "kernel", 4);
console.log(`PTX loaded (shmem=${k.shmem})`);

const hA = new Float32Array(N);
const hB = new Float32Array(N);
const hC = new Float32Array(N);
for (let i = 0; i < N; i++) { hA[i] = i; hB[i] = i * 2; }
const dA = cuAlloc(hA.byteLength), dB = cuAlloc(hB.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer);
cuHtoD(dB, hB.buffer);
cuLaunch(k, [Math.ceil(N / BLOCK), 1, 1], [128, 1, 1], [dA, dB, dC, N]);
cuSync();
cuDtoH(hC.buffer, dC);

let ok = true;
for (let i = 0; i < N; i++) {
  if (Math.abs(hC[i] - (hA[i] + hB[i])) > 0.001) { ok = false; console.log(`mismatch ${i}: ${hC[i]}`); break; }
}
console.log(ok ? `✓ kernel_ttir vector-add correct (${N} elements)` : "✗ FAILED");
cuFree(dA); cuFree(dB); cuFree(dC);
void ffiPtr;
