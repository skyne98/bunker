// test_kernel_ttir_math.ts — arrow-function lift with math (exp) + control flow (for).
import { ffiPtr } from "bun:ffi";
import { kernel_ttir, ptr, i32, f32, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const N = 8192;
const BLOCK = 256;

// Elementwise exp via the arrow-function lift (pointer-tensor style).
const expk = kernel_ttir((A: ptr<f32>, C: ptr<f32>, n: i32) => {
  const pid = programId(0);
  const offs = arange(0, BLOCK) + pid * BLOCK;
  const mask = offs < n;
  const a = load(A + offs, { mask });
  store(C + offs, exp(a), { mask });
});

console.log("=== TTIR (exp kernel) ===");
console.log(expk.ttir);
const k = compileAndLoad(expk.ttir, "kernel", 4);
console.log(`loaded (shmem=${k.shmem})`);

const hA = new Float32Array(N);
const hC = new Float32Array(N);
for (let i = 0; i < N; i++) hA[i] = (Math.random() * 2 - 1);
const dA = cuAlloc(hA.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer);
cuLaunch(k, [Math.ceil(N / BLOCK), 1, 1], [128, 1, 1], [dA, dC, N]);
cuSync();
cuDtoH(hC.buffer, dC);

let maxErr = 0, ok = true;
for (let i = 0; i < N; i++) {
  const ref = Math.exp(hA[i]);
  const d = Math.abs(hC[i] - ref);
  if (d > 1e-3) { ok = false; console.log(`mismatch ${i}: got ${hC[i]} ref ${ref}`); if (i > 3) break; }
  if (d > maxErr) maxErr = d;
}
console.log(ok ? `✓ exp kernel (arrow lift + math) correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");
cuFree(dA); cuFree(dC);
void ffiPtr;
