// test_kernel_ttir_if.ts — arrow-function lift with an `if` (scf.if).
// Block 0 writes A to C; other blocks write zero. Verifies the lift's
// conditional lowering (scalar cond → scf.if with a masked store body).
import { ffiPtr } from "bun:ffi";
import { kernel_ttir, ptr, i32, f32, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const N = 4096;
const BLOCK = 256;

const k = kernel_ttir((A: ptr<f32>, C: ptr<f32>, n: i32) => {
  const pid = programId(0);
  const offs = arange(0, BLOCK) + pid * BLOCK;
  const mask = offs < n;
  const a = load(A + offs, { mask });
  if (pid < 1) {
    store(C + offs, a, { mask });
  }
});

console.log("=== TTIR ===");
console.log(k.ttir);
const loaded = compileAndLoad(k.ttir, "kernel", 4);
console.log(`loaded (shmem=${loaded.shmem})`);

const hA = new Float32Array(N);
const hC = new Float32Array(N);
for (let i = 0; i < N; i++) hA[i] = i + 1;
const dA = cuAlloc(hA.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer);
cuLaunch(loaded, [Math.ceil(N / BLOCK), 1, 1], [128, 1, 1], [dA, dC, N]);
cuSync();
cuDtoH(hC.buffer, dC);

// Only block 0 should have written (BLOCK elements = hA[0..255]).
let ok = true;
for (let i = 0; i < N; i++) {
  const expected = i < BLOCK ? hA[i] : 0;
  if (Math.abs(hC[i] - expected) > 1e-6) { ok = false; console.log(`mismatch ${i}: got ${hC[i]} exp ${expected}`); if (i > 4) break; }
}
console.log(ok ? `✓ arrow-lift if (scf.if) correct` : "✗ FAILED");
cuFree(dA); cuFree(dC);
void ffiPtr;
