// test_kernel_ttir_for.ts — arrow-function lift with a `for` loop (scf.for).
// Body stores the same tile 3× (idempotent) → result C == A, verifying the
// lift's for-loop lowering (init/cond/step + body).
import { ffiPtr } from "bun:ffi";
import { kernel_ttir, ptr, i32, f32, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const N = 4096;
const BLOCK = 256;

const k = kernel_ttir((A: ptr<f32>, C: ptr<f32>, n: i32) => {
  const pid = programId(0);
  const offs = arange(0, BLOCK) + pid * BLOCK;
  const mask = offs < n;
  const a = load(A + offs, { mask });
  for (let i = 0; i < 3; i += 1) {
    store(C + offs, a, { mask });
  }
});

const loaded = compileAndLoad(k.ttir, "kernel", 4);
console.log(`loaded (shmem=${loaded.shmem})`);

const hA = new Float32Array(N);
const hC = new Float32Array(N);
for (let i = 0; i < N; i++) hA[i] = (i + 1) * 0.5;
const dA = cuAlloc(hA.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer);
cuLaunch(loaded, [Math.ceil(N / BLOCK), 1, 1], [128, 1, 1], [dA, dC, N]);
cuSync();
cuDtoH(hC.buffer, dC);

let ok = true;
for (let i = 0; i < N; i++) {
  if (Math.abs(hC[i] - hA[i]) > 1e-6) { ok = false; console.log(`mismatch ${i}: ${hC[i]} vs ${hA[i]}`); break; }
}
console.log(ok ? `✓ arrow-lift for (scf.for) correct — C == A after 3 idempotent stores` : "✗ FAILED");
cuFree(dA); cuFree(dC);
void ffiPtr;
