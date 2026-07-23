// test_ttir_builder.ts — verify the fluent TTIRBuilder produces valid TTIR,
// compiles via the shim, and runs correctly on the 3090. Uses the robust
// compileAndLoad + cu helpers.
import { ffiPtr } from "bun:ffi";
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const N = 4096;
const BLOCK = 256;
const grid = Math.ceil(N / BLOCK);

const b = new TTIRBuilder();
const A = b.param("A", { ptr: "f32" });
const B = b.param("B", { ptr: "f32" });
const C = b.param("C", { ptr: "f32" });
const Nv = b.param("N", "i32");

const pid = b.programId(0);
const base = b.mul(pid, b.i32(BLOCK));
const offs = b.add(b.arange(0, BLOCK), base);
const mask = b.lt(offs, b.splat(Nv, [BLOCK], "i32"));
const Ap = b.addptr(b.splatPtr(A, BLOCK, "f32"), offs);
const Bp = b.addptr(b.splatPtr(B, BLOCK, "f32"), offs);
const a = b.load(Ap, { mask });
const bb = b.load(Bp, { mask });
const c = b.add(a, bb);
const Cp = b.addptr(b.splatPtr(C, BLOCK, "f32"), offs);
b.store(Cp, c, { mask });

const ttir = b.build("add_kernel", 4);
const k = compileAndLoad(ttir, "add_kernel", 4);
console.log(`PTX loaded (shmem=${k.shmem})`);

const hA = new Float32Array(N);
const hB = new Float32Array(N);
const hC = new Float32Array(N);
for (let i = 0; i < N; i++) { hA[i] = i; hB[i] = i * 2; }
const dA = cuAlloc(hA.byteLength), dB = cuAlloc(hB.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer); cuHtoD(dB, hB.buffer);
cuLaunch(k, [grid, 1, 1], [128, 1, 1], [dA, dB, dC, N]);
cuSync();
cuDtoH(hC.buffer, dC);

let ok = true;
for (let i = 0; i < N; i++) {
  if (Math.abs(hC[i] - (hA[i] + hB[i])) > 0.001) { ok = false; console.log(`mismatch ${i}`); break; }
}
console.log(ok ? `✓ All ${N} elements correct` : "✗ FAILED");
cuFree(dA); cuFree(dB); cuFree(dC);
void ffiPtr;
