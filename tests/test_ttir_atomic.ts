// test_ttir_atomic.ts — verify tt.atomic_add via a block-wise scatter-add:
// each thread adds its value to a shared output location, racing → atomic.
// Computes out[0] += sum of all (pid * BLOCK + tid) across the grid.
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const N = 4096;          // total elements / threads
const BLOCK = 256;
const GRID = N / BLOCK;

const b = new TTIRBuilder();
const Out = b.param("Out", { ptr: "f32" });
const pid = b.programId(0);
// offs = arange(0, BLOCK) + pid*BLOCK  → global thread index
const offs = b.add(b.arange(0, BLOCK), b.mul(pid, b.i32(BLOCK)));
// value = offs as f32 (each thread contributes its global index)
const val = b.sitofp(offs, "f32");
// atomic-add each value to Out[0] (a single scalar location).
// Splat Out to a 1-element pointer-tensor, addptr by 0, atomicAdd.
const outPtrs = b.splatPtr(Out, BLOCK, "f32");
const zeroOffs = b.zeros([BLOCK], "i32");
const outAt0 = b.addptr(outPtrs, zeroOffs);
b.atomicAdd(outAt0, val);
// (no store needed — atomicAdd writes back)

const ttir = b.build("atomadd", 4);
console.log("=== TTIR ===");
console.log(ttir);
const k = compileAndLoad(ttir, "atomadd", 4);
console.log(`loaded (shmem=${k.shmem})`);

const hOut = new Float32Array([0]);
const dOut = cuAlloc(4);
cuHtoD(dOut, hOut.buffer);
cuLaunch(k, [GRID, 1, 1], [128, 1, 1], [dOut]);
cuSync();
cuDtoH(hOut.buffer, dOut);

// Reference: sum of 0..N-1
const ref = (N * (N - 1)) / 2;
console.log(`out[0] = ${hOut[0]}, ref = ${ref}`);
console.log(Math.abs(hOut[0] - ref) < 1 ? `✓ atomicAdd correct` : "✗ FAILED");
cuFree(dOut);
