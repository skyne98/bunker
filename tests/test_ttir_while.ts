// test_ttir_while.ts — smoke test scf.while with an iter-arg accumulator.
// Computes out = K via: acc=0; while (i < K) { acc++; i++; } — exercises
// the scf.while before/do regions with iter-args.
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const K = 1000;
const b = new TTIRBuilder();
const Out = b.param("Out", { ptr: "i32" });

const cK = b.i32(K);
const c1 = b.i32(1);
// splat Out to a 1-element pointer-tensor and store the final acc.
const outPtrs = b.splatPtr(Out, 1, "i32");
const zeroOff = b.zeros([1], "i32");
const outAt0 = b.addptr(outPtrs, zeroOff);

const [i, acc] = b.while_(
  [b.i32(0), b.i32(0)],   // init: i=0, acc=0
  (bb, [i0, a0]) => {
    const cond = bb.lt(i0, cK);
    return { cond, iterArgs: [i0, a0] };
  },
  (bb, [i0, a0]) => {
    const i1 = bb.add(i0, c1);
    const a1 = bb.add(a0, c1);
    return [i1, a1];
  },
);
b.store(outAt0, b.splat(acc, [1], "i32"));

const ttir = b.build("whiletest", 4);
console.log("=== TTIR ===");
console.log(ttir);
const k = compileAndLoad(ttir, "whiletest", 4);
console.log(`loaded (shmem=${k.shmem})`);

const hOut = new Int32Array([0]);
const dOut = cuAlloc(4);
cuHtoD(dOut, hOut.buffer);
cuLaunch(k, [1, 1, 1], [128, 1, 1], [dOut]);
cuSync();
cuDtoH(hOut.buffer, dOut);
console.log(`out = ${hOut[0]}, ref = ${K} ${hOut[0] === K ? "✓" : "✗"}`);
cuFree(dOut);
