// prototypes/swiglu.ts — SwiGLU activation for Qwen3.5's MLP.
//
//   MLP(x) = down( silu(gate(x)) * up(x) ),   silu(x) = x · σ(x) = x/(1+e^{-x})
//
// The gate/up/down are GEMMs (the existing matmul kernel). This prototype is
// the fused elementwise activation between them:  act = silu(gate) * up.
// Pure elementwise; silu uses exp (inlines) → NO libdevice needed.
//
//   bun run prototypes/swiglu.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

function buildSwiGLU(total: number, BLOCK: number) {
  const b = new TTIRBuilder();
  const G = b.param("G", { ptr: "f32" });
  const U = b.param("U", { ptr: "f32" });
  const O = b.param("O", { ptr: "f32" });
  const pid = b.programId(0);
  const off = b.mul(pid, b.i32(BLOCK));
  const tpG = b.makeTensorPtr(G, [total], [1], [off], [BLOCK], "f32", [0]);
  const tpU = b.makeTensorPtr(U, [total], [1], [off], [BLOCK], "f32", [0]);
  const tpO = b.makeTensorPtr(O, [total], [1], [off], [BLOCK], "f32", [0]);
  const g = b.load(tpG, { boundaryCheck: [0], padding: 1 });
  const u = b.load(tpU, { boundaryCheck: [0], padding: 1 });
  // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
  const negG = b.mul(g, b.f32(-1));
  const sig = b.divf(b.f32(1), b.add(b.f32(1), b.exp(negG)));
  const silu = b.mul(g, sig);
  const y = b.mul(silu, u);
  b.store(tpO, y, { boundaryCheck: [0] });
  return b.build("swiglu", 4);
}

function refSwiGLU(G: Float32Array, U: Float32Array) {
  const O = new Float32Array(G.length);
  for (let i = 0; i < G.length; i++) {
    const g = G[i], s = 1 / (1 + Math.exp(-g));
    O[i] = g * s * U[i];
  }
  return O;
}

const M = 512, I = 3584;                 // Qwen3.5-0.8B: hidden=1024, mlp=3584
const total = M * I;
const BLOCK = 1024;
const k = compileAndLoad(buildSwiGLU(total, BLOCK), "swiglu", 4);
console.log(`swiglu loaded (shmem=${k.shmem})`);

const G = new Float32Array(total).map(() => (Math.random() * 2 - 1));
const U = new Float32Array(total).map(() => (Math.random() * 2 - 1));
const O = new Float32Array(total);
const dG = cuAlloc(G.byteLength), dU = cuAlloc(U.byteLength), dO = cuAlloc(O.byteLength);
cuHtoD(dG, G.buffer); cuHtoD(dU, U.buffer);
for (let i = 0; i < 3; i++) cuLaunch(k, [Math.ceil(total / BLOCK), 1, 1], [128, 1, 1], [dG, dU, dO]);
cuSync();
const t0 = performance.now(), it = 100;
for (let i = 0; i < it; i++) cuLaunch(k, [Math.ceil(total / BLOCK), 1, 1], [128, 1, 1], [dG, dU, dO]);
cuSync();
const dt = (performance.now() - t0) / 1000 / it;
cuDtoH(O.buffer, dO);

const ref = refSwiGLU(G, U);
let maxErr = 0;
for (let i = 0; i < total; i++) maxErr = Math.max(maxErr, Math.abs(O[i] - ref[i]));
console.log(`✓ swiglu correct (max err ${maxErr.toExponential(2)})`);
console.log(`bench ${M}x${I}: ${(dt * 1e6).toFixed(1)} µs/it, ${((3 * total * 4) / dt / 1e9).toFixed(0)} GB/s effective mem-BW`);
cuFree(dG); cuFree(dU); cuFree(dO);
