// prototypes/rope.ts — Rotary Position Embedding for Qwen3.5 full-attention.
//
// Qwen3.5 RoPE: mrope_interleaved=true (interleaved pairs), partial_rotary=0.25
// (rotate the first 0.25·head_dim dims; the rest pass through). head_dim=256 →
// rotary_dim=64, pairs=32, non-rotary=192. rope_theta=1e7.
//
//   For each position m and pair k (elements 2k, 2k+1):
//     y[2k]   = x[2k]·cos - x[2k+1]·sin
//     y[2k+1] = x[2k+1]·cos + x[2k]·sin
//     y[j]    = x[j]   for j >= rotary_dim      (pass-through)
//
// cos/sin are PRECOMPUTED on the host (exact, no libdevice) and passed as
// [M, pairs] tables — the kernel just loads + rotates. Interleaved pairs are
// formed via strided (stride-2) tiled loads of the even/odd elements.
//
//   bun run prototypes/rope.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const HEAD_DIM = 256, ROTARY = 64, PAIRS = ROTARY / 2, NONROTARY = HEAD_DIM - ROTARY;
const THETA = 1e7;

function buildRoPE(M: number) {
  const b = new TTIRBuilder();
  const X = b.param("X", { ptr: "f32" });
  const Y = b.param("Y", { ptr: "f32" });
  const C = b.param("C", { ptr: "f32" });   // cos table [M, PAIRS]
  const S = b.param("S", { ptr: "f32" });   // sin table [M, PAIRS]
  const pid = b.programId(0);
  const rowOff = b.mul(pid, b.i32(HEAD_DIM));        // m·head_dim
  const tblOff = b.mul(pid, b.i32(PAIRS));           // m·pairs  (cos/sin row)

  // rotary portion: strided (stride 2) loads of even/odd elements → pairs
  const ar = b.arange(0, PAIRS);
  const evenOff = b.add(b.mul(ar, b.i32(2)), rowOff);
  const oddOff  = b.add(evenOff, b.i32(1));
  const cosOff  = b.add(ar, tblOff);
  const even = b.load(b.addptr(b.splatPtr(X, PAIRS, "f32"), evenOff));
  const odd  = b.load(b.addptr(b.splatPtr(X, PAIRS, "f32"), oddOff));
  const c = b.load(b.addptr(b.splatPtr(C, PAIRS, "f32"), cosOff));
  const s = b.load(b.addptr(b.splatPtr(S, PAIRS, "f32"), cosOff));

  const newEven = b.sub(b.mul(even, c), b.mul(odd, s));     // x_even·cos - x_odd·sin
  const newOdd  = b.add(b.mul(odd, c), b.mul(even, s));     // x_odd·cos + x_even·sin

  b.store(b.addptr(b.splatPtr(Y, PAIRS, "f32"), evenOff), newEven);
  b.store(b.addptr(b.splatPtr(Y, PAIRS, "f32"), oddOff), newOdd);

  // non-rotary pass-through: 2D tiled ptr so the row boundary is respected
  // (a flat 1D tile of 256 would overrun into the next row). block [1, NRBLOCK].
  const NRBLOCK = 1 << Math.ceil(Math.log2(NONROTARY));
  const tpNRin = b.makeTensorPtr(X, [M, HEAD_DIM], [HEAD_DIM, 1], [pid, b.i32(ROTARY)], [1, NRBLOCK], "f32", [1, 0]);
  const tpNRout = b.makeTensorPtr(Y, [M, HEAD_DIM], [HEAD_DIM, 1], [pid, b.i32(ROTARY)], [1, NRBLOCK], "f32", [1, 0]);
  b.store(tpNRout, b.load(tpNRin, { boundaryCheck: [0, 1], padding: 1 }), { boundaryCheck: [0, 1] });
  return b.build("rope", 4);
}

// host reference + cos/sin precompute (interleaved, partial)
function precompute(M: number) {
  const cos = new Float32Array(M * PAIRS), sin = new Float32Array(M * PAIRS);
  for (let m = 0; m < M; m++) for (let k = 0; k < PAIRS; k++) {
    const freq = Math.pow(THETA, -k / PAIRS);          // theta^(-k/pairs) = theta^(-2k/rotary)
    const ang = m * freq;
    cos[m * PAIRS + k] = Math.cos(ang);
    sin[m * PAIRS + k] = Math.sin(ang);
  }
  return { cos, sin };
}
function refRoPE(X: Float32Array, cos: Float32Array, sin: Float32Array, M: number) {
  const Y = new Float32Array(M * HEAD_DIM);
  for (let m = 0; m < M; m++) {
    for (let k = 0; k < PAIRS; k++) {
      const e = 2 * k, o = 2 * k + 1, c = cos[m * PAIRS + k], s = sin[m * PAIRS + k];
      const xe = X[m * HEAD_DIM + e], xo = X[m * HEAD_DIM + o];
      Y[m * HEAD_DIM + e] = xe * c - xo * s;
      Y[m * HEAD_DIM + o] = xo * c + xe * s;
    }
    for (let j = ROTARY; j < HEAD_DIM; j++) Y[m * HEAD_DIM + j] = X[m * HEAD_DIM + j];
  }
  return Y;
}

const M = 512;
const k = compileAndLoad(buildRoPE(M), "rope", 4);
console.log(`rope loaded (shmem=${k.shmem}, head_dim=${HEAD_DIM}, rotary=${ROTARY}, interleaved, partial 0.25)`);

const X = new Float32Array(M * HEAD_DIM).map(() => (Math.random() * 2 - 1));
const Y = new Float32Array(M * HEAD_DIM);
const { cos, sin } = precompute(M);
const dX = cuAlloc(X.byteLength), dY = cuAlloc(Y.byteLength), dC = cuAlloc(cos.byteLength), dS = cuAlloc(sin.byteLength);
cuHtoD(dX, X.buffer); cuHtoD(dC, cos.buffer); cuHtoD(dS, sin.buffer);
for (let i = 0; i < 3; i++) cuLaunch(k, [M, 1, 1], [128, 1, 1], [dX, dY, dC, dS]);
cuSync();
const t0 = performance.now(), it = 200;
for (let i = 0; i < it; i++) cuLaunch(k, [M, 1, 1], [128, 1, 1], [dX, dY, dC, dS]);
cuSync();
const dt = (performance.now() - t0) / 1000 / it;
cuDtoH(Y.buffer, dY);

const ref = refRoPE(X, cos, sin, M);
let maxErr = 0;
for (let i = 0; i < M * HEAD_DIM; i++) maxErr = Math.max(maxErr, Math.abs(Y[i] - ref[i]));
console.log(`${maxErr < 1e-4 ? "✓" : "✗"} rope correct (max err ${maxErr.toExponential(2)})`);
console.log(`bench ${M}x${HEAD_DIM}: ${(dt * 1e6).toFixed(1)} µs/it, ${((2 * M * HEAD_DIM + 2 * M * PAIRS) * 4 / dt / 1e9).toFixed(0)} GB/s effective mem-BW`);
cuFree(dX); cuFree(dY); cuFree(dC); cuFree(dS);
