// prototypes/rope.ts — Rotary Position Embedding (Qwen3.5), optimized (Liger-style).
//
// Qwen3.5 RoPE: mrope_interleaved=true, partial_rotary=0.25 (rotate first 64 of
// 256 head dims), rope_theta=1e7.
//
// Optimizations vs the first cut (f32, single-head, tiny bench):
//   • f16 in/out (the model dtype → 2× less bandwidth); rotation in f32 (Liger
//     casts q up to f32, rotates, stores back to f16) for precision.
//   • Liger-style: one kernel for a generic X[M,H,HD]→Y, launched once for Q
//     (H=HQ) and once for K (H=HKV) — cos/sin uploaded once, shared.
//   • benchmarked at realistic scale (M=8192, HQ=8, HKV=2, HD=256).
// cos/sin precomputed on the host (exact, no libdevice). Interleaved pairs via
// pointer-tensor strided loads (splatPtr+addptr, even/odd offsets). Non-rotary
// pass-through via a 2-D tiled pointer (respects row boundaries).
//
//   bun run prototypes/rope.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";
import { f32to16 } from "../src/kernel";

const HEAD_DIM = 256, ROTARY = 64, PAIRS = ROTARY / 2, NONROTARY = HEAD_DIM - ROTARY;
const THETA = 1e7;

function buildRoPE(M: number, H: number) {
  const b = new TTIRBuilder();
  const X = b.param("X", { ptr: "f16" });
  const Y = b.param("Y", { ptr: "f16" });
  const C = b.param("C", { ptr: "f32" });
  const S = b.param("S", { ptr: "f32" });
  const pidM = b.programId(0), pidH = b.programId(1);
  const rowOff = b.mul(b.add(b.mul(pidM, b.i32(H)), pidH), b.i32(HEAD_DIM));
  const tblOff = b.mul(pidM, b.i32(PAIRS));

  const ar = b.arange(0, PAIRS);
  const evenOff = b.add(b.mul(ar, b.i32(2)), rowOff);
  const oddOff  = b.add(evenOff, b.i32(1));
  const cosOff  = b.add(ar, tblOff);
  const even = b.fpext(b.load(b.addptr(b.splatPtr(X, PAIRS, "f16"), evenOff)), "f32");
  const odd  = b.fpext(b.load(b.addptr(b.splatPtr(X, PAIRS, "f16"), oddOff)),  "f32");
  const c = b.load(b.addptr(b.splatPtr(C, PAIRS, "f32"), cosOff));
  const s = b.load(b.addptr(b.splatPtr(S, PAIRS, "f32"), cosOff));
  const newEven = b.sub(b.mul(even, c), b.mul(odd, s));
  const newOdd  = b.add(b.mul(odd, c), b.mul(even, s));
  b.store(b.addptr(b.splatPtr(Y, PAIRS, "f16"), evenOff), b.fptrunc(newEven, "f16"));
  b.store(b.addptr(b.splatPtr(Y, PAIRS, "f16"), oddOff),  b.fptrunc(newOdd,  "f16"));

  const NRBLOCK = 1 << Math.ceil(Math.log2(NONROTARY));
  const rowIdx = b.add(b.mul(pidM, b.i32(H)), pidH);   // flat row = m·H + h
  const tpNRin = b.makeTensorPtr(X, [M * H, HEAD_DIM], [HEAD_DIM, 1], [rowIdx, b.i32(ROTARY)], [1, NRBLOCK], "f16", [1, 0]);
  const tpNRout = b.makeTensorPtr(Y, [M * H, HEAD_DIM], [HEAD_DIM, 1], [rowIdx, b.i32(ROTARY)], [1, NRBLOCK], "f16", [1, 0]);
  b.store(tpNRout, b.load(tpNRin, { boundaryCheck: [0, 1], padding: 1 }), { boundaryCheck: [0, 1] });
  return b.build("rope", 4);
}

function precompute(M: number) {
  const cos = new Float32Array(M * PAIRS), sin = new Float32Array(M * PAIRS);
  for (let m = 0; m < M; m++) for (let k = 0; k < PAIRS; k++) {
    const ang = m * Math.pow(THETA, -k / PAIRS);
    cos[m * PAIRS + k] = Math.cos(ang); sin[m * PAIRS + k] = Math.sin(ang);
  }
  return { cos, sin };
}
// f16→f32 decode (matches f32to16: denorm→0)
function f16to32(a: Uint16Array): Float32Array {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const u = a[i], sign = (u >> 15) & 1, exp = (u >> 10) & 0x1f, frac = u & 0x3ff;
    let f: number;
    if (exp === 0) f = 0; else if (exp === 31) f = frac ? NaN : Infinity;
    else f = Math.pow(2, exp - 15) * (1 + frac / 1024);
    o[i] = sign ? -f : f;
  }
  return o;
}
function refRoPE(Qf: Float32Array, cos: Float32Array, sin: Float32Array, M: number, H: number) {
  const Y = new Float32Array(M * H * HEAD_DIM);
  for (let m = 0; m < M; m++) for (let h = 0; h < H; h++) {
    const b0 = (m * H + h) * HEAD_DIM;
    for (let k = 0; k < PAIRS; k++) {
      const e = 2 * k, o = 2 * k + 1, c = cos[m * PAIRS + k], s = sin[m * PAIRS + k];
      const xe = Qf[b0 + e], xo = Qf[b0 + o];
      Y[b0 + e] = xe * c - xo * s; Y[b0 + o] = xo * c + xe * s;
    }
    for (let j = ROTARY; j < HEAD_DIM; j++) Y[b0 + j] = Qf[b0 + j];
  }
  return Y;
}

const M = Number(process.env.MP||8192), HQ = 8, HKV = 2;
const kQ = compileAndLoad(buildRoPE(M, HQ), "rope", 4);   // H baked into shapes → separate kernel per H
const kK = compileAndLoad(buildRoPE(M, HKV), "rope", 4);
console.log(`rope loaded (f16, head_dim=${HEAD_DIM}, rotary=${ROTARY}, interleaved, partial 0.25)`);

const Qf = new Float32Array(M * HQ * HEAD_DIM).map(() => (Math.random() * 2 - 1) * 0.5);
const Kf = new Float32Array(M * HKV * HEAD_DIM).map(() => (Math.random() * 2 - 1) * 0.5);
const Q = f32to16(Qf), K = f32to16(Kf), OQ = new Uint16Array(M * HQ * HEAD_DIM), OK = new Uint16Array(M * HKV * HEAD_DIM);
const { cos, sin } = precompute(M);
const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dOQ = cuAlloc(OQ.byteLength), dOK = cuAlloc(OK.byteLength);
const dC = cuAlloc(cos.byteLength), dS = cuAlloc(sin.byteLength);
cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dC, cos.buffer); cuHtoD(dS, sin.buffer);

// correctness (Q path)
cuLaunch(kQ, [M, HQ, 1], [128, 1, 1], [dQ, dOQ, dC, dS]); cuSync(); cuDtoH(OQ.buffer, dOQ);
const ref = refRoPE(Qf, cos, sin, M, HQ);
const OQf = f16to32(OQ);
let maxErr = 0;
for (let i = 0; i < OQf.length; i++) maxErr = Math.max(maxErr, Math.abs(OQf[i] - ref[i]));
console.log(`${maxErr < 0.05 ? "✓" : "✗"} rope correct (max err ${maxErr.toExponential(2)}, f16)`);

// bench: Q then K (Liger-style: cos/sin shared), host-timed on the ttir context
// (NOT kernel.ts events — that's a different CUDA context → INVALID_HANDLE).
for (let i = 0; i < 5; i++) { cuLaunch(kQ, [M, HQ, 1], [128, 1, 1], [dQ, dOQ, dC, dS]); cuLaunch(kK, [M, HKV, 1], [128, 1, 1], [dK, dOK, dC, dS]); }
cuSync();
const ITERS = 10;
const t0 = performance.now();
let rcX=0; for (let i = 0; i < ITERS; i++) { rcX=cuLaunch(kQ, [M, HQ, 1], [128, 1, 1], [dQ, dOQ, dC, dS]); cuLaunch(kK, [M, HKV, 1], [128, 1, 1], [dK, dOK, dC, dS]); } cuSync(); console.log("rc="+rcX);
cuSync();
const dt = (performance.now() - t0) / 1000 / ITERS;
const bytes = 2 * 2 * (M * HQ * HEAD_DIM + M * HKV * HEAD_DIM) + 2 * 2 * M * PAIRS * 4 / 2; // f16 q/k read+write; cos+sin f32
console.log(`bench Q+K ${M}x(HQ=${HQ},HKV=${HKV})x${HEAD_DIM}: ${(dt*1e6).toFixed(1)} µs/it, ${(bytes / dt / 1e9).toFixed(0)} GB/s effective mem-BW`);
cuFree(dQ); cuFree(dK); cuFree(dOQ); cuFree(dOK); cuFree(dC); cuFree(dS);
