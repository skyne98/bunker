// prototypes/gdn_fast.ts — fast chunked-parallel GatedDeltaNet prefill (fla-faithful).
//
// Ports fla's chunked structure (gated_delta_rule/chunk.py + common/chunk_h.py
// + chunk_o.py): state-tiled, high-occupancy. The clean gdn_chunk was correct
// but 6× slower than the naive recurrence (Neumann O(C³) WY + per-(b,h) grid).
// This fixes both:
//   • WY on the HOST (efficient forward-substitution, no Neumann, no GPU slicing):
//     A=(I−L)⁻¹, w=A@k_beta, u=A@v_beta.
//   • State-dim tiling: grid [B·H, NV] for fwd_h, [B·H, nChunks, NV] for fwd_o
//     (fla's pattern) → high occupancy.
// Two GPU kernels (fla's split):
//   fwd_h: h[c] stored (before update); v_new = u−w@h; h += kᵀ@v_new. (1 iter-arg;
//          tiled pointers recomputed from iv — carrying 6 iter-args crashed.)
//   fwd_o: o = scale·(q@h + tril(q@kᵀ)@v_new)  (loops BK for full qk).
//   bun run prototypes/gdn_fast.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

function buildFwdH(BH: number, nC: number, C: number, DK: number, DV: number, BV: number) {
  const b = new TTIRBuilder();
  const W = b.param("W", { ptr: "f32" }), U = b.param("U", { ptr: "f32" });
  const K = b.param("K", { ptr: "f32" }), H = b.param("H", { ptr: "f32" }), VN = b.param("VN", { ptr: "f32" });
  const pidBH = b.programId(0), pidV = b.programId(1);
  const vOff = b.mul(pidV, b.i32(BV));
  const bhRow = b.mul(pidBH, b.i32(nC * C)), bhChunk = b.mul(pidBH, b.i32(nC));
  const h0 = b.zeros([DK, BV], "f32");
  b.forIter(b.index(0), b.index(nC), b.index(1), [h0], (bb, iv, [h]) => {
    const c = bb.indexCast(iv, "i32");
    const cRow = bb.add(bhRow, bb.mul(c, b.i32(C)));
    const hRow = bb.mul(bb.add(bhChunk, c), b.i32(DK));
    const tpW = bb.makeTensorPtr(W, [BH * nC * C, DK], [DK, 1], [cRow, b.i32(0)], [C, DK], "f32", [1, 0]);
    const tpU = bb.makeTensorPtr(U, [BH * nC * C, DV], [DV, 1], [cRow, vOff], [C, BV], "f32", [1, 0]);
    const tpK = bb.makeTensorPtr(K, [BH * nC * C, DK], [DK, 1], [cRow, b.i32(0)], [C, DK], "f32", [1, 0]);
    const tpH = bb.makeTensorPtr(H, [BH * nC * DK, DV], [DV, 1], [hRow, vOff], [DK, BV], "f32", [1, 0]);
    const tpVN = bb.makeTensorPtr(VN, [BH * nC * C, DV], [DV, 1], [cRow, vOff], [C, BV], "f32", [1, 0]);
    const w = bb.load(tpW), u = bb.load(tpU), k = bb.load(tpK);
    const vNew = bb.sub(u, bb.dot(w, h));
    bb.store(tpH, h, { boundaryCheck: [0, 1] });
    bb.store(tpVN, vNew, { boundaryCheck: [0, 1] });
    return [bb.add(h, bb.dot(bb.trans(k), vNew))];
  });
  return b.build("gdn_fwd_h", 4);
}

function buildFwdO(BH: number, nC: number, C: number, DK: number, DV: number, BV: number) {
  const b = new TTIRBuilder();
  const Q = b.param("Q", { ptr: "f32" }), K = b.param("K", { ptr: "f32" }), VN = b.param("VN", { ptr: "f32" });
  const H = b.param("H", { ptr: "f32" }), O = b.param("O", { ptr: "f32" });
  const scale = b.f32(1 / Math.sqrt(DK));
  const pidBH = b.programId(0), pidC = b.programId(1), pidV = b.programId(2);
  const vOff = b.mul(pidV, b.i32(BV));
  const tRow = b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(C));
  const tpQ = b.makeTensorPtr(Q, [BH * nC * C, DK], [DK, 1], [tRow, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpK = b.makeTensorPtr(K, [BH * nC * C, DK], [DK, 1], [tRow, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpVN = b.makeTensorPtr(VN, [BH * nC * C, DV], [DV, 1], [tRow, vOff], [C, BV], "f32", [1, 0]);
  const tpH = b.makeTensorPtr(H, [BH * nC * DK, DV], [DV, 1], [b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(DK)), vOff], [DK, BV], "f32", [1, 0]);
  const tpO = b.makeTensorPtr(O, [BH * nC * C, DV], [DV, 1], [tRow, vOff], [C, BV], "f32", [1, 0]);
  const qFull = b.load(tpQ, { boundaryCheck: [0, 1] }), kFull = b.load(tpK, { boundaryCheck: [0, 1] });
  const hBlock = b.load(tpH, { boundaryCheck: [0, 1] });
  const vnewBlk = b.load(tpVN, { boundaryCheck: [0, 1] });
  const oInter = b.dot(qFull, hBlock);
  const qk = b.dot(qFull, b.trans(kFull));
  const ar = b.arange(0, C);
  const lower = b.ge(b.broadcast(b.expandDims(ar, 1), [C, C]), b.broadcast(b.expandDims(ar, 0), [C, C]));
  const qkMasked = b.select(lower, qk, b.splat(b.f32(0), [C, C], "f32"));
  const o = b.add(b.mul(oInter, scale), b.mul(b.dot(qkMasked, vnewBlk), scale));
  b.store(tpO, o, { boundaryCheck: [0, 1] });
  return b.build("gdn_fwd_o", 4);
}

// host WY: A=(I-L)^-1 (forward-substitution), w=A@k_beta, u=A@v_beta
function hostWY(Q: Float32Array, K: Float32Array, V: Float32Array, B: Float32Array,
                BH: number, T: number, DK: number, DV: number, C: number) {
  const nC = T / C;
  const W = new Float32Array(BH * nC * C * DK), U = new Float32Array(BH * nC * C * DV);
  for (let bh = 0; bh < BH; bh++) for (let ci = 0; ci < nC; ci++) {
    const base = (bh * nC + ci) * C;
    const kb = new Float32Array(C * DK), vb = new Float32Array(C * DV);
    for (let i = 0; i < C; i++) { const be = B[bh * T + base + i];
      for (let d = 0; d < DK; d++) kb[i * DK + d] = K[(bh * T + base + i) * DK + d] * be;
      for (let d = 0; d < DV; d++) vb[i * DV + d] = V[(bh * T + base + i) * DV + d] * be; }
    const kkt = new Float32Array(C * C);
    for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) { let s = 0; for (let d = 0; d < DK; d++) s += kb[i * DK + d] * K[(bh * T + base + j) * DK + d]; kkt[i * C + j] = s; }
    const A = new Float32Array(C * C);
    for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) A[i * C + j] = i > j ? -kkt[i * C + j] : 0;
    for (let i = 1; i < C; i++) for (let j = 0; j < i; j++) { let s = 0; for (let k = 0; k < i; k++) s += A[i * C + k] * A[k * C + j]; A[i * C + j] += s; }
    for (let i = 0; i < C; i++) A[i * C + i] += 1;
    const wO = (bh * nC + ci) * C * DK, uO = (bh * nC + ci) * C * DV;
    for (let i = 0; i < C; i++) for (let d = 0; d < DK; d++) { let s = 0; for (let k = 0; k < C; k++) s += A[i * C + k] * kb[k * DK + d]; W[wO + i * DK + d] = s; }
    for (let i = 0; i < C; i++) for (let d = 0; d < DV; d++) { let s = 0; for (let k = 0; k < C; k++) s += A[i * C + k] * vb[k * DV + d]; U[uO + i * DV + d] = s; }
  }
  return { W, U };
}

// reference: naive O(T) recurrence (gdn_clean)
function refRecur(Q: Float32Array, K: Float32Array, V: Float32Array, B: Float32Array,
                 BH: number, T: number, DK: number, DV: number) {
  const scale = 1 / Math.sqrt(DK), O = new Float32Array(BH * T * DV);
  for (let bh = 0; bh < BH; bh++) {
    const S = new Float32Array(DK * DV);
    for (let t = 0; t < T; t++) {
      const q = Q.slice((bh * T + t) * DK, (bh * T + t) * DK + DK).map(x => x * scale);
      const k = K.slice((bh * T + t) * DK, (bh * T + t) * DK + DK);
      const v = V.slice((bh * T + t) * DV, (bh * T + t) * DV + DV);
      const be = B[bh * T + t];
      const kS = new Float32Array(DV); for (let d = 0; d < DV; d++) { let s = 0; for (let i = 0; i < DK; i++) s += k[i] * S[i * DV + d]; kS[d] = s; }
      const delta = new Float32Array(DV); for (let d = 0; d < DV; d++) delta[d] = (v[d] - kS[d]) * be;
      for (let i = 0; i < DK; i++) for (let d = 0; d < DV; d++) S[i * DV + d] += k[i] * delta[d];
      for (let d = 0; d < DV; d++) { let s = 0; for (let i = 0; i < DK; i++) s += q[i] * S[i * DV + d]; O[(bh * T + t) * DV + d] = s; }
    }
  }
  return O;
}

const BH = 8, T = 512, DK = 128, DV = 128, C = 64, BV = 64;
const nC = T / C;
const kH = compileAndLoad(buildFwdH(BH, nC, C, DK, DV, BV), "gdn_fwd_h", 4);
const kO = compileAndLoad(buildFwdO(BH, nC, C, DK, DV, BV), "gdn_fwd_o", 4);
console.log(`gdn_fast loaded (BH=${BH} T=${T} d=${DK} C=${C} BV=${BV}; grids fwd_h=${BH * DV / BV} fwd_o=${BH * nC * DV / BV})`);

const rnd = () => (Math.random() * 2 - 1) * 0.3;
const Q = new Float32Array(BH * T * DK).map(rnd), K = new Float32Array(BH * T * DK).map(rnd), V = new Float32Array(BH * T * DV).map(rnd);
const B = new Float32Array(BH * T).map(() => Math.random() * 0.8 + 0.1);
const { W, U } = hostWY(Q, K, V, B, BH, T, DK, DV, C);
const Hs = new Float32Array(BH * nC * DK * DV), VNh = new Float32Array(BH * nC * C * DV), O = new Float32Array(BH * T * DV);
const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dW = cuAlloc(W.byteLength), dU = cuAlloc(U.byteLength), dH = cuAlloc(Hs.byteLength), dVN = cuAlloc(VNh.byteLength), dO = cuAlloc(O.byteLength);
cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dW, W.buffer); cuHtoD(dU, U.buffer);

const rcH = cuLaunch(kH, [BH, DV / BV, 1], [128, 1, 1], [dW, dU, dK, dH, dVN]); cuSync();
const rcO = cuLaunch(kO, [BH, nC, DV / BV], [128, 1, 1], [dQ, dK, dVN, dH, dO]); cuSync(); cuDtoH(O.buffer, dO);
console.log(`fwd_h rc=${rcH} fwd_o rc=${rcO}`);

const ref = refRecur(Q, K, V, B, BH, T, DK, DV);
let maxErr = 0, ok = true;
for (let i = 0; i < O.length; i++) { const d = Math.abs(O[i] - ref[i]); if (d > maxErr) maxErr = d; if (d > 1e-2 && ok) { ok = false; console.log(`mismatch [${i}]: got ${O[i].toFixed(4)} ref ${ref[i].toFixed(4)}`); } }
console.log(ok ? `✓ gdn_fast correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");

for (let i = 0; i < 3; i++) { cuLaunch(kH, [BH, DV / BV, 1], [128, 1, 1], [dW, dU, dK, dH, dVN]); cuLaunch(kO, [BH, nC, DV / BV], [128, 1, 1], [dQ, dK, dVN, dH, dO]); }
cuSync();
const t0 = performance.now(), it = 50;
for (let i = 0; i < it; i++) { cuLaunch(kH, [BH, DV / BV, 1], [128, 1, 1], [dW, dU, dK, dH, dVN]); cuLaunch(kO, [BH, nC, DV / BV], [128, 1, 1], [dQ, dK, dVN, dH, dO]); }
cuSync();
const dt = (performance.now() - t0) / 1000 / it;
console.log(`bench fast (fwd_h+fwd_o) BH=${BH} T=${T} d=${DK}: ${(dt * 1e6).toFixed(1)} µs/it  (vs naive recurrence 830 µs, clean-chunk 5192 µs)`);
cuFree(dQ); cuFree(dK); cuFree(dW); cuFree(dU); cuFree(dH); cuFree(dVN); cuFree(dO);
