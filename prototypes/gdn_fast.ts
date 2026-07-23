// prototypes/gdn_fast.ts — fast chunked-parallel GatedDeltaNet (fla-faithful, on-GPU WY).
//
// Ports fla's actual kernels: solve_tril (on-GPU (I-L)^-1 via row-by-row
// forward-substitution using where+sum — NO host WY, NO Neumann), recompute_w_u
// (w=A@k_beta, u=A@v_beta), chunk_h (state recurrence), chunk_o (output).
//
//   fwd_intra: A=(I-L)^-1 on GPU (forward-subst loop, where+sum, no slicing);
//              w=A@kb, u=A@vb.   grid [B·H, nChunks]
//   fwd_h:     v_new=u-w@h; store h[c]; h+=kᵀ@v_new.  grid [B·H, NV]
//   fwd_o:     o=scale(q@h + tril(q@kᵀ)@v_new).       grid [B·H, nChunks, NV]
//
//   bun run prototypes/gdn_fast.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

// ── fwd_intra: on-GPU WY (A=(I-L)^-1 via forward-subst, then w,u) ─────
function buildFwdIntra(BH: number, nC: number, C: number, DK: number, DV: number) {
  const b = new TTIRBuilder();
  const K = b.param("K", { ptr: "f32" }), KB = b.param("KB", { ptr: "f32" }), VB = b.param("VB", { ptr: "f32" });
  const W = b.param("W", { ptr: "f32" }), U = b.param("U", { ptr: "f32" });
  const pidBH = b.programId(0), pidC = b.programId(1);
  const row = b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(C));

  const tpK = b.makeTensorPtr(K, [BH * nC * C, DK], [DK, 1], [row, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpKB = b.makeTensorPtr(KB, [BH * nC * C, DK], [DK, 1], [row, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpVB = b.makeTensorPtr(VB, [BH * nC * C, DV], [DV, 1], [row, b.i32(0)], [C, DV], "f32", [1, 0]);
  const tpW = b.makeTensorPtr(W, [BH * nC * C, DK], [DK, 1], [row, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpU = b.makeTensorPtr(U, [BH * nC * C, DV], [DV, 1], [row, b.i32(0)], [C, DV], "f32", [1, 0]);
  const k = b.load(tpK), kb = b.load(tpKB), vb = b.load(tpVB);
  const kkt = b.dot(kb, b.trans(k));                              // [C,C]

  const ar = b.arange(0, C);
  const rowI = b.broadcast(b.expandDims(ar, 1), [C, C]);
  const colI = b.broadcast(b.expandDims(ar, 0), [C, C]);
  const strictLower = b.gt(rowI, colI);
  const zeroCC = b.splat(b.f32(0), [C, C], "f32");
  const kktNeg = b.select(strictLower, b.mul(kkt, b.f32(-1)), zeroCC);  // -L (strict lower)

  // forward-substitution: b_A[i,:i] += sum_j b_A[i,j]*b_A[j,:]  (row-by-row, where+sum)
  const [A] = b.forIter(b.index(0), b.index(C), b.index(1), [kktNeg], (bb, iv, [bA]) => {
    const i = bb.indexCast(iv, "i32");
    const rowMask = bb.eq(rowI, i);                              // row == i  [C,C]
    const colMask = bb.lt(colI, i);                              // col <  i
    const mask = bb.and(rowMask, colMask);
    const bARow = bb.select(mask, kktNeg, zeroCC);              // -L[i,:i] as [C,C] (row i, cols<i)
    const bAVec = bb.sum(bARow, 0);                              // [C]  extract row i
    const contrib = bb.sum(bb.mul(bb.broadcast(bb.expandDims(bAVec, 1), [C, C]), bA), 0);  // [C]
    const newRow = bb.add(bAVec, contrib);
    const newRowBc = bb.broadcast(bb.expandDims(newRow, 1), [C, C]);
    return [bb.select(mask, newRowBc, bA)];                     // update row i, cols<i
  });
  const eye = b.select(b.eq(rowI, colI), b.splat(b.f32(1), [C, C], "f32"), zeroCC);
  const Ainv = b.add(A, eye);                                    // (I-L)^-1
  b.store(tpW, b.dot(Ainv, kb), { boundaryCheck: [0, 1] });      // w = A@kb
  b.store(tpU, b.dot(Ainv, vb), { boundaryCheck: [0, 1] });    // u = A@vb
  return b.build("gdn_intra", 4);
}

// ── fwd_h: state recurrence, grid [B·H, NV], 1 iter-arg (h), ptrs from iv ──
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

// ── fwd_o: o = scale(q@h + tril(q@kᵀ)@v_new) ──────────────────────────
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
  const o = b.add(b.mul(oInter, scale), b.mul(b.dot(b.select(lower, qk, b.splat(b.f32(0), [C, C], "f32")), vnewBlk), scale));
  b.store(tpO, o, { boundaryCheck: [0, 1] });
  return b.build("gdn_fwd_o", 4);
}

// reference: naive O(T) recurrence
function refRecur(Q: Float32Array, K: Float32Array, V: Float32Array, B: Float32Array, BH: number, T: number, DK: number, DV: number) {
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

const BH = 32, T = 512, DK = 128, DV = 128, C = 64, BV = 64;
const nC = T / C;
const kI = compileAndLoad(buildFwdIntra(BH, nC, C, DK, DV), "gdn_intra", 4);
const kH = compileAndLoad(buildFwdH(BH, nC, C, DK, DV, BV), "gdn_fwd_h", 4);
const kO = compileAndLoad(buildFwdO(BH, nC, C, DK, DV, BV), "gdn_fwd_o", 4);
console.log(`gdn_fast loaded (BH=${BH} T=${T} d=${DK} C=${C}; grids intra=${BH * nC} fwd_h=${BH * DV / BV} fwd_o=${BH * nC * DV / BV})`);

const rnd = () => (Math.random() * 2 - 1) * 0.3;
const Q = new Float32Array(BH * T * DK).map(rnd), K = new Float32Array(BH * T * DK).map(rnd), V = new Float32Array(BH * T * DV).map(rnd);
const B = new Float32Array(BH * T).map(() => Math.random() * 0.8 + 0.1);
// fold beta on host: kb=k*β, vb=v*β
const KB = new Float32Array(BH * T * DK), VB = new Float32Array(BH * T * DV);
for (let i = 0; i < BH * T; i++) { for (let d = 0; d < DK; d++) KB[i * DK + d] = K[i * DK + d] * B[i]; for (let d = 0; d < DV; d++) VB[i * DV + d] = V[i * DV + d] * B[i]; }
const W = new Float32Array(BH * nC * C * DK), U = new Float32Array(BH * nC * C * DV);
const Hs = new Float32Array(BH * nC * DK * DV), VNh = new Float32Array(BH * nC * C * DV), O = new Float32Array(BH * T * DV);
const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dKB = cuAlloc(KB.byteLength), dVB = cuAlloc(VB.byteLength);
const dW = cuAlloc(W.byteLength), dU = cuAlloc(U.byteLength), dH = cuAlloc(Hs.byteLength), dVN = cuAlloc(VNh.byteLength), dO = cuAlloc(O.byteLength);
cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dKB, KB.buffer); cuHtoD(dVB, VB.buffer);

// fwd_intra (WY) -> fwd_h (state) -> fwd_o (output)
const rcI = cuLaunch(kI, [BH, nC, 1], [128, 1, 1], [dK, dKB, dVB, dW, dU]); cuSync();
const rcH = cuLaunch(kH, [BH, DV / BV, 1], [128, 1, 1], [dW, dU, dK, dH, dVN]); cuSync();
const rcO = cuLaunch(kO, [BH, nC, DV / BV], [128, 1, 1], [dQ, dK, dVN, dH, dO]); cuSync(); cuDtoH(O.buffer, dO);
console.log(`rc intra=${rcI} fwd_h=${rcH} fwd_o=${rcO}`);

const ref = refRecur(Q, K, V, B, BH, T, DK, DV);
let maxErr = 0, ok = true;
for (let i = 0; i < O.length; i++) { const d = Math.abs(O[i] - ref[i]); if (d > maxErr) maxErr = d; if (d > 1e-2 && ok) { ok = false; console.log(`mismatch [${i}]: got ${O[i].toFixed(4)} ref ${ref[i].toFixed(4)}`); } }
console.log(ok ? `✓ gdn_fast correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");

for (let i = 0; i < 3; i++) { cuLaunch(kI, [BH, nC, 1], [128, 1, 1], [dK, dKB, dVB, dW, dU]); cuLaunch(kH, [BH, DV / BV, 1], [128, 1, 1], [dW, dU, dK, dH, dVN]); cuLaunch(kO, [BH, nC, DV / BV], [128, 1, 1], [dQ, dK, dVN, dH, dO]); }
cuSync();
const t0 = performance.now(), it = 50;
for (let i = 0; i < it; i++) { cuLaunch(kI, [BH, nC, 1], [128, 1, 1], [dK, dKB, dVB, dW, dU]); cuLaunch(kH, [BH, DV / BV, 1], [128, 1, 1], [dW, dU, dK, dH, dVN]); cuLaunch(kO, [BH, nC, DV / BV], [128, 1, 1], [dQ, dK, dVN, dH, dO]); }
cuSync();
const dt = (performance.now() - t0) / 1000 / it;
console.log(`bench (intra+fwd_h+fwd_o) BH=${BH} T=${T} d=${DK}: ${(dt * 1e6).toFixed(1)} µs/it`);
cuFree(dQ); cuFree(dK); cuFree(dKB); cuFree(dVB); cuFree(dW); cuFree(dU); cuFree(dH); cuFree(dVN); cuFree(dO);
