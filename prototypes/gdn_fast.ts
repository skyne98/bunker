// prototypes/gdn_fast.ts — fast chunked GatedDeltaNet (fla-faithful, blocked WY).
//
// 5 GPU kernels, all on-device:
//   1. kkt_fwd:       kkt = kb @ kᵀ  per chunk.        grid [BH, nC]
//   2. solve_tril:    A_inv = (I-L)⁻¹ via blocked 16×16.  grid [nC, BH]
//   3. recompute_w_u: w = A_inv@kb, u = A_inv@vb.       grid [BH, nC]
//   4. fwd_h:         h recurrence (state-tiled).        grid [BH, NV]
//   5. fwd_o:         o = scale(q@h + tril(q@kᵀ)@v_new). grid [BH, nC, NV]
//
//   bun run prototypes/gdn_fast.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";
// f16 tensor-core matmul: cast both inputs to f16, accumulate f32
function dot16(bb: any, a: any, c: any): any { return bb.dot(bb.fptrunc(a, "f16"), bb.fptrunc(c, "f16")); }

// ── 1. kkt_fwd: kkt = kb @ kᵀ  [C,C] per chunk ───────────────────────
function buildKkt(BH: number, nC: number, C: number, DK: number) {
  const b = new TTIRBuilder();
  const K = b.param("K", { ptr: "f32" }), KB = b.param("KB", { ptr: "f32" }), KKT = b.param("KKT", { ptr: "f32" });
  const pidBH = b.programId(0), pidC = b.programId(1);
  const row = b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(C));
  const tpK = b.makeTensorPtr(K, [BH*nC*C, DK], [DK, 1], [row, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpKB = b.makeTensorPtr(KB, [BH*nC*C, DK], [DK, 1], [row, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpKKT = b.makeTensorPtr(KKT, [BH*nC*C, C], [C, 1], [row, b.i32(0)], [C, C], "f32", [1, 0]);
  const k = b.load(tpK, { boundaryCheck: [0, 1] }), kb = b.load(tpKB, { boundaryCheck: [0, 1] });
  b.store(tpKKT, dot16(b, kb, b.trans(k)), { boundaryCheck: [0, 1] });
  return b.build("kkt_fwd", 4);
}

// ── 2. solve_tril: blocked 16×16 (I-L)⁻¹ (from gdn_solve_tril.ts) ─────
function buildSolveTril64(BH: number, nC: number) {
  const C = 64, B16 = 16;
  const b = new TTIRBuilder();
  const A = b.param("A", { ptr: "f32" }), Ai = b.param("Ai", { ptr: "f32" });
  const pidC = b.programId(0), pidBH = b.programId(1);
  const base = b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(C));
  const o_i = b.arange(0, B16);
  const m_A = b.gt(b.broadcast(b.expandDims(o_i, 1), [B16, B16]), b.broadcast(b.expandDims(o_i, 0), [B16, B16]));
  const m_I = b.eq(b.broadcast(b.expandDims(o_i, 1), [B16, B16]), b.broadcast(b.expandDims(o_i, 0), [B16, B16]));
  const zero16 = b.splat(b.f32(0), [B16, B16], "f32");
  const zeroV = b.splat(b.f32(0), [B16], "f32");
  function fwdSubstDiag(br: number): any {
    const blkRow = b.add(base, b.i32(br * B16));
    const tpBlk = b.makeTensorPtr(A, [BH*nC*C, C], [C, 1], [blkRow, b.i32(br * B16)], [B16, B16], "f32", [1, 0]);
    const blk = b.load(tpBlk, { boundaryCheck: [0, 1] });
    const bAi0 = b.select(m_A, b.mul(blk, b.f32(-1)), zero16);
    const [bAi] = b.forIter(b.index(2), b.index(B16), b.index(1), [bAi0], (bb, iv, [bA]) => {
      const i = bb.indexCast(iv, "i32");
      const rMask = bb.broadcast(bb.expandDims(bb.eq(o_i, i), 1), [B16, B16]);
      const baRow = bb.select(rMask, blk, zero16);
      const ba = bb.mul(bb.sum(baRow, 0), b.f32(-1));
      const baMasked = bb.select(bb.lt(o_i, i), ba, zeroV);
      const contrib = bb.sum(bb.mul(bb.broadcast(bb.expandDims(baMasked, 1), [B16, B16]), bA), 0);
      const newRow = bb.add(baMasked, contrib);
      const newRowBc = bb.broadcast(bb.expandDims(newRow, 0), [B16, B16]);
      const updMask = bb.broadcast(bb.expandDims(bb.eq(o_i, i), 1), [B16, B16]);
      return [bb.select(updMask, newRowBc, bA)];
    });
    return b.add(bAi, b.select(m_I, b.splat(b.f32(1), [B16, B16], "f32"), zero16));
  }
  const Ai11 = fwdSubstDiag(0), Ai22 = fwdSubstDiag(1), Ai33 = fwdSubstDiag(2), Ai44 = fwdSubstDiag(3);
  function loadBlock(br: number, bc: number): any {
    return b.load(b.makeTensorPtr(A, [BH*nC*C, C], [C, 1], [b.add(base, b.i32(br*B16)), b.i32(bc*B16)], [B16, B16], "f32", [1, 0]), { boundaryCheck: [0, 1] });
  }
  const A21 = loadBlock(1,0), A31 = loadBlock(2,0), A32 = loadBlock(2,1), A41 = loadBlock(3,0), A42 = loadBlock(3,1), A43 = loadBlock(3,2);
  const Ai21 = b.mul(b.dot(b.dot(Ai22, A21), Ai11), b.f32(-1));
  const Ai32 = b.mul(b.dot(b.dot(Ai33, A32), Ai22), b.f32(-1));
  const Ai43 = b.mul(b.dot(b.dot(Ai44, A43), Ai33), b.f32(-1));
  const Ai31 = b.mul(b.dot(Ai33, b.add(b.dot(A31, Ai11), b.dot(A32, Ai21))), b.f32(-1));
  const Ai42 = b.mul(b.dot(Ai44, b.add(b.dot(A42, Ai22), b.dot(A43, Ai32))), b.f32(-1));
  const Ai41 = b.mul(b.dot(Ai44, b.add(b.add(b.dot(A41, Ai11), b.dot(A42, Ai21)), b.dot(A43, Ai31))), b.f32(-1));
  function storeBlock(br: number, bc: number, val: any) {
    b.store(b.makeTensorPtr(Ai, [BH*nC*C, C], [C, 1], [b.add(base, b.i32(br*B16)), b.i32(bc*B16)], [B16, B16], "f32", [1, 0]), val, { boundaryCheck: [0, 1] });
  }
  storeBlock(0,0,Ai11); storeBlock(1,1,Ai22); storeBlock(2,2,Ai33); storeBlock(3,3,Ai44);
  storeBlock(1,0,Ai21); storeBlock(2,0,Ai31); storeBlock(2,1,Ai32); storeBlock(3,0,Ai41); storeBlock(3,1,Ai42); storeBlock(3,2,Ai43);
  return b.build("solve_tril64", 4);
}

// ── 3. recompute_w_u: w = A_inv@kb, u = A_inv@vb ──────────────────────
function buildWU(BH: number, nC: number, C: number, DK: number, DV: number) {
  const b = new TTIRBuilder();
  const AI = b.param("AI", { ptr: "f32" }), KB = b.param("KB", { ptr: "f32" }), VB = b.param("VB", { ptr: "f32" });
  const W = b.param("W", { ptr: "f32" }), U = b.param("U", { ptr: "f32" });
  const pidBH = b.programId(0), pidC = b.programId(1);
  const row = b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(C));
  const tpAI = b.makeTensorPtr(AI, [BH*nC*C, C], [C, 1], [row, b.i32(0)], [C, C], "f32", [1, 0]);
  const tpKB = b.makeTensorPtr(KB, [BH*nC*C, DK], [DK, 1], [row, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpVB = b.makeTensorPtr(VB, [BH*nC*C, DV], [DV, 1], [row, b.i32(0)], [C, DV], "f32", [1, 0]);
  const tpW = b.makeTensorPtr(W, [BH*nC*C, DK], [DK, 1], [row, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpU = b.makeTensorPtr(U, [BH*nC*C, DV], [DV, 1], [row, b.i32(0)], [C, DV], "f32", [1, 0]);
  const ai = b.load(tpAI, { boundaryCheck: [0, 1] }), kb = b.load(tpKB, { boundaryCheck: [0, 1] }), vb = b.load(tpVB, { boundaryCheck: [0, 1] });
  b.store(tpW, dot16(b, ai, kb), { boundaryCheck: [0, 1] });
  b.store(tpU, dot16(b, ai, vb), { boundaryCheck: [0, 1] });
  return b.build("recompute_wu", 4);
}

// ── 4. fwd_h: state recurrence, grid [BH, NV] ────────────────────────
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
    const tpW = bb.makeTensorPtr(W, [BH*nC*C, DK], [DK, 1], [cRow, b.i32(0)], [C, DK], "f32", [1, 0]);
    const tpU = bb.makeTensorPtr(U, [BH*nC*C, DV], [DV, 1], [cRow, vOff], [C, BV], "f32", [1, 0]);
    const tpK = bb.makeTensorPtr(K, [BH*nC*C, DK], [DK, 1], [cRow, b.i32(0)], [C, DK], "f32", [1, 0]);
    const tpH = bb.makeTensorPtr(H, [BH*nC*DK, DV], [DV, 1], [hRow, vOff], [DK, BV], "f32", [1, 0]);
    const tpVN = bb.makeTensorPtr(VN, [BH*nC*C, DV], [DV, 1], [cRow, vOff], [C, BV], "f32", [1, 0]);
    const w = bb.load(tpW), u = bb.load(tpU), k = bb.load(tpK);
    const vNew = bb.sub(u, dot16(bb, w, h));
    bb.store(tpH, h, { boundaryCheck: [0, 1] });
    bb.store(tpVN, vNew, { boundaryCheck: [0, 1] });
    return [bb.add(h, dot16(bb, bb.trans(k), vNew))];
  });
  return b.build("gdn_fwd_h", 4, 4);
}

// ── 5. fwd_o: o = scale(q@h + tril(q@kᵀ)@v_new) ──────────────────────
function buildFwdO(BH: number, nC: number, C: number, DK: number, DV: number, BV: number) {
  const b = new TTIRBuilder();
  const Q = b.param("Q", { ptr: "f32" }), K = b.param("K", { ptr: "f32" }), VN = b.param("VN", { ptr: "f32" });
  const H = b.param("H", { ptr: "f32" }), O = b.param("O", { ptr: "f32" });
  const scale = b.f32(1 / Math.sqrt(DK));
  const pidBH = b.programId(0), pidC = b.programId(1), pidV = b.programId(2);
  const vOff = b.mul(pidV, b.i32(BV));
  const tRow = b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(C));
  const tpQ = b.makeTensorPtr(Q, [BH*nC*C, DK], [DK, 1], [tRow, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpK = b.makeTensorPtr(K, [BH*nC*C, DK], [DK, 1], [tRow, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpVN = b.makeTensorPtr(VN, [BH*nC*C, DV], [DV, 1], [tRow, vOff], [C, BV], "f32", [1, 0]);
  const tpH = b.makeTensorPtr(H, [BH*nC*DK, DV], [DV, 1], [b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(DK)), vOff], [DK, BV], "f32", [1, 0]);
  const tpO = b.makeTensorPtr(O, [BH*nC*C, DV], [DV, 1], [tRow, vOff], [C, BV], "f32", [1, 0]);
  const q = b.load(tpQ, { boundaryCheck: [0, 1] }), k = b.load(tpK, { boundaryCheck: [0, 1] });
  const h = b.load(tpH, { boundaryCheck: [0, 1] }), vn = b.load(tpVN, { boundaryCheck: [0, 1] });
  const oInter = dot16(b, q, h);
  const qk = dot16(b, q, b.trans(k));
  const ar = b.arange(0, C);
  const lower = b.ge(b.broadcast(b.expandDims(ar, 1), [C, C]), b.broadcast(b.expandDims(ar, 0), [C, C]));
  b.store(tpO, b.add(b.mul(oInter, scale), b.mul(dot16(b, b.select(lower, qk, b.splat(b.f32(0), [C, C], "f32")), vn), scale)), { boundaryCheck: [0, 1] });
  return b.build("gdn_fwd_o", 4, 4);
}

// reference: naive O(T) recurrence
function refRecur(Q: Float32Array, K: Float32Array, V: Float32Array, B: Float32Array, BH: number, T: number, DK: number, DV: number) {
  const scale = 1 / Math.sqrt(DK), O = new Float32Array(BH * T * DV);
  for (let bh = 0; bh < BH; bh++) {
    const S = new Float32Array(DK * DV);
    for (let t = 0; t < T; t++) {
      const q = Q.slice((bh*T+t)*DK, (bh*T+t)*DK+DK).map(x=>x*scale);
      const k = K.slice((bh*T+t)*DK, (bh*T+t)*DK+DK);
      const v = V.slice((bh*T+t)*DV, (bh*T+t)*DV+DV);
      const be = B[bh*T+t];
      const kS = new Float32Array(DV); for (let d=0; d<DV; d++) { let s=0; for (let i=0; i<DK; i++) s += k[i]*S[i*DV+d]; kS[d]=s; }
      const delta = new Float32Array(DV); for (let d=0; d<DV; d++) delta[d] = (v[d]-kS[d])*be;
      for (let i=0; i<DK; i++) for (let d=0; d<DV; d++) S[i*DV+d] += k[i]*delta[d];
      for (let d=0; d<DV; d++) { let s=0; for (let i=0; i<DK; i++) s += q[i]*S[i*DV+d]; O[(bh*T+t)*DV+d] = s; }
    }
  }
  return O;
}

const BH = 32, T = Number(process.env.TP||512), DK = 128, DV = 128, C = 64, BV = 32;
const nC = T / C;
const kKKT = compileAndLoad(buildKkt(BH, nC, C, DK), "kkt_fwd", 4);
const kST  = compileAndLoad(buildSolveTril64(BH, nC), "solve_tril64", 4);
const kWU  = compileAndLoad(buildWU(BH, nC, C, DK, DV), "recompute_wu", 4);
const kH   = compileAndLoad(buildFwdH(BH, nC, C, DK, DV, BV), "gdn_fwd_h", 4);
const kO   = compileAndLoad(buildFwdO(BH, nC, C, DK, DV, BV), "gdn_fwd_o", 4);
console.log(`gdn_fast loaded (BH=${BH} T=${T} d=${DK} C=${C}; 5 kernels: kkt+solve_tril+wu+fwd_h+fwd_o)`);

const rnd = () => (Math.random() * 2 - 1) * 0.3;
const Q = new Float32Array(BH*T*DK).map(rnd), K = new Float32Array(BH*T*DK).map(rnd), V = new Float32Array(BH*T*DV).map(rnd);
const B = new Float32Array(BH*T).map(() => Math.random()*0.8+0.1);
const KB = new Float32Array(BH*T*DK), VB = new Float32Array(BH*T*DV);
for (let i=0; i<BH*T; i++) { for (let d=0; d<DK; d++) KB[i*DK+d] = K[i*DK+d]*B[i]; for (let d=0; d<DV; d++) VB[i*DV+d] = V[i*DV+d]*B[i]; }
const KKT = new Float32Array(BH*nC*C*C), AI = new Float32Array(BH*nC*C*C);
const W = new Float32Array(BH*nC*C*DK), U = new Float32Array(BH*nC*C*DV);
const Hs = new Float32Array(BH*nC*DK*DV), VNh = new Float32Array(BH*nC*C*DV), O = new Float32Array(BH*T*DV);
const dQ=cuAlloc(Q.byteLength), dK=cuAlloc(K.byteLength), dKB=cuAlloc(KB.byteLength), dVB=cuAlloc(VB.byteLength);
const dKKT=cuAlloc(KKT.byteLength), dAI=cuAlloc(AI.byteLength), dW=cuAlloc(W.byteLength), dU=cuAlloc(U.byteLength);
const dH=cuAlloc(Hs.byteLength), dVN=cuAlloc(VNh.byteLength), dO=cuAlloc(O.byteLength);
cuHtoD(dQ,Q.buffer); cuHtoD(dK,K.buffer); cuHtoD(dKB,KB.buffer); cuHtoD(dVB,VB.buffer);

// 5-kernel pipeline
const _t0=performance.now(); const r1 = cuLaunch(kKKT, [BH, nC, 1], [128,1,1], [dK, dKB, dKKT]); cuSync(); const _t1=performance.now();
const r2 = cuLaunch(kST, [nC, BH, 1], [128,1,1], [dKKT, dAI]); cuSync(); const _t2=performance.now();
const r3 = cuLaunch(kWU, [BH, nC, 1], [128,1,1], [dAI, dKB, dVB, dW, dU]); cuSync(); const _t3=performance.now();
const r4 = cuLaunch(kH, [BH, DV/BV, 1], [128,1,1], [dW, dU, dK, dH, dVN]); cuSync(); const _t4=performance.now();
const r5 = cuLaunch(kO, [BH, nC, DV/BV], [128,1,1], [dQ, dK, dVN, dH, dO]); cuSync(); const _t5=performance.now(); cuDtoH(O.buffer, dO);
console.log(`rc: ${r1},${r2},${r3},${r4},${r5}`); console.log(`per-kernel: kkt=${(_t1-_t0).toFixed(0)}µs st=${(_t2-_t1).toFixed(0)}µs wu=${(_t3-_t2).toFixed(0)}µs fh=${(_t4-_t3).toFixed(0)}µs fo=${(_t5-_t4).toFixed(0)}µs`);

const ref = refRecur(Q, K, V, B, BH, T, DK, DV);
let maxErr = 0, ok = true;
for (let i = 0; i < O.length; i++) { const d = Math.abs(O[i] - ref[i]); if (d > maxErr) maxErr = d; if (d > 1e-2 && ok) { ok = false; console.log(`mismatch [${i}]: got ${O[i].toFixed(4)} ref ${ref[i].toFixed(4)}`); } }
console.log(ok ? `✓ gdn_fast correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");

// bench
for (let i = 0; i < 3; i++) { cuLaunch(kKKT,[BH,nC,1],[128,1,1],[dK,dKB,dKKT]); cuLaunch(kST,[nC,BH,1],[128,1,1],[dKKT,dAI]); cuLaunch(kWU,[BH,nC,1],[128,1,1],[dAI,dKB,dVB,dW,dU]); cuLaunch(kH,[BH,DV/BV,1],[128,1,1],[dW,dU,dK,dH,dVN]); cuLaunch(kO,[BH,nC,DV/BV],[128,1,1],[dQ,dK,dVN,dH,dO]); }
cuSync();
const t0 = performance.now(), it = 50;
for (let i = 0; i < it; i++) { cuLaunch(kKKT,[BH,nC,1],[128,1,1],[dK,dKB,dKKT]); cuLaunch(kST,[nC,BH,1],[128,1,1],[dKKT,dAI]); cuLaunch(kWU,[BH,nC,1],[128,1,1],[dAI,dKB,dVB,dW,dU]); cuLaunch(kH,[BH,DV/BV,1],[128,1,1],[dW,dU,dK,dH,dVN]); cuLaunch(kO,[BH,nC,DV/BV],[128,1,1],[dQ,dK,dVN,dH,dO]); }
cuSync();
const dt = (performance.now() - t0) / 1000 / it;
console.log("per-kernel:"); for (const [n,t] of [["kkt",0],["st",0],["wu",0],["fh",0],["fo",0]]) void 0; console.log(`bench (5 kernels) BH=${BH} T=${T} d=${DK}: ${(dt*1e6).toFixed(1)} µs/it  (naive recurrence ~${830*BH/8} µs est)`);
// buffers freed at process exit

// per-kernel steady-state timing
{
  const NK = 20;
  function pk(name: string, fn: () => number) {
    for (let i = 0; i < 3; i++) { fn(); cuSync(); }
    const t0 = performance.now();
    for (let i = 0; i < NK; i++) { fn(); cuSync(); }
    const us = (performance.now() - t0) * 1000 / NK;
    console.log(`  ${name}: ${us.toFixed(0)} µs`);
    return us;
  }
  console.log("per-kernel (steady-state):");
  pk("kkt_fwd",  () => cuLaunch(kKKT,[BH,nC,1],[128,1,1],[dK,dKB,dKKT]));
  pk("solve_tril",() => cuLaunch(kST,[nC,BH,1],[128,1,1],[dKKT,dAI]));
  pk("wu",       () => cuLaunch(kWU,[BH,nC,1],[128,1,1],[dAI,dKB,dVB,dW,dU]));
  pk("fwd_h",    () => cuLaunch(kH,[BH,DV/BV,1],[128,1,1],[dW,dU,dK,dH,dVN]));
  pk("fwd_o",    () => cuLaunch(kO,[BH,nC,DV/BV],[128,1,1],[dQ,dK,dVN,dH,dO]));
}
