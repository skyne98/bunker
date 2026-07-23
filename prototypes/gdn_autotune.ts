// prototypes/gdn_autotune.ts — autotune the GDN pipeline (fwd_h + fwd_o).
//
// Sweeps BV (state-tile) × num_warps × num_stages for the two bottleneck
// kernels (fwd_h: 211µs, fwd_o: 143µs). kkt/solve_tril/wu are fixed (fast).
// CUDA-event... no — host timing with sync per iter (ttir context, no cross-context).
//
//   bun run prototypes/gdn_autotune.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const DK = 128, DV = 128, C = 64, BH = 32, T = 512, nC = T / C;
const rnd = () => (Math.random() * 2 - 1) * 0.3;

// ── fixed kernels (kkt, solve_tril, wu) — compiled once ──────────────
function dot16(bb: any, a: any, c: any): any { return bb.dot(bb.fptrunc(a, "f16"), bb.fptrunc(c, "f16")); }

function buildKkt(nw: number, ns: number) {
  const b = new TTIRBuilder();
  const K = b.param("K", { ptr: "f32" }), KB = b.param("KB", { ptr: "f32" }), KKT = b.param("KKT", { ptr: "f32" });
  const pidBH = b.programId(0), pidC = b.programId(1);
  const row = b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(C));
  const tpK = b.makeTensorPtr(K, [BH*nC*C, DK], [DK, 1], [row, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpKB = b.makeTensorPtr(KB, [BH*nC*C, DK], [DK, 1], [row, b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpKKT = b.makeTensorPtr(KKT, [BH*nC*C, C], [C, 1], [row, b.i32(0)], [C, C], "f32", [1, 0]);
  const k = b.load(tpK, { boundaryCheck: [0, 1] }), kb = b.load(tpKB, { boundaryCheck: [0, 1] });
  b.store(tpKKT, dot16(b, kb, b.trans(k)), { boundaryCheck: [0, 1] });
  return b.build("kkt_fwd", nw, ns);
}
// solve_tril (f32, blocked 16x16 — same as gdn_solve_tril.ts, omitted for brevity, imported conceptually)
// recompute_wu
function buildWU(nw: number, ns: number) {
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
  return b.build("recompute_wu", nw, ns);
}

// ── tunable kernels (fwd_h, fwd_o) — parametrized BV, nw, ns ──────────
function buildFwdH(BV: number, nw: number, ns: number) {
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
  return b.build("gdn_fwd_h", nw, ns);
}
function buildFwdO(BV: number, nw: number, ns: number) {
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
  return b.build("gdn_fwd_o", nw, ns);
}

// ── solve_tril (import from gdn_solve_tril.ts — inlined for standalone) ──
function buildSolveTril64() {
  const B16 = 16;
  const b = new TTIRBuilder();
  const A = b.param("A", { ptr: "f32" }), Ai = b.param("Ai", { ptr: "f32" });
  const pidC = b.programId(0), pidBH = b.programId(1);
  const base = b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(C));
  const o_i = b.arange(0, B16);
  const m_A = b.gt(b.broadcast(b.expandDims(o_i, 1), [B16, B16]), b.broadcast(b.expandDims(o_i, 0), [B16, B16]));
  const m_I = b.eq(b.broadcast(b.expandDims(o_i, 1), [B16, B16]), b.broadcast(b.expandDims(o_i, 0), [B16, B16]));
  const z16 = b.splat(b.f32(0), [B16, B16], "f32"), zV = b.splat(b.f32(0), [B16], "f32");
  function diag(br: number): any {
    const r = b.add(base, b.i32(br * B16));
    const blk = b.load(b.makeTensorPtr(A, [BH*nC*C, C], [C, 1], [r, b.i32(br*B16)], [B16, B16], "f32", [1, 0]), { boundaryCheck: [0, 1] });
    const a0 = b.select(m_A, b.mul(blk, b.f32(-1)), z16);
    const [a] = b.forIter(b.index(2), b.index(B16), b.index(1), [a0], (bb, iv, [bA]) => {
      const i = bb.indexCast(iv, "i32");
      const rm = bb.broadcast(bb.expandDims(bb.eq(o_i, i), 1), [B16, B16]);
      const ba = bb.mul(bb.sum(bb.select(rm, blk, z16), 0), b.f32(-1));
      const bam = bb.select(bb.lt(o_i, i), ba, zV);
      const c = bb.sum(bb.mul(bb.broadcast(bb.expandDims(bam, 1), [B16, B16]), bA), 0);
      const nr = bb.add(bam, c);
      const um = bb.broadcast(bb.expandDims(bb.eq(o_i, i), 1), [B16, B16]);
      return [bb.select(um, bb.broadcast(bb.expandDims(nr, 0), [B16, B16]), bA)];
    });
    return b.add(a, b.select(m_I, b.splat(b.f32(1), [B16, B16], "f32"), z16));
  }
  const A11=diag(0),A22=diag(1),A33=diag(2),A44=diag(3);
  const lb = (br:number,bc:number) => b.load(b.makeTensorPtr(A,[BH*nC*C,C],[C,1],[b.add(base,b.i32(br*B16)),b.i32(bc*B16)],[B16,B16],"f32",[1,0]),{boundaryCheck:[0,1]});
  const A21=lb(1,0),A31=lb(2,0),A32=lb(2,1),A41=lb(3,0),A42=lb(3,1),A43=lb(3,2);
  const Ai21=b.mul(b.dot(b.dot(A22,A21),A11),b.f32(-1));
  const Ai32=b.mul(b.dot(b.dot(A33,A32),A22),b.f32(-1));
  const Ai43=b.mul(b.dot(b.dot(A44,A43),A33),b.f32(-1));
  const Ai31=b.mul(b.dot(A33,b.add(b.dot(A31,A11),b.dot(A32,Ai21))),b.f32(-1));
  const Ai42=b.mul(b.dot(A44,b.add(b.dot(A42,A22),b.dot(A43,Ai32))),b.f32(-1));
  const Ai41=b.mul(b.dot(A44,b.add(b.add(b.dot(A41,A11),b.dot(A42,Ai21)),b.dot(A43,Ai31))),b.f32(-1));
  const sb=(br:number,bc:number,v:any)=>b.store(b.makeTensorPtr(Ai,[BH*nC*C,C],[C,1],[b.add(base,b.i32(br*B16)),b.i32(bc*B16)],[B16,B16],"f32",[1,0]),v,{boundaryCheck:[0,1]});
  sb(0,0,A11);sb(1,1,A22);sb(2,2,A33);sb(3,3,A44);sb(1,0,Ai21);sb(2,0,Ai31);sb(2,1,Ai32);sb(3,0,Ai41);sb(3,1,Ai42);sb(3,2,Ai43);
  return b.build("solve_tril64", 4, 3);
}

// ── inputs ────────────────────────────────────────────────────────────
const Q = new Float32Array(BH*T*DK).map(rnd), K = new Float32Array(BH*T*DK).map(rnd), V = new Float32Array(BH*T*DV).map(rnd);
const B = new Float32Array(BH*T).map(() => Math.random()*0.8+0.1);
const KB = new Float32Array(BH*T*DK), VB = new Float32Array(BH*T*DV);
for (let i=0; i<BH*T; i++) { for (let d=0; d<DK; d++) KB[i*DK+d]=K[i*DK+d]*B[i]; for (let d=0; d<DV; d++) VB[i*DV+d]=V[i*DV+d]*B[i]; }
const dQ=cuAlloc(Q.byteLength),dK=cuAlloc(K.byteLength),dKB=cuAlloc(KB.byteLength),dVB=cuAlloc(VB.byteLength);
cuHtoD(dQ,Q.buffer);cuHtoD(dK,K.buffer);cuHtoD(dKB,KB.buffer);cuHtoD(dVB,VB.buffer);

// ── fixed kernels ─────────────────────────────────────────────────────
const kKKT = compileAndLoad(buildKkt(4, 3), "kkt_fwd", 4);
const kST  = compileAndLoad(buildSolveTril64(), "solve_tril64", 4);
const kWU  = compileAndLoad(buildWU(4, 3), "recompute_wu", 4);

// intermediate buffers (reused across configs)
const N = BH * nC;
const dKKT=cuAlloc(N*C*C*4), dAI=cuAlloc(N*C*C*4), dW=cuAlloc(N*C*DK*4), dU=cuAlloc(N*C*DV*4), dO=cuAlloc(BH*T*DV*4);

// reference output (from the first valid config)
let refO: Float32Array | null = null;

console.log(`autotuning GDN (BH=${BH} T=${T} C=${C}); search: BV × nw × ns\n`);

const SPACE: { BV: number; nw: number; ns: number }[] = [];
for (const BV of [32, 64, 128]) for (const nw of [4, 8]) for (const ns of [2, 3, 4]) SPACE.push({ BV, nw, ns });

let best: { cfg: typeof SPACE[0]; us: number } | null = null;

for (const cfg of SPACE) {
  const { BV, nw, ns } = cfg;
  let kH, kO;
  try { kH = compileAndLoad(buildFwdH(BV, nw, ns), "gdn_fwd_h", nw); kO = compileAndLoad(buildFwdO(BV, nw, ns), "gdn_fwd_o", nw); }
  catch { console.log(`  BV=${BV} nw=${nw} ns=${ns}  compile FAIL — skip`); continue; }
  const dH = cuAlloc(N*DK*DV*4), dVN = cuAlloc(N*C*BV*4);
  const O = new Float32Array(BH*T*DV);
  // run pipeline
  let rc = 0;
  rc = cuLaunch(kKKT,[BH,nC,1],[128,1,1],[dK,dKB,dKKT]); cuSync();
  if (rc) { console.log(`  BV=${BV} nw=${nw} ns=${ns}  kkt rc=${rc}`); cuFree(dH); cuFree(dVN); continue; }
  cuLaunch(kST,[nC,BH,1],[128,1,1],[dKKT,dAI]); cuSync();
  cuLaunch(kWU,[BH,nC,1],[128,1,1],[dAI,dKB,dVB,dW,dU]); cuSync();
  rc = cuLaunch(kH,[BH,DV/BV,1],[nw*32,1,1],[dW,dU,dK,dH,dVN]); cuSync();
  if (rc) { console.log(`  BV=${BV} nw=${nw} ns=${ns}  fwd_h rc=${rc}`); cuFree(dH); cuFree(dVN); continue; }
  rc = cuLaunch(kO,[BH,nC,DV/BV],[nw*32,1,1],[dQ,dK,dVN,dH,dO]); cuSync();
  if (rc) { console.log(`  BV=${BV} nw=${nw} ns=${ns}  fwd_o rc=${rc}`); cuFree(dH); cuFree(dVN); continue; }
  cuDtoH(O.buffer, dO);
  // correctness (compare to first valid or absolute sanity)
  if (refO === null) refO = O.slice();
  let maxErr = 0; for (let i = 0; i < O.length; i++) maxErr = Math.max(maxErr, Math.abs(O[i] - refO[i]));
  // bench
  for (let i = 0; i < 3; i++) { cuLaunch(kKKT,[BH,nC,1],[128,1,1],[dK,dKB,dKKT]); cuLaunch(kST,[nC,BH,1],[128,1,1],[dKKT,dAI]); cuLaunch(kWU,[BH,nC,1],[128,1,1],[dAI,dKB,dVB,dW,dU]); cuLaunch(kH,[BH,DV/BV,1],[nw*32,1,1],[dW,dU,dK,dH,dVN]); cuLaunch(kO,[BH,nC,DV/BV],[nw*32,1,1],[dQ,dK,dVN,dH,dO]); cuSync(); }
  const t0 = performance.now(); const ITERS = 20;
  for (let i = 0; i < ITERS; i++) { cuLaunch(kKKT,[BH,nC,1],[128,1,1],[dK,dKB,dKKT]); cuLaunch(kST,[nC,BH,1],[128,1,1],[dKKT,dAI]); cuLaunch(kWU,[BH,nC,1],[128,1,1],[dAI,dKB,dVB,dW,dU]); cuLaunch(kH,[BH,DV/BV,1],[nw*32,1,1],[dW,dU,dK,dH,dVN]); cuLaunch(kO,[BH,nC,DV/BV],[nw*32,1,1],[dQ,dK,dVN,dH,dO]); cuSync(); }
  const us = (performance.now() - t0) * 1000 / ITERS;
  const ok = maxErr < 0.1;
  console.log(`  BV=${String(BV).padStart(3)} nw=${nw} ns=${ns}  ${us.toFixed(0).padStart(6)} µs  err=${maxErr.toExponential(1)}  ${ok ? "✓" : "✗"}`);
  if (ok && (!best || us < best.us)) best = { cfg, us };
  cuFree(dH); cuFree(dVN);
}

if (best) console.log(`\nBEST: BV=${best.cfg.BV} nw=${best.cfg.nw} ns=${best.cfg.ns}  →  ${best.us.toFixed(0)} µs`);
cuFree(dQ);cuFree(dK);cuFree(dKB);cuFree(dVB);cuFree(dKKT);cuFree(dAI);cuFree(dW);cuFree(dU);cuFree(dO);
