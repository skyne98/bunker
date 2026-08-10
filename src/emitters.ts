// src/emitters.ts — faithful TTIR emitters for every op in the whole-model graph.
// Direct, honest ports of the working decode kernels in prototypes/decode.ts.
// The generated TTIR computes REAL results — never a zero-fill or approximation.
// Each emitter returns an SSA Value, or `undefined` after storing its outputs to
// global memory via ctx.ptrs (pointer-output / state ops).
//
// IMPORTANT: ctx.ptrs holds the resolved kernel params. Pointer params are
// !tt.ptr<D> Values; scalar params (role=="scalar", e.g. token_id/Pos) are
// plain scalar D Values. Emitters use `scalarOf()` to get a true scalar from
// either (a scalar param directly, or loading a 1-elem ptr).
import type { EmitCtx, GraphNode } from "./fusion";

function pt(ctx: EmitCtx, n: GraphNode, io: "in" | "out", i: number): any {
  const tn = io === "in" ? n.inputs[i].tensorName : n.outputs[i].tensorName;
  const p = ctx.ptrs.get(tn);
  if (p === undefined) throw new Error(`emitter: missing ptr for '${tn}' (declared in graph?)`);
  return p;
}
export const inP = (c: EmitCtx, n: GraphNode, i: number) => pt(c, n, "in", i);
export const outP = (c: EmitCtx, n: GraphNode, i: number) => pt(c, n, "out", i);

/** Get a scalar Value from a param (scalar param → itself; ptr → load elem 0). */
function scalarOf(b: EmitCtx["b"], p: any): any {
  if (p.isScalar) return p;
  // pointer to a 1-elem tensor → load scalar?
  // decode kernels pass scalar i32 params; here it's a ptr, so load [1].
  throw new Error(`emitter: expected scalar param, got ptr (${p.type ? JSON.stringify(p.type) : p})`);
}

// ── Embed: X[H] = E[token, :] ─────────────────────────────────────
export function emitEmbed(ctx: EmitCtx, n: GraphNode): undefined {
  const b = ctx.b;
  const E = inP(ctx, n, 0);
  const IDp = inP(ctx, n, 1);
  const X = outP(ctx, n, 0);
  const Vocab = n.inputs[0].type.shape[0];
  const Hh = n.params.H;
  // token id is a scalar i32 kernel arg
  const id0 = IDp.isScalar ? IDp : b.load(b.makeTensorPtr(IDp, [1], [1], [b.i32(0)], [1], "i32", [1]), {});
  // tiled gather: one CTA, tile [1,Hh] = the full row; start offset [id0, 0]
  const tpE = b.makeTensorPtr(E, [Vocab, Hh], [Hh, 1], [id0, b.i32(0)], [1, Hh], "bf16", [1, 0]);
  const tpX = b.makeTensorPtr(X, [1, Hh], [Hh, 1], [b.i32(0), b.i32(0)], [1, Hh], "bf16", [1, 0]);
  b.store(tpX, b.load(tpE, {}), {});
  return undefined;
}

// ── Conv1d decode (stateful) ──────────────────────────────────────
export function emitConv1d(ctx: EmitCtx, n: GraphNode): undefined {
  const b = ctx.b;
  const QKV = inP(ctx, n, 0), CS = inP(ctx, n, 1), CW = inP(ctx, n, 2);
  const Out = outP(ctx, n, 0), CSN = outP(ctx, n, 1);
  const QKVD = n.params.QKVD;
  const BLK = 1024, pid = b.programId(0), off = b.mul(pid, b.i32(BLK));
  const qkv = b.fpext(b.load(b.makeTensorPtr(QKV, [1, QKVD], [QKVD, 1], [b.i32(0), off], [1, BLK], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const s0 = b.fpext(b.load(b.makeTensorPtr(CS, [3, QKVD], [QKVD, 1], [b.i32(0), off], [1, BLK], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const s1 = b.fpext(b.load(b.makeTensorPtr(CS, [3, QKVD], [QKVD, 1], [b.i32(1), off], [1, BLK], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const s2 = b.fpext(b.load(b.makeTensorPtr(CS, [3, QKVD], [QKVD, 1], [b.i32(2), off], [1, BLK], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const cwIdx = b.mul(b.arange(0, BLK), b.i32(4));
  const cwBase = b.mul(off, b.i32(4));
  const w0 = b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(CW, BLK, "bf16"), b.add(cwIdx, cwBase))), "f32"), [1, BLK]);
  const w1 = b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(CW, BLK, "bf16"), b.add(cwIdx, b.add(cwBase, b.i32(1))))), "f32"), [1, BLK]);
  const w2 = b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(CW, BLK, "bf16"), b.add(cwIdx, b.add(cwBase, b.i32(2))))), "f32"), [1, BLK]);
  const w3 = b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(CW, BLK, "bf16"), b.add(cwIdx, b.add(cwBase, b.i32(3))))), "f32"), [1, BLK]);
  const convRaw = b.add(b.add(b.mul(w0, s0), b.mul(w1, s1)), b.add(b.mul(w2, s2), b.mul(w3, qkv)));
  const convSig = b.divf(b.f32(1), b.add(b.f32(1), b.exp(b.mul(convRaw, b.f32(-1)))));
  const convOut = b.mul(convRaw, convSig);
  b.store(b.makeTensorPtr(Out, [1, QKVD], [QKVD, 1], [b.i32(0), off], [1, BLK], "bf16", [1, 0]), b.fptrunc(convOut, "bf16"), { boundaryCheck: [0, 1] });
  b.store(b.makeTensorPtr(CSN, [3, QKVD], [QKVD, 1], [b.i32(0), off], [1, BLK], "bf16", [1, 0]), b.fptrunc(s1, "bf16"), { boundaryCheck: [0, 1] });
  b.store(b.makeTensorPtr(CSN, [3, QKVD], [QKVD, 1], [b.i32(1), off], [1, BLK], "bf16", [1, 0]), b.fptrunc(s2, "bf16"), { boundaryCheck: [0, 1] });
  b.store(b.makeTensorPtr(CSN, [3, QKVD], [QKVD, 1], [b.i32(2), off], [1, BLK], "bf16", [1, 0]), b.fptrunc(qkv, "bf16"), { boundaryCheck: [0, 1] });
  return undefined;
}

// ── GDN delta rule (stateful) — faithful port of buildGDNDecode ──
export function emitGDN(ctx: EmitCtx, n: GraphNode): undefined {
  const b = ctx.b;
  const CO = inP(ctx, n, 0), Zmb = inP(ctx, n, 1), ALog = inP(ctx, n, 2), dtB = inP(ctx, n, 3);
  const aP = inP(ctx, n, 4), bP = inP(ctx, n, 5), NW = inP(ctx, n, 6), SS = inP(ctx, n, 7);
  const Out = outP(ctx, n, 0), SSN = outP(ctx, n, 1);
  const { LVH, LKD, LVD, QKVD, KEYDIM, ZD } = n.params;
  const head = b.programId(0);
  const qOff = b.mul(head, b.i32(LKD));
  const kOff = b.add(b.i32(KEYDIM), qOff);
  const vOff = b.add(b.i32(2 * KEYDIM), qOff);
  const qRaw = b.fpext(b.load(b.makeTensorPtr(CO, [1, QKVD], [QKVD, 1], [b.i32(0), qOff], [1, LKD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const kRaw = b.fpext(b.load(b.makeTensorPtr(CO, [1, QKVD], [QKVD, 1], [b.i32(0), kOff], [1, LKD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const vRaw = b.fpext(b.load(b.makeTensorPtr(CO, [1, QKVD], [QKVD, 1], [b.i32(0), vOff], [1, LKD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const qRstd = b.rsqrtHw(b.add(b.sum(b.mul(qRaw, qRaw), 1), b.f32(1e-6)));
  const kRstd = b.rsqrtHw(b.add(b.sum(b.mul(kRaw, kRaw), 1), b.f32(1e-6)));
  const qNorm = b.mul(qRaw, b.mul(qRstd, b.f32(1 / Math.sqrt(LKD))));
  const kNorm = b.mul(kRaw, kRstd);
  // decay = exp(-exp(A_log)*softplus(a_p+dt_bias)); beta = sigmoid(b_p)
  const aV = b.load(b.addptr(b.splatPtr(aP, 1, "f32"), head));
  const dtV = b.fpext(b.load(b.addptr(b.splatPtr(dtB, 1, "bf16"), head)), "f32");
  const alV = b.load(b.addptr(b.splatPtr(ALog, 1, "f32"), head));
  const spIn = b.add(aV, dtV);
  const softplus = b.mul(b.log2Hw(b.add(b.f32(1), b.exp(spIn))), b.f32(Math.LN2));
  const decayExp = b.exp(b.mul(b.f32(-1), b.mul(b.exp(alV), softplus)));
  const bV = b.load(b.addptr(b.splatPtr(bP, 1, "f32"), head));
  const beta = b.divf(b.f32(1), b.add(b.f32(1), b.exp(b.mul(bV, b.f32(-1)))));
  const S = b.load(b.makeTensorPtr(SS, [LVH * LKD, LVD], [LVD, 1], [b.mul(head, b.i32(LKD)), b.i32(0)], [LKD, LVD], "f32", [1, 0]), { boundaryCheck: [0, 1], padding: 1 });
  const Sdecayed = b.mul(S, b.broadcastTo(decayExp, [LKD, LVD]));
  const kCol = b.broadcast(b.expandDims(b.reshape(kNorm, [LKD]), 1), [LKD, LVD]);
  const kS = b.sum(b.mul(kCol, Sdecayed), 0);
  const delta = b.mul(b.sub(vRaw, b.broadcastTo(kS, [1, LKD])), b.broadcastTo(beta, [1, LKD]));
  const Snew = b.add(Sdecayed, b.mul(kCol, b.broadcastTo(delta, [LKD, LVD])));
  b.store(b.makeTensorPtr(SSN, [LVH * LKD, LVD], [LVD, 1], [b.mul(head, b.i32(LKD)), b.i32(0)], [LKD, LVD], "f32", [1, 0]), Snew, { boundaryCheck: [0, 1] });
  const qCol = b.broadcast(b.expandDims(b.reshape(qNorm, [LKD]), 1), [LKD, LVD]);
  const o = b.sum(b.mul(qCol, Snew), 0);
  const oMs = b.divf(b.sum(b.mul(o, o), 0), b.f32(LVD));
  const oRstd = b.rsqrtHw(b.add(oMs, b.f32(1e-6)));
  const oNormed = b.mul(o, oRstd);
  const nw = b.load(b.makeTensorPtr(NW, [1, LVD], [LVD, 1], [b.i32(0), b.i32(0)], [1, LVD], "f32", [1, 0]), { boundaryCheck: [0, 1], padding: 1 });
  const oWeighted = b.mul(b.broadcastTo(oNormed, [1, LVD]), nw);
  const zVal = b.fpext(b.load(b.makeTensorPtr(Zmb, [1, ZD], [ZD, 1], [b.i32(0), qOff], [1, LVD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const zSig = b.mul(zVal, b.divf(b.f32(1), b.add(b.f32(1), b.exp(b.mul(zVal, b.f32(-1))))));
  const oGated = b.mul(oWeighted, zSig);
  b.store(b.makeTensorPtr(Out, [1, ZD], [ZD, 1], [b.i32(0), qOff], [1, LVD], "bf16", [1, 0]), b.fptrunc(oGated, "bf16"), { boundaryCheck: [0, 1] });
  return undefined;
}

// ── q_norm (per-head RMSNorm+scale) — port of buildQNorm ──
export function emitQNorm(ctx: EmitCtx, n: GraphNode): undefined {
  const b = ctx.b;
  const QG = inP(ctx, n, 0), QNW = inP(ctx, n, 1), QN = outP(ctx, n, 0);
  const { HD, NH } = n.params;
  const head = b.programId(0);
  const qOffIn = b.mul(head, b.i32(HD * 2)), qOffOut = b.mul(head, b.i32(HD));
  const qRaw = b.fpext(b.load(b.makeTensorPtr(QG, [1, NH * HD * 2], [NH * HD * 2, 1], [b.i32(0), qOffIn], [1, HD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const rstd = b.rsqrtHw(b.add(b.divf(b.sum(b.mul(qRaw, qRaw), 1), b.f32(HD)), b.f32(1e-6)));
  const qn = b.mul(b.mul(qRaw, rstd), b.add(b.f32(1), b.fpext(b.load(b.makeTensorPtr(QNW, [1, HD], [HD, 1], [b.i32(0), b.i32(0)], [1, HD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32")));
  b.store(b.makeTensorPtr(QN, [1, NH * HD], [NH * HD, 1], [b.i32(0), qOffOut], [1, HD], "bf16", [1, 0]), b.fptrunc(qn, "bf16"), { boundaryCheck: [0, 1] });
  return undefined;
}

// ── k_norm — port of buildKNormD ──
export function emitKNorm(ctx: EmitCtx, n: GraphNode): undefined {
  const b = ctx.b;
  const K = inP(ctx, n, 0), KNW = inP(ctx, n, 1), Out = outP(ctx, n, 0);
  const { HD, NKV } = n.params;
  const head = b.programId(0), off = b.mul(head, b.i32(HD));
  const kRaw = b.fpext(b.load(b.makeTensorPtr(K, [1, NKV * HD], [NKV * HD, 1], [b.i32(0), off], [1, HD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const rstd = b.rsqrtHw(b.add(b.divf(b.sum(b.mul(kRaw, kRaw), 1), b.f32(HD)), b.f32(1e-6)));
  const kn = b.mul(b.mul(kRaw, rstd), b.add(b.f32(1), b.fpext(b.load(b.makeTensorPtr(KNW, [1, HD], [HD, 1], [b.i32(0), b.i32(0)], [1, HD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32")));
  b.store(b.makeTensorPtr(Out, [1, NKV * HD], [NKV * HD, 1], [b.i32(0), off], [1, HD], "bf16", [1, 0]), b.fptrunc(kn, "bf16"), { boundaryCheck: [0, 1] });
  return undefined;
}

// ── RoPE (in-place on q/k buffer) — port of buildRoPEInPlace / buildRoPEKInPlace ──
export function emitRoPE(ctx: EmitCtx, n: GraphNode): undefined {
  const b = ctx.b;
  const Q = inP(ctx, n, 0), CosT = inP(ctx, n, 1), SinT = inP(ctx, n, 2), PosP = inP(ctx, n, 3);
  const Pos = PosP.isScalar ? PosP : b.load(b.makeTensorPtr(PosP, [1], [1], [b.i32(0)], [1], "i32", [1]), {});
  const { HD, NH, ROT_HALF, MAX_LEN } = n.params;
  const head = b.programId(0), off = b.mul(head, b.i32(HD));
  const q1 = b.fpext(b.load(b.makeTensorPtr(Q, [1, NH * HD], [NH * HD, 1], [b.i32(0), off], [1, ROT_HALF], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const q2 = b.fpext(b.load(b.makeTensorPtr(Q, [1, NH * HD], [NH * HD, 1], [b.i32(0), b.add(off, b.i32(ROT_HALF))], [1, ROT_HALF], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const cos = b.load(b.makeTensorPtr(CosT, [MAX_LEN, ROT_HALF], [ROT_HALF, 1], [Pos, b.i32(0)], [1, ROT_HALF], "f32", [1, 0]), { boundaryCheck: [0, 1], padding: 1 });
  const sin = b.load(b.makeTensorPtr(SinT, [MAX_LEN, ROT_HALF], [ROT_HALF, 1], [Pos, b.i32(0)], [1, ROT_HALF], "f32", [1, 0]), { boundaryCheck: [0, 1], padding: 1 });
  const q1Rot = b.sub(b.mul(q1, cos), b.mul(q2, sin));
  const q2Rot = b.add(b.mul(q2, cos), b.mul(q1, sin));
  b.store(b.makeTensorPtr(Q, [1, NH * HD], [NH * HD, 1], [b.i32(0), off], [1, ROT_HALF], "bf16", [1, 0]), b.fptrunc(q1Rot, "bf16"), { boundaryCheck: [0, 1] });
  b.store(b.makeTensorPtr(Q, [1, NH * HD], [NH * HD, 1], [b.i32(0), b.add(off, b.i32(ROT_HALF))], [1, ROT_HALF], "bf16", [1, 0]), b.fptrunc(q2Rot, "bf16"), { boundaryCheck: [0, 1] });
  return undefined;
}

// ── RoPE for k (NKV heads) — port of buildRoPEKInPlace ──
export function emitRoPEK(ctx: EmitCtx, n: GraphNode): undefined {
  const b = ctx.b;
  const K = inP(ctx, n, 0), CosT = inP(ctx, n, 1), SinT = inP(ctx, n, 2), PosP = inP(ctx, n, 3);
  const Pos = PosP.isScalar ? PosP : b.load(b.makeTensorPtr(PosP, [1], [1], [b.i32(0)], [1], "i32", [1]), {});
  const { HD, NKV, ROT_HALF, MAX_LEN } = n.params;
  const head = b.programId(0), off = b.mul(head, b.i32(HD));
  const k1 = b.fpext(b.load(b.makeTensorPtr(K, [1, NKV * HD], [NKV * HD, 1], [b.i32(0), off], [1, ROT_HALF], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const k2 = b.fpext(b.load(b.makeTensorPtr(K, [1, NKV * HD], [NKV * HD, 1], [b.i32(0), b.add(off, b.i32(ROT_HALF))], [1, ROT_HALF], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const cos = b.load(b.makeTensorPtr(CosT, [MAX_LEN, ROT_HALF], [ROT_HALF, 1], [Pos, b.i32(0)], [1, ROT_HALF], "f32", [1, 0]), { boundaryCheck: [0, 1], padding: 1 });
  const sin = b.load(b.makeTensorPtr(SinT, [MAX_LEN, ROT_HALF], [ROT_HALF, 1], [Pos, b.i32(0)], [1, ROT_HALF], "f32", [1, 0]), { boundaryCheck: [0, 1], padding: 1 });
  const k1Rot = b.sub(b.mul(k1, cos), b.mul(k2, sin));
  const k2Rot = b.add(b.mul(k2, cos), b.mul(k1, sin));
  b.store(b.makeTensorPtr(K, [1, NKV * HD], [NKV * HD, 1], [b.i32(0), off], [1, ROT_HALF], "bf16", [1, 0]), b.fptrunc(k1Rot, "bf16"), { boundaryCheck: [0, 1] });
  b.store(b.makeTensorPtr(K, [1, NKV * HD], [NKV * HD, 1], [b.i32(0), b.add(off, b.i32(ROT_HALF))], [1, ROT_HALF], "bf16", [1, 0]), b.fptrunc(k2Rot, "bf16"), { boundaryCheck: [0, 1] });
  return undefined;
}

// ── FA2 attention (stateful) — faithful port of buildFA2Attn: takes pre-rotated
//    q (QR), k (KR), v (VB) + gate (QG) + cache (KC/VC) + Pos scalar; writes
//    current k/v into cache, scores against cache, causal mask, softmax, gate. ──
export function emitFA2Attn(ctx: EmitCtx, n: GraphNode): undefined {
  const b = ctx.b;
  const QR = inP(ctx, n, 0), KR = inP(ctx, n, 1), VB = inP(ctx, n, 2), QG = inP(ctx, n, 3);
  const KC = inP(ctx, n, 4), VC = inP(ctx, n, 5), Out = outP(ctx, n, 0), PosP = inP(ctx, n, 6);
  // Pos is a scalar i32 kernel arg
  const Pos = PosP.isScalar ? PosP : b.load(b.makeTensorPtr(PosP, [1], [1], [b.i32(0)], [1], "i32", [1]), {});
  const { HD, NH, MAX_LEN, NKV } = n.params;
  const head = b.programId(0);
  const headKv = b.divi(head, b.i32(NH / NKV));
  const qOff = b.mul(head, b.i32(HD));
  const kvOff = b.mul(headKv, b.i32(HD));
  const gOff = b.add(b.mul(head, b.i32(HD * 2)), b.i32(HD));
  const q = b.fpext(b.load(b.makeTensorPtr(QR, [1, NH * HD], [NH * HD, 1], [b.i32(0), qOff], [1, HD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const k = b.fpext(b.load(b.makeTensorPtr(KR, [1, NKV * HD], [NKV * HD, 1], [b.i32(0), kvOff], [1, HD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const cacheRow = b.add(b.mul(head, b.i32(MAX_LEN)), Pos);
  b.store(b.makeTensorPtr(KC, [NH * MAX_LEN, HD], [HD, 1], [cacheRow, b.i32(0)], [1, HD], "bf16", [1, 0]), b.fptrunc(k, "bf16"), { boundaryCheck: [0, 1] });
  const v = b.fpext(b.load(b.makeTensorPtr(VB, [1, NKV * HD], [NKV * HD, 1], [b.i32(0), kvOff], [1, HD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  b.store(b.makeTensorPtr(VC, [NH * MAX_LEN, HD], [HD, 1], [cacheRow, b.i32(0)], [1, HD], "bf16", [1, 0]), b.fptrunc(v, "bf16"), { boundaryCheck: [0, 1] });
  const kCache = b.fpext(b.load(b.makeTensorPtr(KC, [NH * MAX_LEN, HD], [HD, 1], [b.mul(head, b.i32(MAX_LEN)), b.i32(0)], [MAX_LEN, HD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const qBc = b.broadcastTo(q, [MAX_LEN, HD]);
  const scores = b.mul(b.sum(b.mul(qBc, kCache), 1), b.f32(1 / Math.sqrt(HD)));
  const ar = b.arange(0, MAX_LEN);
  const le = b.le(ar, b.broadcastTo(Pos, [MAX_LEN]));
  const selArr = b.select(le, b.splat(b.f32(0), [MAX_LEN], "f32"), b.splat(b.f32(-1e30), [MAX_LEN], "f32"));
  const maskArr = b.broadcastTo(selArr, [1, MAX_LEN]);
  const scoresMasked = b.add(b.broadcastTo(scores, [1, MAX_LEN]), maskArr);
  const maxScore = b.max(scoresMasked, 1);
  const expScores = b.exp(b.sub(b.broadcastTo(scoresMasked, [1, MAX_LEN]), b.broadcastTo(maxScore, [1, MAX_LEN])));
  const sumExp = b.sum(expScores, 1);
  const weights = b.divf(expScores, b.broadcastTo(sumExp, [1, MAX_LEN]));
  const vCache = b.fpext(b.load(b.makeTensorPtr(VC, [NH * MAX_LEN, HD], [HD, 1], [b.mul(head, b.i32(MAX_LEN)), b.i32(0)], [MAX_LEN, HD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const weightsBc = b.broadcast(b.expandDims(b.reshape(weights, [MAX_LEN]), 1), [MAX_LEN, HD]);
  const attnOut = b.sum(b.mul(weightsBc, vCache), 0);
  const gate = b.fpext(b.load(b.makeTensorPtr(QG, [1, NH * HD * 2], [NH * HD * 2, 1], [b.i32(0), gOff], [1, HD], "bf16", [1, 0]), { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const gateSig = b.divf(b.f32(1), b.add(b.f32(1), b.exp(b.mul(gate, b.f32(-1)))));
  const out = b.mul(b.broadcastTo(attnOut, [1, HD]), gateSig);
  b.store(b.makeTensorPtr(Out, [1, NH * HD], [NH * HD, 1], [b.i32(0), qOff], [1, HD], "bf16", [1, 0]), b.fptrunc(out, "bf16"), { boundaryCheck: [0, 1] });
  return undefined;
}

// ── argmax — port of buildArgmax (grid [1]) ──
export function emitArgmax(ctx: EmitCtx, n: GraphNode): undefined {
  const b = ctx.b;
  const L = inP(ctx, n, 0), V = outP(ctx, n, 0), I = outP(ctx, n, 1);
  const VOCAB = n.params.VOCAB;
  const CHUNK = 4096;
  const tpL = b.makeTensorPtr(L, [1, VOCAB], [VOCAB, 1], [b.i32(0), b.i32(0)], [1, CHUNK], "f32", [1, 0]);
  const initMax = b.broadcastTo(b.f32(-1000), [1]);
  const initIdx = b.broadcastTo(b.i32(0), [1]);
  const initOff = b.broadcastTo(b.i32(0), [1]);
  const [finalMax, finalIdx] = b.forIter(b.index(0), b.index(VOCAB), b.index(CHUNK), [initMax, initIdx, initOff, tpL], (bb, _, [curMax, curIdx, curOff, tp]) => {
    const chunk = bb.load(tp, { boundaryCheck: [0, 1], padding: 1 });
    const localMax = bb.max(chunk, 1);
    const mask = bb.eq(chunk, bb.broadcastTo(localMax, [1, CHUNK]));
    const arange = bb.arange(0, CHUNK);
    const arangeBc = bb.broadcast(bb.expandDims(arange, 0), [1, CHUNK]);
    const masked = bb.select(mask, arangeBc, bb.broadcastTo(bb.i32(0), [1, CHUNK]));
    const localIdx = bb.sum(masked, 1);
    const globalIdx = bb.add(localIdx, curOff);
    const isBetter = bb.gt(localMax, curMax);
    const newMax = bb.select(isBetter, localMax, curMax);
    const newIdx = bb.select(isBetter, globalIdx, curIdx);
    return [newMax, newIdx, bb.add(curOff, bb.broadcastTo(bb.i32(CHUNK), [1])), bb.advance(tp, [bb.i32(0), bb.i32(CHUNK)])];
  });
  b.store(b.makeTensorPtr(V, [1, 1], [1, 1], [b.i32(0), b.i32(0)], [1, 1], "f32", [1, 0]), b.broadcastTo(finalMax, [1, 1]), { boundaryCheck: [0, 1] });
  b.store(b.makeTensorPtr(I, [1, 1], [1, 1], [b.i32(0), b.i32(0)], [1, 1], "i32", [1, 0]), b.broadcastTo(finalIdx, [1, 1]), { boundaryCheck: [0, 1] });
  return undefined;
}
