// prototypes/gdn_chunk.ts — GatedDeltaNet chunked-parallel prefill (clean port).
//
// The fast prefill: instead of the O(T) sequential recurrence (gdn_clean), the
// sequence is split into chunks of C and the delta rule is computed in parallel
// within each chunk (matmuls → tensor cores) + a recurrence across chunks
// carrying the state S. Ported from fla's `naive_chunk_gated_delta_rule` /
// `delta_rule_chunkwise` reference.
//
// Key porting trick: the reference computes (I - L)^{-1} (L = strict-lower of
// -(k_beta·kᵀ)) via row-by-row forward substitution (Python slicing). The tile
// model has no dynamic slicing, so we use the **finite Neumann series** — L is
// strict-lower-triangular ⇒ nilpotent (L^C = 0) ⇒ (I-L)^{-1} = I + L + L² + … +
// L^{C-1}, a `scf.for` accumulating matrix powers (pure `tt.dot`, no slicing).
//
// Per chunk (β folded on host: kb=k·β, vb=v·β; q scaled 1/√d_k):
//   L    = strict_lower( -(kb·kᵀ) )            # [C,C]
//   inv  = I + L + L² + … + L^{C-1}            # Neumann (scf.for of C-1 matmuls)
//   u    = inv · vb ;  w = inv · kb            # [C,D]
//   ui   = u - w·S                             # subtract carried state
//   o    = q·S + strict_lower(q·kᵀ) · ui       # [C,D]
//   S   += kᵀ · ui                            # state update [D,D]
// The inter-chunk loop carries S as a scf.for iter-arg (FA2/matmul K-loop pattern).
//
// Non-gated (no decay); gating = layer on the decay (precomputed) — noted below.
//   bun run prototypes/gdn_chunk.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

function buildGDNchunk(BH: number, T: number, DK: number, DV: number, C: number) {
  const b = new TTIRBuilder();
  const Q = b.param("Q", { ptr: "f32" }), K = b.param("K", { ptr: "f32" });
  const KB = b.param("KB", { ptr: "f32" }), VB = b.param("VB", { ptr: "f32" }), O = b.param("O", { ptr: "f32" });
  const pid = b.programId(0);
  const nChunks = T / C;
  const scale = b.f32(1 / Math.sqrt(DK));

  const tpQ  = b.makeTensorPtr(Q,  [BH * T, DK], [DK, 1], [b.mul(pid, b.i32(T)), b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpK  = b.makeTensorPtr(K,  [BH * T, DK], [DK, 1], [b.mul(pid, b.i32(T)), b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpKB = b.makeTensorPtr(KB, [BH * T, DK], [DK, 1], [b.mul(pid, b.i32(T)), b.i32(0)], [C, DK], "f32", [1, 0]);
  const tpVB = b.makeTensorPtr(VB, [BH * T, DV], [DV, 1], [b.mul(pid, b.i32(T)), b.i32(0)], [C, DV], "f32", [1, 0]);
  const tpO  = b.makeTensorPtr(O,  [BH * T, DV], [DV, 1], [b.mul(pid, b.i32(T)), b.i32(0)], [C, DV], "f32", [1, 0]);

  // identity + strict-lower masks [C,C]
  const ar = b.arange(0, C);
  const rowI = b.broadcast(b.expandDims(ar, 1), [C, C]);
  const colI = b.broadcast(b.expandDims(ar, 0), [C, C]);
  const eye = b.select(b.eq(rowI, colI), b.splat(b.f32(1), [C, C], "f32"), b.splat(b.f32(0), [C, C], "f32"));
  const lowerMask = b.gt(rowI, colI);                  // strict lower (row > col)
  const zeroC = b.splat(b.f32(0), [C, C], "f32");

  const S0 = b.zeros([DK, DV], "f32");
  b.forIter(b.index(0), b.index(nChunks), b.index(1), [S0, tpQ, tpK, tpKB, tpVB, tpO],
    (bb, _iv, [S, tpQ2, tpK2, tpKB2, tpVB2, tpO2]) => {
      const q  = bb.mul(bb.load(tpQ2),  scale);        // [C,DK]
      const k  = bb.load(tpK2);                         // [C,DK]
      const kb = bb.load(tpKB2);                        // [C,DK]
      const vb = bb.load(tpVB2);                        // [C,DV]
      const kt = bb.trans(k);                           // [DK,C]
      // L = strict_lower( -(kb·kᵀ) )
      const kk = bb.dot(kb, kt);                         // [C,C]
      const L  = bb.select(lowerMask, bb.mul(kk, bb.f32(-1)), zeroC);
      // Neumann: inv = I + L + L² + … + L^{C-1}
      let inv = eye;
      let p = L;
      const [invF] = bb.forIter(bb.index(0), bb.index(C - 1), bb.index(1), [inv, p],
        (ib, _j, [acc, pw]) => {
          const accN = ib.add(acc, pw);
          const pwN = ib.dot(pw, L);                    // p = p·L
          return [accN, pwN];
        });
      const u = bb.dot(invF, vb);                       // [C,DV]
      const w = bb.dot(invF, kb);                       // [C,DK]
      // inter-chunk
      const wS = bb.dot(w, S);                          // [C,DV]
      const ui = bb.sub(u, wS);                         // [C,DV]
      const oInter = bb.dot(q, S);                      // [C,DV]
      const qk = bb.select(lowerMask, bb.dot(q, kt), zeroC);  // strict_lower(q·kᵀ)
      const o = bb.add(oInter, bb.dot(qk, ui));         // [C,DV]
      bb.store(tpO2, o);
      const Snew = bb.add(S, bb.dot(kt, ui));            // S += kᵀ·ui
      return [Snew,
        bb.advance(tpQ2, [bb.i32(C), bb.i32(0)]), bb.advance(tpK2, [bb.i32(C), bb.i32(0)]),
        bb.advance(tpKB2, [bb.i32(C), bb.i32(0)]), bb.advance(tpVB2, [bb.i32(C), bb.i32(0)]),
        bb.advance(tpO2, [bb.i32(C), bb.i32(0)])];
    });
  return b.build("gdn_chunk", 4);
}

// host reference = fla delta_rule_chunkwise (β folded: kb=k·β, vb=v·β)
function refChunk(Q: Float32Array, K: Float32Array, KB: Float32Array, VB: Float32Array,
                  BH: number, T: number, DK: number, DV: number, C: number) {
  const scale = 1 / Math.sqrt(DK);
  const nC = T / C;
  const O = new Float32Array(BH * T * DV);
  const matmul = (a: number[], b: number[], an: number, am: number, bm: number) => {
    const o = new Float32Array(an * bm);
    for (let i = 0; i < an; i++) for (let j = 0; j < bm; j++) { let s = 0; for (let k = 0; k < am; k++) s += a[i * am + k] * b[k * bm + j]; o[i * bm + j] = s; }
    return o;
  };
  for (let bh = 0; bh < BH; bh++) {
    const S = new Float32Array(DK * DV);
    for (let ci = 0; ci < nC; ci++) {
      const sl = (arr: Float32Array, D: number) => Array.from(arr.slice((bh * T + ci * C) * D, (bh * T + ci * C + C) * D));
      const q = sl(Q, DK).map(x => x * scale), k = sl(K, DK), kb = sl(KB, DK), vb = sl(VB, DV);
      // L = strict_lower(-(kb·kᵀ))
      const kkt = new Float32Array(C * C);
      for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) { let s = 0; for (let d = 0; d < DK; d++) s += kb[i * DK + d] * k[j * DK + d]; kkt[i * C + j] = s; }
      const L = new Float32Array(C * C);
      for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) L[i * C + j] = i > j ? -kkt[i * C + j] : 0;
      // Neumann inv = I + L + L² + ...
      let inv = new Float32Array(C * C); for (let i = 0; i < C; i++) inv[i * C + i] = 1;
      let pw = L;
      for (let it = 0; it < C - 1; it++) { for (let i = 0; i < C * C; i++) inv[i] += pw[i]; pw = matmul(Array.from(pw), Array.from(L), C, C, C); }
      const u = matmul(Array.from(inv), vb, C, C, DV);
      const w = matmul(Array.from(inv), kb, C, C, DK);
      // wS = w @ S
      const wS = new Float32Array(C * DV);
      for (let i = 0; i < C; i++) for (let j = 0; j < DV; j++) { let s = 0; for (let d = 0; d < DK; d++) s += w[i * DK + d] * S[d * DV + j]; wS[i * DV + j] = s; }
      const ui = new Float32Array(C * DV); for (let i = 0; i < C * DV; i++) ui[i] = u[i] - wS[i];
      const oInter = new Float32Array(C * DV);
      for (let i = 0; i < C; i++) for (let j = 0; j < DV; j++) { let s = 0; for (let d = 0; d < DK; d++) s += q[i * DK + d] * S[d * DV + j]; oInter[i * DV + j] = s; }
      const qk = new Float32Array(C * C);
      for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) { if (i > j) { let s = 0; for (let d = 0; d < DK; d++) s += q[i * DK + d] * k[j * DK + d]; qk[i * C + j] = s; } }
      const qkui = matmul(Array.from(qk), Array.from(ui), C, C, DV);
      // store o at [bh, ci*C .. ci*C+C]
      for (let i = 0; i < C; i++) for (let j = 0; j < DV; j++) O[((bh * T + ci * C) + i) * DV + j] = oInter[i * DV + j] + qkui[i * DV + j];
      // S += kᵀ @ ui
      for (let d = 0; d < DK; d++) for (let j = 0; j < DV; j++) { let s = 0; for (let i = 0; i < C; i++) s += k[i * DK + d] * ui[i * DV + j]; S[d * DV + j] += s; }
    }
  }
  return O;
}

const BH = 2, T = 64, DK = 128, DV = 128, C = 16;
const _ttir = buildGDNchunk(BH, T, DK, DV, C); if (process.env.DUMP) { console.log(_ttir); process.exit(0); } const k = compileAndLoad(_ttir, "gdn_chunk", 4);
console.log(`gdn_chunk loaded (BH=${BH} T=${T} d_k=${DK} d_v=${DV} C=${C})`);

const rnd = () => (Math.random() * 2 - 1) * 0.3;
const Q = new Float32Array(BH * T * DK).map(rnd);
const K = new Float32Array(BH * T * DK).map(rnd);
const B = new Float32Array(BH * T).map(() => Math.random() * 0.8 + 0.1);
const V = new Float32Array(BH * T * DV).map(rnd);
const KB = new Float32Array(BH * T * DK), VB = new Float32Array(BH * T * DV);
for (let i = 0; i < BH * T; i++) { for (let d = 0; d < DK; d++) KB[i * DK + d] = K[i * DK + d] * B[i]; for (let d = 0; d < DV; d++) VB[i * DV + d] = V[i * DV + d] * B[i]; }
const O = new Float32Array(BH * T * DV);
const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dKB = cuAlloc(KB.byteLength), dVB = cuAlloc(VB.byteLength), dO = cuAlloc(O.byteLength);
cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dKB, KB.buffer); cuHtoD(dVB, VB.buffer);
cuLaunch(k, [BH, 1, 1], [128, 1, 1], [dQ, dK, dKB, dVB, dO]); cuSync(); cuDtoH(O.buffer, dO);

const ref = refChunk(Q, K, KB, VB, BH, T, DK, DV, C);
let maxErr = 0, ok = true;
for (let i = 0; i < O.length; i++) { const d = Math.abs(O[i] - ref[i]); if (d > maxErr) maxErr = d; if (d > 1e-2 && ok) { ok = false; console.log(`mismatch [${i}]: got ${O[i].toFixed(4)} ref ${ref[i].toFixed(4)}`); } }
console.log(ok ? `✓ gdn_chunk correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");
cuFree(dQ); cuFree(dK); cuFree(dKB); cuFree(dVB); cuFree(dO);

// bench: chunked vs naive O(T) recurrence (gdn_clean) at the same size
{
  const BH = 4, T = 512, DK = 128, DV = 128, C = 16;
  const kc = compileAndLoad(buildGDNchunk(BH, T, DK, DV, C), "gdn_chunk", 4);
  const Q2 = new Float32Array(BH * T * DK).map(rnd), K2 = new Float32Array(BH * T * DK).map(rnd);
  const V2 = new Float32Array(BH * T * DV).map(rnd), B2 = new Float32Array(BH * T).map(() => Math.random() * 0.8 + 0.1);
  const KB2 = new Float32Array(BH * T * DK), VB2 = new Float32Array(BH * T * DV), O2 = new Float32Array(BH * T * DV);
  for (let i = 0; i < BH * T; i++) { for (let d = 0; d < DK; d++) KB2[i * DK + d] = K2[i * DK + d] * B2[i]; for (let d = 0; d < DV; d++) VB2[i * DV + d] = V2[i * DV + d] * B2[i]; }
  const dQ2 = cuAlloc(Q2.byteLength), dK2 = cuAlloc(K2.byteLength), dKB2 = cuAlloc(KB2.byteLength), dVB2 = cuAlloc(VB2.byteLength), dO2 = cuAlloc(O2.byteLength);
  cuHtoD(dQ2, Q2.buffer); cuHtoD(dK2, K2.buffer); cuHtoD(dKB2, KB2.buffer); cuHtoD(dVB2, VB2.buffer);
  for (let i = 0; i < 3; i++) cuLaunch(kc, [BH, 1, 1], [128, 1, 1], [dQ2, dK2, dKB2, dVB2, dO2]);
  cuSync();
  const t0 = performance.now(), it = 50;
  for (let i = 0; i < it; i++) cuLaunch(kc, [BH, 1, 1], [128, 1, 1], [dQ2, dK2, dKB2, dVB2, dO2]);
  cuSync();
  const dt = (performance.now() - t0) / 1000 / it;
  console.log(`bench chunked BH=${BH} T=${T} d=${DK}: ${(dt * 1e6).toFixed(1)} µs/it`);
  cuFree(dQ2); cuFree(dK2); cuFree(dKB2); cuFree(dVB2); cuFree(dO2);
}
