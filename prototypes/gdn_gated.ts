// prototypes/gdn_gated.ts — GatedDeltaNet per-token step WITH the decay gate.
//
// Extends gdn_clean (naive delta-rule recurrence) with the "gated" feature that
// defines GatedDeltaNet: the per-token decay gate  S *= decay[t],  where
// decay[t] = exp(g[t]), g[t] = -exp(A_log)·softplus(a[t]+dt_bias). decay is
// precomputed on the HOST (exact, no libdevice — softplus/log done in TS, like
// RoPE's cos/sin) and passed as a [BH*T] buffer; the kernel only multiplies.
// β stays folded on the host (kb=k·β, vb=v·β), so no per-token scalar-β in the
// kernel. (L2-norm of q/k is a separate pre-pass in fla too — added later.)
//
// Per token (q scaled 1/√d_k):
//   S = S · decay
//   delta = vb − (kb·S)
//   S += k ⊗ delta
//   o = q · S
//
//   bun run prototypes/gdn_gated.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

function buildGDN(BH: number, T: number, DK: number, DV: number) {
  const b = new TTIRBuilder();
  const Q = b.param("Q", { ptr: "f32" }), K = b.param("K", { ptr: "f32" });
  const KB = b.param("KB", { ptr: "f32" }), VB = b.param("VB", { ptr: "f32" });
  const DEC = b.param("DEC", { ptr: "f32" }), O = b.param("O", { ptr: "f32" });
  const pid = b.programId(0);
  const base = b.mul(pid, b.i32(T));
  const scale = b.f32(1 / Math.sqrt(DK));

  const tpQ  = b.makeTensorPtr(Q,  [BH * T * DK], [1], [b.mul(base, b.i32(DK))], [DK], "f32", [0]);
  const tpK  = b.makeTensorPtr(K,  [BH * T * DK], [1], [b.mul(base, b.i32(DK))], [DK], "f32", [0]);
  const tpKB = b.makeTensorPtr(KB, [BH * T * DK], [1], [b.mul(base, b.i32(DK))], [DK], "f32", [0]);
  const tpVB = b.makeTensorPtr(VB, [BH * T * DV], [1], [b.mul(base, b.i32(DV))], [DV], "f32", [0]);
  const tpD  = b.makeTensorPtr(DEC, [BH * T], [1], [base], [1], "f32", [0]);
  const tpO  = b.makeTensorPtr(O,  [BH * T * DV], [1], [b.mul(base, b.i32(DV))], [DV], "f32", [0]);

  const S0 = b.zeros([DK, DV], "f32");
  b.forIter(b.index(0), b.index(T), b.index(1), [S0, tpQ, tpK, tpKB, tpVB, tpD, tpO],
    (bb, _iv, [S, tpQ2, tpK2, tpKB2, tpVB2, tpD2, tpO2]) => {
      const q  = bb.mul(bb.load(tpQ2),  scale);            // [DK]
      const k  = bb.load(tpK2);                             // [DK]
      const kb = bb.load(tpKB2);                           // [DK]
      const vb = bb.load(tpVB2);                           // [DV]
      // S *= decay[t]   (decay is a [1] tile → [1,1] → broadcast [DK,DV])
      const decay1 = bb.load(tpD2);                        // [1]
      const decayBc = bb.broadcast(bb.expandDims(decay1, 0), [DK, DV]);
      const Sd = bb.mul(S, decayBc);                       // [DK, DV]
      // delta = vb - (kb·S)
      const kbBc = bb.broadcast(bb.expandDims(kb, 1), [DK, DV]);
      const kS = bb.sum(bb.mul(kbBc, Sd), 0);              // [DV]
      const delta = bb.sub(vb, kS);                        // [DV]
      // S += k ⊗ delta
      const kBc = bb.broadcast(bb.expandDims(k, 1),    [DK, DV]);
      const dBc = bb.broadcast(bb.expandDims(delta, 0), [DK, DV]);
      const Snew = bb.add(Sd, bb.mul(kBc, dBc));          // [DK, DV]
      // o = q·S
      const qBc = bb.broadcast(bb.expandDims(q, 1), [DK, DV]);
      const o = bb.sum(bb.mul(qBc, Snew), 0);              // [DV]
      bb.store(tpO2, o);
      return [Snew,
        bb.advance(tpQ2, [bb.i32(DK)]), bb.advance(tpK2, [bb.i32(DK)]),
        bb.advance(tpKB2, [bb.i32(DK)]), bb.advance(tpVB2, [bb.i32(DV)]),
        bb.advance(tpD2, [bb.i32(1)]), bb.advance(tpO2, [bb.i32(DV)])];
    });
  return b.build("gdn_gated", 4);
}

function refGDN(Q: Float32Array, K: Float32Array, KB: Float32Array, VB: Float32Array, DEC: Float32Array,
                BH: number, T: number, DK: number, DV: number) {
  const O = new Float32Array(BH * T * DV);
  const scale = 1 / Math.sqrt(DK);
  for (let bh = 0; bh < BH; bh++) {
    const S = new Float32Array(DK * DV);
    for (let t = 0; t < T; t++) {
      const q = Q.slice((bh * T + t) * DK, (bh * T + t) * DK + DK).map(x => x * scale);
      const k = K.slice((bh * T + t) * DK, (bh * T + t) * DK + DK);
      const kb = KB.slice((bh * T + t) * DK, (bh * T + t) * DK + DK);
      const vb = VB.slice((bh * T + t) * DV, (bh * T + t) * DV + DV);
      const decay = DEC[bh * T + t];
      for (let i = 0; i < DK * DV; i++) S[i] *= decay;          // S *= decay
      const kS = new Float32Array(DV);
      for (let d = 0; d < DV; d++) { let s = 0; for (let i = 0; i < DK; i++) s += kb[i] * S[i * DV + d]; kS[d] = s; }
      const delta = new Float32Array(DV); for (let d = 0; d < DV; d++) delta[d] = vb[d] - kS[d];
      for (let i = 0; i < DK; i++) for (let d = 0; d < DV; d++) S[i * DV + d] += k[i] * delta[d];
      for (let d = 0; d < DV; d++) { let s = 0; for (let i = 0; i < DK; i++) s += q[i] * S[i * DV + d]; O[(bh * T + t) * DV + d] = s; }
    }
  }
  return O;
}

const BH = 4, T = 64, DK = 128, DV = 128;
const k = compileAndLoad(buildGDN(BH, T, DK, DV), "gdn_gated", 4);
console.log(`gdn_gated loaded (shmem=${k.shmem}, BH=${BH} T=${T} d_k=${DK} d_v=${DV})`);

const rnd = () => (Math.random() * 2 - 1) * 0.3;
const Q = new Float32Array(BH * T * DK).map(rnd);
const K = new Float32Array(BH * T * DK).map(rnd);
const B = new Float32Array(BH * T).map(() => Math.random() * 0.8 + 0.1);
const V = new Float32Array(BH * T * DV).map(rnd);
const KB = new Float32Array(BH * T * DK), VB = new Float32Array(BH * T * DV);
for (let i = 0; i < BH * T; i++) { for (let d = 0; d < DK; d++) KB[i * DK + d] = K[i * DK + d] * B[i]; for (let d = 0; d < DV; d++) VB[i * DV + d] = V[i * DV + d] * B[i]; }
const DEC = new Float32Array(BH * T).map(() => 0.95 + Math.random() * 0.04);   // realistic decay ~0.95-0.99
const O = new Float32Array(BH * T * DV);
const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dKB = cuAlloc(KB.byteLength), dVB = cuAlloc(VB.byteLength), dDEC = cuAlloc(DEC.byteLength), dO = cuAlloc(O.byteLength);
cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dKB, KB.buffer); cuHtoD(dVB, VB.buffer); cuHtoD(dDEC, DEC.buffer);
cuLaunch(k, [BH, 1, 1], [128, 1, 1], [dQ, dK, dKB, dVB, dDEC, dO]);
cuSync(); cuDtoH(O.buffer, dO);

const ref = refGDN(Q, K, KB, VB, DEC, BH, T, DK, DV);
let maxErr = 0, ok = true;
for (let i = 0; i < O.length; i++) {
  const d = Math.abs(O[i] - ref[i]);
  if (d > maxErr) maxErr = d;
  if (d > 1e-3 && ok) { ok = false; console.log(`mismatch [${i}]: got ${O[i].toFixed(5)} ref ${ref[i].toFixed(5)}`); }
}
console.log(ok ? `✓ gdn_gated correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");
cuFree(dQ); cuFree(dK); cuFree(dKB); cuFree(dVB); cuFree(dDEC); cuFree(dO);
