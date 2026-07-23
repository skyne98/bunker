// prototypes/gdn_clean.ts — GatedDeltaNet (Qwen3.5 linear attention), clean port.
//
// The 18/24 linear-attention layers are the core of Qwen3.5. This is the
// "clean first" port — the naive delta-rule recurrence (fla's
// `delta_rule_recurrence`), proving the recurrence in TTIR (scf.for with the
// state S [d_k×d_v] carried as an iter-arg). Conv1d, the decay gate, and the
// chunked-parallel prefill come next; this validates the core algorithm.
//
// Per token t (β folded on host as kb=k·β, vb=v·β; q scaled by 1/√d_k):
//   delta = vb − (kb·S)           # kb·S = Σ kb[:,None]·S  (reduce axis 0)
//   S    = S + k ⊗ delta          # outer product
//   o    = q · S                  # = Σ q[:,None]·S
//
// Pure elementwise + tt.reduce + broadcast — no dot, no libdevice (the real
// model's q/k L2-norm uses the Newton rsqrt from rmsnorm.ts; here we match the
// naive reference's 1/√d_k scaling for a clean correctness check).
//
//   bun run prototypes/gdn_clean.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

function buildGDN(BH: number, T: number, DK: number, DV: number) {
  const b = new TTIRBuilder();
  const Q  = b.param("Q",  { ptr: "f32" });   // [BH*T, DK]  (q, will be scaled)
  const K  = b.param("K",  { ptr: "f32" });   // [BH*T, DK]  (original k, for outer product)
  const KB = b.param("KB", { ptr: "f32" });   // [BH*T, DK]  (k·β)
  const VB = b.param("VB", { ptr: "f32" });   // [BH*T, DV]  (v·β)
  const O  = b.param("O",  { ptr: "f32" });   // [BH*T, DV]
  const pid = b.programId(0);
  const base = b.mul(pid, b.i32(T));
  const scale = b.f32(1 / Math.sqrt(DK));

  const tpQ  = b.makeTensorPtr(Q,  [BH * T * DK], [1], [b.mul(base, b.i32(DK))], [DK], "f32", [0]);
  const tpK  = b.makeTensorPtr(K,  [BH * T * DK], [1], [b.mul(base, b.i32(DK))], [DK], "f32", [0]);
  const tpKB = b.makeTensorPtr(KB, [BH * T * DK], [1], [b.mul(base, b.i32(DK))], [DK], "f32", [0]);
  const tpVB = b.makeTensorPtr(VB, [BH * T * DV], [1], [b.mul(base, b.i32(DV))], [DV], "f32", [0]);
  const tpO  = b.makeTensorPtr(O,  [BH * T * DV], [1], [b.mul(base, b.i32(DV))], [DV], "f32", [0]);

  const S0 = b.zeros([DK, DV], "f32");

  const [Sf] = b.forIter(b.index(0), b.index(T), b.index(1), [S0, tpQ, tpK, tpKB, tpVB, tpO],
    (bb, _iv, [S, tpQ2, tpK2, tpKB2, tpVB2, tpO2]) => {
      const q  = bb.mul(bb.load(tpQ2),  scale);            // [DK]
      const k  = bb.load(tpK2);                            // [DK]
      const kb = bb.load(tpKB2);                           // [DK]
      const vb = bb.load(tpVB2);                           // [DV]
      // kb·S = Σ kb[:,None]*S  (axis 0) → [DV]
      const kbBc = bb.broadcast(bb.expandDims(kb, 1), [DK, DV]);
      const kbS  = bb.sum(bb.mul(kbBc, S), 0);             // [DV]
      const delta = bb.sub(vb, kbS);                       // [DV]
      // S += k ⊗ delta
      const kBc  = bb.broadcast(bb.expandDims(k, 1),    [DK, DV]);
      const dBc  = bb.broadcast(bb.expandDims(delta, 0), [DK, DV]);
      const Snew = bb.add(S, bb.mul(kBc, dBc));            // [DK, DV]
      // o = q·S = Σ q[:,None]*S  (axis 0) → [DV]
      const qBc = bb.broadcast(bb.expandDims(q, 1), [DK, DV]);
      const o = bb.sum(bb.mul(qBc, Snew), 0);              // [DV]
      bb.store(tpO2, o);
      return [Snew,
        bb.advance(tpQ2,  [bb.i32(DK)]),
        bb.advance(tpK2,  [bb.i32(DK)]),
        bb.advance(tpKB2, [bb.i32(DK)]),
        bb.advance(tpVB2, [bb.i32(DV)]),
        bb.advance(tpO2,  [bb.i32(DV)])];
    });
  void Sf;

  return b.build("gdn_clean", 4);
}

// host reference = fla delta_rule_recurrence (β folded: kb=k·β, vb=v·β)
function refGDN(Q: Float32Array, K: Float32Array, KB: Float32Array, VB: Float32Array,
                BH: number, T: number, DK: number, DV: number) {
  const O = new Float32Array(BH * T * DV);
  const scale = 1 / Math.sqrt(DK);
  for (let bh = 0; bh < BH; bh++) {
    const S = new Float32Array(DK * DV);             // [DK, DV]
    for (let t = 0; t < T; t++) {
      const q = Q.slice((bh * T + t) * DK, (bh * T + t) * DK + DK).map(x => x * scale);
      const k = K.slice((bh * T + t) * DK, (bh * T + t) * DK + DK);
      const kb = KB.slice((bh * T + t) * DK, (bh * T + t) * DK + DK);
      const vb = VB.slice((bh * T + t) * DV, (bh * T + t) * DV + DV);
      const kbS = new Float32Array(DV);
      for (let d = 0; d < DV; d++) { let s = 0; for (let i = 0; i < DK; i++) s += kb[i] * S[i * DV + d]; kbS[d] = s; }
      const delta = new Float32Array(DV); for (let d = 0; d < DV; d++) delta[d] = vb[d] - kbS[d];
      for (let i = 0; i < DK; i++) for (let d = 0; d < DV; d++) S[i * DV + d] += k[i] * delta[d];
      for (let d = 0; d < DV; d++) { let s = 0; for (let i = 0; i < DK; i++) s += q[i] * S[i * DV + d]; O[(bh * T + t) * DV + d] = s; }
    }
  }
  return O;
}

const BH = 4, T = 64, DK = 128, DV = 128;            // Qwen3.5: d_k=d_v=128
const k = compileAndLoad(buildGDN(BH, T, DK, DV), "gdn_clean", 4);
console.log(`gdn_clean loaded (shmem=${k.shmem}, BH=${BH} T=${T} d_k=${DK} d_v=${DV})`);

const rnd = () => (Math.random() * 2 - 1) * 0.3;
const Q = new Float32Array(BH * T * DK).map(rnd);
const K = new Float32Array(BH * T * DK).map(rnd);
const B = new Float32Array(BH * T).map(() => Math.random() * 0.8 + 0.1);  // β ∈ (0.1,0.9)
const V = new Float32Array(BH * T * DV).map(rnd);
const KB = new Float32Array(BH * T * DK), VB = new Float32Array(BH * T * DV);
for (let i = 0; i < BH * T; i++) for (let d = 0; d < DK; d++) KB[i * DK + d] = K[i * DK + d] * B[i];
for (let i = 0; i < BH * T; i++) for (let d = 0; d < DV; d++) VB[i * DV + d] = V[i * DV + d] * B[i];
const O = new Float32Array(BH * T * DV);

const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dKB = cuAlloc(KB.byteLength), dVB = cuAlloc(VB.byteLength), dO = cuAlloc(O.byteLength);
cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dKB, KB.buffer); cuHtoD(dVB, VB.buffer);
cuLaunch(k, [BH, 1, 1], [128, 1, 1], [dQ, dK, dKB, dVB, dO]);
cuSync(); cuDtoH(O.buffer, dO);

const ref = refGDN(Q, K, KB, VB, BH, T, DK, DV);
let maxErr = 0, ok = true;
for (let i = 0; i < O.length; i++) {
  const d = Math.abs(O[i] - ref[i]);
  if (d > maxErr) maxErr = d;
  if (d > 1e-3 && ok) { ok = false; console.log(`mismatch [${i}]: got ${O[i].toFixed(5)} ref ${ref[i].toFixed(5)}`); }
}
console.log(ok ? `✓ gdn_clean correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");
cuFree(dQ); cuFree(dK); cuFree(dKB); cuFree(dVB); cuFree(dO);
