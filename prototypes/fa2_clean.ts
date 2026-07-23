// prototypes/fa2_clean.ts
//
// Prototype: FlashAttention-2 (forward, causal) ported to the bunker TTIR
// builder, translated from Triton's canonical `06-fused-attention.py`
// (`_attn_fwd_inner`). This is the "original clean" reference port — a second
// prototype will port vLLM's optimized `triton_unified_attention` (split-KV
// decode, Q-block/GQA tiling) on top of this structure.
//
// Algorithm (online softmax, per Q-row block, K-loop with iter-args):
//   carry (acc=O, m_i=running max, l_i=running sum) across K/V tiles:
//     qk  = dot(Q, Kᵀ) * scale                  # [BM, BN] f16→f32
//     qk  = select(causal_mask, qk, -inf)
//     m_ij = maximum(m_i, reduce_max(qk, 1))
//     α   = exp(m_i - m_ij)                      # rescale factor
//     p   = exp(qk - m_ij)                      # [BM, BN]
//     acc = dot(p_f16, V_f16, α*acc)            # fused: acc = α*acc + p@V
//     l_i = α*l_i + sum(p, 1);  m_i = m_ij
//   O = acc / l_i
//
// Every op here maps 1:1 to a Triton/TTIR primitive the builder already has:
//   tl.dot            -> b.dot (with acc)        -> mma.sync via accelerate_matmul
//   tl.load/store     -> b.load/b.store (boundaryCheck)
//   tl.make_block_ptr -> b.makeTensorPtr;  tl.advance -> b.advance
//   tl.math.exp2      -> b.exp (natural exp; equivalent up to scale)
//   tl.max/tl.sum      -> b.max/b.sum (tt.reduce)
//   tl.where           -> b.select
//   tl.maximum         -> b.maximum
//   p.to(f16)          -> b.fptrunc
//   k.T                -> b.trans
//   for start_n ...    -> b.forIter (iter-args = acc, m_i, l_i, start_n, tpK, tpV)
//
// No libdevice needed: exp inlines; no rsqrt/sin/cos. Portable today.
//
// Verified hardware: RTX 3090 (sm_86). Usage:
//   bun run prototypes/fa2_clean.ts

import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";
import { f32to16 } from "../src/kernel";

// ── build the FA2 forward kernel ──────────────────────────────────────
function buildFA2(M: number, N: number, D: number, BM: number, BN: number) {
  const b = new TTIRBuilder();
  const Qp = b.param("Q", { ptr: "f16" });
  const Kp = b.param("K", { ptr: "f16" });
  const Vp = b.param("V", { ptr: "f16" });
  const Op = b.param("O", { ptr: "f32" });

  const pidM = b.programId(0);
  const offM = b.mul(pidM, b.i32(BM));
  const scale = b.f32(1 / Math.sqrt(D));

  // tiled pointers: Q[BM,D], K[BN,D], V[BN,D], O[BM,D]  (K/V advance along dim 0)
  const tpQ = b.makeTensorPtr(Qp, [M, D], [D, 1], [offM, b.i32(0)], [BM, D], "f16", [1, 0]);
  const tpK0 = b.makeTensorPtr(Kp, [N, D], [D, 1], [b.i32(0), b.i32(0)], [BN, D], "f16", [1, 0]);
  const tpV0 = b.makeTensorPtr(Vp, [N, D], [D, 1], [b.i32(0), b.i32(0)], [BN, D], "f16", [1, 0]);
  const tpO = b.makeTensorPtr(Op, [M, D], [D, 1], [offM, b.i32(0)], [BM, D], "f32", [1, 0]);

  const q = b.load(tpQ);                                    // [BM, D] f16

  const acc0 = b.zeros([BM, D], "f32");
  const m0 = b.splat(b.f32(-1e30), [BM], "f32");            // running max = -inf
  const l0 = b.splat(b.f32(0), [BM], "f32");                // running sum = 0
  const sn0 = b.i32(0);                                     // start_n counter

  const [acc, , l_i] = b.forIter(
    b.index(0), b.index(N), b.index(BN),
    [acc0, m0, l0, sn0, tpK0, tpV0],
    (bb, _iv, [acc, m_i, l_i, sn, tpK, tpV]) => {
      // --- load K, V tiles; qk = Q @ Kᵀ ---
      const kt = bb.trans(bb.load(tpK));                     // [D, BN] f16
      const qk = bb.mul(bb.dot(q, kt), scale);               // [BM, BN] f32
      const v = bb.load(tpV);                                // [BN, D] f16

      // --- causal mask: rowAbs (offM+arange BM) >= colAbs (sn+arange BN) ---
      const rowAbs = bb.add(offM, bb.arange(0, BM));        // [BM] i32
      const colAbs = bb.add(sn, bb.arange(0, BN));          // [BN] i32
      const rowBc = bb.broadcast(bb.expandDims(rowAbs, 1), [BM, BN]);   // [BM, BN]
      const colBc = bb.broadcast(bb.expandDims(colAbs, 0), [BM, BN]);   // [BN, BN]
      const mask = bb.ge(rowBc, colBc);                     // [BM, BN] i1
      const negInf = bb.splat(bb.f32(-1e30), [BM, BN], "f32");
      const qkM = bb.select(mask, qk, negInf);              // masked qk

      // --- online softmax update ---
      const m_ij = bb.maximum(m_i, bb.max(qkM, 1));         // [BM]
      const alpha = bb.exp(bb.sub(m_i, m_ij));               // [BM]
      const m_ijBc = bb.broadcast(bb.expandDims(m_ij, 1), [BM, BN]);
      const p = bb.exp(bb.sub(qkM, m_ijBc));                 // [BM, BN] f32
      const l_ij = bb.sum(p, 1);                             // [BM]

      // acc = α*acc + p@V  (dot-with-acc fuses the rescale)
      const aBc = bb.broadcast(bb.expandDims(alpha, 1), [BM, D]);
      const accScaled = bb.mul(acc, aBc);                   // [BM, D]
      const p16 = bb.fptrunc(p, "f16");
      const accNew = bb.dot(p16, v, accScaled);              // [BM, D] f32

      const lNew = bb.add(bb.mul(l_i, alpha), l_ij);
      const snNew = bb.add(sn, bb.i32(BN));
      const tpKNext = bb.advance(tpK, [bb.i32(BN), bb.i32(0)]);
      const tpVNext = bb.advance(tpV, [bb.i32(BN), bb.i32(0)]);
      return [accNew, m_ij, lNew, snNew, tpKNext, tpVNext];
    },
  );

  const lBc = b.broadcast(b.expandDims(l_i, 1), [BM, D]);
  const o = b.divf(acc, lBc);
  b.store(tpO, o, { boundaryCheck: [0, 1] });
  return b.build("fa2_fwd", 8);
}

// ── host reference (causal attention, from f32 source; GPU uses f16 so ~0.05 tol) ─
function refAttn(Qf: Float32Array, Kf: Float32Array, Vf: Float32Array, M: number, N: number, D: number) {
  const O = new Float32Array(M * D);
  const scale = 1 / Math.sqrt(D);
  for (let i = 0; i < M; i++) {
    let m = -Infinity;
    for (let j = 0; j <= i; j++) {
      let s = 0; for (let k = 0; k < D; k++) s += Qf[i * D + k] * Kf[j * D + k];
      s *= scale; if (s > m) m = s;
    }
    let sum = 0; const e: number[] = [];
    for (let j = 0; j <= i; j++) {
      let s = 0; for (let k = 0; k < D; k++) s += Qf[i * D + k] * Kf[j * D + k];
      const ev = Math.exp(s * scale - m); e.push(ev); sum += ev;
    }
    for (let j = 0; j <= i; j++) {
      const w = e[j] / sum;
      for (let k = 0; k < D; k++) O[i * D + k] += w * Vf[j * D + k];
    }
  }
  return O;
}
function f16f(x: number): number { void x; return 0; }

// ── run + verify + bench ───────────────────────────────────────────────
async function main() {
  // small correctness run
  {
    const M = 256, N = 256, D = 64, BM = 64, BN = 64;
    const ttir = buildFA2(M, N, D, BM, BN);
    if (process.env.DUMP) { console.log(ttir); process.exit(0); }
    const k = compileAndLoad(ttir, "fa2_fwd", 8);
    console.log(`fa2_fwd loaded (shmem=${k.shmem})`);

    const Qf = new Float32Array(M * D), Kf = new Float32Array(N * D), Vf = new Float32Array(N * D);
    for (let i = 0; i < M * D; i++) Qf[i] = (Math.random() * 2 - 1) * 0.5;
    for (let i = 0; i < N * D; i++) Kf[i] = (Math.random() * 2 - 1) * 0.5;
    for (let i = 0; i < N * D; i++) Vf[i] = (Math.random() * 2 - 1) * 0.5;
    const Q = f32to16(Qf), K = f32to16(Kf), V = f32to16(Vf), O = new Float32Array(M * D);

    const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dV = cuAlloc(V.byteLength), dO = cuAlloc(O.byteLength);
    cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dV, V.buffer);
    cuLaunch(k, [Math.ceil(M / BM), 1, 1], [256, 1, 1], [dQ, dK, dV, dO]);
    cuSync();
    cuDtoH(O.buffer, dO);

    const ref = refAttn(Qf, Kf, Vf, M, N, D);
    let maxErr = 0, ok = true;
    for (let i = 0; i < M && ok; i++) for (let dd = 0; dd < D; dd++) {
      const d = Math.abs(O[i * D + dd] - ref[i * D + dd]);
      if (d > maxErr) maxErr = d;
      if (d > 0.1) { ok = false; console.log(`mismatch [${i},${dd}]: got ${O[i*D+dd].toFixed(4)} ref ${ref[i*D+dd].toFixed(4)}`); break; }
    }
    console.log(ok ? `✓ fa2_clean correct (max err ${maxErr.toFixed(4)})` : "✗ FAILED");
    cuFree(dQ); cuFree(dK); cuFree(dV); cuFree(dO);
  }

  // larger benchmark
  {
    const M = 2048, N = 2048, D = 128, BM = 128, BN = 64;
    const ttir = buildFA2(M, N, D, BM, BN);
    const k = compileAndLoad(ttir, "fa2_fwd", 8);
    const Q = f32to16(new Float32Array(M * D).map(() => (Math.random() * 2 - 1) * 0.5));
    const K = f32to16(new Float32Array(N * D).map(() => (Math.random() * 2 - 1) * 0.5));
    const V = f32to16(new Float32Array(N * D).map(() => (Math.random() * 2 - 1) * 0.5));
    const O = new Float32Array(M * D);
    const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dV = cuAlloc(V.byteLength), dO = cuAlloc(O.byteLength);
    cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dV, V.buffer);
    // warmup
    for (let i = 0; i < 3; i++) cuLaunch(k, [Math.ceil(M / BM), 1, 1], [256, 1, 1], [dQ, dK, dV, dO]);
    cuSync();
    const t0 = performance.now();
    const it = 50;
    for (let i = 0; i < it; i++) cuLaunch(k, [Math.ceil(M / BM), 1, 1], [256, 1, 1], [dQ, dK, dV, dO]);
    cuSync();
    const dt = (performance.now() - t0) / 1000 / it;
    const flops = 4 * M * N * D;              // 2*QK + 2*PV
    console.log(`bench ${M}x${N}x${D}: ${(dt * 1e3).toFixed(3)} ms/it, ${(flops / dt / 1e12).toFixed(2)} TFLOPS`);
    cuFree(dQ); cuFree(dK); cuFree(dV); cuFree(dO);
  }
}
main();
