// prototypes/fa2_vllm.ts
//
// Prototype: optimized FlashAttention-2 (prefill, causal, multi-head GQA)
// ported in the style of vLLM's `triton_unified_attention` / IBM's "Anatomy of
// a Triton Attention Kernel" — built on top of the clean fa2_clean.ts port.
//
// Optimizations layered on top of the clean port:
//   1. Multi-head GQA grid: grid = [num_q_blocks, num_q_heads]. This fixes the
//      clean port's fatal flaw — grid [M/BM, 1] = only 16 programs on 82 SMs
//      (severe under-occupancy → 4.2 TFLOPS). Gridding over heads gives
//      Hq× more programs and saturates the SMs. Each q-head program derives its
//      kv_head = q_head // G and loads that KV head (correct GQA).
//   2. Two-stage causal split (FA2 work partitioning): for q-block at row m,
//      STAGE 1 loops K-tiles [0, m·BM) fully UNMASKED (all-valid lower
//      triangle — skips the ~half of tiles that the clean port masked to -inf
//      and pointlessly processed), then STAGE 2 does just the diagonal block
//      [m·BM, (m+1)·BM) WITH the causal mask. ~halves the work for causal.
//   3. Tuned block (BM=128, BN=64), num_warps=4, num_stages=3.
//
// What this prototype does NOT yet include (the decode-oriented vLLM opts,
// which matter most at BLOCK_M=1 where prefill-style tiling underutilizes):
//   - GQA Q-block KV reuse (load each KV head once per group of G q-heads —
//     big decode bandwidth win; here each q-head program loads its own KV).
//   - Split-KV parallel-tiled-softmax decode (3D grid + reduction kernel) —
//     the signature vLLM decode optimization; a separate kernel to add.
//
// Verified hardware: RTX 3090 (sm_86). Usage:
//   bun run prototypes/fa2_vllm.ts

import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";
import { f32to16 } from "../src/kernel";

// one online-softmax tile update. If `mask` is given, qk is masked to -inf
// where mask is false BEFORE the max/exp (causal diagonal stage).
function attnTile(bb: TTIRBuilder, q: any, kt: any, v: any, acc: any, m_i: any, l_i: any,
                  scale: any, BM: number, BN: number, D: number, mask?: any) {
  const qk = bb.mul(bb.dot(q, kt), scale);                  // [BM, BN] f32
  const qkM = mask ? bb.select(mask, qk, bb.splat(bb.f32(-1e30), [BM, BN], "f32")) : qk;
  const m_ij = bb.maximum(m_i, bb.max(qkM, 1));             // [BM]
  const alpha = bb.exp(bb.sub(m_i, m_ij));                   // [BM]
  const m_ijBc = bb.broadcast(bb.expandDims(m_ij, 1), [BM, BN]);
  const p = bb.exp(bb.sub(qkM, m_ijBc));                     // [BM, BN]
  const l_ij = bb.sum(p, 1);                                 // [BM]
  const aBc = bb.broadcast(bb.expandDims(alpha, 1), [BM, D]);
  const accNew = bb.dot(bb.fptrunc(p, "f16"), v, bb.mul(acc, aBc));  // α·acc + p@V
  const lNew = bb.add(bb.mul(l_i, alpha), l_ij);
  return [accNew, m_ij, lNew] as const;
}

export function buildFA2vllm(Hq: number, Hkv: number, M: number, N: number, D: number,
                      BM: number, BN: number, numWarps = 4, numStages = 3) {
  const G = Hq / Hkv;
  const b = new TTIRBuilder();
  const Qp = b.param("Q", { ptr: "f16" });   // [Hq*M, D] row-major (head-major)
  const Kp = b.param("K", { ptr: "f16" });   // [Hkv*N, D]
  const Vp = b.param("V", { ptr: "f16" });   // [Hkv*N, D]
  const Op = b.param("O", { ptr: "f32" });  // [Hq*M, D]

  const pidM = b.programId(0);
  const qHead = b.programId(1);
  const kvHead = b.divi(qHead, b.i32(G));                    // q_head // G
  const offM = b.mul(pidM, b.i32(BM));
  const qRowOff = b.add(b.mul(qHead, b.i32(M)), offM);       // q_head*M + offM
  const kvBase = b.mul(kvHead, b.i32(N));                    // kv_head*N
  const diagStart = b.mul(pidM, b.i32(BM));                  // m·BM
  const diagEnd = b.add(diagStart, b.i32(BM));               // (m+1)·BM
  const scale = b.f32(1 / Math.sqrt(D));

  const tpQ = b.makeTensorPtr(Qp, [Hq * M, D], [D, 1], [qRowOff, b.i32(0)], [BM, D], "f16", [1, 0]);
  const tpK0 = b.makeTensorPtr(Kp, [Hkv * N, D], [D, 1], [kvBase, b.i32(0)], [BN, D], "f16", [1, 0]);
  const tpV0 = b.makeTensorPtr(Vp, [Hkv * N, D], [D, 1], [kvBase, b.i32(0)], [BN, D], "f16", [1, 0]);
  const tpO = b.makeTensorPtr(Op, [Hq * M, D], [D, 1], [qRowOff, b.i32(0)], [BM, D], "f32", [1, 0]);

  const q = b.load(tpQ);                                     // [BM, D] (loaded once)
  const acc0 = b.zeros([BM, D], "f32");
  const m0 = b.splat(b.f32(-1e30), [BM], "f32");
  const l0 = b.splat(b.f32(0), [BM], "f32");

  // ── STAGE 1: off-diagonal, fully unmasked [0, diagStart) ──────────────
  const r1 = b.forIter(b.index(0), diagStart, b.index(BN), [acc0, m0, l0, tpK0, tpV0],
    (bb, _iv, [acc, m_i, l_i, tpK, tpV]) => {
      const kt = bb.trans(bb.load(tpK));
      const v = bb.load(tpV);
      const [a, m, l] = attnTile(bb, q, kt, v, acc, m_i, l_i, scale, BM, BN, D);
      return [a, m, l, bb.advance(tpK, [bb.i32(BN), bb.i32(0)]), bb.advance(tpV, [bb.i32(BN), bb.i32(0)])];
    });

  // ── STAGE 2: diagonal block [diagStart, diagEnd), masked ─────────────
  const r2 = b.forIter(diagStart, diagEnd, b.index(BN), r1,
    (bb, iv, [acc, m_i, l_i, tpK, tpV]) => {
      const kt = bb.trans(bb.load(tpK));
      const v = bb.load(tpV);
      // causal mask: rowAbs (offM+arange BM) >= colAbs (iv+arange BN)
      const rowAbs = bb.add(offM, bb.arange(0, BM));
      const colAbs = bb.add(bb.indexCast(iv, "i32"), bb.arange(0, BN));
      const rowBc = bb.broadcast(bb.expandDims(rowAbs, 1), [BM, BN]);
      const colBc = bb.broadcast(bb.expandDims(colAbs, 0), [BM, BN]);
      const mask = bb.ge(rowBc, colBc);
      const [a, m, l] = attnTile(bb, q, kt, v, acc, m_i, l_i, scale, BM, BN, D, mask);
      return [a, m, l, bb.advance(tpK, [bb.i32(BN), bb.i32(0)]), bb.advance(tpV, [bb.i32(BN), bb.i32(0)])];
    });

  const [accF, , lF] = r2;
  const o = b.divf(accF, b.broadcast(b.expandDims(lF, 1), [BM, D]));
  b.store(tpO, o, { boundaryCheck: [0, 1] });
  return b.build("fa2_vllm", numWarps, numStages);
}

// ── host reference (multi-head GQA causal attention, from f32 source) ──
export function refAttn(Qf: Float32Array, Kf: Float32Array, Vf: Float32Array,
                 Hq: number, Hkv: number, M: number, N: number, D: number) {
  const G = Hq / Hkv;
  const O = new Float32Array(Hq * M * D);
  const scale = 1 / Math.sqrt(D);
  for (let h = 0; h < Hq; h++) {
    const kv = (h / G) | 0;
    for (let i = 0; i < M; i++) {
      let m = -Infinity;
      for (let j = 0; j <= i; j++) {
        let s = 0; for (let k = 0; k < D; k++) s += Qf[h * M * D + i * D + k] * Kf[kv * N * D + j * D + k];
        s *= scale; if (s > m) m = s;
      }
      let sum = 0; const e: number[] = [];
      for (let j = 0; j <= i; j++) {
        let s = 0; for (let k = 0; k < D; k++) s += Qf[h * M * D + i * D + k] * Kf[kv * N * D + j * D + k];
        const ev = Math.exp(s * scale - m); e.push(ev); sum += ev;
      }
      for (let j = 0; j <= i; j++) {
        const w = e[j] / sum;
        for (let k = 0; k < D; k++) O[h * M * D + i * D + k] += w * Vf[kv * N * D + j * D + k];
      }
    }
  }
  return O;
}

async function main() {
  // correctness
  {
    const Hq = 8, Hkv = 2, M = 256, N = 256, D = 64, BM = Number(process.env.BM||64), BN = Number(process.env.BN||64), G = Hq / Hkv;
    const NW = Number(process.env.WARPS||4), NS = Number(process.env.STAGES||3);
    const ttir = buildFA2vllm(Hq, Hkv, M, N, D, BM, BN, NW, NS);
    if (process.env.DUMP) { console.log(ttir); process.exit(0); }
    const k = compileAndLoad(ttir, "fa2_vllm", NW);
    console.log(`fa2_vllm loaded (shmem=${k.shmem}, GQA G=${G})`);

    const Qf = new Float32Array(Hq * M * D).map(() => (Math.random() * 2 - 1) * 0.5);
    const Kf = new Float32Array(Hkv * N * D).map(() => (Math.random() * 2 - 1) * 0.5);
    const Vf = new Float32Array(Hkv * N * D).map(() => (Math.random() * 2 - 1) * 0.5);
    const Q = f32to16(Qf), K = f32to16(Kf), V = f32to16(Vf), O = new Float32Array(Hq * M * D);
    const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dV = cuAlloc(V.byteLength), dO = cuAlloc(O.byteLength);
    cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dV, V.buffer);
    cuLaunch(k, [Math.ceil(M / BM), Hq, 1], [128, 1, 1], [dQ, dK, dV, dO]);
    cuSync(); cuDtoH(O.buffer, dO);

    const ref = refAttn(Qf, Kf, Vf, Hq, Hkv, M, N, D);
    let maxErr = 0, ok = true;
    for (let i = 0; i < Hq * M * D && ok; i++) {
      const d = Math.abs(O[i] - ref[i]);
      if (d > maxErr) maxErr = d;
      if (d > 0.1) { ok = false; console.log(`mismatch [${i}]: got ${O[i].toFixed(4)} ref ${ref[i].toFixed(4)}`); break; }
    }
    console.log(ok ? `✓ fa2_vllm correct (max err ${maxErr.toFixed(4)})` : "✗ FAILED");
    cuFree(dQ); cuFree(dK); cuFree(dV); cuFree(dO);
  }

  // benchmark
  {
    const Hq = 8, Hkv = 2, M = 2048, N = 2048, D = 128, BM = Number(process.env.BM||32), BN = Number(process.env.BN||32);
    const NW = Number(process.env.WARPS||4), NS = Number(process.env.STAGES||4);
    const ttir = buildFA2vllm(Hq, Hkv, M, N, D, BM, BN, NW, NS);
    const k = compileAndLoad(ttir, "fa2_vllm", NW);
    const Q = f32to16(new Float32Array(Hq * M * D).map(() => (Math.random() * 2 - 1) * 0.5));
    const K = f32to16(new Float32Array(Hkv * N * D).map(() => (Math.random() * 2 - 1) * 0.5));
    const V = f32to16(new Float32Array(Hkv * N * D).map(() => (Math.random() * 2 - 1) * 0.5));
    const O = new Float32Array(Hq * M * D);
    const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dV = cuAlloc(V.byteLength), dO = cuAlloc(O.byteLength);
    cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dV, V.buffer);
    for (let i = 0; i < 3; i++) cuLaunch(k, [Math.ceil(M / BM), Hq, 1], [128, 1, 1], [dQ, dK, dV, dO]);
    cuSync();
    const t0 = performance.now(); const it = 50;
    for (let i = 0; i < it; i++) cuLaunch(k, [Math.ceil(M / BM), Hq, 1], [128, 1, 1], [dQ, dK, dV, dO]);
    cuSync();
    const dt = (performance.now() - t0) / 1000 / it;
    const flops = 2 * Hq * M * N * D * 2;                    // 2·(QK + PV) per q-head
    console.log(`bench ${Hq}h(${Hkv}kv) ${M}x${N}x${D}: ${(dt * 1e3).toFixed(3)} ms/it, ${(flops / dt / 1e12).toFixed(2)} TFLOPS`);
    cuFree(dQ); cuFree(dK); cuFree(dV); cuFree(dO);
  }
}
if (import.meta.main) main();
