// prototypes/fa2_autotune.ts
//
// Autotune the FA2 prefill kernel (`buildFA2vllm`) over (BM, BN, numWarps,
// numStages) using CUDA-event timing — the same pattern as bench/autotune.ts,
// applied to the FA2 port. The framework picks the best config instead of a
// hand-picked one, and can re-tune per problem size / GPU.
//
//   bun run prototypes/fa2_autotune.ts
//   # optional: Hq=8 Hkv=2 SEQ=2048 D=128
import { compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";
import { getCuCtx, createCudaEvents, gpuTimeUs } from "../src/kernel";
import { f32to16 } from "../src/kernel";
import { buildFA2vllm, refAttn } from "./fa2_vllm";

const Hq = Number(process.env.Hq || 8), Hkv = Number(process.env.Hkv || 2);
const M = Number(process.env.SEQ || 2048), N = M, D = Number(process.env.D || 128);
const G = Hq / Hkv;
const flops = 4 * Hq * M * N * D; // 2·(QK+PV) per q-head

// search space (configs that don't compile or are invalid are skipped)
const SPACE: { BM: number; BN: number; w: number; s: number }[] = [];
for (const BM of [32, 64, 128]) for (const BN of [32, 64, 128])
  for (const w of [4, 8]) for (const s of [2, 3, 4])
    SPACE.push({ BM, BN, w, s });

// inputs (shared across configs)
const Q = f32to16(new Float32Array(Hq * M * D).map(() => (Math.random() * 2 - 1) * 0.5));
const K = f32to16(new Float32Array(Hkv * N * D).map(() => (Math.random() * 2 - 1) * 0.5));
const V = f32to16(new Float32Array(Hkv * N * D).map(() => (Math.random() * 2 - 1) * 0.5));
const O = new Float32Array(Hq * M * D);
const dQ = cuAlloc(Q.byteLength), dK = cuAlloc(K.byteLength), dV = cuAlloc(V.byteLength), dO = cuAlloc(O.byteLength);
cuHtoD(dQ, Q.buffer); cuHtoD(dK, K.buffer); cuHtoD(dV, V.buffer);

const cs = getCuCtx();
const ev = createCudaEvents();

let best: { cfg: typeof SPACE[0]; tfs: number } | null = null;
console.log(`autotuning fa2_vllm  ${Hq}h/${Hkv}kv G=${G}  ${M}x${N}x${D}  (${SPACE.length} configs)\n`);

// shared small-correctness inputs (M=256, D=64) — every config must pass these
const CM = 256, CD = 64;
const cQ = f32to16(new Float32Array(Hq * CM * CD).map(() => (Math.random() * 2 - 1) * 0.5));
const cK = f32to16(new Float32Array(Hkv * CM * CD).map(() => (Math.random() * 2 - 1) * 0.5));
const cV = f32to16(new Float32Array(Hkv * CM * CD).map(() => (Math.random() * 2 - 1) * 0.5));
const cQf = Float32Array.from(cQ, (_, i) => cQ[i]), cKf = Float32Array.from(cK, (_, i) => cK[i]), cVf = Float32Array.from(cV, (_, i) => cV[i]);
// f16->f32 decode for the reference: re-expand f16 bits properly
const cQf32 = f16to32(cQ), cKf32 = f16to32(cK), cVf32 = f16to32(cV);
const cO = new Float32Array(Hq * CM * CD);
const dcQ = cuAlloc(cQ.byteLength), dcK = cuAlloc(cK.byteLength), dcV = cuAlloc(cV.byteLength), dcO = cuAlloc(cO.byteLength);
cuHtoD(dcQ, cQ.buffer); cuHtoD(dcK, cK.buffer); cuHtoD(dcV, cV.buffer);

function f16to32(a: Uint16Array): Float32Array {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const u = a[i], sign = (u >> 15) & 1, exp = (u >> 10) & 0x1f, frac = u & 0x3ff;
    let f: number;
    if (exp === 0) f = 0;                              // denorm -> 0 (matches f32to16)
    else if (exp === 31) f = frac ? NaN : Infinity;
    else f = Math.pow(2, exp - 15) * (1 + frac / 1024);
    o[i] = sign ? -f : f;
  }
  return o;
}
void cQf; void cKf; void cVf;

function correct(cfg: { BM: number; BN: number; w: number; s: number }): boolean {
  const bm = Math.min(cfg.BM, CM), bn = Math.min(cfg.BN, 1024);
  let k; try { k = compileAndLoad(buildFA2vllm(Hq, Hkv, CM, CM, CD, bm, bn, cfg.w, cfg.s), "fa2_vllm", cfg.w); } catch { return false; }
  cuLaunch(k, [Math.ceil(CM / bm), Hq, 1], [cfg.w * 32, 1, 1], [dcQ, dcK, dcV, dcO]);
  cuSync(); cuDtoH(cO.buffer, dcO);
  const ref = refAttn(cQf32, cKf32, cVf32, Hq, Hkv, CM, CM, CD);
  let mx = 0; for (let i = 0; i < cO.length; i++) { const d = Math.abs(cO[i] - ref[i]); if (d > mx) mx = d; if (Number.isNaN(d)) return false; }
  return mx < 0.1;
}

for (const cfg of SPACE) {
  if (!correct(cfg)) { console.log(`  BM=${String(cfg.BM).padStart(3)} BN=${String(cfg.BN).padStart(3)} w=${cfg.w} s=${cfg.s}  incorrect ✗ (skip)`); continue; }
  let k;
  try { k = compileAndLoad(buildFA2vllm(Hq, Hkv, M, N, D, cfg.BM, cfg.BN, cfg.w, cfg.s), "fa2_vllm", cfg.w); }
  catch (e) { continue; }
  const grid = [Math.ceil(M / cfg.BM), Hq, 1];
  const block = [cfg.w * 32, 1, 1];
  const args = [dQ, dK, dV, dO];
  let rc0 = -1;
  for (let i = 0; i < 3; i++) rc0 = cuLaunch(k, grid, block, args);
  cuSync();
  if (rc0 !== 0) { console.log(`  BM=${cfg.BM} BN=${cfg.BN} w=${cfg.w} s=${cfg.s}  launch rc=${rc0} (skip)`); continue; }
  const ITERS = 20;
  cs.cuEventRecord(Number(ev.start.readBigUInt64LE(0)), 0);
  for (let i = 0; i < ITERS; i++) cuLaunch(k, grid, block, args);
  cs.cuEventRecord(Number(ev.stop.readBigUInt64LE(0)), 0);
  cs.cuEventSynchronize(Number(ev.stop.readBigUInt64LE(0)));
  const us = gpuTimeUs(cs, ev) / ITERS;
  const tfs = flops / (us / 1e6) / 1e12;
  console.log(`  BM=${String(cfg.BM).padStart(3)} BN=${String(cfg.BN).padStart(3)} w=${cfg.w} s=${cfg.s}  ${us.toFixed(1).padStart(7)} µs  ${tfs.toFixed(2).padStart(6)} TFLOPS  ✓`);
  if (!best || tfs > best.tfs) best = { cfg, tfs };
}

cuFree(dQ); cuFree(dK); cuFree(dV); cuFree(dO);
if (best) console.log(`\nBEST: BM=${best.cfg.BM} BN=${best.cfg.BN} w=${best.cfg.w} s=${best.cfg.s}  →  ${best.tfs.toFixed(2)} TFLOPS`);

// ── correctness-check the winning config against the host reference ─────
if (best) {
  const m = 256, d = 64, bm = Math.min(best.cfg.BM, m), bn = Math.min(best.cfg.BN, d * 4);
  const Qf = new Float32Array(Hq * m * d).map(() => (Math.random() * 2 - 1) * 0.5);
  const Kf = new Float32Array(Hkv * m * d).map(() => (Math.random() * 2 - 1) * 0.5);
  const Vf = new Float32Array(Hkv * m * d).map(() => (Math.random() * 2 - 1) * 0.5);
  const Q = f32to16(Qf), K = f32to16(Kf), V = f32to16(Vf), O = new Float32Array(Hq * m * d);
  const dQ2 = cuAlloc(Q.byteLength), dK2 = cuAlloc(K.byteLength), dV2 = cuAlloc(V.byteLength), dO2 = cuAlloc(O.byteLength);
  cuHtoD(dQ2, Q.buffer); cuHtoD(dK2, K.buffer); cuHtoD(dV2, V.buffer);
  const k = compileAndLoad(buildFA2vllm(Hq, Hkv, m, m, d, bm, bn, best.cfg.w, best.cfg.s), "fa2_vllm", best.cfg.w);
  cuLaunch(k, [Math.ceil(m / bm), Hq, 1], [best.cfg.w * 32, 1, 1], [dQ2, dK2, dV2, dO2]);
  cuSync(); cuDtoH(O.buffer, dO2);
  const ref = refAttn(Qf, Kf, Vf, Hq, Hkv, m, m, d);
  let maxErr = 0;
  for (let i = 0; i < O.length; i++) maxErr = Math.max(maxErr, Math.abs(O[i] - ref[i]));
  console.log(`verify best config: max err ${maxErr.toFixed(4)}  ${maxErr < 0.1 ? "✓ correct" : "✗ WRONG"}`);
  cuFree(dQ2); cuFree(dK2); cuFree(dV2); cuFree(dO2);
}
