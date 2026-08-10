// Time each GEMM family used per GDN layer at M=1 to measure achieved BW.
// traffic = B bytes read (M=1, so A negligible).
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuSync, cuLaunch } from "../src/ttir";
import { performance } from "perf_hooks";

const H = 1024, VOCAB = 248320, NL = 24, FAI = 4, INTER = 3584, QKVD = 6144, ZD = 2048, LVH = 16, NH = 8, HD = 256, KV_DIM = 512, QGATE = 4096;

function buildMM(M, N, K, opts) {
  const BM = Math.min(64, M), BN = Math.min(64, N), BK = Math.min(64, K);
  const b = new TTIRBuilder();
  const outElem = opts?.cast ? "bf16" : "f32";
  const A = b.param("A", { ptr: "bf16" }), B = b.param("B", { ptr: "bf16" });
  const C = b.param("C", { ptr: outElem });
  const params = [A, B, C];
  if (opts?.add) { const R = b.param("R", { ptr: "bf16" }); params.push(R); }
  if (opts?.N2) { const B2 = b.param("B2", { ptr: "bf16" }), C2 = b.param("C2", { ptr: outElem }); params.push(B2, C2); }
  const pM = b.programId(0), pN = b.programId(1);
  const tpA = b.makeTensorPtr(A, [M, K], [K, 1], [b.mul(pM, b.i32(BM)), b.i32(0)], [BM, BK], "bf16", [1, 0]);
  const tpB = b.makeTensorPtr(B, [K, N], [1, K], [b.i32(0), b.mul(pN, b.i32(BN))], [BK, BN], "bf16", [0, 1]);
  const tpC = b.makeTensorPtr(C, [M, N], [N, 1], [b.mul(pM, b.i32(BM)), b.mul(pN, b.i32(BN))], [BM, BN], outElem, [1, 0]);
  let tpR = null;
  if (opts?.add) { const R = params[3]; tpR = b.makeTensorPtr(R, [M, N], [N, 1], [b.mul(pM, b.i32(BM)), b.mul(pN, b.i32(BN))], [BM, BN], "bf16", [1, 0]); }
  let tpB2 = null, tpC2 = null, BN2 = 0;
  if (opts?.N2) {
    BN2 = Math.min(64, opts.N2); const B2 = params[opts?.add ? 4 : 3], C2 = params[opts?.add ? 5 : 4];
    tpB2 = b.makeTensorPtr(B2, [K, opts.N2], [1, K], [b.i32(0), b.mul(pN, b.i32(BN2))], [BK, BN2], "bf16", [0, 1]);
    tpC2 = b.makeTensorPtr(C2, [M, opts.N2], [opts.N2, 1], [b.mul(pM, b.i32(BM)), b.mul(pN, b.i32(BN2))], [BM, BN2], outElem, [1, 0]);
  }
  const a0 = b.zeros([BM, BN], "f32");
  const [acc] = b.forIter(b.index(0), b.index(K), b.index(BK), [a0, tpA, tpB], (bb, _, [a, tA, tB]) => {
    const n = bb.dot(bb.load(tA), bb.load(tB), a);
    return [n, bb.advance(tA, [bb.i32(0), bb.i32(BK)]), bb.advance(tB, [bb.i32(BK), bb.i32(0)])];
  });
  let result = acc;
  if (opts?.add) { const res = b.fpext(b.load(tpR, { boundaryCheck: [0, 1], padding: 1 }), "f32"); result = b.add(result, res); }
  const storeVal = opts?.cast ? b.fptrunc(result, "bf16") : result;
  b.store(tpC, storeVal, { boundaryCheck: [0, 1] });
  if (opts?.N2) {
    const a02 = b.zeros([BM, BN2], "f32");
    const [acc2] = b.forIter(b.index(0), b.index(K), b.index(BK), [a02, tpA, tpB2], (bb, _, [a2, tA2, tB2]) => {
      const n = bb.dot(bb.load(tA2), bb.load(tB2), a2);
      return [n, bb.advance(tA2, [bb.i32(0), bb.i32(BK)]), bb.advance(tB2, [bb.i32(BK), bb.i32(0)])];
    });
    let result2 = acc2;
    const sv2 = opts?.cast ? b.fptrunc(result2, "bf16") : result2;
    b.store(tpC2, sv2, { boundaryCheck: [0, 1] });
  }
  const numParams = 3 + (opts?.add ? 1 : 0) + (opts?.N2 ? 2 : 0);
  return b.build("mm", 4, numParams);
}

// GFLOPS for M=1: K*N*2 (MAC = 2 flops), ignore tiny A
function gflops(M, N, K) { return (M * N * K * 2) / 1e9; }

const cases = [
  { name: "qkv  [1,6144,1024]", M: 1, N: 6144, K: 1024, cast: true },
  { name: "z    [1,2048,1024]", M: 1, N: 2048, K: 1024, cast: true },
  { name: "out  [1,1024,2048]", M: 1, N: 1024, K: 2048, cast: true, add: true },
  { name: "a+b  [1,32,1024]", M: 1, N: 32, K: 1024, N2: 16 },
  { name: "gate [1,3584,1024]", M: 1, N: 3584, K: 1024, N2: 3584 },
  { name: "down [1,1024,3584]", M: 1, N: 1024, K: 3584, cast: true, add: true },
  { name: "qpr  [1,4096,1024]", M: 1, N: 4096, K: 1024, cast: true },
  { name: "kvpr [1,512,1024]", M: 1, N: 512, K: 1024, cast: true },
  { name: "oproj[1,1024,2048]", M: 1, N: 1024, K: 2048, cast: true, add: true },
  { name: "lm   [1,248320,1024]", M: 1, N: 248320, K: 1024 },
];
// Allocate an A (H*2) and a big B pool; B is just read, any contents.
const A = cuAlloc(BigInt(H * 2));
const aBig = new Uint16Array(H); for (let i = 0; i < H; i++) aBig[i] = 0x3f80;
cuHtoD(A, aBig.buffer); cuSync();
const maxB = 1024 * 248320 * 2;
const Bpool = cuAlloc(BigInt(maxB));
const Cpool = cuAlloc(BigInt(248320 * 4));
const Rpool = cuAlloc(BigInt(1024 * 2));

for (const c of cases) {
  const k = compileAndLoad(buildMM(c.M, c.N, c.K, c), "mm", 4);
  const gx = [1, Math.ceil(c.N / 64), 1];
  // args: A, B, C[, R][, B2, C2]
  let args = [A, Bpool, Cpool];
  if (c.add) args.push(Rpool);
  if (c.N2) args = [A, Bpool, Cpool, ...(c.add ? [Rpool] : []), Bpool, Cpool];
  for (let i = 0; i < 5; i++) { cuLaunch(k, gx, [128, 1, 1], args); cuSync(); }
  const runs = 200;
  let best = Infinity;
  for (let r = 0; r < runs; r++) {
    const t0 = performance.now();
    cuLaunch(k, gx, [128, 1, 1], args); cuSync();
    const dt = performance.now() - t0;
    if (dt < best) best = dt;
  }
  const bytes = c.N * c.K * 2; // B read once (M=1)
  const gbs = bytes / 1e6 / (best / 1e3);
  console.log(`${c.name}: best ${best.toFixed(3)} ms | ${(bytes / 1e6).toFixed(1)} MB | ${gbs.toFixed(0)} GB/s (${(gbs / 936 * 100).toFixed(0)}%) | ${gflops(c.M, c.N, c.K)} GFLOP → ${(gflops(c.M, c.N, c.K) / (best / 1e3) / 1e3).toFixed(0)} TFLOP/s`);
}
