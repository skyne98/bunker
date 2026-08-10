// prototypes/test_gemm.ts — GEMM bit-accuracy: graph emitGemm vs decode buildMM.
import { D } from "../src/model";
import { Graph } from "../src/fusion";
import { compilePartition, loadPartition } from "../src/fusion";
import { compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch } from "../src/ttir";
import { TTIRBuilder } from "../src/ttir";

const H = D.H, ZD = D.ZD, VOCAB = D.VOCAB;
const bf16f = (u16: number) => { const b = new ArrayBuffer(4); const u = new Uint8Array(b); const f = new Float32Array(b); u[2] = u16 & 0xFF; u[3] = u16 >> 8; return f[0]; };
const f2bf = (f: number) => { const b = new ArrayBuffer(4); const u = new Uint8Array(b); const f32 = new Float32Array(b); f32[0] = f; return u[2] | (u[3] << 8); };

// decode buildMM (verbatim from decode.ts)
function buildMM(M: number, N: number, K: number, opts?: { cast?: boolean }) {
  const BM = Math.min(64, M), BN = Math.min(64, N), BK = Math.min(64, K);
  const b = new TTIRBuilder();
  const outElem = opts?.cast ? "bf16" : "f32";
  const A = b.param("A", { ptr: "bf16" }), B = b.param("B", { ptr: "bf16" });
  const C = b.param("C", { ptr: outElem });
  const pM = b.programId(0), pN = b.programId(1);
  const tpA = b.makeTensorPtr(A, [M, K], [K, 1], [b.mul(pM, b.i32(BM)), b.i32(0)], [BM, BK], "bf16", [1, 0]);
  const tpB = b.makeTensorPtr(B, [K, N], [1, K], [b.i32(0), b.mul(pN, b.i32(BN))], [BK, BN], "bf16", [0, 1]);
  const tpC = b.makeTensorPtr(C, [M, N], [N, 1], [b.mul(pM, b.i32(BM)), b.mul(pN, b.i32(BN))], [BM, BN], outElem, [1, 0]);
  const a0 = b.zeros([BM, BN], "f32");
  const [acc] = b.forIter(b.index(0), b.index(K), b.index(BK), [a0, tpA, tpB], (bb, _, [a, tA, tB]) => {
    const n = bb.dot(bb.load(tA), bb.load(tB), a);
    return [n, bb.advance(tA, [bb.i32(0), bb.i32(BK)]), bb.advance(tB, [bb.i32(BK), bb.i32(0)])];
  });
  const storeVal = opts?.cast ? b.fptrunc(acc, "bf16") : acc;
  b.store(tpC, storeVal, { boundaryCheck: [0, 1] });
  return b.build("mm", 4, 3);
}

function graphMM(M: number, N: number, K: number): { ttir: string; name: string; args: string[]; grid: [number, number, number] } {
  const g = new Graph();
  g.input("A", { shape: [M, K], dtype: "bf16", strides: [K, 1] }, "data");
  g.input("B", { shape: [K, N], dtype: "bf16", strides: [1, K] }, "weight");
  g.input("C", { shape: [M, N], dtype: "f32", strides: [N, 1] }, "data");
  const n = g.node("gemm",
    [{ tensor: g.t("A"), name: "A" }, { tensor: g.t("B"), name: "B" }],
    [{ name: "out", type: { shape: [M, N], dtype: "f32", strides: [N, 1] } }],
    [1, Math.ceil(N / 64), 1], { M, N, K });
  const plans = compilePartition(g, [[n]]);
  return plans[0];
}

async function main() {
  const stPath = process.env.SAFETENSORS_PATH || "/tmp/qwen35_0.8b.safetensors";
  const data = await Bun.file(stPath).bytes();
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const hl = Number(dv.getBigUint64(0, true));
  const hdr = JSON.parse(new TextDecoder().decode(data.subarray(8, 8 + hl)));
  const ds = 8 + hl;
  const base = cuAlloc(BigInt(data.length - ds));
  cuHtoD(base, data.subarray(ds)); cuSync();
  const W = new Map<string, bigint>();
  for (const [k, v] of Object.entries(hdr)) if (k !== "__metadata__") W.set(k, base + BigInt((v as any).data_offsets[0]));

  // Test out_proj GEMM [1, H=1024, K=ZD=2048]: A = gdn output (bf16), B = out_proj weight
  const M = 1, N = H, K = ZD;
  const wOP = W.get("model.language_model.layers.0.linear_attn.out_proj.weight")!;

  // A: varied small bf16
  const aHost = new Uint16Array(K);
  for (let i = 0; i < K; i++) aHost[i] = f2bf(Math.sin(i * 0.1) * 0.1);
  const aBuf = cuAlloc(BigInt(K * 2));
  cuHtoD(aBuf, aHost.buffer); cuSync();

  const kRef = compileAndLoad(buildMM(M, N, K), "mm", 4);
  const gplan = graphMM(M, N, K);
  const gloaded = loadPartition([gplan]);

  const yRef = cuAlloc(BigInt(N * 4));
  const yG = cuAlloc(BigInt(N * 4));
  cuLaunch(kRef, [1, Math.ceil(N / 64), 1], [128, 1, 1], [aBuf, wOP, yRef]);
  cuSync();
  cuLaunch(gloaded[0].k, [gplan.grid[0], gplan.grid[1], gplan.grid[2]], [128, 1, 1], [aBuf, wOP, yG]);
  cuSync();

  const fullR = new Float32Array(N), fullG = new Float32Array(N);
  cuDtoH(fullR.buffer, yRef, BigInt(N * 4));
  cuDtoH(fullG.buffer, yG, BigInt(N * 4));
  let bad = 0, maxD = 0;
  for (let i = 0; i < N; i++) { const d = Math.abs(fullR[i] - fullG[i]); if (d > 1e-6) { bad++; if (d > maxD) maxD = d; } }
  console.log(`out_proj GEMM [1,${N},${K}]: ${bad}/${N} mismatches (max diff ${maxD.toExponential(2)})`);
  console.log(`  ref[0..7]=${[...fullR.slice(0, 8)].map(v => v.toFixed(4)).join(",")}`);
  console.log(`  gph[0..7]=${[...fullG.slice(0, 8)].map(v => v.toFixed(4)).join(",")}`);
}

await main();
