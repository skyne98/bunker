// bench/autotune_decode.ts — automatically discover a decode GEMM's tile
//
// End-to-end exercise of the measured search: build ONE layer-0 GEMM as a
// graph, let `explore()` sweep the (BN, BK) tile space on the real GPU, keep
// the fastest, and persist it via src/policy.ts. No hand-picked tiles here —
// the number printed as "discovered" is whatever the GPU measured fastest.
//
// Run: SAFETENSORS_PATH=/tmp/qwen35_0.8b.safetensors bun run bench/autotune_decode.ts
import { Graph, explore, compilePartition, measurePartition, DEFAULT_GEMM_TILE } from "../src/fusion";
import { policyFromBest, bestFromPolicy, savePolicy, loadPolicy } from "../src/policy";
import { cuAlloc, cuHtoD, cuSync, cuFree } from "../src/ttir";

const H = 1024, QKVD = 6144;

function W(shape: number[]): { shape: number[]; dtype: "bf16"; strides: number[] } {
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  return { shape, dtype: "bf16", strides };
}

const stPath = process.env.SAFETENSORS_PATH || "/tmp/qwen35_0.8b.safetensors";
const data = await Bun.file(stPath).bytes();
const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
const hl = Number(dv.getBigUint64(0, true));
const hdr = JSON.parse(new TextDecoder().decode(data.subarray(8, 8 + hl)));
const ds = 8 + hl;
const base = cuAlloc(BigInt(data.length - ds));
cuHtoD(base, data.subarray(ds)); cuSync();
const wName = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
const wPtr = base + BigInt((hdr[wName] as any).data_offsets[0]);

// ── one GEMM: in[1x1024] @ w[1024x6144] -> out[1x6144] ──
const g = new Graph();
const IN = g.input("in", W([1, H]));
const gemm = g.node("gemm",
  [{ tensor: IN, name: "A" }, { tensor: g.input("w", W([QKVD, H]), "weight"), name: "B" }],
  [{ name: "out", type: W([1, QKVD]) }], [1, Math.ceil(QKVD / 64), 1], { M: 1, N: QKVD, K: H });

const dA = cuAlloc(BigInt(2 * H)); cuHtoD(dA, Buffer.alloc(2 * H)); cuSync();
const dO = cuAlloc(BigInt(2 * QKVD));
const outName = g.out(gemm);

const env = { resolve: (name: string): bigint => {
  if (name === "in") return dA;
  if (name === "w") return wPtr;
  if (name === outName) return dO;
  throw new Error(`unresolved tensor ${name}`);
} };

console.log(`searching tile space for layer-0 qkv GEMM (N=${QKVD}, K=${H})...`);
const { best } = explore(g, env, { moves: ["tile"], beam: 2, budget: 20, verbose: true });
const d = best.tiles[gemm.id] ?? DEFAULT_GEMM_TILE;
console.log(`\nDISCOVERED tile: BN=${d.BN} x BK=${d.BK}  latency=${best.latencyMs?.toFixed(3)}ms`);

// ── persist + replay-check ──
const path = "policy_decode_qkv_layer0.json";
savePolicy(path, policyFromBest(g, "decode-qkv-layer0", best));
const back = bestFromPolicy(g, loadPolicy(path)!);
if (!back) throw new Error("policy did not round-trip");
const plans = compilePartition(g, back.partition, back.tiles);
const replayMs = measurePartition(plans, env, 5);
console.log(`replay from policy: ${replayMs.toFixed(3)}ms (saved ${best.latencyMs?.toFixed(3)}ms)`);
cuFree(dA); cuFree(dO);
