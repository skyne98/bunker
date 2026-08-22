// bench/autotune_decode.ts — automatic, fast, progress-barred tile discovery
//
// Discovers the best (BN, BK) tile for every distinct layer-0 GEMM shape by
// measured search (`explore`, moves=["tile"]), reusing one per-TTIR compile
// cache so repeated kernels compile once. Prints a live progress bar.
//
// Speed contract: this must finish well under 1 minute on the 3090 (each
// distinct tile is compiled once; the rest is sub-ms measurement). The run
// refuses to call itself successful if it exceeds the budget.
//
// Run: SAFETENSORS_PATH=/tmp/qwen35_0.8b.safetensors bun run bench/autotune_decode.ts
import { Graph, explore, compilePartition, measurePartition, TileConfig, KernelPlan } from "../src/fusion";
import { clearKernelCache } from "../src/fusion";
import { policyFromBest, savePolicy } from "../src/policy";
import { cuAlloc, cuHtoD, cuSync, cuFree } from "../src/ttir";
import { performance } from "perf_hooks";

const H = 1024, ZD = 2048, QKVD = 6144, INTER = 3584;
const LIMIT_MS = 60_000; // hard ceiling: the discovery phase must stay fast

function W(shape: number[]): { shape: number[]; dtype: "bf16"; strides: number[] } {
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  return { shape, dtype: "bf16", strides };
}

interface Shape { name: string; M: number; N: number; K: number; w: string; }

const SHAPES: Shape[] = [
  { name: "qkv",   M: 1, N: QKVD,   K: H,    w: "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" },
  { name: "z",     M: 1, N: ZD,     K: H,    w: "model.language_model.layers.0.linear_attn.in_proj_z.weight" },
  { name: "out_p", M: 1, N: H,      K: ZD,   w: "model.language_model.layers.0.linear_attn.out_proj.weight" },
  { name: "gate",  M: 1, N: INTER,  K: H,    w: "model.language_model.layers.0.mlp.gate_proj.weight" },
  { name: "down",  M: 1, N: H,      K: INTER,w: "model.language_model.layers.0.mlp.down_proj.weight" },
];

const stPath = process.env.SAFETENSORS_PATH || "/tmp/qwen35_0.8b.safetensors";
const data = await Bun.file(stPath).bytes();
const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
const hl = Number(dv.getBigUint64(0, true));
const hdr = JSON.parse(new TextDecoder().decode(data.subarray(8, 8 + hl)));
const ds = 8 + hl;
const base = cuAlloc(BigInt(data.length - ds));
cuHtoD(base, data.subarray(ds)); cuSync();
const wPtr = (name: string): bigint => base + BigInt((hdr[name] as any).data_offsets[0]);

const t0 = performance.now();
const results: { name: string; tile: TileConfig; ms: number }[] = [];

for (const s of SHAPES) {
  const g = new Graph();
  const IN = g.input("in", W([1, s.K]));
  const gemm = g.node("gemm",
    [{ tensor: IN, name: "A" }, { tensor: g.input("w", W([s.N, s.K]), "weight"), name: "B" }],
    [{ name: "out", type: W([1, s.N]) }], [1, Math.ceil(s.N / 64), 1], { M: s.M, N: s.N, K: s.K });
  const dA = cuAlloc(BigInt(2 * s.K)); cuHtoD(dA, Buffer.alloc(2 * s.K)); cuSync();
  const dO = cuAlloc(BigInt(2 * s.N));
  const outName = g.out(gemm);
  const env = { resolve: (name: string): bigint => {
    if (name === "in") return dA; if (name === "w") return wPtr(s.w);
    if (name === outName) return dO; throw new Error(`unresolved ${name}`);
  } };
  // Discovery for one shape. The compile cache makes repeated tile configs free.
  const { best } = explore(g, env, { moves: ["tile"], beam: 2, budget: 12, progress: true });
  const tile = best.tiles[gemm.id] ?? { BN: 64, BK: 64 };
  const ms = best.latencyMs ?? 0;
  results.push({ name: s.name, tile, ms });
  console.log(`\n  ${s.name}: DISCOVERED BN=${tile.BN} x BK=${tile.BK}  ${ms.toFixed(3)}ms`);
  savePolicy(`policy_decode_${s.name}.json`, policyFromBest(g, `decode-${s.name}`, best));
  cuFree(dA); cuFree(dO);
}

const total = performance.now() - t0;
clearKernelCache();
console.log(`\nAll layer-0 GEMM tiles discovered in ${(total / 1000).toFixed(1)}s (limit ${LIMIT_MS / 1000}s)`);
if (total > LIMIT_MS) { console.error("!! discovery exceeded the 1-minute budget"); process.exit(1); }
for (const r of results) console.log(`  ${r.name.padEnd(7)} BN=${r.tile.BN} x BK=${r.tile.BK}  ${r.ms.toFixed(3)}ms`);
