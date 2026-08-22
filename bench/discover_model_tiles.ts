// bench/discover_model_tiles.ts — discover every GEMM tile for the whole model
//
// Derives the distinct GEMM shapes from `buildModelGraph()` (no hardcoded
// list), then measures the best (BN, BK) tile for each shape on the real GPU
// and persists a per-shape policy. Config latency is value-independent, so the
// measurement needs zeroed buffers only — no model weights are required.
//
// Speed contract: with the per-TTIR compile cache this must finish well under
// the 60s budget; it exits 1 otherwise. Progress bar on stderr.
//
// Run: bun run bench/discover_model_tiles.ts
import { buildModelGraph } from "../src/model";
import { Graph, explore, TileConfig, clearKernelCache } from "../src/fusion";
import { policyFromBest, savePolicy } from "../src/policy";
import { cuAlloc, cuFree } from "../src/ttir";
import { performance } from "perf_hooks";

const LIMIT_MS = 60_000;

function W(shape: number[]): { shape: number[]; dtype: "bf16"; strides: number[] } {
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  return { shape, dtype: "bf16", strides };
}

// Distinct (N, K) GEMM shapes across the model graph, in first-seen order.
const model = buildModelGraph();
const shapes: { key: string; N: number; K: number; M: number }[] = [];
const seen = new Set<string>();
for (const n of model.nodes) {
  if (n.op !== "gemm") continue;
  const key = `${n.params.N}x${n.params.K}`;
  if (seen.has(key)) continue;
  seen.add(key);
  shapes.push({ key, N: n.params.N, K: n.params.K, M: n.params.M ?? 1 });
}

console.log(`model: ${model.nodes.length} nodes, ${shapes.length} distinct GEMM shapes, budget ${LIMIT_MS / 1000}s`);
const t0 = performance.now();
const found: { key: string; tile: TileConfig; ms: number }[] = [];

for (const s of shapes) {
  // One GEMM as a graph; zeroed operands (values do not affect latency).
  const g = new Graph();
  const IN = g.input("in", W([1, s.K]));
  const gemm = g.node("gemm",
    [{ tensor: IN, name: "A" }, { tensor: g.input("w", W([s.N, s.K]), "weight"), name: "B" }],
    [{ name: "out", type: W([1, s.N]) }], [1, Math.ceil(s.N / 64), 1], { M: s.M, N: s.N, K: s.K });
  const dA = cuAlloc(BigInt(2 * s.K));
  const dW = cuAlloc(BigInt(2 * s.N * s.K));
  const dO = cuAlloc(BigInt(2 * s.N));
  const outName = g.out(gemm);
  const env = { resolve: (name: string): bigint => {
    if (name === "in") return dA;
    if (name === "w") return dW;
    if (name === outName) return dO;
    throw new Error(`unresolved ${name}`);
  } };

  const { best } = explore(g, env, { moves: ["tile"], beam: 2, budget: 12, progress: true });
  const tile = best.tiles[gemm.id] ?? { BN: 64, BK: 64 };
  const ms = best.latencyMs ?? 0;
  found.push({ key: s.key, tile, ms });
  savePolicy(`policy_shape_${s.key}.json`, policyFromBest(g, `shape-${s.key}`, best));
  cuFree(dA); cuFree(dW); cuFree(dO);
}

clearKernelCache();
const total = performance.now() - t0;
console.log(`\nAll distinct model GEMM tiles discovered in ${(total / 1000).toFixed(1)}s (limit ${LIMIT_MS / 1000}s)`);
if (total > LIMIT_MS) { console.error(`!! discovery exceeded the ${LIMIT_MS / 1000}s budget`); process.exit(1); }
for (const r of found) console.log(`  ${r.key.padEnd(14)} -> BN=${r.tile.BN} x BK=${r.tile.BK}  ${r.ms.toFixed(3)}ms`);
console.log("policies: policy_shape_<N>x<K>.json");
