// tests/test_fusion_search.ts — pure (no GPU) tests for the tile search + policy
//
// Covers the automatic-discovery plumbing added to src/fusion.ts and
// src/policy.ts: tile move generation, candidate identity keys, grid
// derivation for single-GEMM kernels, and policy persistence round-trip.
// No GPU, no measurement — these assert structure and determinism.
import {
  Graph, TileConfig, TileAssign, GEMM_TILE_SEARCH, DEFAULT_GEMM_TILE,
  tileDivides, resolveGemmTile, tileMoves, candKey, codegenGroup,
} from "../src/fusion";
import {
  Policy, graphSignature, policyFromBest, bestFromPolicy, savePolicy, loadPolicy,
} from "../src/policy";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";

function w(shape: number[]): { shape: number[]; dtype: "bf16"; strides: number[] } {
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  return { shape, dtype: "bf16", strides };
}

/** Two GEMMs over a shared row: (1,2048)x(2048,1024) and (1,4096)x(4096,1024). */
function makeGraph(): Graph {
  const g = new Graph();
  const IN = g.input("in", w([1, 1024]));
  g.node("gemm",
    [{ tensor: IN, name: "A" }, { tensor: g.input("w0", w([2048, 1024]), "weight"), name: "B" }],
    [{ name: "out", type: w([1, 2048]) }], [1, 32, 1], { M: 1, N: 2048, K: 1024 });
  g.node("gemm",
    [{ tensor: IN, name: "A" }, { tensor: g.input("w1", w([4096, 1024]), "weight"), name: "B" }],
    [{ name: "out", type: w([1, 4096]) }], [1, 64, 1], { M: 1, N: 4096, K: 1024 });
  return g;
}

let failures = 0;
const check = (name: string, cond: boolean, detail = "") => {
  if (cond) console.log(`  ✓ ${name}`);
  else { failures++; console.log(`  ✗ ${name} ${detail}`); }
};

const g = makeGraph();
const [g0, g1] = g.nodes;
const part: any[][] = [[g0], [g1]];
const partFn = (): any[][] => part.map(grp => [...grp]);

// ── 1. tileMoves: one node, one config, always dividing, never a no-op ──
console.log("tileMoves");
const moves = tileMoves(partFn() as any, {});
const legal = moves.every(m => {
  const changed = Object.keys(m.tiles);
  const cfg = m.tiles[Number(changed[0])];
  const nid = Number(changed[0]);
  const n = nid === g0.id ? g0 : g1;
  return changed.length === 1 &&
    cfg !== undefined && tileDivides(cfg, n.params.N, n.params.K) &&
    !(cfg.BN === DEFAULT_GEMM_TILE.BN && cfg.BK === DEFAULT_GEMM_TILE.BK);
});
check("all moves change exactly one GEMM to a dividing, non-default tile", legal);
// 2048: all 8 search configs divide (BN 16/32/64, BK 64/128/256); the
// default 64x64 is excluded -> 7 per GEMM, 14 total.
check(`move count = 14 (7 per GEMM)`, moves.length === 14, `got ${moves.length}`);

// ── 2. tileDivides / resolveGemmTile clamping ──
console.log("resolveGemmTile");
check("16x256 divides (2048,1024)", tileDivides({ BN: 16, BK: 256 }, 2048, 1024));
check("32x256 divides (4096,1024)", tileDivides({ BN: 32, BK: 256 }, 4096, 1024));
const clamped = resolveGemmTile({ BN: 16, BK: 300 }, 2048, 1024);
check("non-dividing BK clamps to 64", clamped.BK === 64 && clamped.BN === 16, JSON.stringify(clamped));
check("missing tile -> default", resolveGemmTile(undefined, 2048, 1024).BN === 64);

// ── 3. candKey: tiles are part of candidate identity ──
console.log("candKey");
const a: any = { partition: partFn(), tiles: { [g0.id]: { BN: 16, BK: 256 } } };
const b: any = { partition: partFn(), tiles: {} };
check("same partition, different tiles -> different keys", candKey(a) !== candKey(b));
check("same tiles + partition -> same key", candKey(a) === candKey({ ...a }));
check("deterministic ordering", candKey(a) === candKey({ partition: partFn(), tiles: { [g0.id]: { BN: 16, BK: 256 } } }));

// ── 4. codegenGroup derives the launch grid from the searched tile ──
console.log("codegenGroup grid");
const plan16 = codegenGroup(g, [g0], 0, { [g0.id]: { BN: 16, BK: 256 } });
check(`grid.y = 2048/16 = 128`, plan16.grid[1] === 128, JSON.stringify(plan16.grid));
check(`grid.x = 1`, plan16.grid[0] === 1);
const planDefault = codegenGroup(g, [g0], 1, {});
check("no tile -> grid.y = 2048/64 = 32", planDefault.grid[1] === 32, JSON.stringify(planDefault.grid));
check("TTIR mentions a 16x256 dot tile", plan16.ttir.includes("16x256") || plan16.ttir.includes("256x16"));

// ── 5. policy round-trip + stale-graph guard ──
console.log("policy round-trip");
const best: any = { partition: partFn(), tiles: { [g0.id]: { BN: 16, BK: 256 }, [g1.id]: { BN: 32, BK: 128 } }, latencyMs: 1.234 };
const pol = policyFromBest(g, "test", best);
const path = join(tmpdir(), `policy_${process.pid}.json`);
savePolicy(path, pol);
const loaded = loadPolicy(path)!;
const back = bestFromPolicy(g, loaded)!;
try { unlinkSync(path); } catch {}
check("signature covers nodes+ops", pol.graphSignature.startsWith("n2:"), pol.graphSignature);
check("round-trip recovers partition", JSON.stringify(back.partition.map(gr => gr.map(n => n.id))) === JSON.stringify(pol.partition));
check("round-trip recovers tiles", back.tiles[g0.id]!.BN === 16 && back.tiles[g1.id]!.BK === 128);
check("round-trip recovers measured latency", back.latencyMs === 1.234);
const g2 = makeGraph();
g2.node("add", [{ tensor: g2.t("n0_out" as any), name: "a" }, { tensor: g2.t("n1_out" as any), name: "b" }],
  [{ name: "out", type: w([1, 1024]) }], [1, 1, 1], { N: 1024 });
if (graphSignature(g2) !== pol.graphSignature) {
  const stale = bestFromPolicy(g2, loaded);
  check("stale graph -> null (refuses to replay)", stale === null);
}

console.log(failures === 0 ? `\nALL PASS` : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
