// prototypes/fusion_model_full.ts — compose per-layer beam-discovered fusions
// into ONE whole-model partition and run the fused graph decoder.
//
// For each of the 24 layers, the beam explorer (src/fusion.ts explore) searches
// a layer's subgraph for the fastest kernel grouping, measured by real GPU wall
// time. The chosen groups (node ids preserved via subGraphOf) are composed with
// singleton kernels for embed/final-norm/lm_head/argmax into a single Partition
// for the FULL model graph. The runner executes it, verified against the
// reference tokens and timed in tok/s vs the unfused 177 tok/s baseline.
//
// Run: SAFETENSORS_PATH=... TOKENIZER_PATH=... bun run prototypes/fusion_model_full.ts
import { buildModelGraph, D } from "../src/model";
import { Graph, GraphNode, explore, compilePartition, measurePartition, RunEnv, Partition } from "../src/fusion";
import { compileStep, runGraphDecode, topoOrder } from "../src/runner";
import { cuAlloc, cuHtoD, cuSync } from "../src/ttir";
import { performance } from "perf_hooks";

const bufSize = (sh: number[], dtype: string) => sh.reduce((a, b) => a * b, 1) * (dtype === "bf16" ? 2 : 4);

async function main() {
  const stPath = process.env.SAFETENSORS_PATH || "/tmp/qwen35_0.8b.safetensors";
  const data = await Bun.file(stPath).bytes();
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const hl = Number(dv.getBigUint64(0, true));
  const hdr = JSON.parse(new TextDecoder().decode(data.subarray(8, 8 + hl)));
  const ds = 8 + hl;
  const base = cuAlloc(BigInt(data.length - ds));
  cuHtoD(base, data.subarray(ds)); cuSync();
  const Wnames = new Map<string, bigint>();
  for (const [k, v] of Object.entries(hdr)) if (k !== "__metadata__") Wnames.set(k, base + BigInt((v as any).data_offsets[0]));

  const g = buildModelGraph();
  console.log(`graph: ${g.nodes.length} nodes`);

  const dynAllocs = new Map<string, bigint>();
  const allocFor = (name: string): bigint => {
    if (!dynAllocs.has(name)) {
      const t = g.t(name);
      const bytes = bufSize(t.type.shape, t.type.dtype);
      const b = cuAlloc(BigInt(bytes));
      cuHtoD(b, Buffer.alloc(bytes)); cuSync();
      dynAllocs.set(name, b);
    }
    return dynAllocs.get(name)!;
  };
  const ropeFor = (name: "rope_cos" | "rope_sin"): bigint => {
    if (!dynAllocs.has(name)) {
      const b = cuAlloc(BigInt(D.MAX_LEN * D.ROT_HALF * 4));
      const arr = new Float32Array(D.MAX_LEN * D.ROT_HALF);
      for (let p = 0; p < D.MAX_LEN; p++) for (let i = 0; i < D.ROT_HALF; i++) {
        const f = 1 / Math.pow(10000000, 2 * i / D.ROT_DIM);
        arr[p * D.ROT_HALF + i] = name === "rope_cos" ? Math.cos(p * f) : Math.sin(p * f);
      }
      cuHtoD(b, arr.buffer); cuSync();
      dynAllocs.set(name, b);
    }
    return dynAllocs.get(name)!;
  };
  const resolve = (name: string): bigint => {
    if (name === "token_id" || name === "pos") return 0n; // scalars handled at launch
    if (name === "embed.weight" || name === "lm_head.weight") return Wnames.get("model.language_model.embed_tokens.weight")!;
    if (name === "final.norm.weight") return Wnames.get("model.language_model.norm.weight")!;
    if (name === "rope_cos") return ropeFor("rope_cos");
    if (name === "rope_sin") return ropeFor("rope_sin");
    if (name.startsWith("model.layers.")) {
      const exact = Wnames.get(name);
      if (exact !== undefined) return exact;
      const v1 = Wnames.get("model.language_model." + name);
      if (v1 !== undefined) return v1;
    }
    // non-weight layer tensor (A_log, dt_bias) or any scratch/state
    const v2 = Wnames.get("model.language_model." + name);
    if (v2 !== undefined) return v2;
    return allocFor(name);
  };
  const env: RunEnv = { resolve };

  // ── collect per-layer nodes ──
  const layerNodes = new Map<number, GraphNode[]>();
  for (const n of g.nodes) {
    const nm = collectLayer(n);
    if (nm !== null && nm >= 0 && nm < D.NL) {
      if (!layerNodes.has(nm)) layerNodes.set(nm, []);
      layerNodes.get(nm)!.push(n);
    }
  }
  const allIds = new Set(g.nodes.map(n => n.id));
  const byId = new Map<number, GraphNode>();
  for (const n of g.nodes) byId.set(n.id, n);

  // ── per-layer beam ──
  const composed: Partition = [];
  const nTopo = D.NL; // explore layers in order for deterministic reuse
  for (let L = 0; L < D.NL; L++) {
    const nodes = layerNodes.get(L);
    if (!nodes || nodes.length === 0) continue;
    const t0 = performance.now();
    const subG = subGraphOf(g, nodes);
    const { best } = explore(subG, env, { beam: 3, budget: 20, moves: ["merge"], verbose: false });
    if (L === 0) {
      console.log("L0 chosen groups:");
      for (const grp of best.partition) console.log(`   ${grp.map(n=>n.op + (n.params?.M ? `(${n.params.M}x${n.params.N}x${n.params.K})` : "")).join("+")}  grid=${JSON.stringify(grp[0].grid)}`);
    }
    const kernels = best.partition.map(grp => {
      // map subgraph node ids → full-graph nodes (same ids)
      return grp.map(n => byId.get(n.id)!);
    });
    composed.push(...kernels);
    console.log(`L${L}: ${nodes.length} nodes → ${best.partition.length} kernels in ${((performance.now() - t0) / 1000).toFixed(1)}s (${best.latencyMs?.toFixed(3) ?? "?"}ms)`);
  }

  // single-kernel for everything not in a layer (embed, final norm, lm_head gemm, argmax, rope tables)
  const uncoveredIds = new Set<number>();
  for (const grp of composed) for (const n of grp) uncoveredIds.add(n.id);
  for (const n of g.nodes) if (!uncoveredIds.has(n.id)) composed.push([n]);
  const ordered = topoOrder(g, composed);
  console.log(`composed partition: ${ordered.length} kernels (nodes ${composed.reduce((a, g) => a + g.length, 0)})`);

  // ── compile + correctness + speed ──
  const c0 = performance.now();
  const compiled = compileStep(g, ordered);
  console.log(`compiled ${compiled.schedule.length} kernels in ${((performance.now() - c0) / 1000).toFixed(1)}s`);

  // state provider (KV in-place, conv/S rotating) — same as decode_graph
  const stateClean = (name: string) =>
    name.replace(/^n\d+_/, "").replace(/_in$/, "").replace(/_new$/, "").replace(/\.kv_k_in/, ".kv_k").replace(/\.kv_v_in/, ".kv_v");
  const stateBufs = new Map<string, bigint[]>(); const stateCur = new Map<string, number>();
  const nonRotating = new Set<string>();
  for (const nm of g.tensors.keys()) {
    const t = g.t(nm); if (t.role !== "state") continue;
    const c = stateClean(nm); if (stateBufs.has(c)) continue;
    const bytes = bufSize(t.type.shape, t.type.dtype);
    const isKV = c.includes(".kv_k") || c.includes(".kv_v");
    const a = cuAlloc(BigInt(bytes)); cuHtoD(a, Buffer.alloc(bytes)); cuSync();
    if (isKV) { stateBufs.set(c, [a, a]); stateCur.set(c, 0); nonRotating.add(c); }
    else { const b = cuAlloc(BigInt(bytes)); cuHtoD(b, Buffer.alloc(bytes)); cuSync(); stateBufs.set(c, [a, b]); stateCur.set(c, 0); }
  }
  const scratch = new Map<string, bigint>();
  const resolveRun = (name: string): bigint | number => {
    const t = g.t(name);
    if (t.role === "scalar") throw new Error("scalar handled by runner");
    if (t.role === "state") {
      const isNew = name.endsWith("_new") || name.includes("_new");
      const c = stateClean(name);
      const idx = isNew ? 1 - stateCur.get(c)! : stateCur.get(c)!;
      return stateBufs.get(c)![idx];
    }
    if (name === "rope_cos" || name === "rope_sin") return ropeFor(name as any);
    const k = name === "embed.weight" || name === "lm_head.weight" ? "model.language_model.embed_tokens.weight"
      : name === "final.norm.weight" ? "model.language_model.norm.weight"
      : name.startsWith("model.") ? (Wnames.has(name) ? name : "model.language_model." + name.slice("model.".length)) : null;
    if (k && Wnames.has(k)) return Wnames.get(k)!;
    const v2 = name.startsWith("model.") && Wnames.has("model.language_model." + name) ? "model.language_model." + name : null;
    if (v2) return Wnames.get(v2)!;
    if (!scratch.has(name)) {
      const bytes = bufSize(t.type.shape, t.type.dtype);
      const b = cuAlloc(BigInt(bytes)); cuHtoD(b, Buffer.alloc(bytes)); cuSync();
      scratch.set(name, b);
    }
    return scratch.get(name)!;
  };
  const rotateState = () => { for (const c of stateBufs.keys()) if (!nonRotating.has(c)) stateCur.set(c, 1 - stateCur.get(c)!); };

  const genLen = 20;
  const got: number[] = [];
  let last5: number[] = [];
  const t0 = performance.now();
  const { tokens, logits } = await runGraphDecode(compiled, resolveRun, 9419, genLen, (t, l, s) => {
    rotateState(); got.push(t); if (got.length <= 5) last5.push(t);
  });
  const dt = (performance.now() - t0) / 1000;
  const ref = [11, 271, 40, 1044, 3133];
  let matches = 0;
  for (let i = 0; i < Math.min(last5.length, ref.length); i++) if (last5[i] === ref[i]) matches++;
  console.log(`fused graph-decode: ${genLen} tok in ${dt.toFixed(2)}s → ${(genLen / dt).toFixed(1)} tok/s`);
  console.log(`first5: [${last5.join(", ")}]  Match ${matches}/5`);
}

function collectLayer(n: GraphNode): number | null {
  const names = [...n.inputs.map((i: any) => i.tensorName), ...n.outputs.map((o: any) => o.tensorName)];
  for (const nm of names) { const m = nm.match(/model\.layers\.(\d+)\./); if (m) return Number(m[1]); }
  return null;
}

/** Sub-graph of `nodes` with ids preserved (from fusion_model.ts). */
function subGraphOf(g: Graph, nodes: GraphNode[]): Graph {
  const sg = new Graph();
  const ids = new Set(nodes.map(n => n.id));
  const externals = new Set<string>();
  for (const n of nodes) for (const i of n.inputs) {
    const t = g.t(i.tensorName);
    if (!t.producer || !ids.has(t.producer.id)) externals.add(i.tensorName);
  }
  for (const n of nodes) for (const o of n.outputs) {
    const t = g.t(o.tensorName);
    if (t.consumers.filter(c => ids.has(c.id)).length === 0) externals.add(o.tensorName);
  }
  for (const nm of externals) { const t = g.t(nm); if (!sg.tensors.has(nm)) sg.input(nm, t.type, t.role); }
  const sorted = [...nodes].sort((a, b) => a.id - b.id);
  for (const n of sorted) {
    (sg as any).nextId = n.id;
    const ins = n.inputs.map((i: any) => ({ tensor: sg.t(i.tensorName), name: i.name }));
    const outs = n.outputs.map((o: any) => ({ name: o.name, type: o.type, role: o.role }));
    const node = sg.node(n.op, ins, outs, [n.grid[0], n.grid[1], n.grid[2]], n.params);
    if (node.id !== n.id) throw new Error(`subgraph id mismatch: got ${node.id} want ${n.id}`);
  }
  return sg;
}
await main();
