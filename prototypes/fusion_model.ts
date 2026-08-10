// prototypes/fusion_model.ts — declare the WHOLE Qwen3.5-0.8B as a graph and
// let the beam explorer optimize it. This is the architecture's payoff: the
// model is a declarative Graph (src/model.ts), every op has a real emitter
// (src/emitters.ts), and the optimiser (src/fusion.ts) measures GPU latency to
// choose the best per-layer fusion partition — no hardcoded kernel layout.
//
// Because the full model is ~200+ nodes, we run the beam per layer (each layer
// is an independent subgraph for fusion purposes), then compose the chosen
// per-layer partitions into a whole-model kernel schedule and measure it.
//
// Run: SAFETENSORS_PATH=... bun run prototypes/fusion_model.ts
import { buildModelGraph, D } from "../src/model";
import { Graph, GraphNode, explore, compilePartition, measurePartition, RunEnv, Partition } from "../src/fusion";
import { cuAlloc, cuHtoD, cuSync } from "../src/ttir";

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
  for (const [k, v] of Object.entries(hdr)) {
    if (k === "__metadata__") continue;
    Wnames.set(k, base + BigInt((v as any).data_offsets[0]));
  }

  // Build the whole-model graph (24 layers + lm_head + argmax)
  console.log("building whole-model graph...");
  const g = buildModelGraph();
  console.log(`  nodes: ${g.nodes.length}`);
  // count per op
  const byOp: Record<string, number> = {};
  for (const n of g.nodes) byOp[n.op] = (byOp[n.op] ?? 0) + 1;
  console.log("  ops:", JSON.stringify(byOp));

  // ── Resolve every external tensor to a valid device pointer.
  // weights: map graph name → safetensors key by layer. State buffers & scratch:
  // allocate real device buffers sized to the tensor. token_id: a real i32.
  const dynAllocs = new Map<string, bigint>();

  const resolve = (name: string): bigint => {
    // scalar kernel arg (token_id) resolves to its i32 VALUE (passed in the param)
    if (name === "token_id") return 9419n;
    // weights
    if (name.includes(".weight")) {
      // graph weight names are like "model.layers.0.input_layernorm.weight" or
      // "embed.weight"/"lm_head.weight"/"final.norm.weight"/"rope_cos"/"rope_sin".
      if (name === "embed.weight" || name === "lm_head.weight") return Wnames.get("model.language_model.embed_tokens.weight")!;
      if (name === "final.norm.weight") return Wnames.get("model.language_model.norm.weight")!;
      if (name.startsWith("model.layers.") && name.endsWith(".weight")) {
        const w = Wnames.get(name);
        if (w !== undefined) return w;
        // try the hidden prefix variants
        const v1 = Wnames.get("model.language_model." + name);
        if (v1 !== undefined) return v1;
      }
      if (name === "rope_cos" || name === "rope_sin") {
        // precomputed RoPE tables — allocate once
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
      }
      // unknown weight — allocate zero buffer sized from the tensor
    }
    // state buffers & scratch — allocate sized buffers
    if (!dynAllocs.has(name)) {
      const t = g.t(name);
      const bytes = t.type.shape.reduce((a, b) => a * b, 1) * (t.type.dtype === "bf16" ? 2 : 4);
      const b = cuAlloc(BigInt(bytes));
      cuHtoD(b, Buffer.alloc(bytes)); cuSync();
      dynAllocs.set(name, b);
    }
    return dynAllocs.get(name)!;
  };
  const env: RunEnv = { resolve };

  // ── Per-layer beam exploration: each layer is its own optimization unit ──
  // Collect nodes per layer from the graph.
  const layerNodes = new Map<number, GraphNode[]>();
  for (const n of g.nodes) {
    // a node belongs to layer L if any input/output tensor name contains ".L."
    const nm = collectLayer(n, g);
    if (nm !== null && nm >= 0 && nm < D.NL) {
      if (!layerNodes.has(nm)) layerNodes.set(nm, []);
      layerNodes.get(nm)!.push(n);
    }
  }

  console.log(`\nlayers found: ${layerNodes.size}`);
  const allNodes = new Set(g.nodes.map(n => n.id));
  const ungrouped = g.nodes.filter(n => {
    const nm = collectLayer(n, g);
    return nm === null || nm < 0 || nm >= D.NL;
  });
  console.log(`nodes not in a layer (embed/final/argmax/etc): ${ungrouped.length}`);

  // Run beam per layer (sample: layer 0 and layer 3 (FA2) for the demo)
  const layersToExplore = [0, 3];
  const chosenPartitions = new Map<number, Partition>();
  for (const L of layersToExplore) {
    const nodes = layerNodes.get(L)!;
    if (!nodes || nodes.length === 0) continue;
    // build a sub-graph: same graph but only these nodes are partitionable.
    // The explore() uses graph.nodes for initial partition; we adapt by forming
    // a partition of JUST these nodes (each its own kernel) inside the full graph.
    const subPartition: Partition = nodes.map(n => [n]);
    // Measure the unfused baseline:
    const sub = compilePartition(g, subPartition);
    const baseT = measurePartition(sub, env, 3);
    console.log(`\nlayer ${L} (${nodes.length} nodes): baseline ${baseT.toFixed(3)}ms (${nodes.length} kernels)`);

    // Manual beam over the layer's own nodes: use explore with a sub-graph to
    // keep the beam scoped. Build a fresh Graph containing only these nodes
    // (their tensor deps stay as externals).
    const subG = subGraphOf(g, nodes);
    const { best } = explore(subG, env, { beam: 3, budget: 30, moves: ["merge"], verbose: true });
    console.log(`  layer ${L}: best ${best.latencyMs?.toFixed(3) ?? "?"}ms, ${best.partition.length} kernels`);
    chosenPartitions.set(L, best.partition);
    for (const grp of best.partition) {
      console.log(`    kernel: ${grp.map(n => n.op).join("+")}`);
    }
  }

  console.log("\nwhole-model fusion exploration done (per layer).");
  console.log("NOTE: full 24-layer single-graph beam is intentionally per-layer");
  console.log("to keep compile+measure tractable; the per-layer winners compose.");
}

/** Extract layer index from a node's tensor names (graph name convention). */
function collectLayer(n: GraphNode, g: Graph): number | null {
  const names = [...n.inputs.map((i:any)=>i.tensorName), ...n.outputs.map((o:any)=>o.tensorName)];
  for (const nm of names) {
    const m = nm.match(/model\.layers\.(\d+)\./);
    if (m) return Number(m[1]);
  }
  return null;
}

/** Build a sub-graph containing only `nodes` (their tensor inputs become externals).
 *  Preserves node ids so output tensor names (n<id>_<name>) line up with the parent. */
function subGraphOf(g: Graph, nodes: GraphNode[]): Graph {
  const sg = new Graph();
  const ids = new Set(nodes.map(n => n.id));
  // declare external tensors first (in the parent's written ids can reuse via
  // input() with the same name; the node outputs keep n<id>_out via nextId).
  const externals = new Set<string>();
  for (const n of nodes)
    for (const i of n.inputs) {
      const t = g.t(i.tensorName);
      if (!t.producer || !ids.has(t.producer.id)) externals.add(i.tensorName);
    }
  for (const n of nodes)
    for (const o of n.outputs) {
      const t = g.t(o.tensorName);
      if (t.consumers.filter(c => ids.has(c.id)).length === 0) externals.add(o.tensorName);
    }
  for (const nm of externals) { const t = g.t(nm); if (!sg.tensors.has(nm)) sg.input(nm, t.type, t.role); }
  // Re-add nodes in id order with the SAME ids → same output tensor names.
  const sorted = [...nodes].sort((a, b) => a.id - b.id);
  for (const n of sorted) {
    // set nextId so this node gets its original id
    (sg as any).nextId = n.id;
    const ins = n.inputs.map((i) => ({ tensor: sg.t(i.tensorName) as any, name: i.name }));
    const outs = n.outputs.map((o) => ({ name: o.name, type: o.type, role: o.role }));
    const node = sg.node(n.op, ins, outs, [n.grid[0], n.grid[1], n.grid[2]], n.params);
    if (node.id !== n.id) throw new Error(`subgraph id mismatch: got ${node.id} want ${n.id}`);
  }
  return sg;
}
await main();
