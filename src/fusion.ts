// src/fusion.ts — Graph IR with measured-latency fusion discovery.
//
// The decoder model is described as a DAG of `node`s (ops) over `tensor`s.
// A fused program is a **partition** of nodes into kernel groups. Different
// partitions are different fusion strategies. There is no hand-rolled "which
// kernels to fuse" policy: the `explore` engine generates candidate partitions,
// compiles + runs each on the GPU, and keeps the fastest by measured wall time
// via beam search. The winning partition is returned for execution.
//
// Design goals (from AGENTS.md + project direction):
//   * graph is the single source of truth — no hardcoded fusion sequences
//   * fusion = searchable partitioning, scored by real GPU latency
//   * beam search over variants (the architecture's key benefit)
//   * every op emitter is a REAL computation (ported from decode.ts kernels),
//     so measured latency reflects actual work — no placeholder zeros
//   * portability: no sm_86/PTX literals here; kernels compile via the shim

import { TTIRBuilder, compileAndLoad, cuLaunch, cuSync, LoadedKernel } from "./ttir";

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════

export type DType = "f32" | "bf16" | "f16" | "i32" | "i1";

export interface TensorType {
  shape: number[];
  dtype: DType;
  strides: number[];
  custom?: { name: string };
}

export type TensorRole = "data" | "weight" | "state" | "scalar";

export interface NodeIO {
  name: string;
  tensorName: string;
  type: TensorType;
  role: TensorRole;
}

export interface GraphNode {
  id: number;
  op: string;
  inputs: NodeIO[];
  outputs: NodeIO[];
  /** Declared launch grid. Elementwise nodes reinterpret grid.y as row id. */
  grid: [number, number, number];
  params: Record<string, any>;
}

export interface Tensor {
  name: string;
  type: TensorType;
  role: TensorRole;
  producer: GraphNode | null;
  consumers: GraphNode[];
  isFusedAway?: boolean;
}

export type Partition = GraphNode[][];

export interface KernelPlan {
  ttir: string;
  name: string;
  /** External tensor names, in param order. */
  args: string[];
  grid: [number, number, number];
  group: GraphNode[];
}

// ═══════════════════════════════════════════════════════════════════
// Graph — declarative model (single source of truth)
// ═══════════════════════════════════════════════════════════════════

export class Graph {
  nodes: GraphNode[] = [];
  tensors = new Map<string, Tensor>();
  private nextId = 0;

  input(name: string, type: TensorType, role: TensorRole = "data"): Tensor {
    const t: Tensor = { name, type, role, producer: null, consumers: [] };
    this.tensors.set(name, t);
    return t;
  }

  node(
    op: string,
    inputs: { tensor: string | Tensor; name: string }[],
    outputs: { name: string; type: TensorType; role?: TensorRole }[],
    grid: [number, number, number],
    params: Record<string, any> = {},
  ): GraphNode {
    const id = this.nextId++;
    const node: GraphNode = { id, op, inputs: [], outputs: [], grid, params };
    for (const i of inputs) {
      const t = typeof i.tensor === "string" ? this.tensors.get(i.tensor)! : i.tensor;
      node.inputs.push({ name: i.name, tensorName: t.name, type: t.type, role: t.role });
      t.consumers.push(node);
    }
    for (const o of outputs) {
      const fullName = `n${id}_${o.name}`;
      node.outputs.push({ name: o.name, tensorName: fullName, type: o.type, role: o.role ?? "data" });
      const t: Tensor = { name: fullName, type: o.type, role: o.role ?? "data", producer: node, consumers: [] };
      this.tensors.set(fullName, t);
    }
    this.nodes.push(node);
    return node;
  }

  t(name: string): Tensor {
    const t = this.tensors.get(name);
    if (!t) throw new Error(`tensor "${name}" not found`);
    return t;
  }

  out(node: GraphNode, outputName = "out"): string {
    return `n${node.id}_${outputName}`;
  }

  /** External (non-produced) tensors. */
  externals(): string[] {
    return [...this.tensors.values()].filter(t => t.producer === null).map(t => t.name);
  }

  /** Dependencies: node → set of producer node ids (for validity checks). */
  deps(): Map<number, Set<number>> {
    const m = new Map<number, Set<number>>();
    for (const n of this.nodes) m.set(n.id, new Set());
    for (const n of this.nodes)
      for (const inp of n.inputs) {
        const t = this.tensors.get(inp.tensorName)!;
        if (t.producer) m.get(n.id)!.add(t.producer.id);
      }
    return m;
  }
}

// ═══════════════════════════════════════════════════════════════════
// Grid compatibility (a VALIDITY rule, not a fusion policy)
// ═══════════════════════════════════════════════════════════════════

/**
 * Whether two nodes can share a single kernel launch given their declared
 * grids. This is a hard correctness constraint: the codegen reads pid.y as the
 * row for elementwise ops, so same-grid nodes can tile together. Nodes with
 * incompatible grids MUST be in different kernels.
 *
 * The rules here are conservative/correct; the explorer decides which *legal*
 * merges are actually profitable. Do not add per-op heuristics here.
 */
export function gridsCompatible(g1: [number, number, number], g2: [number, number, number]): boolean {
  return g1[0] === g2[0] && g1[1] === g2[1] && g1[2] === g2[2];
}

/** Ops that cannot consume an SSA value (need a global-memory pointer). */
export const POINTER_INPUT_OPS = new Set(["gemm", "gdn_delta_rule", "conv1d_decode", "fa2_prep", "fa2_attn", "argmax", "embed"]);

/**
 * Check whether a candidate group is internally valid:
 *   - no duplicate/conflicting grids
 *   - every POINTER_INPUT op's inputs must be external OR produced by a node
 *     whose output was NOT fused away (i.e. available in global memory)
 * This is enforced structurally; the explorer only proposes partitions it can
 * evaluate, so it must call validGroup before keeping a candidate.
 */
export function validGroup(group: GraphNode[], graph: Graph): boolean {
  if (group.length === 0) return false;
  const g0 = group[0].grid;
  for (const n of group) if (!gridsCompatible(n.grid, g0)) return false;

  const inGroup = new Set(group.map(n => n.id));
  // intermediate tensors produced AND consumed within the group = fused away
  const fusedAway = new Set<string>();
  for (const n of group)
    for (const o of n.outputs) {
      const t = graph.tensors.get(o.tensorName)!;
      if (t.consumers.length > 0 && t.consumers.every(c => inGroup.has(c.id))) fusedAway.add(o.tensorName);
    }
  for (const n of group) {
    if (!POINTER_INPUT_OPS.has(n.op)) continue;
    for (const inp of n.inputs) {
      const t = graph.tensors.get(inp.tensorName)!;
      if (!t.producer) continue; // external — fine
      // producer in group AND fused away → SSA, not usable by a pointer-input op
      if (inGroup.has(t.producer.id) && fusedAway.has(inp.tensorName)) return false;
    }
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════════
// Op emitters — REAL computations, ported from the decode kernels so that
// measured latency is meaningful. Each returns an SSA Value (f32 or bf16),
// or null if it stores to global memory directly (pointer-output ops).
// ═══════════════════════════════════════════════════════════════════

// The emit signature receives the builder, resolved input Value(s), the node,
// and helpers. To keep the port faithful, emitters build on TTIRBuilder ops.
type EmitCtx = {
  b: TTIRBuilder;
  graph: Graph;
  /** resolved input SSA/ptr Values keyed by tensorName */
  vals: Map<string, any>;
  /** external pointer params keyed by tensorName */
  ptrs: Map<string, any>;
  tile: number[];
  pidAxis: 0 | 1 | 2;
};

/** Load an input: SSA if fused-away, else from global memory via ptr. */
function loadInput(ctx: EmitCtx, inp: NodeIO): any {
  if (ctx.vals.has(inp.tensorName)) return ctx.vals.get(inp.tensorName);
  const p = ctx.ptrs.get(inp.tensorName)!;
  const { b, tile, pidAxis } = ctx;
  const N = tile[1];
  const pid = b.programId(pidAxis);
  const off = b.mul(pid, b.i32(N));
  const tn = inp.type.shape[inp.type.shape.length - 1];
  const tp = b.makeTensorPtr(p, [1, tn], [tn, 1], [b.i32(0), off], tile, inp.type.dtype as any, [1, 0]);
  return b.load(tp, { boundaryCheck: [0, 1], padding: 1 });
}

function emitGemm(ctx: EmitCtx, node: GraphNode): any {
  const { b } = ctx;
  const { M, N, K, N2 } = node.params;
  const A = ctx.ptrs.get(node.inputs[0].tensorName)!;
  const B = ctx.ptrs.get(node.inputs[1].tensorName)!;
  const BM = Math.min(64, M), BN = Math.min(64, N), BK = Math.min(64, K);
  const pM = b.programId(0), pN = b.programId(1);
  const tpA = b.makeTensorPtr(A, [M, K], [K, 1], [b.mul(pM, b.i32(BM)), b.i32(0)], [BM, BK], "bf16", [1, 0]);
  const tpB = b.makeTensorPtr(B, [K, N], [1, K], [b.i32(0), b.mul(pN, b.i32(BN))], [BK, BN], "bf16", [0, 1]);
  const a0 = b.zeros([BM, BN], "f32");
  const [acc] = b.forIter(b.index(0), b.index(K), b.index(BK), [a0, tpA, tpB], (bb, _, [a, tA, tB]) => {
    const n = bb.dot(bb.load(tA), bb.load(tB), a);
    return [n, bb.advance(tA, [bb.i32(0), bb.i32(BK)]), bb.advance(tB, [bb.i32(BK), bb.i32(0)])];
  });
  if (node.params.cast) return b.fptrunc(acc, "bf16");
  return acc; // f32 SSA
}

function emitCast(ctx: EmitCtx, node: GraphNode): any {
  const { b } = ctx;
  const x = loadInput(ctx, node.inputs[0]);
  const to = node.params.to as DType;
  return to === "bf16" ? b.fptrunc(x, "bf16") : b.fpext(x, to as any);
}

function emitAdd(ctx: EmitCtx, node: GraphNode): any {
  const { b } = ctx;
  const a = b.fpext(loadInput(ctx, node.inputs[0]), "f32");
  const bb = b.fpext(loadInput(ctx, node.inputs[1]), "f32");
  const out = b.add(a, bb);
  return node.params.cast === false ? out : b.fptrunc(out, "bf16");
}

function emitRmsNorm(ctx: EmitCtx, node: GraphNode): any {
  const { b } = ctx;
  const x = loadInput(ctx, node.inputs[0]);
  const w = loadInput(ctx, node.inputs[1]);
  const N = node.params.N;
  const xf = b.fpext(x, "f32");
  const ms = b.divf(b.sum(b.mul(xf, xf), 1), b.f32(N));
  const msBc = b.broadcast(b.expandDims(ms, 1), [1, N]);
  const rstd = b.rsqrtHw(b.add(msBc, b.f32(1e-6)));
  const y = b.mul(xf, rstd);
  const yw = b.mul(y, b.add(b.f32(1), b.fpext(w, "f32")));
  return b.fptrunc(yw, "bf16");
}

function emitSwiGLU(ctx: EmitCtx, node: GraphNode): any {
  const { b } = ctx;
  const g = loadInput(ctx, node.inputs[0]);
  const u = loadInput(ctx, node.inputs[1]);
  const sig = b.divf(b.f32(1), b.add(b.f32(1), b.exp(b.mul(g, b.f32(-1)))));
  return b.fptrunc(b.mul(b.mul(g, sig), u), "bf16");
}

/**
 * emitNode — dispatch to the right emitter. For any op not yet implemented as
 * an emitter, throws — the engine will NOT silently zero-fill (that would make
 * measured latency meaningless).
 */
export function emitNode(ctx: EmitCtx, node: GraphNode): any {
  switch (node.op) {
    case "gemm": return emitGemm(ctx, node);
    case "cast": return emitCast(ctx, node);
    case "add": return emitAdd(ctx, node);
    case "rmsnorm": return emitRmsNorm(ctx, node);
    case "swiglu": return emitSwiGLU(ctx, node);
    default: throw new Error(`fusion: no emitter for op '${node.op}' — refusing to zero-fill`);
  }
}

// ═══════════════════════════════════════════════════════════════════
// Codegen — turn one group into one TTIR kernel
// ═══════════════════════════════════════════════════════════════════

export function codegenGroup(graph: Graph, group: GraphNode[], groupIndex: number): KernelPlan {
  const b = new TTIRBuilder();
  const inGroup = new Set(group.map(n => n.id));
  const fusedAway = new Set<string>();
  for (const n of group)
    for (const o of n.outputs) {
      const t = graph.tensors.get(o.tensorName)!;
      if (t.consumers.length > 0 && t.consumers.every(c => inGroup.has(c.id))) fusedAway.add(o.tensorName);
    }

  // params: external tensors + produced-but-not-fused-away outputs
  const external = new Set<string>();
  for (const n of group)
    for (const inp of n.inputs) {
      const t = graph.tensors.get(inp.tensorName)!;
      const isExternal = t.role !== "data" || !t.producer || !inGroup.has(t.producer.id);
      if (isExternal) external.add(inp.tensorName);
      else if (!fusedAway.has(inp.tensorName)) external.add(inp.tensorName); // produced but materialized
    }
  for (const n of group)
    for (const o of n.outputs) {
      const t = graph.tensors.get(o.tensorName)!;
      if (!fusedAway.has(o.tensorName) && t.role !== "state") external.add(o.tensorName);
    }

  const args: string[] = [];
  const ptrs = new Map<string, any>();
  for (const name of external) {
    const t = graph.tensors.get(name)!;
    const p = b.param(`arg${args.length}`, { ptr: t.type.dtype as any });
    args.push(name);
    ptrs.set(name, p);
  }

  // tile
  let tile: number[] = [1, 1024];
  let tileFrom = false;
  for (const n of group) {
    if (n.op === "gemm") {
      tile = [1, Math.min(64, n.params.N)];
      tileFrom = true;
      break;
    }
  }
  if (!tileFrom && group.length === 1 && group[0].op === "rmsnorm") tile = [1, group[0].params.N];

  // grid axis: if any node has grid.y > 1, pids index rows along y (2D grid);
  // else grid.x is the single row axis (1D grid).
  const hasY = group.some(n => n.grid[1] > 1);
  // For gemm groups the grid is [1, N/64, 1]: pN=programId(1), pM=0.
  const pidAxis: 0 | 1 = hasY ? 1 : 0;

  const ctx: EmitCtx = { b, graph, vals: new Map(), ptrs, tile, pidAxis: pidAxis as 0 | 1 | 2 };

  // emit each node in group order (topological within group)
  for (const n of group) {
    // For GEMM: the group supplies a single grid; gemm reads A (external) + B (external),
    // writes an SSA f32/bf16 accumulator. It must be the FIRST node for now.
    const out = emitNode(ctx, n);
    // route outputs
    for (const o of n.outputs) {
      const t = graph.tensors.get(o.tensorName)!;
      if (t.role === "state") continue;
      if (fusedAway.has(o.tensorName)) {
        ctx.vals.set(o.tensorName, out);
      } else {
        // materialize to global memory
        const p = ptrs.get(o.tensorName)!;
        const tn = t.type.shape[t.type.shape.length - 1];
        const pid = b.programId(pidAxis as 0 | 1 | 2);
        const off = b.mul(pid, b.i32(ctx.tile[1]));
        const tp = b.makeTensorPtr(p, [1, tn], [tn, 1], [b.i32(0), off], ctx.tile, t.type.dtype as any, [1, 0]);
        const storeVal = (t.type.dtype === "bf16" && out.elem !== "bf16") ? b.fptrunc(out, "bf16") : out;
        b.store(tp, storeVal, { boundaryCheck: [0, 1] });
      }
    }
  }

  const name = `f${groupIndex}`;
  const ttir = b.build(name, 4, args.length);
  return { ttir, name, args, grid: group[0].grid, group };
}

// ═══════════════════════════════════════════════════════════════════
// Runner — compile a partition and measure its real GPU latency
// ═══════════════════════════════════════════════════════════════════

export interface CompiledPartition {
  kernels: { plan: KernelPlan; k: LoadedKernel; paramPtrs: bigint[] }[];
  /** total measured wall time (ms) over the partition execution */
  latencyMs: number;
  partition: Partition;
}

export interface RunEnv {
  /** resolve each external tensor name → device pointer; must cover all args. */
  resolve: (name: string) => bigint;
}

/**
 * Compile a partition into kernels. Throws if any group is invalid or any op
 * is un-emittable — the engine never silently drops work.
 */
export function compilePartition(graph: Graph, partition: Partition): KernelPlan[] {
  const plans: KernelPlan[] = [];
  for (let gi = 0; gi < partition.length; gi++) {
    const group = partition[gi];
    if (!validGroup(group, graph))
      throw new Error(`fusion: invalid group ${gi} (${group.map(n => n.op).join("+")})`);
    try {
      plans.push(codegenGroup(graph, group, gi));
    } catch (e: any) {
      throw new Error(`fusion: codegen failed for group ${gi} (${group.map(n => n.op).join("+")}): ${e.message}`);
    }
  }
  return plans;
}

/** Compile + load a partition's kernels into GPU modules. */
export function loadPartition(plans: KernelPlan[]): { plan: KernelPlan; k: LoadedKernel }[] {
  return plans.map(p => ({ plan: p, k: compileAndLoad(p.ttir, p.name, 4) }));
}

/**
 * Run a partition once on the GPU and measure wall time (ms). Kernels are
 * launched in partition order with no per-kernel sync; one final sync drains.
 * The measured time is the true latency a full decode step would incur from
 * these kernels alone (excludes host-side allocation, as in the live loop).
 */
export function measurePartition(plans: KernelPlan[], env: RunEnv, reps = 5): number {
  const loaded = loadPartition(plans);
  const paramPtrs = plans.map(p => p.args.map(a => env.resolve(a)));
  // warm up (compile+load already happened; run once to page everything in)
  for (const [i, { k, plan }] of loaded.entries())
    cuLaunch(k, [plan.grid[0], plan.grid[1], plan.grid[2]], [128, 1, 1], paramPtrs[i]);
  cuSync();
  // timed
  const t0 = performance.now();
  for (let r = 0; r < reps; r++) {
    for (const [i, { k, plan }] of loaded.entries())
      cuLaunch(k, [plan.grid[0], plan.grid[1], plan.grid[2]], [128, 1, 1], paramPtrs[i]);
    cuSync();
  }
  const dt = (performance.now() - t0) / reps;
  return dt;
}

// ═══════════════════════════════════════════════════════════════════
// Beam-search explorer — discovers a good partition by measuring real latency
// ═══════════════════════════════════════════════════════════════════

export interface ExploreConfig {
  /** beam width (default 4) */
  beam?: number;
  /** max partitions evaluated (default 40) */
  budget?: number;
  /** allowed moves (default: split + merge) */
  moves?: ("split" | "merge")[];
  verbose?: boolean;
  /** if a candidate fails to compile, dump its TTIR for debugging */
  dumpOnError?: boolean;
}

/** A candidate = a partition + its measured latency (maybe null if bad). */
export interface Candidate {
  partition: Partition;
  latencyMs: number | null;
}

function initialPartition(graph: Graph): Partition {
  // start with every node its own kernel (safest, correct baseline)
  return graph.nodes.map(n => [n]);
}

/** All legal single-merge moves: merge group i with i+1 (when valid). */
function mergeMoves(graph: Graph, partition: Partition): Partition[] {
  const out: Partition[] = [];
  for (let i = 0; i < partition.length - 1; i++) {
    const merged = partition.map(g => [...g]);
    merged[i] = [...partition[i], ...partition[i + 1]];
    merged.splice(i + 1, 1);
    if (validGroup(merged[i], graph)) out.push(merged);
  }
  return out;
}

/** All legal single-split moves of multi-node groups. */
function splitMoves(graph: Graph, partition: Partition): Partition[] {
  const out: Partition[] = [];
  for (let i = 0; i < partition.length; i++) {
    const g = partition[i];
    if (g.length < 2) continue;
    for (let s = 1; s < g.length; s++) {
      const a = g.slice(0, s), b = g.slice(s);
      const cand = [...partition];
      cand.splice(i, 1, a, b);
      if (validGroup(a, graph) && validGroup(b, graph)) out.push(cand);
    }
  }
  return out;
}

/** Structural cost: launch count (primary) — lower is generally better, but
 *  the REAL score is measured latency below. Used only to break ties. */
function costOf(partition: Partition): number {
  let fusedAway = 0;
  for (const g of partition) fusedAway += g.length - 1;
  return partition.length - fusedAway; // fewer kernels, more fusions => lower
}

/**
 * Beam search over fusion partitions, scored by MEASURED GPU latency.
 *
 * 1. Start: every node its own kernel.
 * 2. Propose legal moves (merge adjacent groups, split multi-node groups).
 * 3. For each candidate: compile → launch on GPU → measure wall time.
 * 4. Keep beam-best candidates (fewest kernels / lowest measured latency).
 * 5. Repeat until budget exhausted. Return the best measured partition.
 *
 * The score is real GPU wall time (not a heuristic), so different fusion
 * strategies are genuinely A/B-tested — the architecture's key benefit.
 */
export function explore(
  graph: Graph,
  env: RunEnv,
  cfg: ExploreConfig = {},
): { best: Candidate; candidates: Candidate[] } {
  const beam = cfg.beam ?? 4;
  const budget = cfg.budget ?? 40;
  const moves = cfg.moves ?? ["split", "merge"];
  const verbose = cfg.verbose ?? false;

  const partKey = (p: Partition) => p.map(g => g.map(n => n.id).join(",")).join("|");

  let beamCands: Candidate[] = [{ partition: initialPartition(graph), latencyMs: null }];
  const all: Candidate[] = [];
  const seen = new Set<string>();

  const evaluate = (p: Partition): Candidate => {
    const key = partKey(p);
    if (seen.has(key)) return { partition: p, latencyMs: null };
    seen.add(key);
    let latencyMs: number | null = null;
    try {
      const plans = compilePartition(graph, p);
      latencyMs = measurePartition(plans, env, 5);
      if (verbose) console.log(`  ✓ ${p.length} kernels: ${latencyMs.toFixed(3)}ms`);
    } catch (e: any) {
      if (cfg.dumpOnError) {
        console.log("=== ERROR candidate TTIR ===");
        try { for (const g of p) codegenGroup(graph, g, 0); } catch (e2: any) { console.log(e2.message); }
        for (const g of p) {
          try { const pl = codegenGroup(graph, g, 0); console.log(`-- group ${g.map(n => n.op).join("+")} --\n${pl.ttir}`); }
          catch (e2: any) { console.log(`-- group ${g.map(n => n.op).join("+")} FAILED: ${e2.message}`); }
        }
      }
      if (verbose) console.log(`  ✗ ${p.length} kernels: ${e.message.split("\n")[0]}`);
    }
    const c: Candidate = { partition: p, latencyMs };
    all.push(c);
    return c;
  };

  let used = 0;
  while (used < budget) {
    const children = new Map<string, Candidate>();
    for (const cand of beamCands) {
      for (const mv of moves) {
        let moveSet: Partition[] = [];
        if (mv === "split") moveSet = splitMoves(graph, cand.partition);
        else if (mv === "merge") moveSet = mergeMoves(graph, cand.partition);
        for (const p of moveSet) {
          if (used >= budget) break;
          const c = evaluate(p);
          used++;
          if (c.latencyMs !== null) children.set(partKey(p), c);
        }
      }
    }
    const ranked = [...children.values()].sort((a, b) => (a.latencyMs ?? 1e9) - (b.latencyMs ?? 1e9));
    const next = ranked.slice(0, Math.max(1, beam));
    if (!next.length) break; // no new valid candidates
    beamCands = next;
    if (verbose) console.log(`  beam → best ${beamCands[0].latencyMs?.toFixed(3)}ms (${beamCands[0].partition.length} kernels)`);
    // avoid infinite loop if we can't progress
    if (next.length === 1 && seen.has(partKey(next[0].partition))) {
      const inc = ranked.filter(r => !seen.has(partKey(r.partition)));
      if (!inc.length) break;
      beamCands = inc.slice(0, 1);
    }
  }

  const evaluated = all.filter(c => c.latencyMs !== null);
  const best = evaluated.length
    ? evaluated.sort((a, b) => (a.latencyMs ?? 1e9) - (b.latencyMs ?? 1e9))[0]
    : { partition: initialPartition(graph), latencyMs: null };
  return { best, candidates: all };
}

