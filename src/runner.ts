// src/runner.ts — graph-driven decode runtime.
//
// This is the whole point: the model architecture is a Graph (src/model.ts);
// the fusion explorer picks a partition; THIS runtime executes that partition
// to actually GENERATE TOKENS. There is no hand-written launch sequence — the
// graph's topological order, the partition's groups, the tensor→buffer mapping
// (RunEnv), and the state double-buffer carry are the only execution logic.
//
// Steps:
//   1. buildModelGraph() → the DAG for one token step.
//   2. choose a partition (per-layer explored, else every-node-its-own-kernel).
//   3. compile each group → PTX → GPU module; launch in topo order each token.
//   4. state tensors double-buffer across tokens (conv state, S-state, KV cache).
//   5. argmax idx read back → next token.
import { Graph, GraphNode, Partition, KernelPlan, compilePartition, loadPartition, RunEnv } from "./fusion";
import { buildModelGraph, D } from "./model";
import type { LoadedKernel } from "./ttir";
import { cuSync, cuLaunch } from "./ttir";

export { D };

/** A compiled, launchable token-step. */
export interface CompiledStep {
  graph: Graph;
  /** in topological order: each entry = kernel + its resolved args (device ptrs / scalars) */
  schedule: { plan: KernelPlan; k: LoadedKernel; args: (bigint | number)[] }[];
  /** names of all snapshot inputs (weights/state-in/scalars) indexed for buffer setup */
  inputNames: string[];
}

/**
 * Choose a partition for the graph. Strategy: for a full model graph, run the
 * beam per layer (as the explorer does) and compose; the composition is just
 * the union of per-layer chosen groups in topological order. If per-layer
 * search is disabled/cached, fall back to every-node-its-own-kernel (correct,
 * unoptimized).
 *
 * Returned groups cover ALL nodes (embed, layers, lm_head, argmax).
 */
export function choosePartition(graph: Graph, env: RunEnv, opts: { optimize?: boolean } = {}): Partition {
  // Simplest correct partition first — each node its own kernel.
  const partition: Partition = graph.nodes.map(n => [n]);
  if (!opts.optimize) return partition;

  // Per-layer beam search: run explore() on each layer's subgraph, compose.
  // (Full 24-layer exploration is deferred to a cached policy file for now;
  //  this hook is where beam-discovered partitions plug in.)
  return partition;
}

/** Topological order of a partition's groups by node deps. */
export function topoOrder(graph: Graph, partition: Partition): Partition {
  const nodeToGroup = new Map<number, number>();
  partition.forEach((g, i) => g.forEach(n => nodeToGroup.set(n.id, i)));
  // Kahn's algorithm over groups
  const indeg = new Map<number, number>();
  partition.forEach((_, i) => indeg.set(i, 0));
  for (let i = 0; i < partition.length; i++) {
    for (const n of partition[i]) {
      for (const inp of n.inputs) {
        const t = graph.tensors.get(inp.tensorName)!;
        if (t.producer) {
          const pg = nodeToGroup.get(t.producer.id)!;
          if (pg !== i) indeg.set(i, (indeg.get(i) ?? 0) + 1);
        }
      }
    }
  }
  const q: number[] = [];
  for (const [i, d] of indeg) if (d === 0) q.push(i);
  const sorted: Partition = [];
  while (q.length) {
    const i = q.shift()!;
    sorted.push(partition[i]);
    // decrement dependents
    for (const n of partition[i]) {
      for (const out of n.outputs) {
        const t = graph.tensors.get(out.tensorName)!;
        for (const c of t.consumers) {
          const cg = nodeToGroup.get(c.id)!;
          if (cg !== i) {
            indeg.set(cg, (indeg.get(cg) ?? 0) - 1);
            if (indeg.get(cg) === 0) q.push(cg);
          }
        }
      }
    }
  }
  if (sorted.length !== partition.length) throw new Error(`topoOrder: cycle in partition (${sorted.length}/${partition.length})`);
  return sorted;
}

/** Compile a partition into launchable kernels (topological order). */
export function compileStep(graph: Graph, partition: Partition): CompiledStep {
  const ordered = topoOrder(graph, partition);
  const plans = compilePartition(graph, ordered);
  const loaded = loadPartition(plans);
  const ret: CompiledStep = {
    graph,
    schedule: loaded.map(({ plan, k }, i) => ({ plan, k, args: [] })),
    inputNames: [] as string[],
  };
  return ret;
}

/**
 * Run the graph as an autoregressive decode loop.
 *
 * `resolveTensor(name)` returns either a device pointer (bigint) or a scalar
 * value (number) depending on the tensor's role (state/data→ptr, scalar→value).
 * `stateNew(name)` returns the buffer to write state outputs into (double-buffer).
 * The loop: for each token, launch schedule kernels in order with args resolved,
 * sync, read argmax idx, feed it back as the next token_id. No hardcoded kernel
 * order anywhere.
 */
export async function runGraphDecode(
  compiled: CompiledStep,
  resolve: (name: string) => bigint | number,
  initialToken: number,
  steps: number,
  onToken?: (token: number, logit: number, step: number) => void,
): Promise<{ tokens: number[]; logits: number[] }> {
  const { graph, schedule } = compiled;
  const argmaxNode = graph.nodes.find(n => n.op === "argmax");
  if (!argmaxNode) throw new Error("runGraphDecode: graph has no argmax node");
  const idxName = argmaxNode.outputs[1].tensorName;
  const valName = argmaxNode.outputs[0].tensorName;
  const ttir = await import("./ttir");

  const generated: number[] = [];
  const generatedLogits: number[] = [];

  for (let step = 0; step < steps; step++) {
    const token = step === 0 ? initialToken : generated[generated.length - 1];
    const pos = step;

    // resolve all args per kernel (state swap handled by resolve)
    const launchArgs: (bigint | number)[][] = schedule.map(s =>
      s.plan.args.map(a => {
        if (a === "token_id") return token;
        if (a === "pos") return pos;
        return resolve(a);
      }),
    );

    for (let i = 0; i < schedule.length; i++) {
      const { k, plan } = schedule[i];
      cuLaunch(k, [plan.grid[0], plan.grid[1], plan.grid[2]], [128, 1, 1], launchArgs[i]);
    }
    cuSync();

    const ibuf = new Int32Array(1);
    ttir.cuDtoH(ibuf.buffer, resolve(idxName) as bigint, BigInt(4));
    const vbuf = new Float32Array(1);
    ttir.cuDtoH(vbuf.buffer, resolve(valName) as bigint, BigInt(4));

    generated.push(ibuf[0]);
    generatedLogits.push(vbuf[0]);
    if (onToken) onToken(ibuf[0], vbuf[0], step);
  }
  return { tokens: generated, logits: generatedLogits };
}
