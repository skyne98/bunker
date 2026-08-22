// src/policy.ts — persistent cache of measured fusion/tile discoveries
//
// `explore()` (src/fusion.ts) discovers a good (partition, tiles) candidate by
// measuring real GPU latency. This module persists that discovery to a JSON
// file so a later run can replay it without re-measuring. All decisions come
// from measurement; nothing here is hand-picked.
import { Graph, Candidate, TileAssign, GraphNode } from "./fusion";
import { readFileSync, writeFileSync, existsSync } from "fs";

/** A discovered (partition + tiles) policy, stable on disk. */
export interface Policy {
  /** schema version; bump on incompatible changes */
  version: 1;
  /** what the policy was tuned for (e.g. "decode-layer0") */
  target: string;
  /** fingerprint of the graph the policy was discovered on */
  graphSignature: string;
  /** node ids per kernel group, in kernel order */
  partition: number[][];
  /** discovered tile per GEMM node id (string keys for JSON) */
  tiles: Record<string, { BN: number; BK: number }>;
  /** measured latency of the policy, ms */
  measuredMs: number;
}

/** Stable fingerprint: node count + each node's (id, op). */
export function graphSignature(graph: Graph): string {
  const ids = [...graph.nodes].sort((a, b) => a.id - b.id)
    .map(n => `${n.id}:${n.op}`).join(",");
  return `n${graph.nodes.length}:${ids}`;
}

/** Build a persisted policy from an explored best candidate. */
export function policyFromBest(graph: Graph, target: string, best: Candidate): Policy {
  const tiles: Record<string, { BN: number; BK: number }> = {};
  for (const [k, v] of Object.entries(best.tiles)) tiles[k] = { BN: v.BN, BK: v.BK };
  return {
    version: 1,
    target,
    graphSignature: graphSignature(graph),
    partition: best.partition.map(g => g.map(n => n.id)),
    tiles,
    measuredMs: best.latencyMs ?? -1,
  };
}

/**
 * Rebuild a candidate from a policy. Returns null when the policy version is
 * unknown or the graph does not match (stale policy), so a mismatched policy
 * can never replay kernels against the wrong graph.
 */
export function bestFromPolicy(graph: Graph, p: Policy): Candidate | null {
  if (p.version !== 1) return null;
  if (p.graphSignature !== graphSignature(graph)) return null;
  const byId = new Map(graph.nodes.map(n => [n.id, n]));
  const partition: GraphNode[][] = p.partition.map(grp =>
    grp.map((id) => {
      const n = byId.get(id);
      if (!n) throw new Error(`policy: node ${id} absent from graph`);
      return n;
    }));
  const tiles: TileAssign = {};
  for (const [k, v] of Object.entries(p.tiles)) tiles[Number(k)] = { BN: v.BN, BK: v.BK };
  return { partition, tiles, latencyMs: p.measuredMs >= 0 ? p.measuredMs : null };
}

/** Persist a policy. Throws on write errors (caller decides how to surface). */
export function savePolicy(path: string, p: Policy): void {
  writeFileSync(path, JSON.stringify(p, null, 2) + "\n");
}

/** Load a policy; null when the file is missing or malformed. */
export function loadPolicy(path: string): Policy | null {
  try {
    if (!existsSync(path)) return null;
    return JSON.parse(readFileSync(path, "utf-8")) as Policy;
  } catch {
    return null;
  }
}
