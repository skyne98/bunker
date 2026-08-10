// prototypes/bench_graph_decode.ts — measure graph-driven decode tok/s.
import { buildModelGraph, D } from "../src/model";
import { compileStep, runGraphDecode, choosePartition } from "../src/runner";
import { cuAlloc, cuHtoD, cuSync } from "../src/ttir";
import { performance } from "perf_hooks";

const bufSize = (sh: number[], dtype: string) => sh.reduce((a, b) => a * b, 1) * (dtype === "bf16" ? 2 : 4);

async function main() {
  const stPath = process.env.SAFETENSORS_PATH || "/tmp/qwen35_0.8b.safetensors";
  const genLen = parseInt(process.argv[2] || "200");
  const data = await Bun.file(stPath).bytes();
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const hl = Number(dv.getBigUint64(0, true));
  const hdr = JSON.parse(new TextDecoder().decode(data.subarray(8, 8 + hl)));
  const ds = 8 + hl;
  const base = cuAlloc(BigInt(data.length - ds));
  cuHtoD(base, data.subarray(ds)); cuSync();
  const W = new Map<string, bigint>();
  for (const [k, v] of Object.entries(hdr)) if (k !== "__metadata__") W.set(k, base + BigInt((v as any).data_offsets[0]));

  const graph = buildModelGraph();
  const compiled = compileStep(graph, choosePartition(graph, { resolve: () => 0n }, { optimize: false }));
  const stateClean = (name: string) =>
    name.replace(/^n\d+_/, "").replace(/_in$/, "").replace(/_new$/, "").replace(/\.kv_k_in/, ".kv_k").replace(/\.kv_v_in/, ".kv_v");
  const stateBufs = new Map<string, bigint[]>(); const stateCur = new Map<string, number>();
  const nonRotating = new Set<string>();
  for (const nm of graph.tensors.keys()) {
    const t = graph.t(nm); if (t.role !== "state") continue;
    const c = stateClean(nm); if (stateBufs.has(c)) continue;
    const bytes = bufSize(t.type.shape, t.type.dtype);
    const isKV = c.includes(".kv_k") || c.includes(".kv_v");
    const a = cuAlloc(BigInt(bytes)); cuHtoD(a, Buffer.alloc(bytes)); cuSync();
    if (isKV) { stateBufs.set(c, [a, a]); stateCur.set(c, 0); nonRotating.add(c); }
    else { const b = cuAlloc(BigInt(bytes)); cuHtoD(b, Buffer.alloc(bytes)); cuSync(); stateBufs.set(c, [a, b]); stateCur.set(c, 0); }
  }
  const scratch = new Map<string, bigint>();
  const safeKey = (nm: string): string | null => {
    if (nm === "embed.weight" || nm === "lm_head.weight") return "model.language_model.embed_tokens.weight";
    if (nm === "final.norm.weight") return "model.language_model.norm.weight";
    if (nm.startsWith("model.")) { const p = "model.language_model." + nm.slice("model.".length); if (W.has(p)) return p; }
    if (W.has(nm)) return nm; return null;
  };
  const resolve = (name: string): bigint | number => {
    const t = graph.t(name);
    if (t.role === "scalar") throw new Error("scalar");
    if (t.role === "state") {
      const isNew = name.endsWith("_new") || name.includes("_new");
      const c = stateClean(name);
      const idx = isNew ? 1 - stateCur.get(c)! : stateCur.get(c)!;
      return stateBufs.get(c)![idx];
    }
    if (name === "rope_cos" || name === "rope_sin") {
      if (!scratch.has(name)) {
        const b = cuAlloc(BigInt(D.MAX_LEN * D.ROT_HALF * 4));
        const arr = new Float32Array(D.MAX_LEN * D.ROT_HALF);
        for (let p = 0; p < D.MAX_LEN; p++) for (let i = 0; i < D.ROT_HALF; i++) {
          const f = 1 / Math.pow(10000000, 2 * i / D.ROT_DIM);
          arr[p * D.ROT_HALF + i] = name === "rope_cos" ? Math.cos(p * f) : Math.sin(p * f);
        }
        cuHtoD(b, arr.buffer); cuSync(); scratch.set(name, b);
      }
      return scratch.get(name)!;
    }
    const k = safeKey(name);
    if (k && W.has(k)) return W.get(k)!;
    if (!scratch.has(name)) {
      const bytes = bufSize(t.type.shape, t.type.dtype);
      const b = cuAlloc(BigInt(bytes)); cuHtoD(b, Buffer.alloc(bytes)); cuSync();
      scratch.set(name, b);
    }
    return scratch.get(name)!;
  };
  const rotateState = () => { for (const c of stateBufs.keys()) if (!nonRotating.has(c)) stateCur.set(c, 1 - stateCur.get(c)!); };

  const t0 = performance.now();
  await runGraphDecode(compiled, resolve, 9419, genLen, (t, l, s) => { rotateState(); });
  const dt = (performance.now() - t0) / 1000;
  console.log(`graph-decode: ${genLen} tokens in ${dt.toFixed(2)}s → ${(genLen / dt).toFixed(1)} tok/s`);
}
await main();
