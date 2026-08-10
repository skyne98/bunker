// prototypes/fusion_explore.ts — run the beam-search fusion explorer on the
// Qwen3.5-0.8B GDN layer, scored by real GPU latency.
//
// Architectural proof: fusion is treated as a graph partition search. Each
// candidate partition is compiled + run on the GPU, and the beam keeps the
// fastest by measured wall time. No hardcoded "which kernels to fuse".
//
// The emittable op subset is: gemm, cast, add, rmsnorm, swiglu. Ops without an
// emitter (conv1d, gdn delta rule, fa2) are represented as external tensors
// (the layer's in/out activation feeds) so the fusion search stays honest.
//
// Run: SAFETENSORS_PATH=... bun run prototypes/fusion_explore.ts
import { Graph, explore, RunEnv } from "../src/fusion";
import { cuAlloc, cuHtoD, cuSync } from "../src/ttir";

const H = 1024, INTER = 3584, QKVD = 6144, ZD = 2048, LVH = 16, LKD = 128, LVD = 128;

async function main() {
  const stPath = process.env.SAFETENSORS_PATH || "/tmp/qwen35_0.8b.safetensors";
  const data = await Bun.file(stPath).bytes();
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const hl = Number(dv.getBigUint64(0, true));
  const hdr = JSON.parse(new TextDecoder().decode(data.subarray(8, 8 + hl)));
  const ds = 8 + hl;
  const base = cuAlloc(BigInt(data.length - ds));
  cuHtoD(base, data.subarray(ds)); cuSync();

  // ── Build the GDN layer graph over emittable ops ──
  const g = new Graph();
  const T: Record<string, any> = {};
  const inp = {
    x: [[1, H], "bf16", "data"],
    input_layernorm_weight: [[H], "bf16", "weight"],
    qkv_weight: [[QKVD, H], "bf16", "weight"],
    z_weight: [[ZD, H], "bf16", "weight"],
    gdn_out: [[1, ZD], "bf16", "data"],       // conv1d+GDN block output (no emitter → external)
    out_proj_weight: [[H, ZD], "bf16", "weight"],
    post_attn_norm_weight: [[H], "bf16", "weight"],
    gate_weight: [[INTER, H], "bf16", "weight"],
    up_weight: [[INTER, H], "bf16", "weight"],
    down_weight: [[H, INTER], "bf16", "weight"],
  };
  for (const [name, [shape, dtype, role]] of Object.entries(inp)) T[name] = g.input(name, { shape, dtype: dtype as any, strides: shape.map(() => 1) }, role as any);

  // 1. input RMSNorm — grid [1]
  const normed = g.node("rmsnorm",
    [{ tensor: T.x, name: "x" }, { tensor: T.input_layernorm_weight, name: "w" }],
    [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }], [1, 1, 1], { N: H });
  // 2. QKV GEMM (f32, cast→bf16 baked) — grid [1,96]
  const qkv = g.node("gemm",
    [{ tensor: g.out(normed), name: "A" }, { tensor: T.qkv_weight, name: "B" }],
    [{ name: "out", type: { shape: [1, QKVD], dtype: "bf16", strides: [QKVD, 1] } }],
    [1, Math.ceil(QKVD / 64), 1], { M: 1, N: QKVD, K: H, cast: true });
  // 3. Z GEMM (f32, cast→bf16 baked) — grid [1,32]
  const z = g.node("gemm",
    [{ tensor: g.out(normed), name: "A" }, { tensor: T.z_weight, name: "B" }],
    [{ name: "out", type: { shape: [1, ZD], dtype: "bf16", strides: [ZD, 1] } }],
    [1, Math.ceil(ZD / 64), 1], { M: 1, N: ZD, K: H, cast: true });
  // 4. out_proj GEMM (f32) — on GDN output; grid [1,16]
  const outProjF = g.node("gemm",
    [{ tensor: T.gdn_out, name: "A" }, { tensor: T.out_proj_weight, name: "B" }],
    [{ name: "out", type: { shape: [1, H], dtype: "f32", strides: [H, 1] } }],
    [1, Math.ceil(H / 64), 1], { M: 1, N: H, K: ZD });
  // 5. cast out_proj → bf16 — grid [1,16]
  const outProjB = g.node("cast",
    [{ tensor: g.out(outProjF), name: "x" }],
    [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
    [1, Math.ceil(H / 64), 1], { N: H, from: "f32", to: "bf16" });
  // 6. add residual → afterAttn — grid [1,16]
  const afterAttn = g.node("add",
    [{ tensor: g.out(outProjB), name: "a" }, { tensor: T.x, name: "b" }],
    [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
    [1, Math.ceil(H / 64), 1], { N: H });
  // 7. post-attn RMSNorm — grid [1]
  const normed2 = g.node("rmsnorm",
    [{ tensor: g.out(afterAttn), name: "x" }, { tensor: T.post_attn_norm_weight, name: "w" }],
    [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }], [1, 1, 1], { N: H });
  // 8. gate GEMM (f32) — grid [1,56]
  const gate = g.node("gemm",
    [{ tensor: g.out(normed2), name: "A" }, { tensor: T.gate_weight, name: "B" }],
    [{ name: "out", type: { shape: [1, INTER], dtype: "f32", strides: [INTER, 1] } }],
    [1, Math.ceil(INTER / 64), 1], { M: 1, N: INTER, K: H });
  // 9. up GEMM (f32) — grid [1,56]
  const up = g.node("gemm",
    [{ tensor: g.out(normed2), name: "A" }, { tensor: T.up_weight, name: "B" }],
    [{ name: "out", type: { shape: [1, INTER], dtype: "f32", strides: [INTER, 1] } }],
    [1, Math.ceil(INTER / 64), 1], { M: 1, N: INTER, K: H });
  // 10. swiglu (f32→bf16) — grid [4]
  const act = g.node("swiglu",
    [{ tensor: g.out(gate), name: "g" }, { tensor: g.out(up), name: "u" }],
    [{ name: "out", type: { shape: [1, INTER], dtype: "bf16", strides: [INTER, 1] } }],
    [Math.ceil(INTER / 1024), 1, 1], { N: INTER });
  // 11. down_proj GEMM (f32) — grid [1,16]
  const downF = g.node("gemm",
    [{ tensor: g.out(act), name: "A" }, { tensor: T.down_weight, name: "B" }],
    [{ name: "out", type: { shape: [1, H], dtype: "f32", strides: [H, 1] } }],
    [1, Math.ceil(H / 64), 1], { M: 1, N: H, K: INTER });
  // 12. cast down → mlpBf — grid [1,16]
  const mlpBf = g.node("cast",
    [{ tensor: g.out(downF), name: "x" }],
    [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
    [1, Math.ceil(H / 64), 1], { N: H, from: "f32", to: "bf16" });
  // 13. add residual → xNew — grid [1,16]
  const xNew = g.node("add",
    [{ tensor: g.out(mlpBf), name: "a" }, { tensor: g.out(afterAttn), name: "b" }],
    [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
    [1, Math.ceil(H / 64), 1], { N: H });

  console.log(`graph: ${g.nodes.length} nodes`);
  for (const n of g.nodes) console.log(`  n${n.id} ${n.op} grid=[${n.grid}] ${n.outputs[0].tensorName}:${n.outputs[0].type.dtype}`);

  const env: RunEnv = {
    resolve: () => base, // every arg resolves to a valid device buffer; we measure TIME only
  };

  console.log("\n=== Beam exploration (real GPU latency, merge + split) ===");
  const { best, candidates } = explore(g, env, { beam: 3, budget: 40, moves: ["merge", "split"], verbose: true });

  console.log("\n=== best partition (fewer kernels = more fusion wins if faster) ===");
  for (const group of best.partition) {
    console.log(`  kernel: ${group.map(n => `${n.id}:${n.op}`).join(" + ")}`);
  }
  console.log(`best latency: ${best.latencyMs?.toFixed(3) ?? "?"} ms, ${best.partition.length} kernels`);
  console.log(`candidates explored (valid): ${candidates.filter(c => c.latencyMs !== null).length}/${candidates.length}`);
}
await main();
