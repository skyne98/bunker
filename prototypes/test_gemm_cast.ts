// prototypes/test_gemm_cast.ts — does a cast:true (bf16 output) gemm work?
// Reproduce exactly the graph's qkv gemm (M=1, N=QKVD, K=H, cast) standalone.
import { D } from "../src/model";
import { Graph } from "../src/fusion";
import { compilePartition, loadPartition } from "../src/fusion";
import { cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch } from "../src/ttir";

const H = D.H, QKVD = D.QKVD;
const bf16f = (u16: number) => { const b = new ArrayBuffer(4); const u = new Uint8Array(b); const f = new Float32Array(b); u[2] = u16 & 0xFF; u[3] = u16 >> 8; return f[0]; };
const f2bf = (f: number) => { const b = new ArrayBuffer(4); const u = new Uint8Array(b); const f32 = new Float32Array(b); f32[0] = f; return u[2] | (u[3] << 8); };

async function main() {
  const stPath = process.env.SAFETENSORS_PATH || "/tmp/qwen35_0.8b.safetensors";
  const data = await Bun.file(stPath).bytes();
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const hl = Number(dv.getBigUint64(0, true));
  const hdr = JSON.parse(new TextDecoder().decode(data.subarray(8, 8 + hl)));
  const ds = 8 + hl;
  const base = cuAlloc(BigInt(data.length - ds));
  cuHtoD(base, data.subarray(ds)); cuSync();
  const W = new Map<string, bigint>();
  for (const [k, v] of Object.entries(hdr)) if (k !== "__metadata__") W.set(k, base + BigInt((v as any).data_offsets[0]));

  const M = 1, N = QKVD, K = H;
  const wr = W.get("model.language_model.layers.0.linear_attn.in_proj_qkv.weight");

  const aHost = new Uint16Array(K);
  for (let i = 0; i < K; i++) aHost[i] = f2bf(Math.sin(i * 0.1) * 0.1);
  const aBuf = cuAlloc(BigInt(K * 2));
  cuHtoD(aBuf, aHost.buffer); cuSync();

  // non-cast (f32 out) — the control
  const gControl = new Graph();
  gControl.input("A", { shape: [M, K], dtype: "bf16", strides: [K, 1] }, "data");
  gControl.input("B", { shape: [K, N], dtype: "bf16", strides: [1, K] }, "weight");
  gControl.input("C", { shape: [M, N], dtype: "f32", strides: [N, 1] }, "data");
  const nControl = gControl.node("gemm",
    [{ tensor: gControl.t("A"), name: "A" }, { tensor: gControl.t("B"), name: "B" }],
    [{ name: "out", type: { shape: [M, N], dtype: "f32", strides: [N, 1] } }],
    [1, Math.ceil(N / 64), 1], { M, N, K });
  const pc = compilePartition(gControl, [[nControl]]);
  const lc = loadPartition(pc);
  const yC = cuAlloc(BigInt(N * 4));
  cuLaunch(lc[0].k, [1, Math.ceil(N / 64), 1], [128, 1, 1], [aBuf, wr, yC]);
  cuSync();
  const fc = new Float32Array(8);
  cuDtoH(fc.buffer, yC, BigInt(32));
  console.log(`control (f32 out) qkv[0..7] = ${[...fc].map(v => v.toFixed(4)).join(", ")}`);

  // cast:true (bf16 out) — the graph path
  const gCast = new Graph();
  gCast.input("A", { shape: [M, K], dtype: "bf16", strides: [K, 1] }, "data");
  gCast.input("B", { shape: [K, N], dtype: "bf16", strides: [1, K] }, "weight");
  gCast.input("C", { shape: [M, N], dtype: "bf16", strides: [N, 1] }, "data");
  const nCast = gCast.node("gemm",
    [{ tensor: gCast.t("A"), name: "A" }, { tensor: gCast.t("B"), name: "B" }],
    [{ name: "out", type: { shape: [M, N], dtype: "bf16", strides: [N, 1] } }],
    [1, Math.ceil(N / 64), 1], { M, N, K, cast: true });
  const pCast = compilePartition(gCast, [[nCast]]);
  const lCast = loadPartition(pCast);
  const yB = cuAlloc(BigInt(N * 2));
  cuLaunch(lCast[0].k, [1, Math.ceil(N / 64), 1], [128, 1, 1], [aBuf, wr, yB]);
  cuSync();
  const fb = new Uint16Array(8);
  cuDtoH(fb.buffer, yB, BigInt(16));
  console.log(`cast (bf16 out) qkv[0..7] = ${[...fb].map(v => bf16f(v).toFixed(4)).join(", ")}`);

  // independent CPU check: qkv[0] = sum_k A[k]*W[0*K+k] (W is [N,K] with stride 1 inner)
  const ek = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
  const wOff = (hdr[ek] as any).data_offsets[0];
  const bf2 = (o: number) => data[ds + o] | (data[ds + o + 1] << 8);
  let c0 = 0;
  for (let k = 0; k < K; k++) c0 += bf16f(aHost[k]) * bf16f(bf2(wOff + (0 * K + k) * 2));
  console.log(`CPU qkv[0] ≈ ${c0.toFixed(4)}`);

  // DECISIVE: run the emitted cast-gemm kernel but with A = the rmsnorm-produced
  // n1_out buffer from a real graph compile+launch. If nonzero → kernel fine,
  // A data fine → bug is elsewhere (naming/aliasing).
  const { buildModelGraph } = await import("../src/model");
  const { compileStep, choosePartition } = await import("../src/runner");
  const graph2 = buildModelGraph();
  const compiled2 = compileStep(graph2, choosePartition(graph2, () => 0n, { optimize: false }));
  const sched2 = compiled2.schedule;
  const scratch2 = new Map<string, bigint>();
  const resolve2 = (name: string): bigint | number => {
    const t = graph2.t(name);
    if (t.role === "scalar") throw new Error("scalar");
    if (name === "rope_cos" || name === "rope_sin") { const b = cuAlloc(BigInt(D.MAX_LEN * D.ROT_HALF * 4)); scratch2.set(name, b); return b; }
    const k = name.includes(".weight") ? ((W.has(name) ? name : "model.language_model." + name)) : null;
    if (k && W.has(k)) return W.get(k)!;
    if (!scratch2.has(name)) { const bytes = t.type.shape.reduce((a,b)=>a*b,1) * (t.type.dtype==="bf16"?2:4); const b = cuAlloc(BigInt(bytes)); scratch2.set(name, b); return b; }
    return scratch2.get(name)!;
  };
  const args2 = sched2.map(s => s.plan.args.map(a => a === "token_id" ? 9419 : a === "pos" ? 0 : resolve2(a)));
  for (let i = 0; i < 3; i++) {
    const { k, plan } = sched2[i];
    cuLaunch(k, [plan.grid[0], plan.grid[1], plan.grid[2]], [128, 1, 1], args2[i]);
  }
  cuSync();
  const qkv = graph2.nodes.find(n => n.op === "gemm" && n.inputs.some(i => i.tensorName.includes("in_proj_qkv")))!;
  const qv = qkv.outputs[0].tensorName;
  const qp = resolve2(qv) as bigint;
  const qb = new Uint16Array(8);
  cuDtoH(qb.buffer, qp, BigInt(16));
  console.log(`graph-path (3 kernels) qkv[0..7] = ${[...qb].map(v => bf16f(v).toFixed(4)).join(", ")}`);
  const a0 = qkv.inputs[0].tensorName;
  const ap = resolve2(a0) as bigint;
  const ab = new Uint16Array(8);
  cuDtoH(ab.buffer, ap, BigInt(16));
  console.log(`  A(=n1_out)[0..7] = ${[...ab].map(v => bf16f(v).toFixed(4)).join(", ")}`);
}

await main();
