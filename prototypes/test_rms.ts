// prototypes/test_rms.ts — RMSNorm bit-accuracy: graph emitter vs decode kRms.
import { D } from "../src/model";
import { Graph } from "../src/fusion";
import { compilePartition, loadPartition } from "../src/fusion";
import { compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch } from "../src/ttir";
import { TTIRBuilder } from "../src/ttir";

const H = D.H;
const bf16f = (u16: number) => { const b = new ArrayBuffer(4); const u = new Uint8Array(b); const f = new Float32Array(b); u[2] = u16 & 0xFF; u[3] = u16 >> 8; return f[0]; };

// decode's buildRMS (verbatim from decode.ts)
function buildRMS(N: number) {
  const b = new TTIRBuilder();
  const X = b.param("X", { ptr: "bf16" }), Wt = b.param("W", { ptr: "bf16" }), Y = b.param("Y", { ptr: "bf16" });
  const row = b.programId(0);
  const tpX = b.makeTensorPtr(X, [1, N], [N, 1], [row, b.i32(0)], [1, N], "bf16", [1, 0]);
  const tpW = b.makeTensorPtr(Wt, [1, N], [N, 1], [b.i32(0), b.i32(0)], [1, N], "bf16", [1, 0]);
  const tpY = b.makeTensorPtr(Y, [1, N], [N, 1], [row, b.i32(0)], [1, N], "bf16", [1, 0]);
  const x = b.fpext(b.load(tpX, { boundaryCheck: [0, 1], padding: 1 }), "f32");
  const ms = b.divf(b.sum(b.mul(x, x), 1), b.f32(N));
  const msBc = b.broadcast(b.expandDims(ms, 1), [1, N]);
  const yy = b.rsqrtHw(b.add(msBc, b.f32(1e-6)));
  let y = b.mul(x, yy);
  y = b.mul(y, b.add(b.f32(1), b.fpext(b.load(tpW, { boundaryCheck: [0, 1], padding: 1 }), "f32")));
  b.store(tpY, b.fptrunc(y, "bf16"), { boundaryCheck: [0, 1] });
  return b.build("rms", 4, 3);
}

function graphRMS(N: number): { ttir: string; name: string; args: string[]; grid: [number, number, number] } {
  const g = new Graph();
  g.input("x", { shape: [1, N], dtype: "bf16", strides: [N, 1] }, "data");
  g.input("w", { shape: [N], dtype: "bf16", strides: [1] }, "weight");
  g.input("y", { shape: [1, N], dtype: "bf16", strides: [N, 1] }, "data");
  const n = g.node("rmsnorm",
    [{ tensor: g.t("x"), name: "x" }, { tensor: g.t("w"), name: "w" }],
    [{ name: "out", type: { shape: [1, N], dtype: "bf16", strides: [N, 1] } }], [1, 1, 1], { N });
  const plans = compilePartition(g, [[n]]);
  return plans[0];
}

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
  const wn = W.get("model.language_model.layers.0.input_layernorm.weight")!;

  const xHost = new Uint16Array(H);
  for (let i = 0; i < H; i++) xHost[i] = (i * 2654435761) >>> 16;
  const xBuf = cuAlloc(BigInt(H * 2));
  cuHtoD(xBuf, xHost.buffer); cuSync();

  // decode path
  const kRef = compileAndLoad(buildRMS(H), "rms", 4);
  // graph path
  const gplan = graphRMS(H);
  const gloaded = loadPartition([gplan]);
  const gk = gloaded[0].k;

  const yRef = cuAlloc(BigInt(H * 2));
  const yG = cuAlloc(BigInt(H * 2));
  cuLaunch(kRef, [1, 1, 1], [128, 1, 1], [xBuf, wn, yRef]);
  cuSync();
  cuLaunch(gk, [gplan.grid[0], gplan.grid[1], gplan.grid[2]], [128, 1, 1], [xBuf, wn, yG]);
  cuSync();

  const fullR = new Uint16Array(H), fullG = new Uint16Array(H);
  cuDtoH(fullR.buffer, yRef, BigInt(H * 2));
  cuDtoH(fullG.buffer, yG, BigInt(H * 2));
  let bad = 0, maxD = 0;
  for (let i = 0; i < H; i++) if (fullR[i] !== fullG[i]) { bad++; const d = Math.abs(bf16f(fullR[i]) - bf16f(fullG[i])); if (d > maxD) maxD = d; }
  console.log(`RMSNorm: ${bad}/${H} mismatches (max diff ${maxD.toExponential(2)})`);
  console.log(`  ref[0..7]=${[...fullR.slice(0, 8)].map(v => bf16f(v).toFixed(4)).join(",")}`);
  console.log(`  gph[0..7]=${[...fullG.slice(0, 8)].map(v => bf16f(v).toFixed(4)).join(",")}`);
}

await main();
