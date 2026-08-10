// prototypes/test_embed.ts — embed emitter bit-accuracy vs raw weight row.
import { D } from "../src/model";
const H = D.H;
import { Graph } from "../src/fusion";
import { compilePartition, loadPartition } from "../src/fusion";
import { cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch } from "../src/ttir";
import { TTIRBuilder } from "../src/ttir";

const bf16f = (u16: number) => { const b = new ArrayBuffer(4); const u = new Uint8Array(b); const f = new Float32Array(b); u[2] = u16 & 0xFF; u[3] = u16 >> 8; return f[0]; };

// decode's buildEmbed (verbatim)
function buildEmbed() {
  const b = new TTIRBuilder();
  const E = b.param("E", { ptr: "bf16" });
  const X = b.param("X", { ptr: "bf16" });
  const ID = b.param("ID", "i32");
  const Hh = 1024;
  const tpE = b.makeTensorPtr(E, [248320, Hh], [Hh, 1], [ID, b.i32(0)], [1, Hh], "bf16", [1, 0]);
  const tpX = b.makeTensorPtr(X, [1, Hh], [Hh, 1], [b.i32(0), b.i32(0)], [1, Hh], "bf16", [1, 0]);
  b.store(tpX, b.load(tpE, {}), {});
  return b.build("emb", 4, 3);
}

function graphEmbed(): { ttir: string; name: string; args: string[]; grid: [number, number, number] } {
  const g = new Graph();
  g.input("E", { shape: [D.VOCAB, H], dtype: "bf16", strides: [H, 1] }, "weight");
  g.input("id", { shape: [], dtype: "i32", strides: [] }, "scalar");
  g.input("out", { shape: [1, H], dtype: "bf16", strides: [H, 1] }, "data");
  const n = g.node("embed",
    [{ tensor: g.t("E"), name: "E" }, { tensor: g.t("id"), name: "id" }],
    [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }], [1, 1, 1], { H });
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

  // fake GPU weight base is not needed; read row directly from host bytes
  const ek = "model.language_model.embed_tokens.weight";
  const eo = (hdr[ek] as any).data_offsets[0];
  const rowOff = ds + eo + 9419 * H * 2;
  const row = new Uint16Array(H);
  for (let i = 0; i < H; i++) row[i] = data[rowOff + i * 2] | (data[rowOff + i * 2 + 1] << 8);

  // expected
  let expMax = 0;
  for (let i = 0; i < H; i++) expMax = Math.max(expMax, Math.abs(bf16f(row[i])));
  console.log(`expected embed[9419] max|.|=${expMax.toFixed(4)}`);

  // graph emit + run (identity weight ptr = host? no — need GPU. Upload just the row region to a tiny buf)
  // For the test, treat the row itself as GPU data.
  const rowBuf = cuAlloc(BigInt(H * 2));
  cuHtoD(rowBuf, row.buffer); cuSync();

  const gplan = graphEmbed();
  // override: the graph's E input is [VOCAB,H]; we pass rowBuf as if it were the base (id=0)
  const gloaded = loadPartition([gplan]);
  const outG = cuAlloc(BigInt(H * 2));
  cuLaunch(gloaded[0].k, [1, 1, 1], [128, 1, 1], [rowBuf, outG, 0]);
  cuSync();

  const fullG = new Uint16Array(H);
  cuDtoH(fullG.buffer, outG, BigInt(H * 2));
  let bad = 0, maxD = 0;
  for (let i = 0; i < H; i++) { if (fullG[i] !== row[i]) { bad++; maxD = Math.max(maxD, Math.abs(bf16f(fullG[i]) - bf16f(row[i]))); } }
  console.log(`embed(id=0): ${bad}/${H} mismatches (max diff ${maxD.toExponential(2)})`);
  let gMax = 0;
  for (let i = 0; i < H; i++) gMax = Math.max(gMax, Math.abs(bf16f(fullG[i])));
  console.log(`  graph out max|.|=${gMax.toFixed(4)}  (we passed id=0, so compare against embed[0] magnitude not 9419)`);
}

await main();
