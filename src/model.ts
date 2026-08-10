// src/model.ts — whole-model declaration for Qwen3.5-0.8B decode, as a graph.
//
// The model is declared DYNAMICALLY as a Graph: every compute step is a node
// (embed, rmsnorm, gemm, conv1d, gdn delta rule, qnorm, knorm, rope, ropek,
// fa2 attention, swiglu, argmax) and every intermediate is a tensor with a
// role. State tensors (conv state, GDN S-state, KV cache) are role="state",
// double-buffered across tokens exactly like decode.ts.
//
// The fusion explorer (src/fusion.ts) partitions these nodes into kernels and
// measures GPU latency to choose the fastest fusion — NO hardcoded kernel
// layout here. The graph + emitters (src/emitters.ts) are the source of truth.
import { Graph } from "./fusion";

export const D = {
  H: 1024, VOCAB: 248320, NL: 24, FAI: 4, INTER: 3584, QKVD: 6144, ZD: 2048, EPS: 1e-6,
  NH: 8, NKV: 2, HD: 256, LKH: 16, LVH: 16, LKD: 128, LVD: 128, MAX_LEN: 128,
  QGATE: 4096, KV_DIM: 512, ROT_DIM: 64, ROT_HALF: 32, CHUNK: 4096,
};
const { H, VOCAB, NL, FAI, INTER, QKVD, ZD, NH, NKV, HD, MAX_LEN, LVH, LVD } = D;

/** row-major bf16/f32 tensor type */
function w(shape: number[]): { shape: number[]; dtype: "bf16"; strides: number[] } {
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  return { shape, dtype: "bf16", strides };
}
function f32sh(shape: number[]): { shape: number[]; dtype: "f32"; strides: number[] } {
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  return { shape, dtype: "f32", strides };
}
function i32sh(shape: number[]): { shape: number[]; dtype: "i32"; strides: number[] } {
  const strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  return { shape, dtype: "i32", strides };
}

/** Number of grid.y tiles for a GEMM of width N (tile 64), or full 1D span. */
const gy = (N: number) => Math.ceil(N / 64);

/**
 * Build the full Qwen3.5-0.8B decode graph for ONE token step.
 * Externals include per-layer weights AND per-layer state-in buffers
 * ("conv_state_in"/"s_state_in"/"kv_k_in"/"kv_v_in"). Outputs end at
 * "argmax idx"/"argmax val". State "new" outputs are the same names the
 * runtime will swap (the graph declares them as distinct tensors; the runtime
 * passes base pointers for in and a double-buffer for *_new).
 */
export function buildModelGraph(): Graph {
  const g = new Graph();
  const Tk = g.input("token_id", { shape: [], dtype: "i32", strides: [] }, "scalar");
  const Pos = g.input("pos", { shape: [], dtype: "i32", strides: [] }, "scalar");
  const embed_w = g.input("embed.weight", w([VOCAB, H]), "weight");

  // ── Embedding ──
  const emb = g.node("embed",
    [{ tensor: embed_w, name: "E" }, { tensor: Tk, name: "id" }],
    [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
    [1, 1, 1], { H });

  let h = loc(g, emb); // current hidden-state tensor name

  for (let layer = 0; layer < NL; layer++) {
    const full = layer % FAI === FAI - 1;
    const pfx = full ? "self_attn" : "linear_attn";

    // input RMSNorm
    const normed = g.node("rmsnorm",
      [{ tensor: h, name: "x" }, { tensor: g.input(`${L(layer)}.input_layernorm.weight`, w([H]), "weight"), name: "w" }],
      [{ name: "out", type: w([1, H]) }], [1, 1, 1], { N: H });

    if (!full) {
      // GDN layer
      const qkv = g.node("gemm",
        [{ tensor: loc(g, normed), name: "A" }, { tensor: g.input(`${L(layer)}.linear_attn.in_proj_qkv.weight`, w([QKVD, H]), "weight"), name: "B" }],
        [{ name: "out", type: w([1, QKVD]) }], [1, gy(QKVD), 1], { M: 1, N: QKVD, K: H, cast: true });
      const zG = g.node("gemm",
        [{ tensor: loc(g, normed), name: "A" }, { tensor: g.input(`${L(layer)}.linear_attn.in_proj_z.weight`, w([ZD, H]), "weight"), name: "B" }],
        [{ name: "out", type: w([1, ZD]) }], [1, gy(ZD), 1], { M: 1, N: ZD, K: H, cast: true });
      const convOut = g.node("conv1d_decode",
        [
          { tensor: loc(g, qkv), name: "qkv" },
          { tensor: g.input(`${L(layer)}.conv_state_in`, { shape: [3, QKVD], dtype: "bf16", strides: [QKVD, 1] }, "state"), name: "state" },
          { tensor: g.input(`${L(layer)}.linear_attn.conv1d.weight`, w([QKVD, 4]), "weight"), name: "w" },
        ],
        [
          { name: "out", type: w([1, QKVD]) },
          { name: `${L(layer)}.conv_state_new`, type: { shape: [3, QKVD], dtype: "bf16", strides: [QKVD, 1] }, role: "state" },
        ],
        [Math.ceil(QKVD / 1024), 1, 1], { QKVD });
      const aP = g.node("gemm",
        [{ tensor: loc(g, normed), name: "A" }, { tensor: g.input(`${L(layer)}.linear_attn.in_proj_a.weight`, w([LVH, H]), "weight"), name: "B" }],
        [{ name: "out", type: f32sh([1, LVH]) }], [1, 1, 1], { M: 1, N: LVH, K: H });
      const bP = g.node("gemm",
        [{ tensor: loc(g, normed), name: "A" }, { tensor: g.input(`${L(layer)}.linear_attn.in_proj_b.weight`, w([LVH, H]), "weight"), name: "B" }],
        [{ name: "out", type: f32sh([1, LVH]) }], [1, 1, 1], { M: 1, N: LVH, K: H });
      const gdn = g.node("gdn_delta_rule",
        [
          { tensor: loc(g, convOut), name: "conv_out" },
          { tensor: loc(g, zG), name: "z" },
          { tensor: g.input(`${L(layer)}.linear_attn.A_log`, f32sh([LVH]), "weight"), name: "A_log" },
          { tensor: g.input(`${L(layer)}.linear_attn.dt_bias`, w([LVH]), "weight"), name: "dt_bias" },
          { tensor: loc(g, aP), name: "a_p" },
          { tensor: loc(g, bP), name: "b_p" },
          { tensor: g.input(`${L(layer)}.linear_attn.norm.weight`, f32sh([LVD]), "weight"), name: "norm_w" },
          { tensor: g.input(`${L(layer)}.s_state_in`, { shape: [LVH, D.LKD, LVD], dtype: "f32", strides: [D.LKD * LVD, LVD, 1] }, "state"), name: "s_state" },
        ],
        [
          { name: "out", type: w([1, ZD]) },
          { name: `${L(layer)}.s_state_new`, type: { shape: [LVH, D.LKD, LVD], dtype: "f32", strides: [D.LKD * LVD, LVD, 1] }, role: "state" },
        ],
        [LVH, 1, 1], { LVH, LKD: D.LKD, LVD, QKVD, KEYDIM: LVH * D.LKD, ZD });
      const outProj = g.node("gemm",
        [{ tensor: loc(g, gdn), name: "A" }, { tensor: g.input(`${L(layer)}.linear_attn.out_proj.weight`, w([H, ZD]), "weight"), name: "B" }],
        [{ name: "out", type: f32sh([1, H]) }], [1, gy(H), 1], { M: 1, N: H, K: ZD });
      const castO = g.node("cast",
        [{ tensor: loc(g, outProj), name: "x" }], [{ name: "out", type: w([1, H]) }], [1, gy(H), 1], { to: "bf16", N: H });
      const afterAttn = g.node("add",
        [{ tensor: loc(g, castO), name: "a" }, { tensor: h, name: "b" }], [{ name: "out", type: w([1, H]) }], [1, gy(H), 1], { N: H });
      h = loc(g, afterAttn);
    } else {
      // FA2 layer (matches decode: q_proj,qnorm,rope,k_proj,knorm,ropek, fa2_attn, o_proj,cast,add)
      const qp = g.node("gemm",
        [{ tensor: loc(g, normed), name: "A" }, { tensor: g.input(`${L(layer)}.self_attn.q_proj.weight`, w([D.QGATE, H]), "weight"), name: "B" }],
        [{ name: "out", type: w([1, D.QGATE]) }], [1, gy(D.QGATE), 1], { M: 1, N: D.QGATE, K: H, cast: true });
      const kp = g.node("gemm",
        [{ tensor: loc(g, normed), name: "A" }, { tensor: g.input(`${L(layer)}.self_attn.k_proj.weight`, w([D.KV_DIM, H]), "weight"), name: "B" }],
        [{ name: "out", type: w([1, D.KV_DIM]) }], [1, gy(D.KV_DIM), 1], { M: 1, N: D.KV_DIM, K: H, cast: true });
      const vp = g.node("gemm",
        [{ tensor: loc(g, normed), name: "A" }, { tensor: g.input(`${L(layer)}.self_attn.v_proj.weight`, w([D.KV_DIM, H]), "weight"), name: "B" }],
        [{ name: "out", type: w([1, D.KV_DIM]) }], [1, gy(D.KV_DIM), 1], { M: 1, N: D.KV_DIM, K: H, cast: true });
      const qNorm = g.node("qnorm",
        [{ tensor: loc(g, qp), name: "q" }, { tensor: g.input(`${L(layer)}.self_attn.q_norm.weight`, w([HD]), "weight"), name: "w" }],
        [{ name: "out", type: w([1, NH * HD]) }], [NH, 1, 1], { HD, NH });
      // q_buf is produced by qnorm, consumed by rope (in-place) and fa2_attn
      const ropeQ = g.node("rope",
        [{ tensor: loc(g, qNorm, "out"), name: "Q" }, { tensor: g.input(`rope_cos`, f32sh([MAX_LEN, D.ROT_HALF]), "weight"), name: "C" },
         { tensor: g.input(`rope_sin`, f32sh([MAX_LEN, D.ROT_HALF]), "weight"), name: "S" }, { tensor: Pos, name: "P" }],
        [], [NH, 1, 1], { HD, NH, ROT_HALF: D.ROT_HALF, MAX_LEN });
      const kNorm = g.node("knorm",
        [{ tensor: loc(g, kp), name: "k" }, { tensor: g.input(`${L(layer)}.self_attn.k_norm.weight`, w([HD]), "weight"), name: "w" }],
        [{ name: "out", type: w([1, D.KV_DIM]) }], [NKV, 1, 1], { HD, NKV });
      const ropeK = g.node("ropek",
        [{ tensor: loc(g, kNorm, "out"), name: "K" }, { tensor: g.input(`rope_cos`, f32sh([MAX_LEN, D.ROT_HALF]), "weight"), name: "C" },
         { tensor: g.input(`rope_sin`, f32sh([MAX_LEN, D.ROT_HALF]), "weight"), name: "S" }, { tensor: Pos, name: "P" }],
        [], [NKV, 1, 1], { HD, NKV, ROT_HALF: D.ROT_HALF, MAX_LEN });
      const fa2 = g.node("fa2_attn",
        [
          { tensor: loc(g, qNorm, "out"), name: "q" }, { tensor: loc(g, kNorm, "out"), name: "k" }, { tensor: loc(g, vp), name: "v" },
          { tensor: loc(g, qp), name: "qgate" },
          { tensor: g.input(`${L(layer)}.kv_k_in`, { shape: [NH * MAX_LEN, HD], dtype: "bf16", strides: [HD, 1] }, "state"), name: "kc_in" },
          { tensor: g.input(`${L(layer)}.kv_v_in`, { shape: [NH * MAX_LEN, HD], dtype: "bf16", strides: [HD, 1] }, "state"), name: "vc_in" },
          { tensor: Pos, name: "P" },
        ],
        [
          { name: "out", type: w([1, NH * HD]) },
          { name: `${L(layer)}.kv_k_new`, type: { shape: [NH * MAX_LEN, HD], dtype: "bf16", strides: [HD, 1] }, role: "state" },
          { name: `${L(layer)}.kv_v_new`, type: { shape: [NH * MAX_LEN, HD], dtype: "bf16", strides: [HD, 1] }, role: "state" },
        ],
        [NH, 1, 1], { HD, NH, MAX_LEN, NKV });
      const oProj = g.node("gemm",
        [{ tensor: loc(g, fa2), name: "A" }, { tensor: g.input(`${L(layer)}.self_attn.o_proj.weight`, w([H, NH * HD]), "weight"), name: "B" }],
        [{ name: "out", type: f32sh([1, H]) }], [1, gy(H), 1], { M: 1, N: H, K: NH * HD });
      const castO = g.node("cast",
        [{ tensor: loc(g, oProj), name: "x" }], [{ name: "out", type: w([1, H]) }], [1, gy(H), 1], { to: "bf16", N: H });
      const afterAttn = g.node("add",
        [{ tensor: loc(g, castO), name: "a" }, { tensor: h, name: "b" }], [{ name: "out", type: w([1, H]) }], [1, gy(H), 1], { N: H });
      h = loc(g, afterAttn);
    }

    // post-attn norm + MLP (identical both layer kinds)
    const normed2 = g.node("rmsnorm",
      [{ tensor: h, name: "x" }, { tensor: g.input(`${L(layer)}.post_attention_layernorm.weight`, w([H]), "weight"), name: "w" }],
      [{ name: "out", type: w([1, H]) }], [1, 1, 1], { N: H });
    const gate = g.node("gemm",
      [{ tensor: loc(g, normed2), name: "A" }, { tensor: g.input(`${L(layer)}.mlp.gate_proj.weight`, w([INTER, H]), "weight"), name: "B" }],
      [{ name: "out", type: f32sh([1, INTER]) }], [1, gy(INTER), 1], { M: 1, N: INTER, K: H });
    const up = g.node("gemm",
      [{ tensor: loc(g, normed2), name: "A" }, { tensor: g.input(`${L(layer)}.mlp.up_proj.weight`, w([INTER, H]), "weight"), name: "B" }],
      [{ name: "out", type: f32sh([1, INTER]) }], [1, gy(INTER), 1], { M: 1, N: INTER, K: H });
    const act = g.node("swiglu",
      [{ tensor: loc(g, gate), name: "g" }, { tensor: loc(g, up), name: "u" }],
      [{ name: "out", type: w([1, INTER]) }], [Math.ceil(INTER / 1024), 1, 1], { N: INTER });
    const down = g.node("gemm",
      [{ tensor: loc(g, act), name: "A" }, { tensor: g.input(`${L(layer)}.mlp.down_proj.weight`, w([H, INTER]), "weight"), name: "B" }],
      [{ name: "out", type: f32sh([1, H]) }], [1, gy(H), 1], { M: 1, N: H, K: INTER });
    const castD = g.node("cast",
      [{ tensor: loc(g, down), name: "x" }], [{ name: "out", type: w([1, H]) }], [1, gy(H), 1], { to: "bf16", N: H });
    const xNew = g.node("add",
      [{ tensor: h, name: "a" }, { tensor: loc(g, castD), name: "b" }], [{ name: "out", type: w([1, H]) }], [1, gy(H), 1], { N: H });
    h = loc(g, xNew);
  }

  // final norm + lm_head (tied embed) + argmax
  const fn = g.node("rmsnorm",
    [{ tensor: h, name: "x" }, { tensor: g.input(`final.norm.weight`, w([H]), "weight"), name: "w" }],
    [{ name: "out", type: w([1, H]) }], [1, 1, 1], { N: H });
  const logits = g.node("gemm",
    [{ tensor: loc(g, fn), name: "A" }, { tensor: g.input(`lm_head.weight`, w([VOCAB, H]), "weight"), name: "B" }],
    [{ name: "out", type: f32sh([1, VOCAB]) }], [1, gy(VOCAB), 1], { M: 1, N: VOCAB, K: H });
  const argmax = g.node("argmax",
    [{ tensor: loc(g, logits), name: "logits" }],
    [{ name: "val", type: { shape: [1], dtype: "f32", strides: [1] } }, { name: "idx", type: { shape: [1], dtype: "i32", strides: [1] } }],
    [1, 1, 1], { VOCAB });

  return g;
}

const L = (layer: number) => `model.layers.${layer}`;
/** locate a node's first "out" tensor name */
function loc(g: Graph, n: import("./fusion").GraphNode): string {
  return g.out(n, "out");
}
