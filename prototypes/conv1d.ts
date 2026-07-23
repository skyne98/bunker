// prototypes/conv1d.ts — depthwise causal conv1d (k=4) + silu for GatedDeltaNet (f16).
//
// Ported from fla's `causal_conv1d_fwd_kernel`. Qwen3.5's GatedDeltaNet applies
// a depthwise causal conv1d (kernel=4) + silu to the qkv projection before
// splitting into q/k/v. f16 in/out (model dtype), f32 accumulate/silu.
//
//   y[t,c] = silu( Σ_{j=0..W-1} weight[c,j] · x[t-(W-1)+j, c] )     (x[n<0]=0)
//
// Process BT output positions at once, summing W shifted input windows ×
// per-channel tap weights (strided pointer-tensor loads). Host-pads W-1 leading
// zeros per batch row so all offsets are non-negative (no cross-batch bleed).
//
//   bun run prototypes/conv1d.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";
import { f32to16 } from "../src/kernel";

const W = 4;

function buildConv1d(B: number, T: number, D: number, BT: number, BD: number) {
  const Tp = T + (W - 1);
  const b = new TTIRBuilder();
  const X = b.param("X", { ptr: "f16" }), WT = b.param("W", { ptr: "f16" }), Y = b.param("Y", { ptr: "f16" });
  const iD = b.programId(0), iT = b.programId(1), iB = b.programId(2);
  const tBase = b.add(b.mul(iB, b.i32(Tp)), b.mul(iT, b.i32(BT)));
  const yBase = b.add(b.mul(iB, b.i32(T)), b.mul(iT, b.i32(BT)));
  const dBase = b.mul(iD, b.i32(BD));
  const tpY = b.makeTensorPtr(Y, [B * T, D], [D, 1], [yBase, dBase], [BT, BD], "f16", [1, 0]);
  let acc = b.zeros([BT, BD], "f32");
  for (let tap = 0; tap < W; tap++) {
    const tOff = b.add(tBase, b.i32(tap));
    const tpX = b.makeTensorPtr(X, [B * Tp, D], [D, 1], [tOff, dBase], [BT, BD], "f16", [1, 0]);
    const x = b.fpext(b.load(tpX, { boundaryCheck: [0, 1], padding: 1 }), "f32");
    const wOff = b.add(b.mul(dBase, b.i32(W)), b.i32(tap));
    const w = b.fpext(b.load(b.addptr(b.splatPtr(WT, BD, "f16"), b.add(b.mul(b.arange(0, BD), b.i32(W)), wOff))), "f32");
    acc = b.add(acc, b.mul(x, b.broadcast(b.expandDims(w, 0), [BT, BD])));
  }
  const sig = b.divf(b.f32(1), b.add(b.f32(1), b.exp(b.mul(acc, b.f32(-1)))));
  b.store(tpY, b.fptrunc(b.mul(acc, sig), "f16"), { boundaryCheck: [0, 1] });
  return b.build("conv1d", 4);
}

function f16to32(a: Uint16Array): Float32Array {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const u = a[i], sign = (u >> 15) & 1, exp = (u >> 10) & 0x1f, frac = u & 0x3ff;
    let f: number;
    if (exp === 0) f = 0; else if (exp === 31) f = frac ? NaN : Infinity;
    else f = Math.pow(2, exp - 15) * (1 + frac / 1024);
    o[i] = sign ? -f : f;
  }
  return o;
}
function refConv1d(Xf: Float32Array, Wf: Float32Array, B: number, T: number, D: number) {
  const Y = new Float32Array(B * T * D);
  for (let bi = 0; bi < B; bi++) for (let t = 0; t < T; t++) for (let c = 0; c < D; c++) {
    let s = 0;
    for (let j = 0; j < W; j++) { const tt = t - (W - 1) + j; if (tt >= 0) s += Wf[c * W + j] * Xf[(bi * T + tt) * D + c]; }
    Y[(bi * T + t) * D + c] = s / (1 + Math.exp(-s));
  }
  return Y;
}

const B = 8, T = 512, D = 6144, BT = 32, BD = 128;   // Qwen3.5 conv_dim=6144
const k = compileAndLoad(buildConv1d(B, T, D, BT, BD), "conv1d", 4);
console.log(`conv1d loaded (f16, B=${B} T=${T} D=${D} W=${W})`);

const Xf = new Float32Array(B * T * D).map(() => (Math.random() * 2 - 1) * 0.5);
const Wf = new Float32Array(D * W).map(() => (Math.random() * 2 - 1) * 0.3);
const Xp = f32to16(new Float32Array(B * (T + W - 1) * D));   // padded, zeros
for (let bi = 0; bi < B; bi++) { const src = f32to16(Xf.slice(bi * T * D, (bi + 1) * T * D)); Xp.set(src, bi * (T + W - 1) * D + (W - 1) * D); }
const Wt = f32to16(Wf), Y = new Uint16Array(B * T * D);
const dX = cuAlloc(Xp.byteLength), dW = cuAlloc(Wt.byteLength), dY = cuAlloc(Y.byteLength);
cuHtoD(dX, Xp.buffer); cuHtoD(dW, Wt.buffer);

// correctness
cuLaunch(k, [Math.ceil(D / BD), Math.ceil(T / BT), B], [128, 1, 1], [dX, dW, dY]); cuSync(); cuDtoH(Y.buffer, dY);
const ref = refConv1d(Xf, Wf, B, T, D);
const Yf = f16to32(Y);
let maxErr = 0, ok = true;
for (let i = 0; i < Yf.length; i++) { const d = Math.abs(Yf[i] - ref[i]); if (d > maxErr) maxErr = d; if (d > 0.05 && ok) { ok = false; console.log(`mismatch [${i}] got ${Yf[i].toFixed(4)} ref ${ref[i].toFixed(4)}`); } }
console.log(ok ? `✓ conv1d correct (max err ${maxErr.toExponential(2)}, f16)` : "✗ FAILED");

// bench (host-timed on the ttir context)
for (let i = 0; i < 3; i++) cuLaunch(k, [Math.ceil(D / BD), Math.ceil(T / BT), B], [128, 1, 1], [dX, dW, dY]);
cuSync();
const ITERS = 50, t0 = performance.now();
for (let i = 0; i < ITERS; i++) cuLaunch(k, [Math.ceil(D / BD), Math.ceil(T / BT), B], [128, 1, 1], [dX, dW, dY]);
cuSync();
const dt = (performance.now() - t0) / 1000 / ITERS;
const bytes = (B * T * D + B * T * D + D * W) * 2;   // f16 read X + write Y + read W
console.log(`bench ${B}x${T}x${D}: ${(dt * 1e6).toFixed(1)} µs/it, ${(bytes / dt / 1e9).toFixed(0)} GB/s effective mem-BW`);
cuFree(dX); cuFree(dW); cuFree(dY);
