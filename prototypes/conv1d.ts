// prototypes/conv1d.ts — depthwise causal conv1d (k=4) + silu for GatedDeltaNet.
//
// Ported from fla's `causal_conv1d_fwd_kernel` (the Triton reference). Qwen3.5's
// GatedDeltaNet applies a depthwise causal conv1d (kernel=4) + silu to the qkv
// projection before splitting into q/k/v. This prototype is that conv:
//
//   y[t, c] = silu( Σ_{j=0..W-1} weight[c, j] · x[t-(W-1)+j, c] )     (x[n<0]=0)
//
// Process a block of BT output positions at once, summing W shifted input
// windows (boundary_check zero-pads t<0). Per-channel tap weights loaded via
// strided pointer-tensors (weight is [D, W] row-major → column = stride W).
// silu uses exp (inlines) → no libdevice.
//
//   bun run prototypes/conv1d.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const W = 4;

function buildConv1d(B: number, T: number, D: number, BT: number, BD: number) {
  const Tp = T + (W - 1);   // host-padded time dim (W-1 leading zeros per batch row)
  const b = new TTIRBuilder();
  const X = b.param("X", { ptr: "f32" });        // [B*Tp, D]  (padded)
  const WT = b.param("W", { ptr: "f32" });       // [D, W] depthwise weights
  const Y = b.param("Y", { ptr: "f32" });        // [B*T, D]  (unpadded output)
  const iD = b.programId(0), iT = b.programId(1), iB = b.programId(2);
  const tBase = b.add(b.mul(iB, b.i32(Tp)), b.mul(iT, b.i32(BT)));   // padded input base
  const yBase = b.add(b.mul(iB, b.i32(T)), b.mul(iT, b.i32(BT)));   // output base
  const dBase = b.mul(iD, b.i32(BD));

  const tpY = b.makeTensorPtr(Y, [B * T, D], [D, 1], [yBase, dBase], [BT, BD], "f32", [1, 0]);
  let acc = b.zeros([BT, BD], "f32");
  for (let tap = 0; tap < W; tap++) {
    const tOff = b.add(tBase, b.i32(tap));                          // padded[t..t+W-1], all >= 0
    const tpX = b.makeTensorPtr(X, [B * Tp, D], [D, 1], [tOff, dBase], [BT, BD], "f32", [1, 0]);
    const x = b.load(tpX, { boundaryCheck: [0, 1], padding: 1 });  // [BT, BD]
    const wOff = b.add(b.mul(dBase, b.i32(W)), b.i32(tap));
    const wAr = b.mul(b.arange(0, BD), b.i32(W));
    const w = b.load(b.addptr(b.splatPtr(WT, BD, "f32"), b.add(wAr, wOff)));  // [BD]
    const wBc = b.broadcast(b.expandDims(w, 0), [BT, BD]);
    acc = b.add(acc, b.mul(x, wBc));
  }
  const sig = b.divf(b.f32(1), b.add(b.f32(1), b.exp(b.mul(acc, b.f32(-1)))));
  b.store(tpY, b.mul(acc, sig), { boundaryCheck: [0, 1] });
  return b.build("conv1d", 4);
}

function refConv1d(X: Float32Array, Wt: Float32Array, B: number, T: number, D: number) {
  const Y = new Float32Array(B * T * D);
  for (let bi = 0; bi < B; bi++) for (let t = 0; t < T; t++) for (let c = 0; c < D; c++) {
    let s = 0;
    for (let j = 0; j < W; j++) { const tt = t - (W - 1) + j; if (tt >= 0) s += Wt[c * W + j] * X[(bi * T + tt) * D + c]; }
    Y[(bi * T + t) * D + c] = s / (1 + Math.exp(-s));   // silu
  }
  return Y;
}

const B = 4, T = 128, D = 512, BT = 64, BD = 128;
const k = compileAndLoad(buildConv1d(B, T, D, BT, BD), "conv1d", 4);
console.log(`conv1d loaded (shmem=${k.shmem}, B=${B} T=${T} D=${D} W=${W})`);

const X = new Float32Array(B * T * D).map(() => (Math.random() * 2 - 1) * 0.5);
// host-pad: W-1 leading zeros per batch row → [B*(T+W-1)*D]
const Xp = new Float32Array(B * (T + W - 1) * D);
for (let bi = 0; bi < B; bi++) Xp.set(X.slice(bi * T * D, (bi + 1) * T * D), bi * (T + W - 1) * D + (W - 1) * D);
const Wt = new Float32Array(D * W).map(() => (Math.random() * 2 - 1) * 0.3);
const Y = new Float32Array(B * T * D);
const dX = cuAlloc(Xp.byteLength), dW = cuAlloc(Wt.byteLength), dY = cuAlloc(Y.byteLength);
cuHtoD(dX, Xp.buffer); cuHtoD(dW, Wt.buffer);
for (let i = 0; i < 3; i++) cuLaunch(k, [Math.ceil(D / BD), Math.ceil(T / BT), B], [128, 1, 1], [dX, dW, dY]);
cuSync();
const t0 = performance.now(), it = 100;
for (let i = 0; i < it; i++) cuLaunch(k, [Math.ceil(D / BD), Math.ceil(T / BT), B], [128, 1, 1], [dX, dW, dY]);
cuSync();
const dt = (performance.now() - t0) / 1000 / it;
cuDtoH(Y.buffer, dY);

const ref = refConv1d(X, Wt, B, T, D);
let maxErr = 0, ok = true;
for (let i = 0; i < Y.length; i++) { const d = Math.abs(Y[i] - ref[i]); if (d > maxErr) maxErr = d; if (d > 1e-4 && ok) { ok = false; console.log(`mismatch [${i}] got ${Y[i]} ref ${ref[i]}`); } }
console.log(ok ? `✓ conv1d correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");
console.log(`bench ${B}x${T}x${D}: ${(dt * 1e6).toFixed(1)} µs/it, ${((2 * B * T * D + D * W) * 4 / dt / 1e9).toFixed(0)} GB/s effective mem-BW`);
cuFree(dX); cuFree(dW); cuFree(dY);
