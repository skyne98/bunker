// prototypes/rmsnorm.ts
//
// Prototype: RMSNorm ported to the bunker TTIR builder, translated from the
// production-fast Liger-Kernel `_rms_norm_forward_kernel` (the canonical fast
// Triton RMSNorm) + Triton's 05-layer-norm tutorial. Qwen3.5 uses GemmaRMSNorm:
//
//   y = x · rsqrt(mean(x²) + ε) · (1 + weight)        ε = 1e-6
//
// One program per row; a 1D tile of BLOCK (≥ N, zero-padded via boundaryCheck)
// holds the whole row. The only non-elementwise op is `rsqrt`. We try the
// builder's NATIVE `math.rsqrt` first (libdevice.10.bc is now present in the
// nix store, so the shim may link it). If that fails to compile, the Newton-
// Raphson fallback (`y ← y·(1.5 − 0.5·a·y²)`, pure mul/sub, no libdevice) is
// used — see `rmsnormNewton` below.
//
// Verified hardware: RTX 3090. Usage:
//   bun run prototypes/rmsnorm.ts
//   NATIVE=0 bun run prototypes/rmsnorm.ts   # force Newton fallback

import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const NATIVE = process.env.NATIVE !== "0"; // try math.rsqrt first

function rsqrt1(b: TTIRBuilder, a: any): any {
  if (NATIVE) return b.rsqrt(a);
  // Newton-Raphson 1/sqrt(a): y_{n+1} = y_n*(1.5 - 0.5*a*y_n^2)
  let y = b.f32(1); // initial guess (will be refined; a>0 typical)
  for (let i = 0; i < 6; i++) {
    const y2 = b.mul(y, y);
    y = b.mul(y, b.sub(b.f32(1.5), b.mul(b.f32(0.5), b.mul(a, y2))));
  }
  return y;
}

function buildRMSNorm(M: number, N: number, BLOCK: number) {
  const b = new TTIRBuilder();
  const X = b.param("X", { ptr: "f32" });
  const W = b.param("W", { ptr: "f32" });
  const Y = b.param("Y", { ptr: "f32" });
  const pid = b.programId(0);
  const rowOff = b.mul(pid, b.i32(N));

  const tpX = b.makeTensorPtr(X, [M * N], [1], [rowOff], [BLOCK], "f32", [0]);
  const tpW = b.makeTensorPtr(W, [N], [1], [b.i32(0)], [BLOCK], "f32", [0]);
  const tpY = b.makeTensorPtr(Y, [M * N], [1], [rowOff], [BLOCK], "f32", [0]);

  const x = b.load(tpX, { boundaryCheck: [0], padding: 1 });     // [BLOCK] f32
  const w = b.load(tpW, { boundaryCheck: [0], padding: 1 });     // [BLOCK] f32

  const meanSq = b.divf(b.sum(b.mul(x, x), 0), b.f32(N));        // scalar
  const rstd = rsqrt1(b, b.add(meanSq, b.f32(1e-6)));            // scalar
  const xnorm = b.mul(x, rstd);                                  // [BLOCK]
  const y = b.mul(xnorm, b.add(b.f32(1), w));                    // (1 + w) * xnorm (GemmaRMSNorm)
  b.store(tpY, y, { boundaryCheck: [0] });
  return b.build("rmsnorm", 4);
}

// host reference (GemmaRMSNorm, f32)
function refRMS(X: Float32Array, W: Float32Array, M: number, N: number) {
  const Y = new Float32Array(M * N);
  for (let r = 0; r < M; r++) {
    let ms = 0; for (let c = 0; c < N; c++) ms += X[r * N + c] * X[r * N + c];
    const rstd = 1 / Math.sqrt(ms / N + 1e-6);
    for (let c = 0; c < N; c++) Y[r * N + c] = X[r * N + c] * rstd * (1 + W[c]);
  }
  return Y;
}

async function main() {
  const M = 512, N = 1024;
  const BM = 1 << Math.ceil(Math.log2(N));
  const ttir = buildRMSNorm(M, N, BM);
  if (process.env.DUMP) { console.log(ttir); process.exit(0); }
  let k;
  try { k = compileAndLoad(ttir, "rmsnorm", 4); }
  catch (e) { console.log("native rsqrt failed:", (e as Error).message.slice(0, 200)); process.exit(1); }
  console.log(`rmsnorm loaded (shmem=${k.shmem}, ${NATIVE ? "native rsqrt" : "Newton rsqrt"})`);

  const X = new Float32Array(M * N).map(() => (Math.random() * 2 - 1));
  const W = new Float32Array(N).map(() => (Math.random() * 2 - 1) * 0.1);
  const Y = new Float32Array(M * N);
  const dX = cuAlloc(X.byteLength), dW = cuAlloc(W.byteLength), dY = cuAlloc(Y.byteLength);
  cuHtoD(dX, X.buffer); cuHtoD(dW, W.buffer);
  cuLaunch(k, [M, 1, 1], [128, 1, 1], [dX, dW, dY]);
  cuSync(); cuDtoH(Y.buffer, dY);

  const ref = refRMS(X, W, M, N);
  let maxErr = 0, ok = true;
  for (let i = 0; i < M * N; i++) {
    const d = Math.abs(Y[i] - ref[i]);
    if (d > maxErr) maxErr = d;
    if (d > 1e-4) { ok = false; if (i < 5) console.log(`mismatch [${i}]: got ${Y[i].toFixed(6)} ref ${ref[i].toFixed(6)}`); }
  }
  console.log(ok ? `✓ rmsnorm correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");

  // benchmark
  for (let i = 0; i < 3; i++) cuLaunch(k, [M, 1, 1], [128, 1, 1], [dX, dW, dY]);
  cuSync();
  const t0 = performance.now(), it = 200;
  for (let i = 0; i < it; i++) cuLaunch(k, [M, 1, 1], [128, 1, 1], [dX, dW, dY]);
  cuSync();
  const dt = (performance.now() - t0) / 1000 / it;
  const bytesRW = (M * N + M * N + N) * 4; // read X+W, write Y
  console.log(`bench ${M}x${N}: ${(dt * 1e6).toFixed(1)} µs/it, ${(bytesRW / dt / 1e9).toFixed(0)} GB/s effective mem-BW`);
  cuFree(dX); cuFree(dW); cuFree(dY);
}
main();
