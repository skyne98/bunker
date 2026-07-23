import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch } from "../src/ttir";

const b = new TTIRBuilder();
const X = b.param("X", {ptr: "f32"});
const Y = b.param("Y", {ptr: "f32"});
const row = b.programId(0);
const tpX = b.makeTensorPtr(X, [1, 4], [4, 1], [row, b.i32(0)], [1, 4], "f32", [1, 0]);
const tpY = b.makeTensorPtr(Y, [1, 4], [4, 1], [row, b.i32(0)], [1, 4], "f32", [1, 0]);
const x = b.load(tpX, {boundaryCheck: [0, 1], padding: 1});
// rsqrt(x) = exp(-0.5 * log(x))
const r = b.exp(b.mul(b.f32(-0.5), b.log(x)));
b.store(tpY, r, {boundaryCheck: [0, 1]});
const ttir = b.build("test_log", 4, 3);
try {
  const k = compileAndLoad(ttir, "test_log", 4);
  const d = cuAlloc(BigInt(16)); // 4 f32
  const inp = new Float32Array([1.0, 4.0, 100.0, 0.0001]);
  cuHtoD(d, inp.buffer); cuSync();
  const out = cuAlloc(BigInt(16));
  cuLaunch(k, [1, 1, 1], [128, 1, 1], [d, out]);
  cuSync();
  const h = new Float32Array(4);
  cuDtoH(h.buffer, out, BigInt(16));
  console.log("rsqrt via exp/log:", h);
  console.log("expected:", [1.0, 0.5, 0.1, 100.0]);
} catch (e: any) {
  console.log("FAILED:", e.message);
}
