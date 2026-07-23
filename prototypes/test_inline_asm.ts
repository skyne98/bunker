import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch } from "../src/ttir";

const b = new TTIRBuilder();
const X = b.param("X", {ptr: "f32"});
const Y = b.param("Y", {ptr: "f32"});
const row = b.programId(0);
const tpX = b.makeTensorPtr(X, [1, 8], [8, 1], [row, b.i32(0)], [1, 8], "f32", [1, 0]);
const tpY = b.makeTensorPtr(Y, [1, 8], [8, 1], [row, b.i32(0)], [1, 8], "f32", [1, 0]);
const x = b.load(tpX, {boundaryCheck: [0, 1], padding: 1});
const y = b.rsqrtHw(x);
b.store(tpY, y, {boundaryCheck: [0, 1]});
const ttir = b.build("test_asm", 4, 3);

const k = compileAndLoad(ttir, "test_asm", 4);
const inp = new Float32Array([1.0, 4.0, 100.0, 0.0001, 192.0, 1.5, 0.00031, 2.0]);
const d = cuAlloc(BigInt(32)); cuHtoD(d, inp.buffer); cuSync();
const out = cuAlloc(BigInt(32));
cuLaunch(k, [1, 1, 1], [128, 1, 1], [d, out]); cuSync();
const h = new Float32Array(8);
cuDtoH(h.buffer, out, BigInt(32));
console.log("rsqrtHw:", h);
console.log("expected:", [1, 0.5, 0.1, 100, 1/Math.sqrt(192), 1/Math.sqrt(1.5), 1/Math.sqrt(0.00031), 1/Math.sqrt(2)].map(v=>+v.toFixed(4)));
