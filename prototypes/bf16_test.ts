// Test bf16 support in the builder.
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

// bf16 matmul: A[64,128] bf16 @ B[128,64] bf16 -> C[64,64] f32 (tensor cores)
const b = new TTIRBuilder();
const A = b.param("A", { ptr: "bf16" });
const B = b.param("B", { ptr: "bf16" });
const C = b.param("C", { ptr: "f32" });
const tpA = b.makeTensorPtr(A, [64, 128], [128, 1], [b.i32(0), b.i32(0)], [64, 128], "bf16", [1, 0]);
const tpB = b.makeTensorPtr(B, [128, 64], [64, 1], [b.i32(0), b.i32(0)], [128, 64], "bf16", [1, 0]);
const tpC = b.makeTensorPtr(C, [64, 64], [64, 1], [b.i32(0), b.i32(0)], [64, 64], "f32", [1, 0]);
const a = b.load(tpA);
const bb = b.load(tpB);
const acc = b.zeros([64, 64], "f32");
const c = b.dot(a, bb, acc);
b.store(tpC, c, { boundaryCheck: [0, 1] });
const k = compileAndLoad(b.build("bf16_matmul", 4), "bf16_matmul", 4);
console.log(`bf16 matmul loaded (shmem=${k.shmem})`);

// test data: bf16 values (same byte layout as f16, but different interpretation)
// We'll just use raw bytes that represent bf16 floats
const M = 64, N = 64, K = 128;
const hA = new Uint16Array(M * K);  // bf16 = 2 bytes
const hB = new Uint16Array(K * N);
// fill with small bf16 values (0.5 = 0x3F00 in bf16)
for (let i = 0; i < M * K; i++) hA[i] = 0x3F00; // 0.5 in bf16
for (let i = 0; i < K * N; i++) hB[i] = 0x3F00;
const hC = new Float32Array(M * N);

const dA = cuAlloc(hA.byteLength), dB = cuAlloc(hB.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer); cuHtoD(dB, hB.buffer);
cuLaunch(k, [1, 1, 1], [128, 1, 1], [dA, dB, dC]);
cuSync(); cuDtoH(hC.buffer, dC);

// reference: 0.5 * 0.5 * 128 = 32 (each element)
const ref = 0.5 * 0.5 * 128;
let maxErr = 0;
for (let i = 0; i < M * N; i++) maxErr = Math.max(maxErr, Math.abs(hC[i] - ref));
console.log(`bf16 matmul: ref=${ref}, got=${hC[0]}, max err=${maxErr.toFixed(6)} ${maxErr < 0.01 ? "✓" : "✗"}`);
cuFree(dA); cuFree(dB); cuFree(dC);
