import { cuAlloc, cuHtoD, cuDtoH, cuSync } from "../src/ttir";

// Measure CPU-side costs
const logits = cuAlloc(BigInt(248320 * 4));
const hLogits = new Float32Array(248320);

// Fill with data
for (let i = 0; i < 248320; i++) hLogits[i] = Math.random();
cuHtoD(logits, hLogits.buffer); cuSync();

// cuDtoH for logits (993KB)
let t = performance.now();
for (let i = 0; i < 100; i++) cuDtoH(hLogits.buffer, logits, BigInt(248320 * 4));
const dtohMs = (performance.now() - t) / 100;
console.log(`cuDtoH logits (993KB): ${dtohMs.toFixed(4)} ms`);

// CPU argmax
t = performance.now();
for (let i = 0; i < 100; i++) {
  let best = 0, bestV = -Infinity;
  for (let j = 0; j < 248320; j++) if (hLogits[j] > bestV) { bestV = hLogits[j]; best = j; }
}
const argmaxMs = (performance.now() - t) / 100;
console.log(`CPU argmax (248K floats): ${argmaxMs.toFixed(4)} ms`);

// cuHtoD for embedding (2KB)
const emb = new Uint16Array(1024);
const embD = cuAlloc(BigInt(2048));
t = performance.now();
for (let i = 0; i < 100; i++) cuHtoD(embD, emb.buffer);
const htoDMs = (performance.now() - t) / 100;
console.log(`cuHtoD embedding (2KB): ${htoDMs.toFixed(4)} ms`);

// cuHtoD for mask (512B)
const mask = new Float32Array(128);
const maskD = cuAlloc(BigInt(512));
t = performance.now();
for (let i = 0; i < 100; i++) cuHtoD(maskD, mask.buffer);
const maskMs = (performance.now() - t) / 100;
console.log(`cuHtoD mask (512B): ${maskMs.toFixed(4)} ms`);

// cuAlloc overhead
t = performance.now();
for (let i = 0; i < 100; i++) cuAlloc(BigInt(4096));
const allocMs = (performance.now() - t) / 100;
console.log(`cuAlloc (4KB): ${allocMs.toFixed(4)} ms`);

// Count total cuAllocs per token
// Per GDN layer: ~10 allocs (qkv, z, a, b, qkvB, zB, convOut, gdnOut, normed, attnF32, attnBf, afterAttn, normed2, gate, up, act, mlpOut, mlpBf, x) = ~19
// Per FA2 layer: ~12 allocs (qgF32, kF32, vF32, qgB, vB, kB, kNormB, fa2Out, normed, attnF32, attnBf, afterAttn, normed2, gate, up, act, mlpOut, mlpBf, x) = ~19
// Total: 24 * 19 + 2 = ~458 allocs... but many are small
console.log(`\nEstimated allocs/token: ~400`);
console.log(`Alloc overhead: ${(400 * allocMs).toFixed(2)} ms`);

// Total CPU time
console.log(`\n=== CPU overhead per token ===`);
console.log(`cuDtoH logits:     ${dtohMs.toFixed(3)} ms`);
console.log(`CPU argmax:        ${argmaxMs.toFixed(3)} ms`);
console.log(`cuHtoD embedding:  ${htoDMs.toFixed(3)} ms`);
console.log(`cuHtoD mask:       ${maskMs.toFixed(3)} ms`);
console.log(`cuAlloc ×400:      ${(400 * allocMs).toFixed(3)} ms`);
console.log(`Total CPU:         ${(dtohMs + argmaxMs + htoDMs + maskMs + 400*allocMs).toFixed(3)} ms`);
