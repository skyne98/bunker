import { cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch } from "../src/ttir";

// Measure: how many kernel launches per token, and the overhead of each launch
const stPath = "/tmp/qwen35_0.8b.safetensors";
const data = await Bun.file(stPath).bytes();
const buf = cuAlloc(BigInt(1024*2));
cuHtoD(buf, data.subarray(0, 2048).buffer);

// Measure raw cuLaunch overhead (no-op kernel equivalent)
// We can't launch a no-op, but we can measure cuSync overhead
const N = 1000;
const t0 = performance.now();
for (let i = 0; i < N; i++) cuSync();
const dt = (performance.now() - t0) / N;
console.log(`cuSync overhead: ${dt.toFixed(6)} ms (${(dt*1000).toFixed(2)} µs)`);

// Measure cuDtoH for logits (VOCAB*4 = ~1MB)
const logits = cuAlloc(BigInt(248320 * 4));
const hLogits = new Float32Array(248320);
cuHtoD(logits, hLogits.buffer); cuSync();
const t1 = performance.now();
for (let i = 0; i < 100; i++) cuDtoH(hLogits.buffer, logits, BigInt(248320*4));
const dt2 = (performance.now() - t1) / 100;
console.log(`cuDtoH logits (993KB): ${dt2.toFixed(4)} ms`);

// Measure cuHtoD for mask (512B)
const mask = new Float32Array(128);
const maskD = cuAlloc(BigInt(512));
const t3 = performance.now();
for (let i = 0; i < 100; i++) cuHtoD(maskD, mask.buffer);
const dt3 = (performance.now() - t3) / 100;
console.log(`cuHtoD mask (512B): ${dt3.toFixed(4)} ms`);

// Measure cuHtoD for embedding (2KB)
const emb = new Uint16Array(1024);
const embD = cuAlloc(BigInt(2048));
const t4 = performance.now();
for (let i = 0; i < 100; i++) cuHtoD(embD, emb.buffer);
const dt4 = (performance.now() - t4) / 100;
console.log(`cuHtoD embedding (2KB): ${dt4.toFixed(4)} ms`);

// Count kernel launches per token
console.log("\n=== Kernel launches per token ===");
let launches = 0;
// GDN layer (18 layers):
launches += 18 * (4 + 2 + 1 + 1 + 1 + 1 + 2 + 2); // qkv,z,a,b GEMMs + casts + conv1d + GDN + out_proj + rms + cast + add + rms + mlp(4) + cast + add
// Actually let me count more carefully per layer:
// Per GDN layer: rms(1) + qkv(1) + z(1) + a(1) + b(1) + csQ(1) + csZ(1) + conv1d(1) + gdn(1) + out_proj(1) + cs(1) + add(1) + rms2(1) + gp(1) + up(1) + sg(1) + dp(1) + cs2(1) + add2(1) = 19
// Per FA2 layer: rms(1) + qp(1) + kp(1) + vp(1) + csQG(1) + csV(1) + csK(1) + qn(1) + rope(1) + kn(1) + ropek(1) + fa2(1) + op(1) + cs(1) + add(1) + rms2(1) + gp(1) + up(1) + sg(1) + dp(1) + cs2(1) + add2(1) = 22
let gdnLaunches = 19;
let fa2Launches = 22;
let total = 18 * gdnLaunches + 6 * fa2Launches + 2; // +2 for final rms + lm_head
console.log(`GDN layers: ${gdmLaunches} launches × 18 = ${gdmLaunches * 18}`);
console.log(`FA2 layers: ${fa2Launches} launches × 6 = ${fa2Launches * 6}`);
console.log(`Final: 2 (rms + lm_head)`);
console.log(`Total: ${total} launches/token`);
console.log(`At 0.001ms overhead each: ${(total * 0.001).toFixed(2)} ms overhead`);
