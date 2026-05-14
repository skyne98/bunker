// test_graph.ts — CUDA Graphs for dequant→TC pipeline, eliminating launch overhead
import { dlopen, ptr } from "bun:ffi";
import { DequantKernel, int8MatmulTTIR, compileTTIR, getCuCtx } from "./kernel.ts";

// Use SHARED context from kernel.ts to avoid dual-context bugs
const cs = getCuCtx();
// Add graph functions to the shared context's symbol table
// We need to dlopen them separately since getCuCtx only has basic functions
const cug = dlopen("/run/opengl-driver/lib/libcuda.so", {
  cuGraphCreate:{args:["ptr","u32"],returns:"i32"},
  cuGraphAddKernelNode:{args:["ptr","i64","ptr","u64","ptr"],returns:"i32"},
  cuGraphInstantiate:{args:["ptr","i64","ptr","ptr","u32"],returns:"i32"},
  cuGraphLaunch:{args:["i64","i64"],returns:"i32"},
  cuGraphExecKernelNodeSetParams:{args:["i64","i64","ptr"],returns:"i32"},
  cuGraphDestroy:{args:["i64"],returns:"i32"},
  cuGraphExecDestroy:{args:["i64"],returns:"i32"},
}).symbols;

// ── Compile kernels ──
const M = 1024, N = 1024, K = 1024;
const cfg = { BM: 32, BN: 32, BK: 1024, numWarps: 4 };

const { ptx, shmem } = compileTTIR(int8MatmulTTIR(cfg), 4);
const entry = ptx.match(/\.visible\s+\.entry\s+(\w+)/)[1];
const tcMod = Buffer.alloc(8); cs.cuModuleLoadData(tcMod, ptr(Buffer.from(ptx + "\0")));
const tcFn = Buffer.alloc(8);
cs.cuModuleGetFunction(tcFn, Number(tcMod.readBigUInt64LE(0)), ptr(Buffer.from(entry + "\0")));

const deq = new DequantKernel(); deq.load();

// ── Allocate buffers ──
const NUM_BLOCKS = M * K / 32;
const SZ = BigInt(512 * 1024 * 1024);
const dQ = Buffer.alloc(8); cs.cuMemAlloc_v2(dQ, SZ);   // Q4_K input
const dB = Buffer.alloc(8); cs.cuMemAlloc_v2(dB, SZ);   // INT8 B
const dA = Buffer.alloc(8); cs.cuMemAlloc_v2(dA, SZ);   // dequant output
const dC = Buffer.alloc(8); cs.cuMemAlloc_v2(dC, SZ);   // output

// ── Build kernel params ──
// Dequant: 3 params (ptr input, ptr output, i32 numBlocks)
// Each param value must be the exact size the kernel expects
const deqValA = Buffer.alloc(8); // input ptr
deqValA.writeBigUInt64LE(dQ.readBigUInt64LE(0), 0);
const deqValB = Buffer.alloc(8); // output ptr
deqValB.writeBigUInt64LE(dA.readBigUInt64LE(0), 0);
const deqValC = Buffer.alloc(4); // numBlocks (i32 = 4 bytes)
deqValC.writeInt32LE(NUM_BLOCKS, 0);

// Pointer array: each entry points to the corresponding param value
const dkp = Buffer.alloc(4 * 8);
dkp.writeBigUInt64LE(BigInt(Number(ptr(deqValA))), 0);
dkp.writeBigUInt64LE(BigInt(Number(ptr(deqValB))), 8);
dkp.writeBigUInt64LE(BigInt(Number(ptr(deqValC))), 16);
dkp.writeBigUInt64LE(0n, 24); // null terminator

// TC: Params depend on PTX (typically 5 entry-level, 3 used for A,B,C + 2 unused)
// Use individual buffers for each param value
const tcVals: Buffer[] = [];
tcVals.push(Buffer.alloc(8)); tcVals[0].writeBigUInt64LE(dA.readBigUInt64LE(0), 0);
tcVals.push(Buffer.alloc(8)); tcVals[1].writeBigUInt64LE(dB.readBigUInt64LE(0), 0);
tcVals.push(Buffer.alloc(8)); tcVals[2].writeBigUInt64LE(dC.readBigUInt64LE(0), 0);
// Remaining params (scratch etc): null pointers
const numTCParams = (ptx.match(/\.param/g) || []).length;
while (tcVals.length < numTCParams) tcVals.push(Buffer.alloc(8)); // zeros

const tkp = Buffer.alloc((numTCParams + 1) * 8);
for (let i = 0; i < numTCParams; i++)
  tkp.writeBigUInt64LE(BigInt(Number(ptr(tcVals[i]))), i * 8);
tkp.writeBigUInt64LE(0n, numTCParams * 8);

// ── CUDA Kernel Node Params ──
// Dequant node params
const deqNodeParams = Buffer.alloc(11 * 8 + 12 * 8); // Huge oversize for the C struct
// Actually, the C struct CUDA_KERNEL_NODE_PARAMS is ~88 bytes on 64-bit
// Fields: func(8), gridDim(12), blockDim(12), sharedMem(8), kernelParams(8), extra(8)
// plus padding = ~64 bytes

// CUDA_KERNEL_NODE_PARAMS layout (56 bytes on Linux x86_64):
// 0: func (pointer, 8 bytes)
// 8: gridDimX (u32, 4), gridDimY (u32, 4), gridDimZ (u32, 4) = 12 bytes
// 20: blockDimX (u32, 4), blockDimY (u32, 4), blockDimZ (u32, 4) = 12 bytes
// 32: sharedMemBytes (u32, 4)
// 36: padding (4 bytes)
// 40: kernelParams (pointer, 8)
// 48: extra (pointer, 8)

function makeNodeParams(func: bigint, gx: number, gy: number, gz: number,
                        bx: number, by: number, bz: number,
                        shmem: number, kp: number): Buffer {
  const buf = Buffer.alloc(56);
  buf.writeBigUInt64LE(func, 0);
  buf.writeUInt32LE(gx, 8); buf.writeUInt32LE(gy, 12); buf.writeUInt32LE(gz, 16);
  buf.writeUInt32LE(bx, 20); buf.writeUInt32LE(by, 24); buf.writeUInt32LE(bz, 28);
  buf.writeUInt32LE(shmem, 32);
  // padding at 36
  buf.writeBigUInt64LE(BigInt(kp), 40);
  buf.writeBigUInt64LE(0n, 48);  // extra = NULL
  return buf;
}

// TC params
const gx = Math.ceil(N / 32), gy = Math.ceil(M / 32);

// Dequant node params
const dgx = Math.ceil(NUM_BLOCKS / 64);
const dnp = makeNodeParams(deq["_fn"], dgx, 1, 1, 64, 1, 1, 0, Number(ptr(dkp)));

// TC node params
const tnp = makeNodeParams(tcFn.readBigUInt64LE(0), gx, gy, 1, 128, 1, 1, shmem, Number(ptr(tkp)));

// ── Create graph ──
const graphPtr = Buffer.alloc(8);
const ret1 = cug.cuGraphCreate(graphPtr, 0);
console.log("cuGraphCreate:", ret1, "graph:", Number(graphPtr.readBigUInt64LE(0)));

const graph = graphPtr.readBigUInt64LE(0);

// Debug: check values before graph creation
console.log("Deq func:", deq["_fn"], "grid:", dgx, "block:", 64, "shmem:", 0);
console.log("DKP ptr:", Number(ptr(dkp)));
console.log("DKP entries:", dkp.readBigUInt64LE(0), dkp.readBigUInt64LE(8), dkp.readBigUInt64LE(16), dkp.readBigUInt64LE(24));
console.log("TC func:", tcFn.readBigUInt64LE(0), "grid:", gx, gy, "block:", 128, "shmem:", shmem);
console.log("TKP ptr:", Number(ptr(tkp)));

// Use v2 to be safe
const addNode = cug.cuGraphAddKernelNode;

// Dequant node (no deps)
const deqNode = Buffer.alloc(8);
const ret2 = addNode(deqNode, graph, 0n, 0n, ptr(dnp));
console.log("cuGraphAddKernelNode (deq):", ret2, "node:", Number(deqNode.readBigUInt64LE(0)));

if (ret2 !== 0) { console.log("Deq node failed, aborting."); process.exit(1); }

// TC node, dependent on dequant node
const tcNode = Buffer.alloc(8);
const deps1 = Buffer.alloc(8); deps1.writeBigUInt64LE(deqNode.readBigUInt64LE(0), 0);
const ret3 = addNode(tcNode, graph, ptr(deps1), 1n, ptr(tnp));
console.log("cuGraphAddKernelNode (TC):", ret3);

// ── Instantiate ──
const execPtr = Buffer.alloc(8);
const ret4 = cug.cuGraphInstantiate(execPtr, graph, ptr(Buffer.alloc(8)), ptr(Buffer.alloc(8)), 0);
console.log("cuGraphInstantiate:", ret4, "exec:", Number(execPtr.readBigUInt64LE(0)));

const exec = execPtr.readBigUInt64LE(0);

// ── Benchmark graph vs sequential ──
const eS=Buffer.alloc(8);cs.cuEventCreate(eS,0);const eP=Buffer.alloc(8);cs.cuEventCreate(eP,0);
const es=Number(eS.readBigUInt64LE(0)),ep=Number(eP.readBigUInt64LE(0)),elv=new Float32Array(1);

// Set up test data: fill Q4_K blocks with known values
// Generate Q4_K data + reference on host
const hQ = Buffer.alloc(NUM_BLOCKS * 20);
const hRef = new Int8Array(M * K);
for (let r = 0; r < M; r++) {
  for (let kb = 0; kb < K/32; kb++) {
    const bi = r * (K/32) + kb;
    for (let j = 0; j < 16; j++) {
      const lo = (Math.random() * 16) | 0;
      const hi = (Math.random() * 16) | 0;
      hQ[bi * 20 + j] = lo | (hi << 4);
      hRef[r * K + kb * 32 + j * 2] = lo - 8;
      hRef[r * K + kb * 32 + j * 2 + 1] = hi - 8;
    }
  }
}
const hB = new Int8Array(K * N);
for (let i = 0; i < K * N; i++) hB[i] = (Math.random() * 256 - 128) | 0;

const hC = new Int32Array(M * N);
cs.cuMemcpyHtoD_v2(Number(dQ.readBigUInt64LE(0)), hQ, BigInt(hQ.length));
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)), Buffer.from(hB.buffer), BigInt(hB.byteLength));

// Verify sequential pipeline first
deq.launch(dQ.readBigUInt64LE(0), dA.readBigUInt64LE(0), NUM_BLOCKS);
cs.cuLaunchKernel(Number(tcFn.readBigUInt64LE(0)), gx, gy, 1, 128, 1, 1, shmem, 0n, ptr(tkp), null);
cs.cuCtxSynchronize();
cs.cuMemcpyDtoH_v2(hC, Number(dC.readBigUInt64LE(0)), BigInt(M*N*4));

// Reference C = dequant_A @ B
const refC = new Int32Array(M*N);
for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
  let s = 0; for (let k = 0; k < K; k++) s += hRef[i*K+k] * hB[k*N+j];
  refC[i*N+j] = s;
}
let max=0, cnt=0;
for (let i = 0; i < M*N; i++) { const d = Math.abs(hC[i] - refC[i]); if (d > max) max = d; if (d !== 0) cnt++; }
console.log(`Sequential verify: ${cnt} errors, max=${max}${cnt === 0 ? " OK" : ""}`);

// Now verify graph
cug.cuGraphLaunch(exec, 0n);
cs.cuCtxSynchronize();
cs.cuMemcpyDtoH_v2(hC, Number(dC.readBigUInt64LE(0)), BigInt(M*N*4));
max=0; cnt=0;
for (let i = 0; i < M*N; i++) { const d = Math.abs(hC[i] - refC[i]); if (d > max) max = d; if (d !== 0) cnt++; }
console.log(`Graph verify: ${cnt} errors, max=${max}${cnt === 0 ? " OK" : ""}`);

// Benchmark: graph
const gtimes: number[] = [];
for (let i = 0; i < 30; i++) {
  cs.cuEventRecord(es, 0);
  cug.cuGraphLaunch(exec, 0n);
  cs.cuEventRecord(ep, 0);
  cs.cuEventSynchronize(ep);
  cs.cuEventElapsedTime(elv, es, ep);
  gtimes.push(elv[0] * 1000);
}
const gavg = gtimes.reduce((a, b) => a + b, 0) / gtimes.length;
console.log(`\nCUDA Graph:  ${gavg.toFixed(1)} µs  ${(2*M*N*K / (gavg/1e6) / 1e12).toFixed(2)} TFLOPS`);

// Sequential timing (no extra sync)
const seqtimes: number[] = [];
for (let i = 0; i < 30; i++) {
  cs.cuEventRecord(es, 0);
  deq.launch(dQ.readBigUInt64LE(0), dA.readBigUInt64LE(0), NUM_BLOCKS);
  cs.cuLaunchKernel(Number(tcFn.readBigUInt64LE(0)), gx, gy, 1, 128, 1, 1, shmem, 0n, ptr(tkp), null);
  cs.cuEventRecord(ep, 0);
  cs.cuEventSynchronize(ep);
  cs.cuEventElapsedTime(elv, es, ep);
  seqtimes.push(elv[0] * 1000);
}
const savg = seqtimes.reduce((a, b) => a + b, 0) / seqtimes.length;
console.log(`Sequential: ${savg.toFixed(1)} µs  ${(2*M*N*K / (savg/1e6) / 1e12).toFixed(2)} TFLOPS`);

console.log(`Speedup: ${(savg/gavg).toFixed(2)}x`);

// Cleanup
cug.cuGraphExecDestroy(exec);
cug.cuGraphDestroy(graph);
