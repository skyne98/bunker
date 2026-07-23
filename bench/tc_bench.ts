// tc_bench.ts — Tensor core matmul benchmark
import { dlopen, ptr, CString } from "bun:ffi";
import { execSync } from "child_process";
import { writeFileSync } from "fs";

const WRAPPER = "/tmp/triton_wrap";

function compile(ttir: string): { ptx: string; shmem: number } {
  const f = `/tmp/tc_${process.pid}.mlir`;
  writeFileSync(f, ttir);
  const out = execSync(`${WRAPPER} ${f} 4`, { cwd: __dirname, encoding: "utf-8", maxBuffer: 50*1024*1024 });
  // Shared mem size: hardcoded for known configs
  const smem = BM <= 16 ? 16384 : 32768;
  return { ptx: out, shmem: smem };
}

const CUDA = "/run/opengl-driver/lib/libcuda.so";
const cs = dlopen(CUDA, {
  cuInit:{args:["u32"],returns:"i32"},cuDeviceGet:{args:["ptr","i32"],returns:"i32"},
  cuDeviceGetName:{args:["ptr","i32","i32"],returns:"i32"},
  cuCtxCreate_v2:{args:["ptr","u32","i64"],returns:"i32"},
  cuModuleLoadData:{args:["ptr","ptr"],returns:"i32"},
  cuModuleGetFunction:{args:["ptr","i64","ptr"],returns:"i32"},
  cuMemAlloc_v2:{args:["ptr","i64"],returns:"i32"},
  cuLaunchKernel:{args:["i64","u32","u32","u32","u32","u32","u32","u32","ptr","ptr","ptr"],returns:"i32"},
  cuCtxSynchronize:{args:[],returns:"i32"},
  cuMemFree_v2:{args:["i64"],returns:"i32"},
}).symbols;

cs.cuInit(0); const dev = new Int32Array(1); cs.cuDeviceGet(dev, 0);
const nb = new Uint8Array(256); cs.cuDeviceGetName(nb, 256, dev[0]);
cs.cuCtxCreate_v2(Buffer.alloc(8), 0, BigInt(dev[0]));
console.log(`GPU: ${new CString(ptr(nb))}`);

// Best config: 32×1024×32, 128 mma.sync, 2.1M flops/block
const BM=32, BN=32, BK=1024;
const FLOP = 2 * BM * BN * BK;

const ttir = ()=>`module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %cA = arith.constant ${BM} : i64
    %cB = arith.constant ${BN} : i64
    %cK = arith.constant ${BK} : i64
    %cN = arith.constant 4096 : i64
    %c1 = arith.constant 1 : i64
    %zero = arith.constant dense<0.000000e+00> : tensor<${BM}x${BN}xf32>
    %tpA = tt.make_tensor_ptr %A, [%cN, %cK], [%cK, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BM}x${BK}xf16>>
    %tpB = tt.make_tensor_ptr %B, [%cK, %cN], [%cN, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BK}x${BN}xf16>>
    %a_tile = tt.load %tpA {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<${BM}x${BK}xf16>>
    %b_tile = tt.load %tpB {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<${BK}x${BN}xf16>>
    %c_tile = tt.dot %a_tile, %b_tile, %zero : tensor<${BM}x${BK}xf16> * tensor<${BK}x${BN}xf16> -> tensor<${BM}x${BN}xf32>
    %tpC = tt.make_tensor_ptr %C, [%cA, %cB], [%cB, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BM}x${BN}xf32>>
    tt.store %tpC, %c_tile {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<${BM}x${BN}xf32>>
    tt.return
  }
}`;

console.log(`Compiling ${BM}×${BK}×${BN}...`);
console.time("compile");
const {ptx, shmem} = compile(ttir());
console.timeEnd("compile");
const mma = (ptx.match(/mma\.sync/g) || []).length;
console.log(`PTX: ${(ptx.length/1024).toFixed(0)}KB, ${mma} mma.sync, ${(FLOP/1e6).toFixed(2)}M flops/block`);

const mod = Buffer.alloc(8); cs.cuModuleLoadData(mod, ptr(Buffer.from(ptx)));
const fn = Buffer.alloc(8); cs.cuModuleGetFunction(fn, Number(mod.readBigUInt64LE(0)), ptr(Buffer.from("matmul\0")));
const fh = Number(fn.readBigUInt64LE(0));

const SZ = BigInt(128 * 1024 * 1024);
const dA = Buffer.alloc(8); cs.cuMemAlloc_v2(dA, SZ);
const dB = Buffer.alloc(8); cs.cuMemAlloc_v2(dB, SZ);
const dC = Buffer.alloc(8); cs.cuMemAlloc_v2(dC, SZ);

const pb = Buffer.alloc(5*8); pb.writeBigUInt64LE(dA.readBigUInt64LE(0),0);
pb.writeBigUInt64LE(dB.readBigUInt64LE(0),8); pb.writeBigUInt64LE(dC.readBigUInt64LE(0),16);
const kp = Buffer.alloc(6*8); const pp=Number(ptr(pb));
for(let i=0;i<5;i++) kp.writeBigUInt64LE(BigInt(pp+i*8), i*8); kp.writeBigUInt64LE(0n,40);

const ITERS = 10, PEAK = 71;

if (cs.cuLaunchKernel(fh, 1,1,1, 128,1,1, shmem, 0n, ptr(kp), null) !== 0) process.exit(1);
cs.cuCtxSynchronize();
console.log(`Shared mem: ${shmem} bytes\n`);

for (const blocks of [82, 656, 5248, 16384, 65536, 262144]) {
  const total = FLOP * blocks;
  const gx = Math.ceil(Math.sqrt(blocks)), gy = Math.ceil(blocks/gx);
  for (let w=0;w<3;w++) cs.cuLaunchKernel(fh, gx,gy,1, 128,1,1, shmem, 0n, ptr(kp), null);
  cs.cuCtxSynchronize();
  const t: number[] = [];
  for (let i=0;i<ITERS;i++) {
    const t0=performance.now();
    const lr=cs.cuLaunchKernel(fh, gx,gy,1, 128,1,1, shmem, 0n, ptr(kp), null);
    if (lr!==0) break;
    cs.cuCtxSynchronize();
    t.push(performance.now()-t0);
  }
  if (!t.length) break;
  const avg = t.reduce((a,b)=>a+b,0)/t.length;
  const tops = total / (avg/1000) / 1e12;
  console.log(`${String(blocks).padStart(7)} blks | ${(avg*1000).toFixed(1).padStart(7)}µs | ${tops.toFixed(2).padStart(7)} TFLOPS | ${(tops/PEAK*100).toFixed(1).padStart(5)}%`);
}

cs.cuMemFree_v2(Number(dA.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dB.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dC.readBigUInt64LE(0)));
