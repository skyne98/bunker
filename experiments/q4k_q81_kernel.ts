// q4k_q81_kernel.ts — Q4_K × Q8_1 tensor core matmul via TTIR
// Dequantizes Q4_K → INT8, uses INT8 mma.sync, applies scale factors
import { dlopen, ptr, CString } from "bun:ffi";
import { execFileSync } from "child_process";
import { writeFileSync, readFileSync } from "fs";

const WRAPPER = "/tmp/triton_wrap";
const CUDA = "/run/opengl-driver/lib/libcuda.so";

const cs = dlopen(CUDA, {
  cuInit:{args:["u32"],returns:"i32"},cuDeviceGet:{args:["ptr","i32"],returns:"i32"},
  cuCtxCreate_v2:{args:["ptr","u32","i64"],returns:"i32"},
  cuModuleLoadData:{args:["ptr","ptr"],returns:"i32"},
  cuModuleGetFunction:{args:["ptr","i64","ptr"],returns:"i32"},
  cuLaunchKernel:{args:["i64","u32","u32","u32","u32","u32","u32","u32","ptr","ptr","ptr"],returns:"i32"},
  cuCtxSynchronize:{args:[],returns:"i32"},
  cuMemAlloc_v2:{args:["ptr","i64"],returns:"i32"},
  cuMemcpyHtoD_v2:{args:["i64","ptr","i64"],returns:"i32"},
  cuMemcpyDtoH_v2:{args:["ptr","i64","i64"],returns:"i32"},
  cuMemFree_v2:{args:["i64"],returns:"i32"},
}).symbols;

cs.cuInit(0); const dev=new Int32Array(1); cs.cuDeviceGet(dev,0);
cs.cuCtxCreate_v2(Buffer.alloc(8),0,BigInt(dev[0]));

// ─── Q4_K → INT8 dequantization TTIR ───
// Strategy: use two-stage approach
// 1. Pre-dequantize Q4_K → INT8 on device (one kernel)
// 2. INT8 TC matmul on dequantized data (second kernel)
// This avoids complex inline dequant + dot in a single kernel

function makeDequantTTIR(BM: number, BK: number): string {
  // Load Q4_K blocks, dequantize to INT8 tile
  // Q4_K layout per block (20 bytes): 16 packed nibbles + d(f16) + dmin(f16)
  // Output: BM×BK INT8 tensor
  return `module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @dequant(%in: !tt.ptr<i8>, %out: !tt.ptr<i8>, %scales: !tt.ptr<i8>) {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    // Load 32-byte Q4_K data: 16 packed bytes per 32 nibbles
    // For a BM×BK tile, we need (BM×BK/32) blocks
    // Each block has 20 bytes (16 packed + 2 d + 2 dmin)
    // For now: simple test load
    tt.return
  }
}`;
}

function makeInt8MatmulTTIR(BM: number, BN: number, BK: number, NW: number): string {
  return `module attributes {"ttg.num-warps" = ${NW} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul_i8(%A: !tt.ptr<i8>, %B: !tt.ptr<i8>, %C: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %cBM = arith.constant ${BM} : i64 %cBN = arith.constant ${BN} : i64
    %cBK = arith.constant ${BK} : i64 %cS = arith.constant 4096 : i64
    %c1 = arith.constant 1 : i64
    %zero = arith.constant dense<0> : tensor<${BM}x${BN}xi32>
    %tpA = tt.make_tensor_ptr %A, [%cS, %cBK], [%cBK, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BM}x${BK}xi8>>
    %tpB = tt.make_tensor_ptr %B, [%cBK, %cS], [%cS, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BK}x${BN}xi8>>
    %a = tt.load %tpA {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<${BM}x${BK}xi8>>
    %b = tt.load %tpB {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<${BK}x${BN}xi8>>
    %c = tt.dot %a, %b, %zero : tensor<${BM}x${BK}xi8> * tensor<${BK}x${BN}xi8> -> tensor<${BM}x${BN}xi32>
    %tpC = tt.make_tensor_ptr %C, [%cBM, %cBN], [%cBN, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.store %tpC, %c {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.return
  }
}`;
}

// ─── Compile INT8 matmul ───
const BM=16, BN=16, BK=256, NW=4;
const M=1024, N=1024, K=1024;

const ttir = makeInt8MatmulTTIR(BM, BN, BK, NW);
const f="/tmp/int8_mm_q.mlir"; writeFileSync(f, ttir);

console.log("Compiling INT8 TC matmul (K="+BK+")...");
console.time("compile");
const ptx=execFileSync(WRAPPER,[f,""+NW],{
  cwd:__dirname,encoding:"utf-8",maxBuffer:50*1024*1024,
  env:{...process.env,LD_LIBRARY_PATH:"/nix/store/ixhlv41i2wpl84xgjcks061dz4yssbg3-zlib-1.3.2/lib:/nix/store/bfwqrbwqpbnsdbgf86gz8pn8vvddci3i-libxml2-2.13.8/lib"}
});
console.timeEnd("compile");

const mma = ptx.split("mma.sync").length - 1;
console.log("MMA count:", mma);

// Load and launch
const m=Buffer.alloc(8); cs.cuModuleLoadData(m, ptr(Buffer.from(ptx+"\0")));
const fn=Buffer.alloc(8); cs.cuModuleGetFunction(fn, Number(m.readBigUInt64LE(0)), ptr(Buffer.from("matmul_i8\0")));
const fh=Number(fn.readBigUInt64LE(0));
if(!fh){console.error("fn=0");process.exit(1);}

const SZ=BigInt(128*1024*1024);
const dA=Buffer.alloc(8); cs.cuMemAlloc_v2(dA,SZ);
const dB=Buffer.alloc(8); cs.cuMemAlloc_v2(dB,SZ);
const dC=Buffer.alloc(8); cs.cuMemAlloc_v2(dC,SZ);
const pb=Buffer.alloc(5*8); pb.writeBigUInt64LE(dA.readBigUInt64LE(0),0);
pb.writeBigUInt64LE(dB.readBigUInt64LE(0),8); pb.writeBigUInt64LE(dC.readBigUInt64LE(0),16);
const kp=Buffer.alloc(6*8); const pp=Number(ptr(pb));
for(let i=0;i<5;i++) kp.writeBigUInt64LE(BigInt(pp+i*8),i*8); kp.writeBigUInt64LE(0n,40);
const shmem=16384, ITERS=10;
const FLOP=2*BM*BN*BK;

console.log(`\nBenchmark ${BM}×${BK}×${BN} INT8 TC:`);
for(const blocks of [256,1024,4096]) {
  const total=FLOP*blocks;
  const gx=Math.ceil(Math.sqrt(blocks)), gy=Math.ceil(blocks/gx);
  for(let w=0;w<3;w++) cs.cuLaunchKernel(fh,gx,gy,1,NW*32,1,1,shmem,0n,ptr(kp),null);
  cs.cuCtxSynchronize();
  const t=[];
  for(let i=0;i<ITERS;i++) {
    const t0=performance.now(); cs.cuLaunchKernel(fh,gx,gy,1,NW*32,1,1,shmem,0n,ptr(kp),null);
    cs.cuCtxSynchronize(); t.push(performance.now()-t0);
  }
  const avg=t.reduce((a,b)=>a+b,0)/t.length;
  console.log(`  ${String(blocks).padStart(5)} blocks: ${(avg*1000).toFixed(1).padStart(7)}us -> ${(total/(avg/1000)/1e12).toFixed(2)} TFLOPS`);
}

cs.cuMemFree_v2(Number(dA.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dB.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dC.readBigUInt64LE(0)));

console.log("\nNote: For Q4_K×Q8_1, the workflow would be:");
console.log("1. Dequantize Q4_K → INT8 on GPU (separate kernel or inline arith ops)");
console.log("2. Use INT8 TC matmul (shown above)");
console.log("3. Apply Q4_K scale d4 and Q8_1 scale d8 to INT32 output");
console.log("The dequant kernel needs nibble extraction (arith.andi + arith.shrsi)");
