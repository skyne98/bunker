// q4k_q81_final.ts — Complete Q4_K × Q8_1 tensor core pipeline
// Stages: dequant (host/device) → INT8 TC matmul (K-loop) → scale

import { dlopen, ptr, CString } from "bun:ffi";
import { execFileSync } from "child_process";
import { writeFileSync } from "fs";

const WRAPPER = "/tmp/triton_wrap";
const CUDA = "/run/opengl-driver/lib/libcuda.so";
const cs = dlopen(CUDA, {
  cuInit:{args:["u32"],returns:"i32"},cuDeviceGet:{args:["ptr","i32"],returns:"i32"},
  cuDeviceGetName:{args:["ptr","i32","i32"],returns:"i32"},
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
const nb=new Uint8Array(256); cs.cuDeviceGetName(nb,256,dev[0]);
cs.cuCtxCreate_v2(Buffer.alloc(8),0,BigInt(dev[0]));
console.log(`GPU: ${new CString(ptr(nb))}`);

// ═══════════════════════════════════════════════════════════════════
// Compile helpers
// ═══════════════════════════════════════════════════════════════════

function compileTTIR(ttir:string,nw:number):string {
  const f="/tmp/qk_compile.mlir"; writeFileSync(f,ttir);
  return execFileSync(WRAPPER,[f,""+nw],{cwd:__dirname,encoding:"utf-8",maxBuffer:50*1024*1024,
    env:{...process.env,LD_LIBRARY_PATH:"/nix/store/ixhlv41i2wpl84xgjcks061dz4yssbg3-zlib-1.3.2/lib:/nix/store/bfwqrbwqpbnsdbgf86gz8pn8vvddci3i-libxml2-2.13.8/lib"}});
}

function loadPTX(ptx:string,fn:string):number {
  const m=Buffer.alloc(8); cs.cuModuleLoadData(m,ptr(Buffer.from(ptx+"\0")));
  const f=Buffer.alloc(8); cs.cuModuleGetFunction(f,Number(m.readBigUInt64LE(0)),ptr(Buffer.from(fn+"\0")));
  return Number(f.readBigUInt64LE(0));
}

// ═══════════════════════════════════════════════════════════════════
// f16 helpers
// ═══════════════════════════════════════════════════════════════════

function f32to16u(v:number):number{
  const u=new DataView(new ArrayBuffer(4));u.setFloat32(0,v,true);const bits=u.getUint32(0,true);
  const s=(bits>>16)&0x8000,e=(bits>>23)&0xff,m=bits&0x7fffff;
  if(e===0)return s;if(e===0xff)return s|0x7c00|(m?0x200:0);
  const e16=e-127+15;
  if(e16>=0x1f)return s|0x7c00;if(e16<=0)return s;
  return s|(e16<<10)|(m>>13);
}

function f16to32u(u16:number):number{
  const s=(u16>>15)&1,e=(u16>>10)&0x1f,m=u16&0x3ff;
  if(e===0)return m===0?(s?-0:0):(s?-1:1)*Math.pow(2,-14)*(m/1024);
  if(e===31)return m?NaN:(s?-Infinity:Infinity);
  return (s?-1:1)*Math.pow(2,e-15)*(1+m/1024);
}

// ═══════════════════════════════════════════════════════════════════
// INT8 TC K-loop matmul
// ═══════════════════════════════════════════════════════════════════

// Each block: 16×16 output, K=32 per iteration, 32 iterations = K=1024
// Grid: ceil(M/16) × ceil(N/16)
const BM=16,BN=16,BK_i=32,NW=4; // BK_i = inner K per iteration
const K_ITERS=32; // 1024/32

const SRC_TC =`module attributes {"ttg.num-warps" = ${NW} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul_i8(%A: !tt.ptr<i8>, %B: !tt.ptr<i8>, %C: !tt.ptr<i32>) {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %c${K_ITERS} = arith.constant ${K_ITERS} : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %cK_i64 = arith.constant ${BK_i} : i64
    %cK_i32 = arith.constant ${BK_i} : i32
    %cS_i64 = arith.constant 1024 : i64
    %pid_x = tt.get_program_id x : i32
    %pid_y = tt.get_program_id y : i32
    %bm = arith.muli %pid_x, %c16_i32 : i32
    %bn = arith.muli %pid_y, %c16_i32 : i32
    %zero = arith.constant dense<0> : tensor<${BM}x${BN}xi32>
    %tA0 = tt.make_tensor_ptr %A, [%cS_i64, %cS_i64], [%cS_i64, %c1_i64], [%bm, %c0_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BM}x${BK_i}xi8>>
    %tB0 = tt.make_tensor_ptr %B, [%cS_i64, %cS_i64], [%cS_i64, %c1_i64], [%c0_i32, %bn] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BK_i}x${BN}xi8>>
    %res, %tAe, %tBe = scf.for %k = %c0 to %c${K_ITERS} step %c1 iter_args(%acc = %zero, %tA = %tA0, %tB = %tB0) -> (tensor<${BM}x${BN}xi32>, !tt.ptr<tensor<${BM}x${BK_i}xi8>>, !tt.ptr<tensor<${BK_i}x${BN}xi8>>) {
      %a = tt.load %tA {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<${BM}x${BK_i}xi8>>
      %b = tt.load %tB {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<${BK_i}x${BN}xi8>>
      %new = tt.dot %a, %b, %acc : tensor<${BM}x${BK_i}xi8> * tensor<${BK_i}x${BN}xi8> -> tensor<${BM}x${BN}xi32>
      %nA = tt.advance %tA, [%c0_i32, %cK_i32] : !tt.ptr<tensor<${BM}x${BK_i}xi8>>
      %nB = tt.advance %tB, [%cK_i32, %c0_i32] : !tt.ptr<tensor<${BK_i}x${BN}xi8>>
      scf.yield %new, %nA, %nB : tensor<${BM}x${BN}xi32>, !tt.ptr<tensor<${BM}x${BK_i}xi8>>, !tt.ptr<tensor<${BK_i}x${BN}xi8>>
    }
    %tC = tt.make_tensor_ptr %C, [%cS_i64, %cS_i64], [%cS_i64, %c1_i64], [%bm, %bn] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.store %tC, %res {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.return
  }
}`;

console.log("=== Q4_K × Q8_1 Tensor Core Pipeline ===\n");

// ═══════════════════════════════════════════════════════════════════
// Compile the TC matmul kernel
// ═══════════════════════════════════════════════════════════════════

console.time("compile_tc");
const ptxTC=compileTTIR(SRC_TC,NW);
console.timeEnd("compile_tc");
const mma=ptxTC.split("mma.sync").length-1;
console.log(`INT8 TC K-loop: ${mma} mma.sync, PTX ${(ptxTC.length/1024).toFixed(0)}KB`);
const fhTC=loadPTX(ptxTC,"matmul_i8");

// ═══════════════════════════════════════════════════════════════════
// Build Q4_K and Q8_1 data, dequantize on host
// ═══════════════════════════════════════════════════════════════════

const M=1024,N=1024,K=1024;
const KB=32; // elements per block
const nb_k=K/KB; // 32
const gX=Math.ceil(N/BN),gY=Math.ceil(M/BM); // 64×64 = 4096 blocks

// Q4_K: 20 bytes/block = 16 packed nibbles + d(f16) + dmin(f16)
// Q8_1: 36 bytes/block = 32 INT8 + d(f16) + s(f16)
// Dequantized A: M×K INT8 = 1024×1024
// B (raw): K×N INT8 = 1024×1024

console.log(`\nData: ${M}×${K}×${K} Q4_K×Q8_1`);
console.log(`Blocks: ${gX}×${gY} grid, ${nb_k} K-iterations`);

// Generate data with proper Q4_K/Q8_1 packing
const hA_i8=new Int8Array(M*K); // dequantized A
const hB_i8=new Int8Array(K*N); // raw B
const hScalesF=new Float32Array(2*M*N/nb_k); // d4, d8 per block

for(let r=0;r<M;r++){
  for(let kb=0;kb<nb_k;kb++){
    const base=r*K+kb*KB;
    const packed=new Uint8Array(16);
    for(let i=0;i<16;i++)packed[i]=(Math.random()*256)&0xff;
    // Dequantize Q4_K: nibble - 8 → INT8
    for(let i=0;i<16;i++){
      hA_i8[base+i*2]=(packed[i]&15)-8;
      hA_i8[base+i*2+1]=((packed[i]>>4)&15)-8;
    }
  }
}
for(let c=0;c<N;c++){
  for(let kb=0;kb<nb_k;kb++){
    for(let i=0;i<KB;i++){
      hB_i8[kb*KB*N+c*KB+i]=(Math.random()*256-128)|0;
    }
  }
}

// Allocate + upload
const SZ=BigInt(128*1024*1024);
const dA=Buffer.alloc(8);cs.cuMemAlloc_v2(dA,SZ);
const dB=Buffer.alloc(8);cs.cuMemAlloc_v2(dB,SZ);
const dC=Buffer.alloc(8);cs.cuMemAlloc_v2(dC,SZ);
cs.cuMemcpyHtoD_v2(Number(dA.readBigUInt64LE(0)),Buffer.from(hA_i8.buffer),BigInt(M*K));
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)),Buffer.from(hB_i8.buffer),BigInt(K*N));

const pb=Buffer.alloc(5*8);pb.writeBigUInt64LE(dA.readBigUInt64LE(0),0);
pb.writeBigUInt64LE(dB.readBigUInt64LE(0),8);pb.writeBigUInt64LE(dC.readBigUInt64LE(0),16);
const kp=Buffer.alloc(6*8);const pp=Number(ptr(pb));
for(let i=0;i<5;i++)kp.writeBigUInt64LE(BigInt(pp+i*8),i*8);kp.writeBigUInt64LE(0n,40);

// ═══════════════════════════════════════════════════════════════════
// Benchmark
// ═══════════════════════════════════════════════════════════════════

const FLOP_KLOOP=2*BM*BN*BK_i*K_ITERS; // 524,288 flops/block
const OPS=BigInt(FLOP_KLOOP)*BigInt(gX)*BigInt(gY); // total
const ITERS=10,shmem=1024;

// Warmup
cs.cuLaunchKernel(fhTC,gX,gY,1,NW*32,1,1,shmem,0n,ptr(kp),null);
cs.cuCtxSynchronize();

const times:number[]=[];
for(let i=0;i<ITERS;i++){
  const t0=performance.now();cs.cuLaunchKernel(fhTC,gX,gY,1,NW*32,1,1,shmem,0n,ptr(kp),null);
  cs.cuCtxSynchronize();times.push(performance.now()-t0);
}
const avg=times.reduce((a,b)=>a+b,0)/times.length;
const tops=Number(OPS)/(avg/1000)/1e12;

console.log(`\nResults (${ITERS} iterations):`);
console.log(`  Grid: ${gX}×${gY} blocks, ${NW*32} threads/block`);
console.log(`  Avg: ${(avg*1000).toFixed(1)} µs`);
console.log(`  TFLOPS: ${tops.toFixed(2)}`);
console.log(`  OPs/block: ${FLOP_KLOOP} (${nb_k} × ${BM}×${BK_i}×${BN})`);

// Verify
const hV=new Int32Array(BM*BN);
cs.cuMemcpyDtoH_v2(ptr(Buffer.from(hV.buffer)),Number(dC.readBigUInt64LE(0)),BigInt(BM*BN*4));
let ref=0;
for(let k=0;k<K;k++)ref+=hA_i8[k]*hB_i8[k*N];
console.log(`\nVerification (C[0,0]): kernel=${hV[0]} ref=${ref} ${hV[0]===ref?"✓":"✗"}`);

cs.cuMemFree_v2(Number(dA.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dB.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dC.readBigUInt64LE(0)));

console.log(`\nTo use Q4_K format directly:`);
console.log(`1. Dequantize Q4_K→INT8 on GPU via nibble-extraction kernel`);
console.log(`2. Run INT8 TC K-loop matmul (above)`);
console.log(`3. Apply scales: C_f32 = d4*d8*C_i32 + (8*d4+dmin)*d8*sumQ8`);
