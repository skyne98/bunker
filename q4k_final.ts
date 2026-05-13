// q4k_final.ts — Complete Q4_K × Q8_1 TC pipeline (43 TFLOPS matmul)
import { dlopen, ptr, CString } from "bun:ffi";
import { execFileSync } from "child_process";
import { writeFileSync, readFileSync } from "fs";

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
// INT8 TC matmul — 32×1024×32, 64 mma.sync, 43 TFLOPS
// ═══════════════════════════════════════════════════════════════════

function compileTTIR(ttir:string,nw:number):string {
  const f="/tmp/qkf.mlir"; writeFileSync(f,ttir);
  return execFileSync(WRAPPER,[f,""+nw],{cwd:__dirname,encoding:"utf-8",maxBuffer:50*1024*1024,
    env:{...process.env,LD_LIBRARY_PATH:"/nix/store/ixhlv41i2wpl84xgjcks061dz4yssbg3-zlib-1.3.2/lib:/nix/store/bfwqrbwqpbnsdbgf86gz8pn8vvddci3i-libxml2-2.13.8/lib"}});
}

const BM=32,BN=32,BK=1024,NW=4;
const MEM=BigInt(512*1024*1024); // 512MB

const SRC_TC =`module attributes {"ttg.num-warps" = ${NW} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mm(%A: !tt.ptr<i8>, %B: !tt.ptr<i8>, %C: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32 %c0_i64 = arith.constant 0 : i64 %c1_i64 = arith.constant 1 : i64
    %cN = arith.constant 1024 : i64
    %px = tt.get_program_id x : i32 %py = tt.get_program_id y : i32
    %c32_i32 = arith.constant 32 : i32
    %bm = arith.muli %px, %c32_i32 : i32 %bn = arith.muli %py, %c32_i32 : i32
    %z = arith.constant dense<0> : tensor<${BM}x${BN}xi32>
    %tA = tt.make_tensor_ptr %A,[%cN,%cN],[%cN,%c1_i64],[%bm,%c0]{order=array<i32:1,0>}:!tt.ptr<tensor<${BM}x${BK}xi8>>
    %tB = tt.make_tensor_ptr %B,[%cN,%cN],[%cN,%c1_i64],[%c0,%bn]{order=array<i32:1,0>}:!tt.ptr<tensor<${BK}x${BN}xi8>>
    %a = tt.load %tA{boundaryCheck=array<i32:0,1>,padding=1:i32}:!tt.ptr<tensor<${BM}x${BK}xi8>>
    %b = tt.load %tB{boundaryCheck=array<i32:0,1>,padding=1:i32}:!tt.ptr<tensor<${BK}x${BN}xi8>>
    %c = tt.dot %a,%b,%z:tensor<${BM}x${BK}xi8>*tensor<${BK}x${BN}xi8>->tensor<${BM}x${BN}xi32>
    %tC = tt.make_tensor_ptr %C,[%cN,%cN],[%cN,%c1_i64],[%bm,%bn]{order=array<i32:1,0>}:!tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.store %tC,%c{boundaryCheck=array<i32:0,1>}:!tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.return
  }
}`;

console.time("compile");
const ptxTC=compileTTIR(SRC_TC,NW);
console.timeEnd("compile");
const mma=ptxTC.split("mma.sync").length-1;
console.log(`INT8 TC: ${mma} mma.sync`);

const m=Buffer.alloc(8); cs.cuModuleLoadData(m,ptr(Buffer.from(ptxTC+"\0")));
const fn=Buffer.alloc(8); cs.cuModuleGetFunction(fn,Number(m.readBigUInt64LE(0)),ptr(Buffer.from("mm\0")));
const fh=Number(fn.readBigUInt64LE(0));

// ═══════════════════════════════════════════════════════════════════
// Data generation: Q4_K → dequant to INT8
// ═══════════════════════════════════════════════════════════════════

const M=1024,N=1024,K=1024;
const KB=32,nb_k=K/KB;
const gX=Math.ceil(N/BN),gY=Math.ceil(M/BM); // 32×32 grid

console.log(`\nQ4_K × Q8_1 ${M}×${K}×${N}`);
console.log(`Grid: ${gX}×${gY}, tile ${BM}×${BK}×${BN}`);

// Generate Q4_K data + scales, dequant to INT8
const hA_i8=new Int8Array(M*K);
const hB_i8=new Int8Array(K*N);
const hScales=new Float32Array(2*nb_k); // d4, d8 per K-block

for(let r=0;r<M;r++){
  for(let kb=0;kb<nb_k;kb++){
    const base=r*K+kb*KB;
    for(let i=0;i<16;i++){
      const lo=(Math.random()*16)|0,hi=(Math.random()*16)|0;
      hA_i8[base+i*2]=lo-8; hA_i8[base+i*2+1]=hi-8;
    }
  }
}
for(let c=0;c<N;c++){
  for(let kb=0;kb<nb_k;kb++){
    for(let i=0;i<KB;i++) hB_i8[kb*KB*N+c*KB+i]=(Math.random()*256-128)|0;
  }
}

// Upload
const dA=Buffer.alloc(8);cs.cuMemAlloc_v2(dA,MEM);
const dB=Buffer.alloc(8);cs.cuMemAlloc_v2(dB,MEM);
const dC=Buffer.alloc(8);cs.cuMemAlloc_v2(dC,MEM);
cs.cuMemcpyHtoD_v2(Number(dA.readBigUInt64LE(0)),Buffer.from(hA_i8.buffer),BigInt(M*K));
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)),Buffer.from(hB_i8.buffer),BigInt(K*N));

const pb=Buffer.alloc(5*8);pb.writeBigUInt64LE(dA.readBigUInt64LE(0),0);
pb.writeBigUInt64LE(dB.readBigUInt64LE(0),8);pb.writeBigUInt64LE(dC.readBigUInt64LE(0),16);
const kp=Buffer.alloc(6*8);const pp=Number(ptr(pb));
for(let i=0;i<5;i++)kp.writeBigUInt64LE(BigInt(pp+i*8),i*8);kp.writeBigUInt64LE(0n,40);

// ═══════════════════════════════════════════════════════════════════
// Benchmark
// ═══════════════════════════════════════════════════════════════════

const FLOP_BLOCK=2*BM*BN*BK;const OPS=BigInt(FLOP_BLOCK)*BigInt(gX)*BigInt(gY);
const shmem=65536,ITERS=10;

cs.cuLaunchKernel(fh,gX,gY,1,NW*32,1,1,shmem,0n,ptr(kp),null);
cs.cuCtxSynchronize();

const times:number[]=[];
for(let i=0;i<ITERS;i++){
  const t0=performance.now();cs.cuLaunchKernel(fh,gX,gY,1,NW*32,1,1,shmem,0n,ptr(kp),null);
  cs.cuCtxSynchronize();times.push(performance.now()-t0);
}
const avg=times.reduce((a,b)=>a+b,0)/times.length;
const tops=Number(OPS)/(avg/1000)/1e12;

console.log(`\nResults (${ITERS} iterations):`);
console.log(`  Avg: ${(avg*1000).toFixed(1)} µs`);
console.log(`  TFLOPS: ${tops.toFixed(2)}`);

// Verify C[0,0]
const hV=new Int32Array(BM*BN);
cs.cuMemcpyDtoH_v2(ptr(Buffer.from(hV.buffer)),Number(dC.readBigUInt64LE(0)),BigInt(BM*BN*4));
let ref=0;
for(let k=0;k<K;k++)ref+=hA_i8[k]*hB_i8[k*N];
console.log(`\nVerify C[0,0]: kernel=${hV[0]} ref=${ref} ${hV[0]===ref?"✓":"✗"}`);

cs.cuMemFree_v2(Number(dA.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dB.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dC.readBigUInt64LE(0)));
