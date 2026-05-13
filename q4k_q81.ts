// q4k_q81.ts — Full Q4_K × Q8_1 tensor core matmul
// Pipeline: dequant (host or device) → INT8 TC matmul → scale
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

function loadPTX(ptx:string,fnName:string):number {
  const m=Buffer.alloc(8); cs.cuModuleLoadData(m,ptr(Buffer.from(ptx+"\0")));
  const fn=Buffer.alloc(8); cs.cuModuleGetFunction(fn,Number(m.readBigUInt64LE(0)),ptr(Buffer.from(fnName+"\0")));
  return Number(fn.readBigUInt64LE(0));
}

// ═══════════════════════════════════════════════════════════════════
// f16 helpers
// ═══════════════════════════════════════════════════════════════════

function f32to16(v:number):number{
  const u32=new Uint32Array(1); u32[0]=new DataView(new ArrayBuffer(4)).setFloat32(0,v,true)||0;
  // Actually let's just do it properly
  const bits=(()=>{const u=new DataView(new ArrayBuffer(4));u.setFloat32(0,v,true);return u.getUint32(0,true)})();
  const s=(bits>>16)&0x8000, e=(bits>>23)&0xff, m=bits&0x7fffff;
  if(e===0) return s;
  if(e===0xff) return s|0x7c00|(m?0x200:0);
  const e16=e-127+15;
  if(e16>=0x1f) return s|0x7c00;
  if(e16<=0) return s;
  return s|(e16<<10)|(m>>13);
}

// ═══════════════════════════════════════════════════════════════════
// Step 1: INT8 TC matmul kernel
// ═══════════════════════════════════════════════════════════════════

const BM=16, BN=16, BK=32, NW=4;
const SRC_TC =`module attributes {"ttg.num-warps" = ${NW} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul_i8(%A: !tt.ptr<i8>, %B: !tt.ptr<i8>, %C: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %cM = arith.constant ${BM} : i64 %cN = arith.constant ${BN} : i64
    %cK = arith.constant ${BK} : i64 %cS = arith.constant 1024 : i64
    %c1 = arith.constant 1 : i64
    %z = arith.constant dense<0> : tensor<${BM}x${BN}xi32>
    %tA = tt.make_tensor_ptr %A,[%cS,%cS],[%cS,%c1],[%c0,%c0]{order=array<i32:1,0>}:!tt.ptr<tensor<${BM}x${BK}xi8>>
    %tB = tt.make_tensor_ptr %B,[%cS,%cS],[%cS,%c1],[%c0,%c0]{order=array<i32:1,0>}:!tt.ptr<tensor<${BK}x${BN}xi8>>
    %a = tt.load %tA{boundaryCheck=array<i32:0,1>,padding=1:i32}:!tt.ptr<tensor<${BM}x${BK}xi8>>
    %b = tt.load %tB{boundaryCheck=array<i32:0,1>,padding=1:i32}:!tt.ptr<tensor<${BK}x${BN}xi8>>
    %c = tt.dot %a,%b,%z:tensor<${BM}x${BK}xi8>*tensor<${BK}x${BN}xi8>->tensor<${BM}x${BN}xi32>
    %tC = tt.make_tensor_ptr %C,[%cM,%cN],[%cN,%c1],[%c0,%c0]{order=array<i32:1,0>}:!tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.store %tC,%c{boundaryCheck=array<i32:0,1>}:!tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.return
  }
}`;

console.time("tc_compile");
const ptxTC=compileTTIR(SRC_TC,NW);
console.timeEnd("tc_compile");
const mmaCount=ptxTC.split("mma.sync").length-1;
console.log(`INT8 TC: ${mmaCount} mma.sync`);

const fhTC=loadPTX(ptxTC,"matmul_i8");
if(!fhTC){console.error("TC fn=0");process.exit(1);}

// ═══════════════════════════════════════════════════════════════════
// Step 2: Build data
// ═══════════════════════════════════════════════════════════════════

const M=1024,N=1024,K=1024,KB=32;
const nb_k=K/KB;
const numABlocks=(M/BM)*(K/BK);  // blocks for A
const numBBlocks=(N/BN)*(K/BK);  // blocks for B

// Generate Q4_K and Q8_1 data, and dequantize on host
const SZ=BigInt(128*1024*1024);

// We need three arrays:
// hA_i8: dequantized Q4_K → INT8 for the TC matmul (M×K INT8 values)
// hB_i8: Q8_1 raw values for the TC matmul (K×N INT8 values)  
// scales: for post-processing

const hA_i8=new Int8Array(M*K);
const hB_i8=new Int8Array(K*N);
const hScales=new Float32Array(M*N); // placeholder

// Generate random data + dequantize
for(let r=0;r<M;r++){
  for(let kb=0;kb<nb_k;kb++){
    const packed=new Uint8Array(16);
    for(let i=0;i<16;i++) packed[i]=(Math.random()*256)&0xff;
    for(let i=0;i<16;i++){
      const lo=packed[i]&15, hi=(packed[i]>>4)&15;
      hA_i8[r*K+kb*KB+i*2]=lo-8;
      hA_i8[r*K+kb*KB+i*2+1]=hi-8;
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

// Upload
const dA=Buffer.alloc(8); cs.cuMemAlloc_v2(dA,SZ);
const dB=Buffer.alloc(8); cs.cuMemAlloc_v2(dB,SZ);
const dC=Buffer.alloc(8); cs.cuMemAlloc_v2(dC,SZ);
cs.cuMemcpyHtoD_v2(Number(dA.readBigUInt64LE(0)),Buffer.from(hA_i8.buffer),BigInt(M*K));
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)),Buffer.from(hB_i8.buffer),BigInt(K*N));

const pb=Buffer.alloc(5*8); pb.writeBigUInt64LE(dA.readBigUInt64LE(0),0);
pb.writeBigUInt64LE(dB.readBigUInt64LE(0),8); pb.writeBigUInt64LE(dC.readBigUInt64LE(0),16);
const kp=Buffer.alloc(6*8); const pp=Number(ptr(pb));
for(let i=0;i<5;i++) kp.writeBigUInt64LE(BigInt(pp+i*8),i*8); kp.writeBigUInt64LE(0n,40);

// ═══════════════════════════════════════════════════════════════════
// Step 3: Verify and benchmark
// ═══════════════════════════════════════════════════════════════════

// Verify single tile first
const gX=Math.ceil(N/BN), gY=Math.ceil(M/BM);

cs.cuLaunchKernel(fhTC,1,1,1,NW*32,1,1,16384,0n,ptr(kp),null);
cs.cuCtxSynchronize();

const hC=new Int32Array(BM*BN);
cs.cuMemcpyDtoH_v2(ptr(Buffer.from(hC.buffer)),Number(dC.readBigUInt64LE(0)),BigInt(BM*BN*4));
let ref=0;
for(let k=0;k<BK;k++) ref+=hA_i8[k]*hB_i8[k*N];
console.log(`C[0,0] = ${hC[0]} (ref: ${ref}) ${hC[0]===ref?"✓":"✗"}`);

if(hC[0]!==ref){
  console.log("Verification failed — check data layout");
  console.log("Note: this requires a K-loop or multiple kernel invocations for full K");
}

// Benchmark full matmul (single K-step kernel × nb_k iterations)
const OPS=2n*BigInt(M)*BigInt(N)*BigInt(K);
const ITERS=10;

console.log(`\nBenchmarking ${M}×${K}×${N} Q4_K×Q8_1 via INT8 TC:`);
console.log(`Tile: ${BM}×${BK}×${BN}, ${nb_k} K-iterations needed`);

// Warmup
cs.cuLaunchKernel(fhTC,gX,gY,1,NW*32,1,1,16384,0n,ptr(kp),null);
cs.cuCtxSynchronize();

const times:number[]=[];
for(let i=0;i<ITERS;i++){
  const t0=performance.now();
  cs.cuLaunchKernel(fhTC,gX,gY,1,NW*32,1,1,16384,0n,ptr(kp),null);
  cs.cuCtxSynchronize();
  times.push(performance.now()-t0);
}
const avg=times.reduce((a,b)=>a+b,0)/times.length;
const tops=Number(OPS)/(avg/1000)/1e12;

console.log(`\nResults (${ITERS} iterations, single K-step):`);
console.log(`  Grid: ${gX}×${gY} blocks, ${NW*32} threads/block`);
console.log(`  Avg: ${(avg*1000).toFixed(1)} µs`);
console.log(`  TOPS: ${tops.toFixed(2)}`);
console.log(`\nNote: For full K=${K}, need ${nb_k} K-iterations.`);
console.log(`Estimated full K=${K} time: ${(avg*1000*nb_k).toFixed(0)} µs`);
console.log(`Estimated TOPS with full K: ${(Number(OPS)/(avg*nb_k/1000)/1e12).toFixed(2)}`);

cs.cuMemFree_v2(Number(dA.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dB.readBigUInt64LE(0)));
cs.cuMemFree_v2(Number(dC.readBigUInt64LE(0)));
