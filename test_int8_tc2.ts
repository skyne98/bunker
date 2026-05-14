// test_int8_tc2.ts — Exact copy of test_int8_tc.ts
import { dlopen, ptr } from "bun:ffi";
import { int8MatmulTTIR, compileTTIR } from "./kernel.ts";

const cs = dlopen("/run/opengl-driver/lib/libcuda.so", {
  cuInit:{args:["u32"],returns:"i32"},cuDeviceGet:{args:["ptr","i32"],returns:"i32"},
  cuCtxCreate_v2:{args:["ptr","u32","i64"],returns:"i32"},
  cuModuleLoadData:{args:["ptr","ptr"],returns:"i32"},
  cuModuleGetFunction:{args:["ptr","i64","ptr"],returns:"i32"},
  cuLaunchKernel:{args:["i64","u32","u32","u32","u32","u32","u32","u32","ptr","ptr","ptr"],returns:"i32"},
  cuMemAlloc_v2:{args:["ptr","i64"],returns:"i32"},
  cuMemcpyHtoD_v2:{args:["i64","ptr","i64"],returns:"i32"},
  cuMemcpyDtoH_v2:{args:["ptr","i64","i64"],returns:"i32"},
  cuMemFree_v2:{args:["i64"],returns:"i32"},cuCtxSynchronize:{args:[],returns:"i32"},
}).symbols;
cs.cuInit(0);const dev=new Int32Array(1);cs.cuDeviceGet(dev,0);
cs.cuCtxCreate_v2(Buffer.alloc(8),0,BigInt(dev[0]));

const M=1024,N=1024,K=1024;
const cfg={BM:32,BN:32,BK:1024,numWarps:4};
const ttir=int8MatmulTTIR(cfg);
const {ptx,shmem}=compileTTIR(ttir,cfg.numWarps);
console.log(`PTX: ${(ptx.match(/mma\.sync/g)||[]).length} mma.sync, ${(ptx.match(/\.param/g)||[]).length} params, shmem=${shmem}`);

const entryM=ptx.match(/\.visible\s+\.entry\s+(\w+)/);
const fnName=entryM?entryM[1]:"?";
console.log(`Entry: ${fnName}`);

const tcMod=Buffer.alloc(8);
cs.cuModuleLoadData(tcMod,ptr(Buffer.from(ptx+"\0")));
const tcFn=Buffer.alloc(8);
cs.cuModuleGetFunction(tcFn,Number(tcMod.readBigUInt64LE(0)),ptr(Buffer.from(fnName+"\0")));
const fhTC=Number(tcFn.readBigUInt64LE(0));

const hA=new Int8Array(M*K);
const hB=new Int8Array(K*N);
const hC=new Int32Array(M*N);
for(let i=0;i<M*K;i++)hA[i]=(Math.random()*256-128)|0;
for(let i=0;i<K*N;i++)hB[i]=(Math.random()*256-128)|0;

const SZ=BigInt(512*1024*1024);
const dA=Buffer.alloc(8);cs.cuMemAlloc_v2(dA,SZ);
const dB=Buffer.alloc(8);cs.cuMemAlloc_v2(dB,SZ);
const dC=Buffer.alloc(8);cs.cuMemAlloc_v2(dC,SZ);
cs.cuMemcpyHtoD_v2(Number(dA.readBigUInt64LE(0)),Buffer.from(hA.buffer),BigInt(hA.byteLength));
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)),Buffer.from(hB.buffer),BigInt(hB.byteLength));

const numParams=(ptx.match(/\.param/g)||[]).length;
const tcArgs=Buffer.alloc(numParams*8);
tcArgs.writeBigUInt64LE(dA.readBigUInt64LE(0),0);
tcArgs.writeBigUInt64LE(dB.readBigUInt64LE(0),8);
tcArgs.writeBigUInt64LE(dC.readBigUInt64LE(0),16);
for(let i=3;i<numParams;i++)tcArgs.writeBigUInt64LE(0n,i*8);

const ppTC=Number(ptr(tcArgs));
const tcKp=Buffer.alloc((numParams+1)*8);
for(let i=0;i<numParams;i++)tcKp.writeBigUInt64LE(BigInt(ppTC+i*8),i*8);
tcKp.writeBigUInt64LE(0n,numParams*8);

const gx=Math.ceil(N/32),gy=Math.ceil(M/32);
const FLOP=2n*BigInt(M)*BigInt(N)*BigInt(K);
const ITERS=10;

cs.cuLaunchKernel(fhTC,gx,gy,1,cfg.numWarps*32,1,1,shmem,0n,ptr(tcKp),null);
cs.cuCtxSynchronize();

const times:number[]=[];
for(let i=0;i<ITERS;i++){
  const t0=performance.now();
  cs.cuLaunchKernel(fhTC,gx,gy,1,cfg.numWarps*32,1,1,shmem,0n,ptr(tcKp),null);
  cs.cuCtxSynchronize();
  times.push(performance.now()-t0);
}
const avg=times.reduce((a,b)=>a+b,0)/times.length;
const tops=Number(FLOP)/(avg/1000)/1e12;
console.log(`INT8 TC matmul: ${(avg*1000).toFixed(0)} µs → ${tops.toFixed(2)} TFLOPS`);

cs.cuMemcpyDtoH_v2(hC,Number(dC.readBigUInt64LE(0)),BigInt(hC.byteLength));
const refC=new Int32Array(M*N);
for(let i=0;i<M;i++){
  for(let j=0;j<N;j++){
    let s=0;
    for(let k=0;k<K;k++)s+=hA[i*K+k]*hB[k*N+j];
    refC[i*N+j]=s;
  }
}
let maxErr=0,errCount=0;
for(let i=0;i<M*N;i++){
  const err=Math.abs(hC[i]-refC[i]);
  if(err>maxErr)maxErr=err;
  if(err!==0)errCount++;
}
console.log(`Max error: ${maxErr}, errors: ${errCount}/${M*N}`);
if(maxErr===0)console.log("✓ Correct!");else console.log("✗ ERRORS");
