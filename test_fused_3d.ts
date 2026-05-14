import { ptr } from "bun:ffi";
import { compileTTIR, getCuCtx, createCudaEvents } from "./kernel.ts";
import { readFileSync } from "fs";

const cs = getCuCtx();
const ev = createCudaEvents();
const eS = Number(ev.start.readBigUInt64LE(0)), eP = Number(ev.stop.readBigUInt64LE(0));
const el = new Float32Array(1);

const ttir = readFileSync("/tmp/fused_3d.ttir", "utf-8");
const { ptx, shmem } = compileTTIR(ttir, 4);
console.log(`PTX: ${(ptx.match(/mma\.sync/g)||[]).length} mma.sync, ${(ptx.match(/ld\.global/g)||[]).length} loads`);

const mod=Buffer.alloc(8);cs.cuModuleLoadData(mod,ptr(Buffer.from(ptx+"\0")));
const fn=Buffer.alloc(8);cs.cuModuleGetFunction(fn,Number(mod.readBigUInt64LE(0)),ptr(Buffer.from("fused\0")));
const fh=Number(fn.readBigUInt64LE(0));

const M=1024,N=1024,K=1024,BLK=32,BLKS_ROW=K/BLK,Q4K_SZ=20,NB=M*BLKS_ROW;
const hQ=Buffer.alloc(NB*Q4K_SZ);const hA_ref=new Int8Array(M*K);const hB=new Int8Array(K*N);const hC=new Int32Array(M*N);
for(let r=0;r<M;r++)for(let kb=0;kb<BLKS_ROW;kb++){const bi=r*BLKS_ROW+kb;const base=bi*Q4K_SZ;for(let j=0;j<16;j++){const lo=(Math.random()*16)|0,hi=(Math.random()*16)|0;hQ[base+j]=lo|(hi<<4);hA_ref[r*K+kb*BLK+j*2]=lo-8;hA_ref[r*K+kb*BLK+j*2+1]=hi-8;}}
for(let i=0;i<K*N;i++)hB[i]=(Math.random()*256-128)|0;

const SZ=BigInt(512*1024*1024);
const dQ=Buffer.alloc(8);cs.cuMemAlloc_v2(dQ,SZ);
const dB=Buffer.alloc(8);cs.cuMemAlloc_v2(dB,SZ);
const dC=Buffer.alloc(8);cs.cuMemAlloc_v2(dC,SZ);
cs.cuMemcpyHtoD_v2(Number(dQ.readBigUInt64LE(0)),hQ,BigInt(hQ.length));
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)),Buffer.from(hB.buffer),BigInt(hB.byteLength));

const np=(ptx.match(/\.param/g)||[]).length;
const kv:Buffer[]=[];kv.push(Buffer.alloc(8));kv[0].writeBigUInt64LE(dQ.readBigUInt64LE(0),0);
kv.push(Buffer.alloc(8));kv[1].writeBigUInt64LE(dB.readBigUInt64LE(0),0);
kv.push(Buffer.alloc(8));kv[2].writeBigUInt64LE(dC.readBigUInt64LE(0),0);
while(kv.length<np)kv.push(Buffer.alloc(8));
const kp=Buffer.alloc((np+1)*8);for(let i=0;i<np;i++)kp.writeBigUInt64LE(BigInt(Number(ptr(kv[i]))),i*8);kp.writeBigUInt64LE(0n,np*8);

const gx=Math.ceil(N/32),gy=Math.ceil(M/32);
cs.cuLaunchKernel(fh,gx,gy,1,128,1,1,shmem,0n,ptr(kp),null);
cs.cuCtxSynchronize();

// CUDA event timing
const t:number[]=[];for(let i=0;i<30;i++){cs.cuEventRecord(eS,0);cs.cuLaunchKernel(fh,gx,gy,1,128,1,1,shmem,0n,ptr(kp),null);cs.cuEventRecord(eP,0);cs.cuEventSynchronize(eP);cs.cuEventElapsedTime(el,eS,eP);t.push(el[0]*1000);}
const avg=t.reduce((a,b)=>a+b,0)/t.length;
const tops=2*M*N*K/(avg/1e6)/1e12;
console.log(`\n3D Fused: ${avg.toFixed(1)} µs → ${tops.toFixed(2)} TFLOPS`);

cs.cuMemcpyDtoH_v2(hC,Number(dC.readBigUInt64LE(0)),BigInt(M*N*4));
const refC=new Int32Array(M*N);for(let i=0;i<M;i++)for(let j=0;j<N;j++){let s=0;for(let k=0;k<K;k++)s+=hA_ref[i*K+k]*hB[k*N+j];refC[i*N+j]=s;}
let max=0,cnt=0;for(let i=0;i<M*N;i++){const d=Math.abs(hC[i]-refC[i]);if(d>max)max=d;if(d!==0)cnt++;}
console.log(`Errors: ${cnt}/${M*N}, max=${max}${max===0?" ✓":" ✗"}`);
