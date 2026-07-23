// Test with inlined compileTTIR
import { dlopen, ptr } from "bun:ffi";
import { execSync } from "child_process";
import { writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

function compileTTIR(ttir: string, numWarps: number): { ptx: string; shmem: number } {
  const f = join(tmpdir(), `k_${process.pid}_${Math.random().toString(36).slice(2)}.mlir`);
  writeFileSync(f, ttir);
  const out = execSync(`/tmp/triton_wrap ${f} ${numWarps}`, {
    cwd: __dirname, encoding: "utf-8", maxBuffer: 50 * 1024 * 1024,
    env: { ...process.env, LD_LIBRARY_PATH: "/nix/store/ixhlv41i2wpl84xgjcks061dz4yssbg3-zlib-1.3.2/lib:/nix/store/bfwqrbwqpbnsdbgf86gz8pn8vvddci3i-libxml2-2.13.8/lib" }
  });
  if (out.startsWith("ERROR:") || out.startsWith("loc(")) throw Error(out.substring(0, 500));
  return { ptx: out, shmem: 32768 };
}

const ttir = `module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul(%A: !tt.ptr<i8>, %B: !tt.ptr<i8>, %C: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32 %cBM = arith.constant 32 : i32 %cBN = arith.constant 32 : i32
    %c0_i64 = arith.constant 0 : i64 %c1_i64 = arith.constant 1 : i64
    %cBK = arith.constant 1024 : i64 %cS = arith.constant 1024 : i64
    %px = tt.get_program_id x : i32 %py = tt.get_program_id y : i32
    %bm = arith.muli %px, %cBM : i32 %bn = arith.muli %py, %cBN : i32
    %z = arith.constant dense<0> : tensor<32x32xi32>
    %tA = tt.make_tensor_ptr %A,[%cS,%cS],[%cS,%c1_i64],[%bm,%c0]{order=array<i32:1,0>}:!tt.ptr<tensor<32x1024xi8>>
    %tB = tt.make_tensor_ptr %B,[%cS,%cS],[%cS,%c1_i64],[%c0,%bn]{order=array<i32:1,0>}:!tt.ptr<tensor<1024x32xi8>>
    %a = tt.load %tA:!tt.ptr<tensor<32x1024xi8>>
    %b = tt.load %tB:!tt.ptr<tensor<1024x32xi8>>
    %c = tt.dot %a,%b,%z:tensor<32x1024xi8>*tensor<1024x32xi8>->tensor<32x32xi32>
    %tC = tt.make_tensor_ptr %C,[%cS,%cS],[%cS,%c1_i64],[%bm,%bn]{order=array<i32:1,0>}:!tt.ptr<tensor<32x32xi32>>
    tt.store %tC,%c:!tt.ptr<tensor<32x32xi32>>
    tt.return
  }
}`;

const {ptx, shmem} = compileTTIR(ttir, 4);
console.log(`PTX: ${(ptx.match(/mma\.sync/g)||[]).length} mma.sync`);

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
const hA=new Int8Array(M*K);for(let i=0;i<M*K;i++)hA[i]=(Math.random()*256-128)|0;
const hB=new Int8Array(K*N);for(let i=0;i<K*N;i++)hB[i]=(Math.random()*256-128)|0;
const hC=new Int32Array(M*N);
const SZ=BigInt(512*1024*1024);
const dA=Buffer.alloc(8);cs.cuMemAlloc_v2(dA,SZ);
const dB=Buffer.alloc(8);cs.cuMemAlloc_v2(dB,SZ);
const dCb=Buffer.alloc(8);cs.cuMemAlloc_v2(dCb,SZ);
cs.cuMemcpyHtoD_v2(Number(dA.readBigUInt64LE(0)),Buffer.from(hA.buffer),BigInt(hA.byteLength));
cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)),Buffer.from(hB.buffer),BigInt(hB.byteLength));

const mod=Buffer.alloc(8);cs.cuModuleLoadData(mod,ptr(Buffer.from(ptx+"\0")));
const fn=Buffer.alloc(8);cs.cuModuleGetFunction(fn,Number(mod.readBigUInt64LE(0)),ptr(Buffer.from("matmul\0")));
const fh=Number(fn.readBigUInt64LE(0));

const np=(ptx.match(/\.param/g)||[]).length;
const a=Buffer.alloc(np*8);
a.writeBigUInt64LE(dA.readBigUInt64LE(0),0);a.writeBigUInt64LE(dB.readBigUInt64LE(0),8);a.writeBigUInt64LE(dCb.readBigUInt64LE(0),16);
for(let i=3;i<np;i++)a.writeBigUInt64LE(0n,i*8);
const pp=Number(ptr(a));const kp=Buffer.alloc((np+1)*8);
for(let i=0;i<np;i++)kp.writeBigUInt64LE(BigInt(pp+i*8),i*8);kp.writeBigUInt64LE(0n,np*8);

const gx=Math.ceil(N/32),gy=Math.ceil(M/32);
cs.cuLaunchKernel(fh,gx,gy,1,128,1,1,32768,0n,ptr(kp),null);
cs.cuCtxSynchronize();
cs.cuMemcpyDtoH_v2(hC,Number(dCb.readBigUInt64LE(0)),BigInt(M*N*4));
const ref=new Int32Array(M*N);
for(let i=0;i<M;i++)for(let j=0;j<N;j++){let s=0;for(let k=0;k<K;k++)s+=hA[i*K+k]*hB[k*N+j];ref[i*N+j]=s;}
let max=0,cnt=0;
for(let i=0;i<M*N;i++){const d=Math.abs(hC[i]-ref[i]);if(d>max)max=d;if(d!==0)cnt++;}
console.log("Inlined Compile + matmul TTIR:",cnt,"errors, max=",max);
if(max===0)console.log("✓ Correct!"); else console.log("✗ ERRORS");
