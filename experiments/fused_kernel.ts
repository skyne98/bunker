// fused_kernel.ts — Fused matmul via LLVM IR + inline PTX mma.sync
// Warp: 16 A rows × 8 B cols, K-loop 32× mma.sync (K=1024)
import { dlopen, ptr, CString } from "bun:ffi";
import { writeFileSync } from "fs";

const LLVM = "/nix/store/6r234y6pkbyyr8pk1wh7nfsmnzdxyswx-llvm-19.1.7-lib/lib/libLLVM-19.so";

function gen(): string {
  const E: string[] = [];
  const emit = (s: string) => E.push(`  ${s}`);

  emit(`%bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()`);
  emit(`%by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()`);
  emit(`%tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()`);
  emit(`%lid = and i32 %tx, 31`);
  emit(`%wid = lshr i32 %tx, 5`);
  emit(`%wo = shl i32 %wid, 4`);
  emit(`%wco_t = shl i32 %wid, 3`);
  emit(`%wco = and i32 %wco_t, 8`);
  emit(`%rb = add i32 %bx, %wo`);
  emit(`%cb = add i32 %by, %wco`);

  // K-loop: 32 iterations, phi for accumulators
  emit(`br label %lp`);
  emit(`lp:`);
  emit(`%ki = phi i32 [ 0, %entry ], [ %kn, %bd ]`);
  emit(`%p0 = phi i32 [ 0, %entry ], [ %r0, %bd ]`);
  emit(`%p1 = phi i32 [ 0, %entry ], [ %r1, %bd ]`);
  emit(`%p2 = phi i32 [ 0, %entry ], [ %r2, %bd ]`);
  emit(`%p3 = phi i32 [ 0, %entry ], [ %r3, %bd ]`);
  emit(`%go = icmp ult i32 %ki, 32`);
  emit(`br i1 %go, label %bd, label %dn`);

  emit(`bd:`);
  emit(`%koff = mul i32 %ki, 32`);

  // A: each thread loads 4 INT8 values, packs into 4× i32
  emit(`%arg = lshr i32 %lid, 2`);
  emit(`%arow = add i32 %rb, %arg`);
  emit(`%acg = and i32 %lid, 3`);
  emit(`%acoo = shl i32 %acg, 2`);
  emit(`%akt = add i32 %koff, %acoo`);
  emit(`%ai0 = mul i32 %arow, 1024`);
  emit(`%ai = add i32 %ai0, %akt`);
  emit(`%ap = getelementptr i8, ptr %A, i32 %ai`);
  // Volatile loads to prevent DCE
  for (let j = 0; j < 4; j++) {
    if (j > 0) emit(`%ag${j} = getelementptr i8, ptr %ap, i32 ${j}`);
    const off = j === 0 ? "%ap" : `%ag${j}`;
    emit(`%al${j} = load volatile i8, ptr ${off}`);
    emit(`%az${j} = zext i8 %al${j} to i32`);
  }
  emit(`%as1 = shl i32 %az1, 8`);
  emit(`%as2 = shl i32 %az2, 16`);
  emit(`%as3 = shl i32 %az3, 24`);
  emit(`%ao1 = or i32 %az0, %as1`);
  emit(`%ao2 = or i32 %ao1, %as2`);
  emit(`%apk = or i32 %ao2, %as3`);
  emit(`%a0v = add i32 %apk, 0`);
  emit(`%a1v = add i32 %apk, 0`);
  emit(`%a2v = add i32 %apk, 0`);
  emit(`%a3v = add i32 %apk, 0`);

  // B: each thread loads 8 INT8, packs into 2× i32
  emit(`%brow = add i32 %koff, %lid`);
  emit(`%bi0 = mul i32 %brow, 1024`);
  emit(`%bi = add i32 %bi0, %cb`);
  emit(`%bp = getelementptr i8, ptr %B, i32 %bi`);
  for (let j = 0; j < 8; j++) {
    if (j > 0) emit(`%bg${j} = getelementptr i8, ptr %bp, i32 ${j}`);
    const off = j === 0 ? "%bp" : `%bg${j}`;
    emit(`%bl${j} = load volatile i8, ptr ${off}`);
    emit(`%bz${j} = zext i8 %bl${j} to i32`);
  }
  emit(`%bs1 = shl i32 %bz1, 8`);  emit(`%bs2 = shl i32 %bz2, 16`);  emit(`%bs3 = shl i32 %bz3, 24`);
  emit(`%bo1 = or i32 %bz0, %bs1`);  emit(`%bo2 = or i32 %bo1, %bs2`);  emit(`%bpk0 = or i32 %bo2, %bs3`);
  emit(`%bs5 = shl i32 %bz5, 8`);  emit(`%bs6 = shl i32 %bz6, 16`);  emit(`%bs7 = shl i32 %bz7, 24`);
  emit(`%bo3 = or i32 %bz4, %bs5`);  emit(`%bo4 = or i32 %bo3, %bs6`);  emit(`%bpk1 = or i32 %bo4, %bs7`);

  // mma.sync: D[0..3]=A[0..3]×B[0..1]+C[0..3]
  emit(`%mma = call { i32, i32, i32, i32 } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { $0, $1, $2, $3 }, { $4, $5, $6, $7 }, { $8, $9 }, { $10, $11, $12, $13 }", "=r,=r,=r,=r,r,r,r,r,r,r,0,1,2,3"`);
  emit(`  (i32 %a0v, i32 %a1v, i32 %a2v, i32 %a3v, i32 %bpk0, i32 %bpk1, i32 %p0, i32 %p1, i32 %p2, i32 %p3)`);
  emit(`%r0 = extractvalue { i32, i32, i32, i32 } %mma, 0`);
  emit(`%r1 = extractvalue { i32, i32, i32, i32 } %mma, 1`);
  emit(`%r2 = extractvalue { i32, i32, i32, i32 } %mma, 2`);
  emit(`%r3 = extractvalue { i32, i32, i32, i32 } %mma, 3`);
  emit(`%kn = add i32 %ki, 1`);
  emit(`br label %lp`);

  // Store final accumulators
  emit(`dn:`);
  emit(`%co = mul i32 %rb, 1024`);
  emit(`%cc = add i32 %co, %cb`);
  emit(`%g0 = getelementptr i32, ptr %C, i32 %cc`);
  emit(`store i32 %p0, ptr %g0`);
  emit(`%g1 = getelementptr i32, ptr %g0, i32 1`);
  emit(`store i32 %p1, ptr %g1`);
  emit(`%g2 = getelementptr i32, ptr %g0, i32 2`);
  emit(`store i32 %p2, ptr %g2`);
  emit(`%g3 = getelementptr i32, ptr %g0, i32 3`);
  emit(`store i32 %p3, ptr %g3`);
  emit(`ret void`);

  return `define void @fused(ptr %A, ptr %B, ptr %C) {
entry:
${E.join("\n")}
}
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
!0 = !{ptr @fused, !"kernel", i32 1}
!nvvm.annotations = !{!0}
`;
}

// ── Compile + test ──
if (import.meta.main) {
  const llvm = gen();
  writeFileSync("/tmp/fused_v3.ll", llvm);

  const ls = dlopen(LLVM, {
    LLVMContextCreate:{args:[],returns:"pointer"},
    LLVMParseIRInContext:{args:["pointer","pointer","pointer","pointer"],returns:"i32"},
    LLVMCreateMemoryBufferWithMemoryRange:{args:["pointer","i64","pointer","i32"],returns:"pointer"},
    LLVMGetTargetFromTriple:{args:["pointer","pointer","pointer"],returns:"i32"},
    LLVMCreateTargetMachine:{args:["pointer","pointer","pointer","pointer","i32","i32","i32"],returns:"pointer"},
    LLVMTargetMachineEmitToMemoryBuffer:{args:["pointer","pointer","i32","pointer","pointer"],returns:"i32"},
    LLVMGetBufferSize:{args:["pointer"],returns:"i64"},
    LLVMGetBufferStart:{args:["pointer"],returns:"pointer"},
    LLVMInitializeNVPTXTargetInfo:{args:[],returns:"void"},
    LLVMInitializeNVPTXTarget:{args:[],returns:"void"},
    LLVMInitializeNVPTXTargetMC:{args:[],returns:"void"},
    LLVMInitializeNVPTXAsmPrinter:{args:[],returns:"void"},
  }).symbols;
  ls.LLVMInitializeNVPTXTargetInfo();ls.LLVMInitializeNVPTXTarget();
  ls.LLVMInitializeNVPTXTargetMC();ls.LLVMInitializeNVPTXAsmPrinter();

  const ctx=ls.LLVMContextCreate();
  const irBuf=Buffer.from(llvm+"\0");
  const mb=ls.LLVMCreateMemoryBufferWithMemoryRange(ptr(irBuf),BigInt(irBuf.length-1),ptr(Buffer.from("k.ll\0")),1);
  const ma=new BigUint64Array(1);const er=new BigUint64Array(1);
  const err=ls.LLVMParseIRInContext(ctx,mb,ptr(ma),ptr(er));
  if(err){const msg=er[0]!==0n?new CString(Number(er[0])).toString():"??";console.log("Parse error:",msg.substring(0,500));process.exit(1);}
  const mod=Number(ma[0]);
  const tp=Buffer.from("nvptx64-nvidia-cuda\0");const ta=new BigUint64Array(1);
  ls.LLVMGetTargetFromTriple(ptr(tp),ptr(ta),ptr(new BigUint64Array(1)));
  const tm=ls.LLVMCreateTargetMachine(Number(ta[0]),ptr(tp),ptr(Buffer.from("sm_86\0")),ptr(Buffer.from("\0")),2,0,0);
  const pa=new BigUint64Array(1);
  ls.LLVMTargetMachineEmitToMemoryBuffer(tm,mod,0,ptr(new BigUint64Array(1)),ptr(pa));
  const ptx=new CString(ls.LLVMGetBufferStart(Number(pa[0])),0,Number(ls.LLVMGetBufferSize(Number(pa[0])))).toString();
  console.log(`PTX: ${(ptx.match(/mma\.sync/g)||[]).length} mma.sync, ${(ptx.match(/ld\.global/g)||[]).length} loads`);

  // ── CUDA test ──
  const CUDA="/run/opengl-driver/lib/libcuda.so";
  const cs=dlopen(CUDA,{
    cuInit:{args:["u32"],returns:"i32"},cuDeviceGet:{args:["ptr","i32"],returns:"i32"},
    cuCtxCreate_v2:{args:["ptr","u32","i64"],returns:"i32"},
    cuModuleLoadData:{args:["ptr","ptr"],returns:"i32"},
    cuModuleGetFunction:{args:["ptr","i64","ptr"],returns:"i32"},
    cuLaunchKernel:{args:["i64","u32","u32","u32","u32","u32","u32","u32","ptr","ptr","ptr"],returns:"i32"},
    cuEventCreate:{args:["ptr","u32"],returns:"i32"},cuEventRecord:{args:["i64","i64"],returns:"i32"},
    cuEventSynchronize:{args:["i64"],returns:"i32"},cuEventElapsedTime:{args:["ptr","i64","i64"],returns:"i32"},
    cuMemAlloc_v2:{args:["ptr","i64"],returns:"i32"},cuMemcpyHtoD_v2:{args:["i64","ptr","i64"],returns:"i32"},
    cuMemcpyDtoH_v2:{args:["ptr","i64","i64"],returns:"i32"},cuMemFree_v2:{args:["i64"],returns:"i32"},
    cuCtxSynchronize:{args:[],returns:"i32"},
  }).symbols;
  cs.cuInit(0);const dev=new Int32Array(1);cs.cuDeviceGet(dev,0);
  cs.cuCtxCreate_v2(Buffer.alloc(8),0,BigInt(dev[0]));
  const eS=Buffer.alloc(8);cs.cuEventCreate(eS,0);const eP=Buffer.alloc(8);cs.cuEventCreate(eP,0);
  const ess=Number(eS.readBigUInt64LE(0)),epp=Number(eP.readBigUInt64LE(0)),elv=new Float32Array(1);

  const mod2=Buffer.alloc(8);cs.cuModuleLoadData(mod2,ptr(Buffer.from(ptx+"\0")));
  const fn2=Buffer.alloc(8);cs.cuModuleGetFunction(fn2,Number(mod2.readBigUInt64LE(0)),ptr(Buffer.from("fused\0")));
  const fh=Number(fn2.readBigUInt64LE(0));
  if(fh===0){console.log("Fn not found");process.exit(1);}

  const SZ=BigInt(512*1024*1024);
  const dA=Buffer.alloc(8);cs.cuMemAlloc_v2(dA,SZ);const dB=Buffer.alloc(8);cs.cuMemAlloc_v2(dB,SZ);const dC=Buffer.alloc(8);cs.cuMemAlloc_v2(dC,SZ);
  
  // Create test data: all-ones A and B → each result = K = 1024
  const M=1024,N=1024,K=1024;
  const hA=new Int8Array(M*K);hA.fill(1);
  const hB=new Int8Array(K*N);hB.fill(1);
  const hC=new Int32Array(M*N);
  cs.cuMemcpyHtoD_v2(Number(dA.readBigUInt64LE(0)),Buffer.from(hA.buffer),BigInt(hA.byteLength));
  cs.cuMemcpyHtoD_v2(Number(dB.readBigUInt64LE(0)),Buffer.from(hB.buffer),BigInt(hB.byteLength));
  
  // Quick param setup: 3 ptr params
  const args=Buffer.alloc(3*8);
  args.writeBigUInt64LE(dA.readBigUInt64LE(0),0);
  args.writeBigUInt64LE(dB.readBigUInt64LE(0),8);
  args.writeBigUInt64LE(dC.readBigUInt64LE(0),16);
  const abase=Number(ptr(args));const kp=Buffer.alloc(4*8);
  kp.writeBigUInt64LE(BigInt(abase),0);kp.writeBigUInt64LE(BigInt(abase+8),8);
  kp.writeBigUInt64LE(BigInt(abase+16),16);kp.writeBigUInt64LE(0n,24);

  const gx=Math.ceil(N/16),gy=Math.ceil(M/8);
  for(let i=0;i<3;i++)cs.cuLaunchKernel(fh,gx,gy,1,128,1,1,0,0n,ptr(kp),null);
  cs.cuCtxSynchronize();

  // CUDA event timing
  const ts:number[]=[];for(let i=0;i<10;i++){
    cs.cuEventRecord(ess,0);cs.cuLaunchKernel(fh,gx,gy,1,128,1,1,0,0n,ptr(kp),null);
    cs.cuEventRecord(epp,0);cs.cuEventSynchronize(epp);cs.cuEventElapsedTime(elv,ess,epp);ts.push(elv[0]*1000);
  }
  const avg=ts.reduce((a,b)=>a+b,0)/ts.length;
  const tops=2*1024*1024*1024/(avg/1e6)/1e12;
  console.log(`Fused: ${avg.toFixed(1)} µs  ${tops.toFixed(2)} TFLOPS`);

  // Verify
  cs.cuMemcpyDtoH_v2(hC,Number(dC.readBigUInt64LE(0)),BigInt(M*N*4));
  let max=0,cnt=0;
  for(let i=0;i<M*N;i++){const d=Math.abs(hC[i]-K);if(d>max)max=d;if(d!==0)cnt++;}
  console.log(`Errors: ${cnt}/${M*N}, max=${max}`);
}
