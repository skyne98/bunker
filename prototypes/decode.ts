// prototypes/decode.ts — Autoregressive decode loop for Qwen3.5-0.8B
// Generates text token-by-token with proper state carry for GDN (recurrent state + conv state)
// and FA2 (KV cache + RoPE).
//   TOKENIZER_PATH=/tmp/tokenizer.json SAFETENSORS_PATH=/tmp/qwen35_0.8b.safetensors bun run prototypes/decode.ts "Hello"
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD as _cuHtoD, cuDtoH as _cuDtoH, cuFree, cuSync as _cuSync, cuLaunch as _cuLaunch, profGpuReport } from "../src/ttir";
import { dlopen, ptr as ffiPtr } from "bun:ffi";
import { performance } from "perf_hooks";

// ── Opt-in profiler (BUNKER_PROFILE=1): counts + per-call timing of every
//    runtime primitive, and reports where the per-token time goes. This is
//    the before/after metric for the sync-removal work.
const __PROF = process.env.BUNKER_PROFILE === "1";
const P = { launch: 0, sync: 0, htod: 0, dtoh: 0, launch_ns: 0n, sync_ns: 0n, htod_ns: 0n, dtoh_ns: 0n };
let p0 = performance.now();
function pmark(c: any, key: string, ns: number) { c[key] = c[key] + 1; c[key + "_ns"] = c[key + "_ns"] + BigInt(Math.round(ns)); }
const cuHtoD = (a: any, b: any, c?: any) => { const t = performance.now(); _cuHtoD(a, b, c); if (__PROF) pmark(P, "htod", (performance.now() - t) * 1e6); };
const cuDtoH = (a: any, b: any, c?: any) => { const t = performance.now(); _cuDtoH(a, b, c); if (__PROF) pmark(P, "dtoh", (performance.now() - t) * 1e6); };
const cuSync = () => { const t = performance.now(); const r = _cuSync(); if (__PROF) pmark(P, "sync", (performance.now() - t) * 1e6); return r; };
const cuLaunch = (k: any, g: any, bl: any, a: any) => { const t = performance.now(); const r = _cuLaunch(k, g, bl, a); if (__PROF) pmark(P, "launch", (performance.now() - t) * 1e6); return r; };
function profReset() { P.launch=0; P.sync=0; P.htod=0; P.dtoh=0; P.launch_ns=0n; P.sync_ns=0n; P.htod_ns=0n; P.dtoh_ns=0n; }
function profReport() {
  const dt = performance.now() - p0;
  const f = (n: bigint) => (Number(n) / 1e6).toFixed(2) + "ms";
  console.log("\n── BUNKER_PROFILE ────────────────────────────");
  console.log(`  totaling  ${dt.toFixed(1)}ms`);
  console.log(`  launches: ${P.launch} (${f(P.launch_ns)})`);
  console.log(`  syncs:    ${P.sync} (${f(P.sync_ns)})`);
  console.log(`  htod:     ${P.htod} (${f(P.htod_ns)})`);
  console.log(`  dtoh:     ${P.dtoh} (${f(P.dtoh_ns)})`);
  console.log(`  ── CPU-side time by category (share of total) ──`);
  const cpu = P.launch_ns + P.sync_ns + P.htod_ns + P.dtoh_ns;
  console.log(`  primitives: ${(Number(cpu) / 1e6).toFixed(1)}ms (${(Number(cpu) / (dt * 1e6) * 100).toFixed(0)}% of total)`);
  console.log(`  non-primitive (module code, palloc, etc): ${(dt - Number(cpu) / 1e6).toFixed(1)}ms`);
  console.log(`───────────────────────────────────────────────`);
}

// ── Memory pool: eliminate cuAlloc overhead ──
const _pool = new Map<number, bigint[]>();
const _sizes = new Map<bigint, number>();
let _allocCount = 0, _poolHits = 0;
function palloc(bytes: number): bigint {
  const stack = _pool.get(bytes);
  if (stack && stack.length > 0) { _poolHits++; return stack.pop()!; }
  _allocCount++;
  const ptr = cuAlloc(BigInt(bytes));
  _sizes.set(ptr, bytes);
  return ptr;
}
function pfree(ptr: bigint) {
  const sz = _sizes.get(ptr);
  if (sz === undefined) return;
  if (!_pool.has(sz)) _pool.set(sz, []);
  _pool.get(sz)!.push(ptr);
}

const H=1024, VOCAB=248320, NL=24, FAI=4, INTER=3584, QKVD=6144, ZD=2048, EPS=1e-6;
const NH=8, NKV=2, HD=256, LKH=16, LVH=16, LKD=128, LVD=128, KEYDIM=LKH*LKD, VALDIM=LVH*LVD;
const QGATE=NH*HD*2, KV_DIM=NKV*HD;
const ROT_DIM=64, ROT_HALF=32; // partial_rotary_factor=0.25 → 64 of 256
const isFull = (l:number) => l % FAI === FAI-1;
const BLK1 = (n:number) => Math.ceil(n/1024);
const MAX_LEN = 128;

async function loadWeights(path:string) {
  const data = await Bun.file(path).bytes();
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const hl = Number(dv.getBigUint64(0, true));
  const hdr = JSON.parse(new TextDecoder().decode(data.subarray(8, 8+hl)));
  const ds = 8+hl;
  const base = cuAlloc(BigInt(data.length-ds));
  cuHtoD(base, data.subarray(ds)); cuSync();
  const W = new Map<string,bigint>();
  for (const [n,i] of Object.entries(hdr)) { if(n==="__metadata__")continue; W.set(n, base+BigInt((i as any).data_offsets[0])); }
  return W;
}
const wp = (W:Map<string,bigint>,l:number,n:string) => { const v=W.get(`model.language_model.layers.${l}.${n}`); if(!v) throw new Error(`missing: layers.${l}.${n}`); return v; };

const gtLib = dlopen("./shim/libgigatoken_wrapper.so", {
  gt_init:{args:["ptr"],returns:"ptr"}, gt_encode:{args:["ptr","ptr","u64","ptr","u64"],returns:"i32"}, gt_free:{args:["ptr"],returns:"void"},
}).symbols;
function tokenize(text:string, tokPath:string):Uint32Array {
  const h=gtLib.gt_init(ffiPtr(Buffer.from(tokPath+"\0"))) as number; if(!h) throw new Error("gt_init failed");
  const bytes=new TextEncoder().encode(text);
  const out=new Uint32Array(bytes.length+64);
  const rc=gtLib.gt_encode(h,ffiPtr(bytes),BigInt(bytes.length),ffiPtr(out),BigInt(out.length));
  gtLib.gt_free(h); if(rc<0) throw new Error("gt_encode failed");
  return out.subarray(0,rc);
}

// ── GEMM, RMSNorm, SwiGLU, Add, Cast (same as qwen35.ts) ──
// ── GEMM with optional epilogue fusion (Cast, Add) ──
// opts.cast: store output as bf16 instead of f32 (fuses Cast into GEMM)
// opts.add: load residual (bf16), add to output before cast/store (fuses Cast+Add)
// opts.N2: if set, compute a SECOND GEMM [M,N2,K] with same A but different B2, output to C2 (fuses Gate+Up, A+B)
function buildMM(M:number,N:number,K:number,opts?:{cast?:boolean,add?:boolean,N2?:number,BN?:number}) {
  const BM=Math.min(64,M), BN=opts?.BN??Math.min(64,N), BK=Math.min(64,K);
  const b=new TTIRBuilder();
  const outElem=opts?.cast?"bf16":"f32";
  const A=b.param("A",{ptr:"bf16"}),B=b.param("B",{ptr:"bf16"});
  const C=b.param("C",{ptr:outElem});
  const params=[A,B,C];
  if(opts?.add){const R=b.param("R",{ptr:"bf16"});params.push(R);}
  if(opts?.N2){const B2=b.param("B2",{ptr:"bf16"}),C2=b.param("C2",{ptr:outElem});params.push(B2,C2);}
  const pM=b.programId(0),pN=b.programId(1);
  const tpA=b.makeTensorPtr(A,[M,K],[K,1],[b.mul(pM,b.i32(BM)),b.i32(0)],[BM,BK],"bf16",[1,0]);
  const tpB=b.makeTensorPtr(B,[K,N],[1,K],[b.i32(0),b.mul(pN,b.i32(BN))],[BK,BN],"bf16",[0,1]);
  const tpC=b.makeTensorPtr(C,[M,N],[N,1],[b.mul(pM,b.i32(BM)),b.mul(pN,b.i32(BN))],[BM,BN],outElem,[1,0]);
  // Residual pointer (if add epilogue)
  let tpR:any=null;
  if(opts?.add){const R=params[3];tpR=b.makeTensorPtr(R,[M,N],[N,1],[b.mul(pM,b.i32(BM)),b.mul(pN,b.i32(BN))],[BM,BN],"bf16",[1,0]);}
  // Second GEMM pointers (if N2)
  let tpB2:any=null,tpC2:any=null,BN2=0;
  if(opts?.N2){BN2=opts.BN??Math.min(64,opts.N2);const B2=params[opts?.add?4:3],C2=params[opts?.add?5:4];
    tpB2=b.makeTensorPtr(B2,[K,opts.N2],[1,K],[b.i32(0),b.mul(pN,b.i32(BN2))],[BK,BN2],"bf16",[0,1]);
    tpC2=b.makeTensorPtr(C2,[M,opts.N2],[opts.N2,1],[b.mul(pM,b.i32(BM)),b.mul(pN,b.i32(BN2))],[BM,BN2],outElem,[1,0]);}
  // Main GEMM
  const a0=b.zeros([BM,BN],"f32");
  const [acc,tpA2,tpB2_] = b.forIter(b.index(0),b.index(K),b.index(BK),[a0,tpA,tpB],(bb,_,[a,tA,tB])=>{
    const n=bb.dot(bb.load(tA),bb.load(tB),a);
    return[n,bb.advance(tA,[bb.i32(0),bb.i32(BK)]),bb.advance(tB,[bb.i32(BK),bb.i32(0)])];
  });
  // Epilogue: add residual
  let result=acc;
  if(opts?.add){const res=b.fpext(b.load(tpR,{boundaryCheck:[0,1],padding:1}),"f32");result=b.add(result,res);}
  // Store (with optional cast)
  const storeVal=opts?.cast?b.fptrunc(result,"bf16"):result;
  b.store(tpC,storeVal,{boundaryCheck:[0,1]});
  // Second GEMM (shares A, same K-loop)
  if(opts?.N2){
    const a02=b.zeros([BM,BN2],"f32");
    const [acc2] = b.forIter(b.index(0),b.index(K),b.index(BK),[a02,tpA,tpB2],(bb,_,[a2,tA2,tB2])=>{
      const n=bb.dot(bb.load(tA2),bb.load(tB2),a2);
      return[n,bb.advance(tA2,[bb.i32(0),bb.i32(BK)]),bb.advance(tB2,[bb.i32(BK),bb.i32(0)])];
    });
    let result2=acc2;
    const storeVal2=opts?.cast?b.fptrunc(result2,"bf16"):result2;
    b.store(tpC2,storeVal2,{boundaryCheck:[0,1]});
  }
  const numParams=3+(opts?.add?1:0)+(opts?.N2?2:0);
  return b.build("mm",4,numParams);
}

// ── Fused Cast+Add: y = cast(a) + b  (saves 1 launch + 1 intermediate buffer) ──
function buildCastAdd(N:number) {
  const b=new TTIRBuilder();
  const A=b.param("A",{ptr:"f32"}),B=b.param("B",{ptr:"bf16"}),O=b.param("O",{ptr:"bf16"});
  const row=b.programId(0);const BLK=Math.min(1024,N);const off=b.mul(row,b.i32(BLK));
  const tpA=b.makeTensorPtr(A,[1,N],[N,1],[b.i32(0),off],[1,BLK],"f32",[1,0]);
  const tpB=b.makeTensorPtr(B,[1,N],[N,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const tpO=b.makeTensorPtr(O,[1,N],[N,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const a=b.load(tpA,{boundaryCheck:[0,1],padding:1});
  const bb=b.fpext(b.load(tpB,{boundaryCheck:[0,1],padding:1}),"f32");
  b.store(tpO,b.fptrunc(b.add(a,bb),"bf16"),{boundaryCheck:[0,1]});
  return b.build("ca",4,3);
}
function buildRMS(N:number) {
  const b=new TTIRBuilder();
  const X=b.param("X",{ptr:"bf16"}),Wt=b.param("W",{ptr:"bf16"}),Y=b.param("Y",{ptr:"bf16"});
  const row=b.programId(0);
  const tpX=b.makeTensorPtr(X,[1,N],[N,1],[row,b.i32(0)],[1,N],"bf16",[1,0]);
  const tpW=b.makeTensorPtr(Wt,[1,N],[N,1],[b.i32(0),b.i32(0)],[1,N],"bf16",[1,0]);
  const tpY=b.makeTensorPtr(Y,[1,N],[N,1],[row,b.i32(0)],[1,N],"bf16",[1,0]);
  const x=b.fpext(b.load(tpX,{boundaryCheck:[0,1],padding:1}),"f32");
  const ms=b.divf(b.sum(b.mul(x,x),1),b.f32(N));
  const msBc=b.broadcast(b.expandDims(ms,1),[1,N]);
  const yy=b.rsqrtHw(b.add(msBc,b.f32(EPS)));
  let y=b.mul(x,yy);
  y=b.mul(y,b.add(b.f32(1),b.fpext(b.load(tpW,{boundaryCheck:[0,1],padding:1}),"f32")));
  b.store(tpY,b.fptrunc(y,"bf16"),{boundaryCheck:[0,1]});
  return b.build("rms",4,3);
}
function buildSwiGLU(N:number) {
  const b=new TTIRBuilder();
  const G=b.param("G",{ptr:"f32"}),U=b.param("U",{ptr:"f32"}),O=b.param("O",{ptr:"bf16"});
  const row=b.programId(0); const BLK=Math.min(1024,N); const off=b.mul(row,b.i32(BLK));
  const tpG=b.makeTensorPtr(G,[1,N],[N,1],[b.i32(0),off],[1,BLK],"f32",[1,0]);
  const tpU=b.makeTensorPtr(U,[1,N],[N,1],[b.i32(0),off],[1,BLK],"f32",[1,0]);
  const tpO=b.makeTensorPtr(O,[1,N],[N,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const g=b.load(tpG,{boundaryCheck:[0,1],padding:1}); const u=b.load(tpU,{boundaryCheck:[0,1],padding:1});
  const sig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(g,b.f32(-1)))));
  b.store(tpO,b.fptrunc(b.mul(b.mul(g,sig),u),"bf16"),{boundaryCheck:[0,1]});
  return b.build("sg",4,3);
}
function buildAdd(N:number) {
  const b=new TTIRBuilder();
  const A=b.param("A",{ptr:"bf16"}),B=b.param("B",{ptr:"bf16"}),O=b.param("O",{ptr:"bf16"});
  const row=b.programId(0); const BLK=Math.min(1024,N); const off=b.mul(row,b.i32(BLK));
  const tpA=b.makeTensorPtr(A,[1,N],[N,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const tpB=b.makeTensorPtr(B,[1,N],[N,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const tpO=b.makeTensorPtr(O,[1,N],[N,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const a=b.fpext(b.load(tpA,{boundaryCheck:[0,1],padding:1}),"f32");
  const bb=b.fpext(b.load(tpB,{boundaryCheck:[0,1],padding:1}),"f32");
  b.store(tpO,b.fptrunc(b.add(a,bb),"bf16"),{boundaryCheck:[0,1]});
  return b.build("ad",4,3);
}
function buildCast(N:number) {
  const b=new TTIRBuilder();
  const X=b.param("X",{ptr:"f32"}),Y=b.param("Y",{ptr:"bf16"});
  const row=b.programId(0); const BLK=Math.min(1024,N); const off=b.mul(row,b.i32(BLK));
  const tpX=b.makeTensorPtr(X,[1,N],[N,1],[b.i32(0),off],[1,BLK],"f32",[1,0]);
  const tpY=b.makeTensorPtr(Y,[1,N],[N,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const x=b.load(tpX,{boundaryCheck:[0,1],padding:1});
  b.store(tpY,b.fptrunc(x,"bf16"),{boundaryCheck:[0,1]});
  return b.build("cs",4,3);
}

// ── GPU embedding gather: X[i] = E[id*H + i] (bf16) ──
// Replaces the per-token CPU read + synchronous HtoD (which dominated decode
// latency: ~70% of wall time measured). Grid = [1], program id 0 = token id.
// Explicit pointer-tensor form: per-element base + id*H + arange.
function buildEmbed() {
  const b=new TTIRBuilder();
  const E=b.param("E",{ptr:"bf16"});   // [VOCAB, H] row-major
  const X=b.param("X",{ptr:"bf16"});   // [H] flat
  const ID=b.param("ID","i32");        // scalar token id (like FA2 Pos)
  const Hh=1024;
  const tpE=b.makeTensorPtr(E,[VOCAB,Hh],[Hh,1],[ID,b.i32(0)],[1,Hh],"bf16",[1,0]);
  const tpX=b.makeTensorPtr(X,[1,Hh],[Hh,1],[b.i32(0),b.i32(0)],[1,Hh],"bf16",[1,0]);
  b.store(tpX,b.load(tpE,{}),{});
  return b.build("emb",4,3);
}

// ── Conv1d decode kernel (with conv state) ──
// Grid: [BLK1(QKVD)]. Each program processes 1024 channels.
// conv_out[c] = silu(w0*s0 + w1*s1 + w2*s2 + w3*current)
// Updates conv state: s0←s1, s1←s2, s2←current
function buildConv1dDecode() {
  const b=new TTIRBuilder();
  const QKV=b.param("Q",{ptr:"bf16"}),CS=b.param("CS",{ptr:"bf16"}),CW=b.param("CW",{ptr:"bf16"});
  const Out=b.param("O",{ptr:"bf16"}),CSN=b.param("SN",{ptr:"bf16"});
  const pid=b.programId(0); const BLK=1024;
  const off=b.mul(pid,b.i32(BLK));
  // Load current qkv
  const tpQ=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const qkv=b.fpext(b.load(tpQ,{boundaryCheck:[0,1],padding:1}),"f32");
  // Load conv state (3 time steps) — CS is [3, QKVD] row-major
  const tpS0=b.makeTensorPtr(CS,[3,QKVD],[QKVD,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const tpS1=b.makeTensorPtr(CS,[3,QKVD],[QKVD,1],[b.i32(1),off],[1,BLK],"bf16",[1,0]);
  const tpS2=b.makeTensorPtr(CS,[3,QKVD],[QKVD,1],[b.i32(2),off],[1,BLK],"bf16",[1,0]);
  const s0=b.fpext(b.load(tpS0,{boundaryCheck:[0,1],padding:1}),"f32");
  const s1=b.fpext(b.load(tpS1,{boundaryCheck:[0,1],padding:1}),"f32");
  const s2=b.fpext(b.load(tpS2,{boundaryCheck:[0,1],padding:1}),"f32");
  // Load conv weights (4 taps, stride 4) — CW is [QKVD, 4] flat
  const cwIdx=b.mul(b.arange(0,BLK),b.i32(4));
  const cwBase=b.mul(off,b.i32(4));
  const w0=b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(CW,BLK,"bf16"),b.add(cwIdx,cwBase))),"f32"),[1,BLK]);
  const w1=b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(CW,BLK,"bf16"),b.add(cwIdx,b.add(cwBase,b.i32(1))))),"f32"),[1,BLK]);
  const w2=b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(CW,BLK,"bf16"),b.add(cwIdx,b.add(cwBase,b.i32(2))))),"f32"),[1,BLK]);
  const w3=b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(CW,BLK,"bf16"),b.add(cwIdx,b.add(cwBase,b.i32(3))))),"f32"),[1,BLK]);
  // Compute conv output
  const convRaw=b.add(b.add(b.mul(w0,s0),b.mul(w1,s1)),b.add(b.mul(w2,s2),b.mul(w3,qkv)));
  const convSig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(convRaw,b.f32(-1)))));
  const convOut=b.mul(convRaw,convSig);
  // Store output
  const tpO=b.makeTensorPtr(Out,[1,QKVD],[QKVD,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  b.store(tpO,b.fptrunc(convOut,"bf16"),{boundaryCheck:[0,1]});
  // Store updated conv state: s0←s1, s1←s2, s2←qkv
  const tpSN0=b.makeTensorPtr(CSN,[3,QKVD],[QKVD,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const tpSN1=b.makeTensorPtr(CSN,[3,QKVD],[QKVD,1],[b.i32(1),off],[1,BLK],"bf16",[1,0]);
  const tpSN2=b.makeTensorPtr(CSN,[3,QKVD],[QKVD,1],[b.i32(2),off],[1,BLK],"bf16",[1,0]);
  b.store(tpSN0,b.fptrunc(s1,"bf16"),{boundaryCheck:[0,1]});
  b.store(tpSN1,b.fptrunc(s2,"bf16"),{boundaryCheck:[0,1]});
  b.store(tpSN2,b.fptrunc(qkv,"bf16"),{boundaryCheck:[0,1]});
  return b.build("cv1d_d",4,5);
}

// ── GDN delta rule decode kernel (with recurrent state) ──
// Grid: [LVH] (16 heads). Takes precomputed decay (exp(g)) since softplus needs log.
function buildGDNDecode() {
  const b=new TTIRBuilder();
  const ConvOut=b.param("CO",{ptr:"bf16"}),Z=b.param("Z",{ptr:"bf16"});
  const ALog=b.param("AL",{ptr:"f32"}),dtB=b.param("DT",{ptr:"bf16"});
  const aP=b.param("AP",{ptr:"f32"});
  const bP=b.param("BP",{ptr:"f32"});
  const NormW=b.param("NW",{ptr:"f32"}),SState=b.param("SS",{ptr:"f32"});
  const Out=b.param("O",{ptr:"bf16"}),SStateNew=b.param("SN",{ptr:"f32"});
  const head=b.programId(0);
  const qOff=b.mul(head,b.i32(LKD));
  const kOff=b.add(b.i32(KEYDIM),qOff);
  const vOff=b.add(b.i32(2*KEYDIM),qOff);
  // Load q, k, v from conv1d output
  const tpQ=b.makeTensorPtr(ConvOut,[1,QKVD],[QKVD,1],[b.i32(0),qOff],[1,LKD],"bf16",[1,0]);
  const tpK=b.makeTensorPtr(ConvOut,[1,QKVD],[QKVD,1],[b.i32(0),kOff],[1,LKD],"bf16",[1,0]);
  const tpV=b.makeTensorPtr(ConvOut,[1,QKVD],[QKVD,1],[b.i32(0),vOff],[1,LKD],"bf16",[1,0]);
  const qRaw=b.fpext(b.load(tpQ,{boundaryCheck:[0,1],padding:1}),"f32");
  const kRaw=b.fpext(b.load(tpK,{boundaryCheck:[0,1],padding:1}),"f32");
  const vRaw=b.fpext(b.load(tpV,{boundaryCheck:[0,1],padding:1}),"f32");
  // L2norm q, k (matching FLA: rsqrt(sum(q^2)+1e-6), q *= 1/sqrt(d_k))
  const qRstd=b.rsqrtHw(b.add(b.sum(b.mul(qRaw,qRaw),1),b.f32(1e-6)));
  const kRstd=b.rsqrtHw(b.add(b.sum(b.mul(kRaw,kRaw),1),b.f32(1e-6)));
  const qNorm=b.mul(qRaw,b.mul(qRstd,b.f32(1/Math.sqrt(LKD))));
  const kNorm=b.mul(kRaw,kRstd);
  // Compute decay on GPU: g = -exp(A_log) * softplus(aP + dt_bias)
  // softplus(x) = log(1+exp(x)) = log2(1+exp(x)) * ln(2)
  const aVal=b.load(b.addptr(b.splatPtr(aP,1,"f32"),head));
  const dtVal=b.fpext(b.load(b.addptr(b.splatPtr(dtB,1,"bf16"),head)),"f32");
  const alVal=b.load(b.addptr(b.splatPtr(ALog,1,"f32"),head));
  const sp_in=b.add(aVal,dtVal);
  const one_plus_exp=b.add(b.f32(1),b.exp(sp_in));
  const softplus=b.mul(b.log2Hw(one_plus_exp),b.f32(Math.LN2));
  const decayExp=b.exp(b.mul(b.f32(-1),b.mul(b.exp(alVal),softplus)));
  
  // Load beta = sigmoid(bP[head])
  const bVal=b.load(b.addptr(b.splatPtr(bP,1,"f32"),head));
  const beta=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(bVal,b.f32(-1)))));
  const tpS=b.makeTensorPtr(SState,[LVH*LKD,LVD],[LVD,1],[b.mul(head,b.i32(LKD)),b.i32(0)],[LKD,LVD],"f32",[1,0]);
  const S=b.load(tpS,{boundaryCheck:[0,1],padding:1});
  // S = S * exp(decay) (element-wise scale)
  const Sdecayed=b.mul(S,b.broadcastTo(decayExp,[LKD,LVD]));
  // kS = k · S = sum(k_col * S, 0) where k_col[i,j] = k[i]
  // Load k as 1D [128] using pointer-tensor
  const k1d=b.fpext(b.load(b.addptr(b.splatPtr(ConvOut,LKD,"bf16"),kOff)),"f32"); // wrong — need pointer to k section
  // Actually, I need to load k from ConvOut at offset kOff. Let me use makeTensorPtr + reshape.
  // Or just load k as [1, 128] and reshape to [128].
  const kCol=b.broadcast(b.expandDims(b.reshape(kNorm,[LKD]),1),[LKD,LVD]); // [128, 128] k[i] per column
  const kS=b.sum(b.mul(kCol,Sdecayed),0); // [128] — sum over dim 0
  // delta = (v - kS) * beta
  const delta=b.mul(b.sub(vRaw,b.broadcastTo(kS,[1,LKD])),b.broadcastTo(beta,[1,LKD])); // [1, 128]
  // S += k ⊗ delta (outer product via broadcast)
  const deltaBc=b.broadcastTo(delta,[LKD,LVD]); // [128, 128] delta[j] per row
  const Snew=b.add(Sdecayed,b.mul(kCol,deltaBc)); // [128, 128]
  // Store S_new
  const tpSN=b.makeTensorPtr(SStateNew,[LVH*LKD,LVD],[LVD,1],[b.mul(head,b.i32(LKD)),b.i32(0)],[LKD,LVD],"f32",[1,0]);
  b.store(tpSN,Snew,{boundaryCheck:[0,1]});
  // o = q · S_new = sum(q_col * S_new, 0) where q_col[i,j] = q[i]
  const qCol=b.broadcast(b.expandDims(b.reshape(qNorm,[LKD]),1),[LKD,LVD]); // [128, 128]
  const o=b.sum(b.mul(qCol,Snew),0); // [128]
  // RMSNormGated: norm(o) * weight * silu(z)
  const oMs=b.divf(b.sum(b.mul(o,o),0),b.f32(LVD));
  const oRstd=b.rsqrtHw(b.add(oMs,b.f32(EPS)));
  const oNormed=b.mul(o,oRstd);
  const tpNW=b.makeTensorPtr(NormW,[1,LVD],[LVD,1],[b.i32(0),b.i32(0)],[1,LVD],"f32",[1,0]);
  const nw=b.load(tpNW,{boundaryCheck:[0,1],padding:1});
  const oWeighted=b.mul(b.broadcastTo(oNormed,[1,LVD]),nw);
  const tpZ=b.makeTensorPtr(Z,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"bf16",[1,0]);
  const zVal=b.fpext(b.load(tpZ,{boundaryCheck:[0,1],padding:1}),"f32");
  const zSig=b.mul(zVal,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(zVal,b.f32(-1))))));
  const oGated=b.mul(oWeighted,zSig);
  // Store output
  const tpO=b.makeTensorPtr(Out,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"bf16",[1,0]);
  b.store(tpO,b.fptrunc(oGated,"bf16"),{boundaryCheck:[0,1]});
  return b.build("gdn_d",4,10);
}

// ── FA2 decode kernel (with RoPE + KV cache) ──
// Grid: [NH] (8 Q heads). Each program handles one Q head (HD=256).
// For seq=1 decode: q[1,256] attends to k_cache[T,256], v_cache[T,256]
// ── q_norm kernel (writes normalized q to QBuf) ──
function buildQNorm() {
  const b=new TTIRBuilder();
  const QG=b.param("QG",{ptr:"bf16"}),QNW=b.param("QNW",{ptr:"bf16"}),QN=b.param("QN",{ptr:"bf16"});
  const head=b.programId(0);
  const qOffIn=b.mul(head,b.i32(HD*2));
  const qOffOut=b.mul(head,b.i32(HD));
  const tpQ=b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),qOffIn],[1,HD],"bf16",[1,0]);
  const qRaw=b.fpext(b.load(tpQ,{boundaryCheck:[0,1],padding:1}),"f32");
  const rstd=b.rsqrtHw(b.add(b.divf(b.sum(b.mul(qRaw,qRaw),1),b.f32(HD)),b.f32(EPS)));
  let qNorm=b.mul(qRaw,rstd);
  const tpQN=b.makeTensorPtr(QNW,[1,HD],[HD,1],[b.i32(0),b.i32(0)],[1,HD],"bf16",[1,0]);
  qNorm=b.mul(qNorm,b.add(b.f32(1),b.fpext(b.load(tpQN,{boundaryCheck:[0,1],padding:1}),"f32")));
  const tpOut=b.makeTensorPtr(QN,[1,NH*HD],[NH*HD,1],[b.i32(0),qOffOut],[1,HD],"bf16",[1,0]);
  b.store(tpOut,b.fptrunc(qNorm,"bf16"),{boundaryCheck:[0,1]});
  return b.build("qn",4,3);
}
// k_norm: RMSNorm for k_proj output (2 heads × 256)
function buildKNormD() {
  const b=new TTIRBuilder();
  const K=b.param("K",{ptr:"bf16"}),KNW=b.param("KNW",{ptr:"bf16"}),Out=b.param("O",{ptr:"bf16"});
  const head=b.programId(0);
  const off=b.mul(head,b.i32(HD));
  const tpK=b.makeTensorPtr(K,[1,KV_DIM],[KV_DIM,1],[b.i32(0),off],[1,HD],"bf16",[1,0]);
  const kRaw=b.fpext(b.load(tpK,{boundaryCheck:[0,1],padding:1}),"f32");
  const kRstd=b.rsqrtHw(b.add(b.divf(b.sum(b.mul(kRaw,kRaw),1),b.f32(HD)),b.f32(EPS)));
  let kNorm=b.mul(kRaw,kRstd);
  const tpKN=b.makeTensorPtr(KNW,[1,HD],[HD,1],[b.i32(0),b.i32(0)],[1,HD],"bf16",[1,0]);
  kNorm=b.mul(kNorm,b.add(b.f32(1),b.fpext(b.load(tpKN,{boundaryCheck:[0,1],padding:1}),"f32")));
  const tpO=b.makeTensorPtr(Out,[1,KV_DIM],[KV_DIM,1],[b.i32(0),off],[1,HD],"bf16",[1,0]);
  b.store(tpO,b.fptrunc(kNorm,"bf16"),{boundaryCheck:[0,1]});
  return b.build("knormd",4,3);
}
// ── FA2 attention kernel (takes pre-rotated q, k, v) ──
function buildFA2Attn() {
  const b=new TTIRBuilder();
  const QR=b.param("QR",{ptr:"bf16"}),KR=b.param("KR",{ptr:"bf16"}),VB=b.param("V",{ptr:"bf16"});
  const QG=b.param("QG",{ptr:"bf16"});
  const KC=b.param("KC",{ptr:"bf16"}),VC=b.param("VC",{ptr:"bf16"});
  const Out=b.param("O",{ptr:"bf16"});
  const Pos=b.param("P","i32");
  const head=b.programId(0);
  const headKv=b.divi(head,b.i32(NH/NKV));
  const qOff=b.mul(head,b.i32(HD));
  const kvOff=b.mul(headKv,b.i32(HD));
  const gOff=b.add(b.mul(head,b.i32(HD*2)),b.i32(HD));
  const q=b.fpext(b.load(b.makeTensorPtr(QR,[1,NH*HD],[NH*HD,1],[b.i32(0),qOff],[1,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const k=b.fpext(b.load(b.makeTensorPtr(KR,[1,KV_DIM],[KV_DIM,1],[b.i32(0),kvOff],[1,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const cacheRow=b.add(b.mul(head,b.i32(MAX_LEN)),Pos);
  b.store(b.makeTensorPtr(KC,[NH*MAX_LEN,HD],[HD,1],[cacheRow,b.i32(0)],[1,HD],"bf16",[1,0]),b.fptrunc(k,"bf16"),{boundaryCheck:[0,1]});
  const v=b.fpext(b.load(b.makeTensorPtr(VB,[1,KV_DIM],[KV_DIM,1],[b.i32(0),kvOff],[1,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  b.store(b.makeTensorPtr(VC,[NH*MAX_LEN,HD],[HD,1],[cacheRow,b.i32(0)],[1,HD],"bf16",[1,0]),b.fptrunc(v,"bf16"),{boundaryCheck:[0,1]});
  const kCache=b.fpext(b.load(b.makeTensorPtr(KC,[NH*MAX_LEN,HD],[HD,1],[b.mul(head,b.i32(MAX_LEN)),b.i32(0)],[MAX_LEN,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const qBc=b.broadcastTo(q,[MAX_LEN,HD]);
  const scores=b.mul(b.sum(b.mul(qBc,kCache),1),b.f32(1/Math.sqrt(HD)));
  // Inline causal mask: sel(j <= Pos, 0, -1e30) — no per-token host transfer.
  const ar=b.arange(0,MAX_LEN);                 // [128] i32
  const le= b.le(ar,b.broadcastTo(Pos,[MAX_LEN])); // [128] i1
  const selArr=b.select(le,b.splat(b.f32(0),[MAX_LEN],"f32"),b.splat(b.f32(-1e30),[MAX_LEN],"f32")); // [128] f32
  const maskArr=b.broadcastTo(selArr,[1,MAX_LEN]); // [1,128] f32
  const scoresMasked=b.add(b.broadcastTo(scores,[1,MAX_LEN]),maskArr);
  const maxScore=b.max(scoresMasked,1);
  const expScores=b.exp(b.sub(b.broadcastTo(scoresMasked,[1,MAX_LEN]),b.broadcastTo(maxScore,[1,MAX_LEN])));
  const sumExp=b.sum(expScores,1);
  const weights=b.divf(expScores,b.broadcastTo(sumExp,[1,MAX_LEN]));
  const vCache=b.fpext(b.load(b.makeTensorPtr(VC,[NH*MAX_LEN,HD],[HD,1],[b.mul(head,b.i32(MAX_LEN)),b.i32(0)],[MAX_LEN,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const weightsBc=b.broadcast(b.expandDims(b.reshape(weights,[MAX_LEN]),1),[MAX_LEN,HD]);
  const attnOut=b.sum(b.mul(weightsBc,vCache),0);
  const gate=b.fpext(b.load(b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),gOff],[1,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const gateSig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(gate,b.f32(-1)))));
  const out=b.mul(b.broadcastTo(attnOut,[1,HD]),gateSig);
  b.store(b.makeTensorPtr(Out,[1,NH*HD],[NH*HD,1],[b.i32(0),qOff],[1,HD],"bf16",[1,0]),b.fptrunc(out,"bf16"),{boundaryCheck:[0,1]});
  return b.build("fa2a",4,7);
}

// ── RoPE in-place: modify q_norm buffer directly (no CPU!) ──
function buildRoPEInPlace() {
  const b=new TTIRBuilder();
  const Q=b.param("Q",{ptr:"bf16"}),CosT=b.param("C",{ptr:"f32"}),SinT=b.param("S",{ptr:"f32"});
  const Pos=b.param("P","i32");
  const head=b.programId(0);
  const off=b.mul(head,b.i32(HD));
  const q1=b.fpext(b.load(b.makeTensorPtr(Q,[1,NH*HD],[NH*HD,1],[b.i32(0),off],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const q2=b.fpext(b.load(b.makeTensorPtr(Q,[1,NH*HD],[NH*HD,1],[b.i32(0),b.add(off,b.i32(ROT_HALF))],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const cos=b.load(b.makeTensorPtr(CosT,[MAX_LEN,ROT_HALF],[ROT_HALF,1],[Pos,b.i32(0)],[1,ROT_HALF],"f32",[1,0]),{boundaryCheck:[0,1],padding:1});
  const sin=b.load(b.makeTensorPtr(SinT,[MAX_LEN,ROT_HALF],[ROT_HALF,1],[Pos,b.i32(0)],[1,ROT_HALF],"f32",[1,0]),{boundaryCheck:[0,1],padding:1});
  const q1Rot=b.sub(b.mul(q1,cos),b.mul(q2,sin));
  const q2Rot=b.add(b.mul(q2,cos),b.mul(q1,sin));
  b.store(b.makeTensorPtr(Q,[1,NH*HD],[NH*HD,1],[b.i32(0),off],[1,ROT_HALF],"bf16",[1,0]),b.fptrunc(q1Rot,"bf16"),{boundaryCheck:[0,1]});
  b.store(b.makeTensorPtr(Q,[1,NH*HD],[NH*HD,1],[b.i32(0),b.add(off,b.i32(ROT_HALF))],[1,ROT_HALF],"bf16",[1,0]),b.fptrunc(q2Rot,"bf16"),{boundaryCheck:[0,1]});
  return b.build("rope",4,4);
}
// ── RoPE in-place for k (NKV heads) ──
function buildRoPEKInPlace() {
  const b=new TTIRBuilder();
  const K=b.param("K",{ptr:"bf16"}),CosT=b.param("C",{ptr:"f32"}),SinT=b.param("S",{ptr:"f32"});
  const Pos=b.param("P","i32");
  const head=b.programId(0);
  const off=b.mul(head,b.i32(HD));
  const k1=b.fpext(b.load(b.makeTensorPtr(K,[1,KV_DIM],[KV_DIM,1],[b.i32(0),off],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const k2=b.fpext(b.load(b.makeTensorPtr(K,[1,KV_DIM],[KV_DIM,1],[b.i32(0),b.add(off,b.i32(ROT_HALF))],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const cos=b.load(b.makeTensorPtr(CosT,[MAX_LEN,ROT_HALF],[ROT_HALF,1],[Pos,b.i32(0)],[1,ROT_HALF],"f32",[1,0]),{boundaryCheck:[0,1],padding:1});
  const sin=b.load(b.makeTensorPtr(SinT,[MAX_LEN,ROT_HALF],[ROT_HALF,1],[Pos,b.i32(0)],[1,ROT_HALF],"f32",[1,0]),{boundaryCheck:[0,1],padding:1});
  const k1Rot=b.sub(b.mul(k1,cos),b.mul(k2,sin));
  const k2Rot=b.add(b.mul(k2,cos),b.mul(k1,sin));
  b.store(b.makeTensorPtr(K,[1,KV_DIM],[KV_DIM,1],[b.i32(0),off],[1,ROT_HALF],"bf16",[1,0]),b.fptrunc(k1Rot,"bf16"),{boundaryCheck:[0,1]});
  b.store(b.makeTensorPtr(K,[1,KV_DIM],[KV_DIM,1],[b.i32(0),b.add(off,b.i32(ROT_HALF))],[1,ROT_HALF],"bf16",[1,0]),b.fptrunc(k2Rot,"bf16"),{boundaryCheck:[0,1]});
  return b.build("ropek",4,4);
}

// ── Fused FA2: q_norm+RoPE+k_norm+RoPE+KV cache+attention+gate — ALL on GPU ──
// Grid: [NH]. No CPU round-trips.
const PASS1 = 128, PASS2 = 64; // 64+128+64 = 256 (all powers of 2)
function buildFA2Fused() {
  const b=new TTIRBuilder();
  const QG=b.param("QG",{ptr:"bf16"}),KB=b.param("K",{ptr:"bf16"}),VB=b.param("V",{ptr:"bf16"});
  const QNW=b.param("QNW",{ptr:"bf16"}),KNW=b.param("KNW",{ptr:"bf16"});
  const CosT=b.param("CT",{ptr:"f32"}),SinT=b.param("ST",{ptr:"f32"});
  const QBuf=b.param("QB",{ptr:"bf16"});
  const KC=b.param("KC",{ptr:"bf16"}),VC=b.param("VC",{ptr:"bf16"});
  const Mask=b.param("M",{ptr:"f32"}),Out=b.param("O",{ptr:"bf16"});
  const Pos=b.param("P","i32");
  const head=b.programId(0);
  const headKv=b.divi(head,b.i32(NH/NKV));
  const qOffIn=b.mul(head,b.i32(HD*2));
  const gOff=b.add(qOffIn,b.i32(HD));
  const kvOff=b.mul(headKv,b.i32(HD));
  const qOffBuf=b.mul(head,b.i32(HD));
  const cacheRow=b.add(b.mul(head,b.i32(MAX_LEN)),Pos);
  // ── q_norm + RoPE ──
  // Load full q[256] for rstd
  const tpQF=b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),qOffIn],[1,HD],"bf16",[1,0]);
  const qF=b.fpext(b.load(tpQF,{boundaryCheck:[0,1],padding:1}),"f32");
  const rstd=b.rsqrtHw(b.add(b.divf(b.sum(b.mul(qF,qF),1),b.f32(HD)),b.f32(EPS)));
  // Load cos/sin for this position [1, 32]
  const cos=b.load(b.makeTensorPtr(CosT,[MAX_LEN,ROT_HALF],[ROT_HALF,1],[Pos,b.i32(0)],[1,ROT_HALF],"f32",[1,0]),{boundaryCheck:[0,1],padding:1});
  const sin=b.load(b.makeTensorPtr(SinT,[MAX_LEN,ROT_HALF],[ROT_HALF,1],[Pos,b.i32(0)],[1,ROT_HALF],"f32",[1,0]),{boundaryCheck:[0,1],padding:1});
  // Helper: normalize a slice of q and apply weight
  // q1[0:32], q2[32:64] → apply RoPE
  const q1R=b.fpext(b.load(b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),qOffIn],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const q2R=b.fpext(b.load(b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),b.add(qOffIn,b.i32(ROT_HALF))],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const w1=b.fpext(b.load(b.makeTensorPtr(QNW,[1,ROT_HALF],[ROT_HALF,1],[b.i32(0),b.i32(0)],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const w2=b.fpext(b.load(b.makeTensorPtr(QNW,[1,ROT_HALF],[ROT_HALF,1],[b.i32(0),b.i32(ROT_HALF)],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const q1=b.mul(q1R,b.mul(rstd,b.add(b.f32(1),w1)));
  const q2=b.mul(q2R,b.mul(rstd,b.add(b.f32(1),w2)));
  // RoPE: non-interleaved, cos/sin duplicated in halves
  const q1Rot=b.sub(b.mul(q1,cos),b.mul(q2,sin));
  const q2Rot=b.add(b.mul(q2,cos),b.mul(q1,sin));
  // Store rotated parts to QBuf
  b.store(b.makeTensorPtr(QBuf,[1,NH*HD],[NH*HD,1],[b.i32(0),qOffBuf],[1,ROT_HALF],"bf16",[1,0]),b.fptrunc(q1Rot,"bf16"),{boundaryCheck:[0,1]});
  b.store(b.makeTensorPtr(QBuf,[1,NH*HD],[NH*HD,1],[b.i32(0),b.add(qOffBuf,b.i32(ROT_HALF))],[1,ROT_HALF],"bf16",[1,0]),b.fptrunc(q2Rot,"bf16"),{boundaryCheck:[0,1]});
  // Pass-through parts [64:192] and [192:256]
  const q3R=b.fpext(b.load(b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),b.add(qOffIn,b.i32(ROT_DIM))],[1,PASS1],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const w3=b.fpext(b.load(b.makeTensorPtr(QNW,[1,PASS1],[PASS1,1],[b.i32(0),b.i32(ROT_DIM)],[1,PASS1],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const q3=b.mul(q3R,b.mul(rstd,b.add(b.f32(1),w3)));
  b.store(b.makeTensorPtr(QBuf,[1,NH*HD],[NH*HD,1],[b.i32(0),b.add(qOffBuf,b.i32(ROT_DIM))],[1,PASS1],"bf16",[1,0]),b.fptrunc(q3,"bf16"),{boundaryCheck:[0,1]});
  const q4R=b.fpext(b.load(b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),b.add(qOffIn,b.i32(ROT_DIM+PASS1))],[1,PASS2],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const w4=b.fpext(b.load(b.makeTensorPtr(QNW,[1,PASS2],[PASS2,1],[b.i32(0),b.i32(ROT_DIM+PASS1)],[1,PASS2],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const q4=b.mul(q4R,b.mul(rstd,b.add(b.f32(1),w4)));
  b.store(b.makeTensorPtr(QBuf,[1,NH*HD],[NH*HD,1],[b.i32(0),b.add(qOffBuf,b.i32(ROT_DIM+PASS1))],[1,PASS2],"bf16",[1,0]),b.fptrunc(q4,"bf16"),{boundaryCheck:[0,1]});
  // ── k_norm + RoPE (same structure, store to KV cache) ──
  const tpKF=b.makeTensorPtr(KB,[1,KV_DIM],[KV_DIM,1],[b.i32(0),kvOff],[1,HD],"bf16",[1,0]);
  const kF=b.fpext(b.load(tpKF,{boundaryCheck:[0,1],padding:1}),"f32");
  const kRstd=b.rsqrtHw(b.add(b.divf(b.sum(b.mul(kF,kF),1),b.f32(HD)),b.f32(EPS)));
  const k1R=b.fpext(b.load(b.makeTensorPtr(KB,[1,KV_DIM],[KV_DIM,1],[b.i32(0),kvOff],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const k2R=b.fpext(b.load(b.makeTensorPtr(KB,[1,KV_DIM],[KV_DIM,1],[b.i32(0),b.add(kvOff,b.i32(ROT_HALF))],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const kw1=b.fpext(b.load(b.makeTensorPtr(KNW,[1,ROT_HALF],[ROT_HALF,1],[b.i32(0),b.i32(0)],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const kw2=b.fpext(b.load(b.makeTensorPtr(KNW,[1,ROT_HALF],[ROT_HALF,1],[b.i32(0),b.i32(ROT_HALF)],[1,ROT_HALF],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const k1=b.mul(k1R,b.mul(kRstd,b.add(b.f32(1),kw1)));
  const k2=b.mul(k2R,b.mul(kRstd,b.add(b.f32(1),kw2)));
  const k1Rot=b.sub(b.mul(k1,cos),b.mul(k2,sin));
  const k2Rot=b.add(b.mul(k2,cos),b.mul(k1,sin));
  b.store(b.makeTensorPtr(KC,[NH*MAX_LEN,HD],[HD,1],[cacheRow,b.i32(0)],[1,ROT_HALF],"bf16",[1,0]),b.fptrunc(k1Rot,"bf16"),{boundaryCheck:[0,1]});
  b.store(b.makeTensorPtr(KC,[NH*MAX_LEN,HD],[HD,1],[cacheRow,b.i32(ROT_HALF)],[1,ROT_HALF],"bf16",[1,0]),b.fptrunc(k2Rot,"bf16"),{boundaryCheck:[0,1]});
  const k3R=b.fpext(b.load(b.makeTensorPtr(KB,[1,KV_DIM],[KV_DIM,1],[b.i32(0),b.add(kvOff,b.i32(ROT_DIM))],[1,PASS1],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const kw3=b.fpext(b.load(b.makeTensorPtr(KNW,[1,PASS1],[PASS1,1],[b.i32(0),b.i32(ROT_DIM)],[1,PASS1],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const k3=b.mul(k3R,b.mul(kRstd,b.add(b.f32(1),kw3)));
  b.store(b.makeTensorPtr(KC,[NH*MAX_LEN,HD],[HD,1],[cacheRow,b.i32(ROT_DIM)],[1,PASS1],"bf16",[1,0]),b.fptrunc(k3,"bf16"),{boundaryCheck:[0,1]});
  const k4R=b.fpext(b.load(b.makeTensorPtr(KB,[1,KV_DIM],[KV_DIM,1],[b.i32(0),b.add(kvOff,b.i32(ROT_DIM+PASS1))],[1,PASS2],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const kw4=b.fpext(b.load(b.makeTensorPtr(KNW,[1,PASS2],[PASS2,1],[b.i32(0),b.i32(ROT_DIM+PASS1)],[1,PASS2],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const k4=b.mul(k4R,b.mul(kRstd,b.add(b.f32(1),kw4)));
  b.store(b.makeTensorPtr(KC,[NH*MAX_LEN,HD],[HD,1],[cacheRow,b.i32(ROT_DIM+PASS1)],[1,PASS2],"bf16",[1,0]),b.fptrunc(k4,"bf16"),{boundaryCheck:[0,1]});
  // Store v to cache
  const v=b.fpext(b.load(b.makeTensorPtr(VB,[1,KV_DIM],[KV_DIM,1],[b.i32(0),kvOff],[1,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  b.store(b.makeTensorPtr(VC,[NH*MAX_LEN,HD],[HD,1],[cacheRow,b.i32(0)],[1,HD],"bf16",[1,0]),b.fptrunc(v,"bf16"),{boundaryCheck:[0,1]});
  return b.build("fa2n",4,11);
}

// ── FA2 attention (separate kernel — ensures cache stores are visible) ──
function buildFA2Attn2() {
  const b=new TTIRBuilder();
  const QBuf=b.param("QB",{ptr:"bf16"}),QG=b.param("QG",{ptr:"bf16"});
  const KC=b.param("KC",{ptr:"bf16"}),VC=b.param("VC",{ptr:"bf16"});
  const Mask=b.param("M",{ptr:"f32"}),Out=b.param("O",{ptr:"bf16"});
  const head=b.programId(0);
  const qOffBuf=b.mul(head,b.i32(HD));
  const gOff=b.add(b.mul(head,b.i32(HD*2)),b.i32(HD));
  // Load rotated q from QBuf
  const qRot=b.fpext(b.load(b.makeTensorPtr(QBuf,[1,NH*HD],[NH*HD,1],[b.i32(0),qOffBuf],[1,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  // Load k_cache [MAX_LEN, HD]
  const kCache=b.fpext(b.load(b.makeTensorPtr(KC,[NH*MAX_LEN,HD],[HD,1],[b.mul(head,b.i32(MAX_LEN)),b.i32(0)],[MAX_LEN,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const qBc=b.broadcastTo(qRot,[MAX_LEN,HD]);
  const scores=b.mul(b.sum(b.mul(qBc,kCache),1),b.f32(1/Math.sqrt(HD)));
  // Mask + softmax
  const maskArr=b.load(b.makeTensorPtr(Mask,[1,MAX_LEN],[MAX_LEN,1],[b.i32(0),b.i32(0)],[1,MAX_LEN],"f32",[1,0]),{boundaryCheck:[0,1],padding:1});
  const scoresMasked=b.add(b.broadcastTo(scores,[1,MAX_LEN]),maskArr);
  const maxScore=b.max(scoresMasked,1);
  const expScores=b.exp(b.sub(b.broadcastTo(scoresMasked,[1,MAX_LEN]),b.broadcastTo(maxScore,[1,MAX_LEN])));
  const sumExp=b.sum(expScores,1);
  const weights=b.divf(expScores,b.broadcastTo(sumExp,[1,MAX_LEN]));
  // Output = weights @ v_cache
  const vCache=b.fpext(b.load(b.makeTensorPtr(VC,[NH*MAX_LEN,HD],[HD,1],[b.mul(head,b.i32(MAX_LEN)),b.i32(0)],[MAX_LEN,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const weightsBc=b.broadcast(b.expandDims(b.reshape(weights,[MAX_LEN]),1),[MAX_LEN,HD]);
  const attnOut=b.sum(b.mul(weightsBc,vCache),0);
  // Gate
  const gate=b.fpext(b.load(b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),gOff],[1,HD],"bf16",[1,0]),{boundaryCheck:[0,1],padding:1}),"f32");
  const gateSig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(gate,b.f32(-1)))));
  const out=b.mul(b.broadcastTo(attnOut,[1,HD]),gateSig);
  b.store(b.makeTensorPtr(Out,[1,NH*HD],[NH*HD,1],[b.i32(0),qOffBuf],[1,HD],"bf16",[1,0]),b.fptrunc(out,"bf16"),{boundaryCheck:[0,1]});
  return b.build("fa2a2",4,6);
}

// CPU RoPE: rotate first rotDim dims (non-interleaved)
function floatToBf16(f:number):number{const buf=new ArrayBuffer(4);const f32=new Float32Array(buf);const u8=new Uint8Array(buf);f32[0]=f;return u8[2]|(u8[3]<<8);}
function applyRoPECPU(data:Uint16Array,cos:Float32Array,sin:Float32Array,numHeads:number,headDim:number,rotDim:number){
  const half=rotDim/2;
  for(let h=0;h<numHeads;h++){const off=h*headDim;for(let i=0;i<half;i++){
    const a=bf16ToFloat(data[off+i]);const b=bf16ToFloat(data[off+i+half]);
    data[off+i]=floatToBf16(a*cos[i]-b*sin[i]);data[off+i+half]=floatToBf16(b*cos[i]+a*sin[i]);
  }}
}

function buildFA2Decode() {
  const b=new TTIRBuilder();
  const QG=b.param("QG",{ptr:"bf16"}),VB=b.param("V",{ptr:"bf16"}),KB=b.param("K",{ptr:"bf16"});
  const QNW=b.param("QNW",{ptr:"bf16"}),KNW=b.param("KNW",{ptr:"bf16"});
  const KCache=b.param("KC",{ptr:"bf16"}),VCache=b.param("VC",{ptr:"bf16"});
  const CosT=b.param("CT",{ptr:"f32"}),SinT=b.param("ST",{ptr:"f32"});
  const Mask=b.param("M",{ptr:"f32"}); // [MAX_LEN] attention mask
  const Out=b.param("O",{ptr:"bf16"});
  const Pos=b.param("P","i32"); // current position (scalar)
  const head=b.programId(0);
  const headKv=b.divi(head,b.i32(NH/NKV));
  const qOffIn=b.mul(head,b.i32(HD*2));
  const gOffIn=b.add(qOffIn,b.i32(HD));
  const qOffOut=b.mul(head,b.i32(HD));
  const vOff=b.mul(headKv,b.i32(HD));
  const hd2=b.i32(HD*2);
  // Load q from interleaved q_proj output
  const tpQ=b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),qOffIn],[1,HD],"bf16",[1,0]);
  const qRaw=b.fpext(b.load(tpQ,{boundaryCheck:[0,1],padding:1}),"f32");
  // RMSNorm q (per head, d=HD=256)
  const qRstd=b.rsqrtHw(b.add(b.divf(b.sum(b.mul(qRaw,qRaw),1),b.f32(HD)),b.f32(EPS)));
  let qNorm=b.mul(qRaw,qRstd);
  const tpQN=b.makeTensorPtr(QNW,[1,HD],[HD,1],[b.i32(0),b.i32(0)],[1,HD],"bf16",[1,0]);
  qNorm=b.mul(qNorm,b.add(b.f32(1),b.fpext(b.load(tpQN,{boundaryCheck:[0,1],padding:1}),"f32")));
  // RoPE: rotate first ROT_DIM=64 dims (non-interleaved: swap halves)
  // q'[0:32] = q[0:32]*cos - q[32:64]*sin
  // q'[32:64] = q[32:64]*cos + q[0:32]*sin
  // Load q[0:32] and q[32:64] from qNorm (already loaded as [1, HD])
  // Use makeTensorPtr on the original QG at the right offsets, then normalize
  const tpQ1=b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),qOffIn],[1,ROT_HALF],"bf16",[1,0]);
  const tpQ2=b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),b.add(qOffIn,b.i32(ROT_HALF))],[1,ROT_HALF],"bf16",[1,0]);
  // We need the normalized halves. Since we already have qNorm[1,HD], let's
  // load cos/sin and apply RoPE to qNorm using slices.
  // Actually, qNorm is a computed value, not a pointer. We can't makeTensorPtr it.
  // Instead, let's load cos/sin and do the rotation on the loaded q values.
  const cos=b.load(b.makeTensorPtr(CosT,[1,ROT_HALF],[ROT_HALF,1],[b.i32(0),b.i32(0)],[1,ROT_HALF],"f32",[1,0]),{boundaryCheck:[0,1],padding:1});
  const sin=b.load(b.makeTensorPtr(SinT,[1,ROT_HALF],[ROT_HALF,1],[b.i32(0),b.i32(0)],[1,ROT_HALF],"f32",[1,0]),{boundaryCheck:[0,1],padding:1});
  // Re-normalize q[0:32] and q[32:64] separately
  const q1Raw=b.fpext(b.load(tpQ1,{boundaryCheck:[0,1],padding:1}),"f32");
  const q2Raw=b.fpext(b.load(tpQ2,{boundaryCheck:[0,1],padding:1}),"f32");
  // For the full q_norm, we already computed it. Let's just apply RoPE to
  // the full qNorm by splitting it. Since we can't split a computed value,
  // let's re-compute: the RMSNorm of q[0:32] uses the full head statistics.
  // Actually, RMSNorm is over the full 256 dims, so we can't split it.
  // The correct approach: qNorm is [1,256]. We need to apply RoPE to [0:64].
  // Since we can't slice a computed tensor, let's just skip RoPE for now
  // and use the trivial attention. At position 0, RoPE is identity anyway.
  // For position > 0, this will be wrong but we'll fix it later.
  
  // Load v (GQA: headKv = head // (NH/NKV))
  const tpV=b.makeTensorPtr(VB,[1,KV_DIM],[KV_DIM,1],[b.i32(0),vOff],[1,HD],"bf16",[1,0]);
  const vVal=b.fpext(b.load(tpV,{boundaryCheck:[0,1],padding:1}),"f32");
  // Load k, normalize, and store to KV cache
  const tpK=b.makeTensorPtr(KB,[1,KV_DIM],[KV_DIM,1],[b.i32(0),vOff],[1,HD],"bf16",[1,0]);
  const kRaw=b.fpext(b.load(tpK,{boundaryCheck:[0,1],padding:1}),"f32");
  const kRstd=b.rsqrtHw(b.add(b.divf(b.sum(b.mul(kRaw,kRaw),1),b.f32(HD)),b.f32(EPS)));
  let kNorm=b.mul(kRaw,kRstd);
  const tpKN=b.makeTensorPtr(KNW,[1,HD],[HD,1],[b.i32(0),b.i32(0)],[1,HD],"bf16",[1,0]);
  kNorm=b.mul(kNorm,b.add(b.f32(1),b.fpext(b.load(tpKN,{boundaryCheck:[0,1],padding:1}),"f32")));
  // Store k_norm to k_cache[head, pos, :]
  const cacheRow=b.add(b.mul(head,b.i32(MAX_LEN)),Pos);
  const tpKCw=b.makeTensorPtr(KCache,[NH*MAX_LEN,HD],[HD,1],[cacheRow,b.i32(0)],[1,HD],"bf16",[1,0]);
  b.store(tpKCw,b.fptrunc(kNorm,"bf16"),{boundaryCheck:[0,1]});
  // Store v to v_cache[head, pos, :]
  const tpVCw=b.makeTensorPtr(VCache,[NH*MAX_LEN,HD],[HD,1],[cacheRow,b.i32(0)],[1,HD],"bf16",[1,0]);
  b.store(tpVCw,b.fptrunc(vVal,"bf16"),{boundaryCheck:[0,1]});
  
  // Load gate
  const tpG=b.makeTensorPtr(QG,[1,QGATE],[QGATE,1],[b.i32(0),gOffIn],[1,HD],"bf16",[1,0]);
  const gate=b.fpext(b.load(tpG,{boundaryCheck:[0,1],padding:1}),"f32");
  const gateSig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(gate,b.f32(-1)))));
  
  // Store current k, v into KV cache at position Pos
  // k_cache[head, pos, :] = k_norm[head, :]
  // v_cache[head, pos, :] = v[head, :]
  // For simplicity, we skip k_norm and RoPE for the cached k.
  // This means attention uses unnormalized k, which is wrong.
  // TODO: fix k_norm + RoPE for cached k.
  
  // Attention: q @ k_cache^T * scale → scores [MAX_LEN]
  // Using broadcast+sum: scores[j] = sum_i q[i] * k_cache[j, i]
  // Load k_cache for this head: [MAX_LEN, HD]
  // KV cache layout: [NH, MAX_LEN, HD] flat = [NH*MAX_LEN, HD]
  // head h starts at row h*MAX_LEN
  const tpKC2=b.makeTensorPtr(KCache,[NH*MAX_LEN,HD],[HD,1],[b.mul(head,b.i32(MAX_LEN)),b.i32(0)],[MAX_LEN,HD],"bf16",[1,0]);
  const kCache=b.fpext(b.load(tpKC2,{boundaryCheck:[0,1],padding:1}),"f32"); // [MAX_LEN, HD]
  // q broadcast to [MAX_LEN, HD]
  const qBc=b.broadcastTo(qNorm,[MAX_LEN,HD]);
  // scores = sum(q_bc * k_cache, 1) → [MAX_LEN]
  const scale=b.f32(1/Math.sqrt(HD));
  const scores=b.mul(b.sum(b.mul(qBc,kCache),1),scale); // [MAX_LEN]
  // Apply mask: add -1e30 for positions > Pos
  const maskArr=b.load(b.makeTensorPtr(Mask,[1,MAX_LEN],[MAX_LEN,1],[b.i32(0),b.i32(0)],[1,MAX_LEN],"f32",[1,0]),{boundaryCheck:[0,1],padding:1});
  const scoresMasked=b.add(b.broadcastTo(scores,[1,MAX_LEN]),maskArr); // [1, MAX_LEN]
  // Softmax: subtract max, exp, normalize
  const maxScore=b.max(scoresMasked,1); // [1] — max over the MAX_LEN dimension
  const expScores=b.exp(b.sub(b.broadcastTo(scoresMasked,[1,MAX_LEN]),b.broadcastTo(maxScore,[1,MAX_LEN]))); // [1, MAX_LEN]
  const sumExp=b.sum(expScores,1); // [1]
  const weights=b.divf(expScores,b.broadcastTo(sumExp,[1,MAX_LEN])); // [1, MAX_LEN]
  // output = weights @ v_cache → [1, HD]
  // = sum_j weights[j] * v_cache[j, :] → [HD]
  const tpVC=b.makeTensorPtr(VCache,[NH*MAX_LEN,HD],[HD,1],[b.mul(head,b.i32(MAX_LEN)),b.i32(0)],[MAX_LEN,HD],"bf16",[1,0]);
  const vCache=b.fpext(b.load(tpVC,{boundaryCheck:[0,1],padding:1}),"f32"); // [MAX_LEN, HD]
  const weightsBc=b.broadcast(b.expandDims(b.reshape(weights,[MAX_LEN]),1),[MAX_LEN,HD]); // [MAX_LEN, HD]
  const attnOut=b.sum(b.mul(weightsBc,vCache),0); // [HD]
  // Apply gate
  const out=b.mul(b.broadcastTo(attnOut,[1,HD]),gateSig);
  // Store output
  const tpO=b.makeTensorPtr(Out,[1,NH*HD],[NH*HD,1],[b.i32(0),qOffOut],[1,HD],"bf16",[1,0]);
  b.store(tpO,b.fptrunc(out,"bf16"),{boundaryCheck:[0,1]});
  return b.build("fa2_d",4,10);
}

// ── GPU argmax: two-stage PARALLEL reduction over logits [VOCAB] f32 ──
// Stage 1 (buildArgmax, grid [SPLIT]): each program reduces a contiguous
// VOCAB/SPLIT range. The range is power-of-two padded to 2048 with -inf via an
// explicit mask (NOT load-NaN padding), so the max reducer sees only real
// values — fully deterministic. SPLIT*2048 > VOCAB; the tail blocks hold -inf.
// Stage 2 (buildArgmaxComb, grid [1]): walks the SPLIT partials sequentially
// with a strict-> winner update (keeps the LOWEST index on ties), exactly the
// semamtics of the old single-block argmax, so results match the reference.
const ARGMAX_SPLIT = 128;
const ARGMAX_PER = 2048; // power-of-two tile per block
const ARGMAX_NEGINF = -3.40282347e38;
function buildArgmax() {
  const b=new TTIRBuilder();
  const L=b.param("L",{ptr:"f32"}),PV=b.param("PV",{ptr:"f32"}),PI=b.param("PI",{ptr:"i32"});
  const pid=b.programId(0);
  const start=b.mul(pid,b.i32(ARGMAX_PER));
  const idxBase=b.add(start,b.arange(0,ARGMAX_PER)); // [PER] i32 global indices
  const mask=b.lt(idxBase,b.i32(VOCAB)); // [PER] i1 — false past the vocab end
  const ptrs=b.addptr(b.splatPtr(L,ARGMAX_PER,"f32"),idxBase); // [PER] !tt.ptr<f32>
  const chunk1=b.load(ptrs,{mask,other:b.splat(b.f32(ARGMAX_NEGINF),[ARGMAX_PER],"f32")}); // [PER] f32, -inf padded
  const chunk=b.expandDims(chunk1,0); // [1, PER]
  const localMax=b.max(chunk,1); // [1]
  // Index of the block-local max within the block's range (unique-max assumption).
  const maskV=b.eq(chunk,b.broadcastTo(localMax,[1,ARGMAX_PER]));
  const arangeBc=b.broadcast(b.expandDims(b.arange(0,ARGMAX_PER),0),[1,ARGMAX_PER]);
  const localIdx=b.sum(b.select(maskV,arangeBc,b.broadcastTo(b.i32(0),[1,ARGMAX_PER])),1); // [1]
  const globalIdx=b.add(localIdx,b.broadcastTo(start,[1])); // [1]
  const tpPV=b.makeTensorPtr(PV,[1,ARGMAX_SPLIT],[ARGMAX_SPLIT,1],[b.i32(0),pid],[1,1],"f32",[1,0]);
  const tpPI=b.makeTensorPtr(PI,[1,ARGMAX_SPLIT],[ARGMAX_SPLIT,1],[b.i32(0),pid],[1,1],"i32",[1,0]);
  b.store(tpPV,b.broadcastTo(localMax,[1,1]),{});
  b.store(tpPI,b.broadcastTo(globalIdx,[1,1]),{});
  return b.build("argmaxp",4,3);
}
function buildArgmaxComb() {
  const b=new TTIRBuilder();
  const PV=b.param("PV",{ptr:"f32"}),PI=b.param("PI",{ptr:"i32"}),OutVal=b.param("V",{ptr:"f32"}),OutIdx=b.param("I",{ptr:"i32"});
  const initMax=b.broadcastTo(b.f32(ARGMAX_NEGINF),[1,1]);
  const initIdx=b.broadcastTo(b.i32(0),[1,1]);
  // Sequential strict-> scan over the 128 partials: keeps the lowest-index winner.
  const [gMax,gIdx]=b.forIter(b.index(0),b.index(ARGMAX_SPLIT),b.index(1),[initMax,initIdx],(bb,i,[curMax,curIdx])=>{
    const ii=bb.indexCast(i,"i32");
    const tpV=bb.makeTensorPtr(PV,[1,ARGMAX_SPLIT],[ARGMAX_SPLIT,1],[b.i32(0),ii],[1,1],"f32",[1,0]);
    const tpI=bb.makeTensorPtr(PI,[1,ARGMAX_SPLIT],[ARGMAX_SPLIT,1],[b.i32(0),ii],[1,1],"i32",[1,0]);
    const v=bb.load(tpV,{});      // [1,1]
    const cand=bb.load(tpI,{});   // [1,1]
    const better=bb.gt(v,curMax);
    return [bb.select(better,v,curMax),bb.select(better,cand,curIdx)];
  });
  const tpO=b.makeTensorPtr(OutVal,[1,1],[1,1],[b.i32(0),b.i32(0)],[1,1],"f32",[1,0]);
  const tpOI=b.makeTensorPtr(OutIdx,[1,1],[1,1],[b.i32(0),b.i32(0)],[1,1],"i32",[1,0]);
  b.store(tpO,b.broadcastTo(gMax,[1,1]),{});
  b.store(tpOI,b.broadcastTo(gIdx,[1,1]),{});
  return b.build("argmaxc",4,4);
}

function launch(k:any,g:number[],bl:number[],args:any[],label:string) {
  cuLaunch(k,[g[0],g[1],g[2]] as [number,number,number],[bl[0],bl[1],bl[2]] as [number,number,number],args);
  // No cuSync — cuDtoH/cuHtoD block implicitly. GPU ops queue on default stream.
}
function launchSync(k:any,g:number[],bl:number[],args:any[],label:string) {
  const rc=cuLaunch(k,[g[0],g[1],g[2]] as [number,number,number],[bl[0],bl[1],bl[2]] as [number,number,number],args);
  const sr=cuSync(); if(rc)console.log(`  [!]${label} rc=${rc}`); if(sr)console.log(`  [!]${label} sync=${sr}`);
  return rc||sr;
}

function bf16ToFloat(u16:number):number {
  const buf=new ArrayBuffer(4); const u8=new Uint8Array(buf); const f32=new Float32Array(buf);
  u8[2]=u16&0xFF; u8[3]=u16>>8; return f32[0];
}

// ── Main ──
const tokPath=process.env.TOKENIZER_PATH||"/tmp/tokenizer.json";
const stPath=process.env.SAFETENSORS_PATH||"/tmp/qwen35_0.8b.safetensors";
const prompt=process.argv[2]||"Hello";
const genLen=parseInt(process.argv[3]||"30");
console.log("=== Qwen3.5-0.8B Decode ===");
const W=await loadWeights(stPath);
const tokenIds=tokenize(prompt,tokPath);
console.log(`prompt: "${prompt}" → ${tokenIds.length} tokens: [${tokenIds}]`);

// Compile kernels
console.log("compiling kernels...");
// Fused GEMM variants — automatically fuse Cast/Add/dual-GEMM into the epilogue
const mm=(M:number,N:number,K:number,opts?:any,label?:string,warps?:number)=>compileAndLoad(buildMM(M,N,K,opts),"mm",warps??4,label??"mm");
const mmF=(M:number,N:number,K:number,opts?:any,label?:string,warps?:number)=>compileAndLoad(buildMM(M,N,K,opts),"mm",warps??4,label??"mm");
// GDN: qkv+cast, z+cast, a+b dual-GEMM, out_proj+cast+add, down_proj+cast+add
const kQKV=mmF(1,QKVD,H,{cast:true,BN:16},"mm_qkv");           // GEMM+Cast, BN=16 => 384 blocks
const kZ=mmF(1,ZD,H,{cast:true,BN:16},"mm_z");              // GEMM+Cast, BN=16 => 128 blocks
const kAB=mmF(1,LVH*2,H,{N2:LVH},"mm_ab");                  // Dual GEMM (a+b) — shares input
const kOutProj=mmF(1,H,ZD,{cast:true,add:true,BN:16},"mm_outp");  // GEMM+Cast+Add (residual), BN=16
// FA2: q+cast, k+cast, v+cast, o+cast+add
const kQProj=mmF(1,QGATE,H,{cast:true,BN:16},"mm_q");       // GEMM+Cast, BN=16 => 256 blocks
const kKVProj=mmF(1,KV_DIM,H,{cast:true,BN:16},"mm_kv");   // GEMM+Cast, BN=16 => 32 blocks
const kOProj=mmF(1,H,NH*HD,{cast:true,add:true,BN:16},"mm_o");  // GEMM+Cast+Add, BN=16
// MLP: gate+up dual-GEMM, down+cast+add
const kGPUP=mmF(1,INTER,H,{N2:INTER,BN:16},"mm_gate");     // Dual GEMM (gate+up), BN=16 => 224 blocks
const kDP=mmF(1,H,INTER,{cast:true,add:true,BN:16},"mm_down"); // BN=16 => 64 col blocks (bit-exact BW fix)
// lm_head (no fusion — argmax reads f32)
const kLM=mm(1,VOCAB,H,"mm_lm");
// Fused Cast+Add (for cases where GEMM isn't the producer)
const kCastAdd=compileAndLoad(buildCastAdd(H),"ca",4);
const kRms=compileAndLoad(buildRMS(H),"rms",4);
const kSg=compileAndLoad(buildSwiGLU(INTER),"sg",4);
const kAd=compileAndLoad(buildAdd(H),"ad",4);
const kCs=compileAndLoad(buildCast(H),"cs",4);
const kCsQKV=compileAndLoad(buildCast(QKVD),"cs",4);
const kCsZD=compileAndLoad(buildCast(ZD),"cs",4);
const kCsQG=compileAndLoad(buildCast(QGATE),"cs",4);
const kCsKV=compileAndLoad(buildCast(KV_DIM),"cs",4);
const kConv1dD=compileAndLoad(buildConv1dDecode(),"cv1d_d",4);
const kGDND=compileAndLoad(buildGDNDecode(),"gdn_d",4);
const kFA2D=compileAndLoad(buildFA2Decode(),"fa2_d",4);
const kFA2N=compileAndLoad(buildFA2Fused(),"fa2n",4);
const kFA2A2=compileAndLoad(buildFA2Attn2(),"fa2a2",4);
const kRoPE=compileAndLoad(buildRoPEInPlace(),"rope",4);
const kRoPEK=compileAndLoad(buildRoPEKInPlace(),"ropek",4);
const kQNorm=compileAndLoad(buildQNorm(),"qn",4);
const kKNormD=compileAndLoad(buildKNormD(),"knormd",4);
const kFA2A=compileAndLoad(buildFA2Attn(),"fa2a",4);
const kArgmax=compileAndLoad(buildArgmax(),"argmaxp",4);
const kArgmaxC=compileAndLoad(buildArgmaxComb(),"argmaxc",4);
const kEmb=compileAndLoad(buildEmbed(),"emb",4);
const argmaxPV=cuAlloc(BigInt(ARGMAX_SPLIT*4)); // f32 partial maxes
const argmaxPI=cuAlloc(BigInt(ARGMAX_SPLIT*4)); // i32 partial indices
const argmaxVal=cuAlloc(BigInt(4)); // f32 max value
const argmaxIdx=cuAlloc(BigInt(4)); // i32 argmax index
// Precompute RoPE cos/sin tables [MAX_LEN, 32] f32
const ropeCosTable=new Float32Array(MAX_LEN*ROT_HALF);
const ropeSinTable=new Float32Array(MAX_LEN*ROT_HALF);
for(let p=0;p<MAX_LEN;p++) for(let i=0;i<ROT_HALF;i++){
  const freq=1/Math.pow(10000000,2*i/ROT_DIM);
  ropeCosTable[p*ROT_HALF+i]=Math.cos(p*freq);
  ropeSinTable[p*ROT_HALF+i]=Math.sin(p*freq);
}
const ropeCosD=cuAlloc(BigInt(MAX_LEN*ROT_HALF*4)); cuHtoD(ropeCosD,ropeCosTable.buffer); cuSync();
const ropeSinD=cuAlloc(BigInt(MAX_LEN*ROT_HALF*4)); cuHtoD(ropeSinD,ropeSinTable.buffer); cuSync();
// Pre-allocate QBuf (temp buffer for rotated q)
const qBuf=cuAlloc(BigInt(NH*HD*2)); // reused each token
console.log("  done");

// Embedding
const embedW=W.get("model.language_model.embed_tokens.weight")!;
function embedToken(id:number):bigint {
  const hEmb=new Uint16Array(H);
  // Read embedding from safetensors directly
  // We already have the weights on GPU, but we need the CPU data for embedding
  // Actually, let's read from the safetensors file
  return embedW!; // We'll handle this differently
}

// Load embedding data to CPU for quick lookup
const stData = await Bun.file(stPath).bytes();
const stDv = new DataView(stData.buffer, stData.byteOffset, stData.byteLength);
const stHl = Number(stDv.getBigUint64(0, true));
const stHdr = JSON.parse(new TextDecoder().decode(stData.subarray(8, 8+stHl)));
const stDs = 8+stHl;
const embInfo = stHdr["model.language_model.embed_tokens.weight"];
const embOff = stDs + embInfo.data_offsets[0];

function getEmbedding(id:number):Uint16Array {
  const h=new Uint16Array(H);
  const rs=embOff + id*H*2;
  for(let j=0;j<H;j++) h[j]=stData[rs+j*2]|(stData[rs+j*2+1]<<8);
  return h;
}

// Load A_log and dt_bias to CPU for decay computation
function loadF32Arr(name:string):Float32Array {
  const info = stHdr[name]; if(!info) throw new Error("missing "+name);
  const off = stDs + info.data_offsets[0];
  const n = (info.data_offsets[1]-info.data_offsets[0])/4;
  const arr = new Float32Array(n);
  for(let i=0;i<n;i++) arr[i]=stDv.getFloat32(off+i*4,true);
  return arr;
}
function loadBf16Arr(name:string):Float32Array {
  const info = stHdr[name]; if(!info) throw new Error("missing "+name);
  const off = stDs + info.data_offsets[0];
  const n = (info.data_offsets[1]-info.data_offsets[0])/2;
  const arr = new Float32Array(n);
  for(let i=0;i<n;i++){const u16=stData[off+i*2]|(stData[off+i*2+1]<<8);const buf=new ArrayBuffer(4);const u8=new Uint8Array(buf);const f32=new Float32Array(buf);u8[2]=u16&0xFF;u8[3]=u16>>8;arr[i]=f32[0];}
  return arr;
}
// Precompute per-layer A_log and dt_bias on CPU
const layerALog:Float32Array[]=[];
const layerDtB:Float32Array[]=[];
for(let l=0;l<NL;l++){
  if(!isFull(l)){
    layerALog.push(loadF32Arr(`model.language_model.layers.${l}.linear_attn.A_log`));
    layerDtB.push(loadBf16Arr(`model.language_model.layers.${l}.linear_attn.dt_bias`));
  }else{ layerALog.push(new Float32Array(0)); layerDtB.push(new Float32Array(0)); }
}

// Decode loop
const allTokens = [...tokenIds];
const refTokens = [9419, 11, 271, 40, 1044, 3133, 440, 264, 12654, 5148, 421, 5533, 279, 1510, 2311, 22233, 71962, 63, 4536, 310, 11290, 264, 2397, 59229, 22868, 13, 561, 5148, 369, 5995, 310];

console.log(`\nGenerating ${genLen} tokens...`);

// Allocate state buffers (per layer)
const convStates = new Array(NL).fill(0).map(()=>cuAlloc(BigInt(3*QKVD*2))); // [3, QKVD] bf16, init to 0
const convStatesNew = new Array(NL).fill(0).map(()=>cuAlloc(BigInt(3*QKVD*2)));
const sStates = new Array(NL).fill(0).map(()=>cuAlloc(BigInt(LVH*LKD*LVD*4))); // [16, 128, 128] f32, init to 0
const sStatesNew = new Array(NL).fill(0).map(()=>cuAlloc(BigInt(LVH*LKD*LVD*4)));
// KV cache for FA2 layers (preallocated, grows with seq)
const kvCacheK = new Array(NL).fill(0).map(()=>cuAlloc(BigInt(NH*HD*MAX_LEN*2))); // [8, 256, MAX_LEN] bf16
const kvCacheV = new Array(NL).fill(0).map(()=>cuAlloc(BigInt(NH*HD*MAX_LEN*2)));
const kvCacheKNew = new Array(NL).fill(0).map(()=>cuAlloc(BigInt(NH*HD*MAX_LEN*2)));
const kvCacheVNew = new Array(NL).fill(0).map(()=>cuAlloc(BigInt(NH*HD*MAX_LEN*2)));

let x:bigint;
const fStart = performance.now();
if (__PROF) { profReset(); p0 = fStart; }

for (let step = 0; step < tokenIds.length + genLen; step++) {
  const tokenId = allTokens[step];
  // Wait, let me simplify: 
  // For step 0..tokenIds.length-1: process prompt tokens
  // For step tokenIds.length..: generate new tokens
  
  // Embed current token — GPU gather kernel (scalar ID, no host copy)
  x = palloc(H*2);
  cuLaunch(kEmb,[1,1,1],[128,1,1],[embedW,x,tokenId],`emb`);
  
  // Forward pass (1 token, with state carry)
  for (let layer=0; layer<NL; layer++) {
    const full=isFull(layer);
    // 1. input_layernorm
    const normed=palloc(H*2);
    launch(kRms,[1,1,1],[128,1,1],[x,wp(W,layer,"input_layernorm.weight"),normed],`rms1.${layer}.${step}`);
    
    let afterAttn:bigint;
    
    if (!full) {
      // GDN decode — fused GEMMs (Cast epilogue, dual A+B, Cast+Add for out_proj)
      const qkvB=palloc(QKVD*2); // bf16 — GEMM+Cast outputs bf16 directly
      launch(kQKV,[1,Math.ceil(QKVD/16),1],[128,1,1],[normed,wp(W,layer,"linear_attn.in_proj_qkv.weight"),qkvB],`qkv.${layer}`);
      const zB=palloc(ZD*2); // bf16 — GEMM+Cast
      launch(kZ,[1,Math.ceil(ZD/16),1],[128,1,1],[normed,wp(W,layer,"linear_attn.in_proj_z.weight"),zB],`z.${layer}`);
      const aP=palloc(LVH*4); const bP=palloc(LVH*4); // f32 — dual GEMM (a+b)
      launch(kAB,[1,1,1],[128,1,1],[normed,wp(W,layer,"linear_attn.in_proj_a.weight"),aP,wp(W,layer,"linear_attn.in_proj_b.weight"),bP],`ab.${layer}`);
      // Conv1d decode (with conv state)
      const convOut=palloc(QKVD*2);
      launch(kConv1dD,[BLK1(QKVD),1,1],[128,1,1],[qkvB,convStates[layer],wp(W,layer,"linear_attn.conv1d.weight"),convOut,convStatesNew[layer]],`cv1d.${layer}`);
      const tmpCs = convStates[layer]; convStates[layer] = convStatesNew[layer]; convStatesNew[layer] = tmpCs;
      pfree(qkvB);
      // GDN delta rule (with recurrent state)
      const gdnOut=palloc(ZD*2);
      launch(kGDND,[LVH,1,1],[128,1,1],[convOut,zB,wp(W,layer,"linear_attn.A_log"),wp(W,layer,"linear_attn.dt_bias"),aP,bP,wp(W,layer,"linear_attn.norm.weight"),sStates[layer],gdnOut,sStatesNew[layer]],`gdn.${layer}`);
      const tmpSs = sStates[layer]; sStates[layer] = sStatesNew[layer]; sStatesNew[layer] = tmpSs;
      pfree(convOut); pfree(zB); pfree(aP); pfree(bP);
      // Fused out_proj+Cast+Add: output = bf16(gdnOut @ W + x)
      afterAttn=palloc(H*2); // bf16 output with residual
      launch(kOutProj,[1,Math.ceil(H/16),1],[128,1,1],[gdnOut,wp(W,layer,"linear_attn.out_proj.weight"),afterAttn,x],`op.${layer}`);
      pfree(gdnOut); pfree(x);
    } else {
      // Fused FA2: Q+Cast, K+Cast, V+Cast, O+Cast+Add — ALL on GPU
      const qgB=palloc(QGATE*2); // bf16 — GEMM+Cast
      launch(kQProj,[1,Math.ceil(QGATE/16),1],[128,1,1],[normed,wp(W,layer,"self_attn.q_proj.weight"),qgB],`qp.${layer}`);
      const vB=palloc(KV_DIM*2); // bf16 — GEMM+Cast
      launch(kKVProj,[1,Math.ceil(KV_DIM/16),1],[128,1,1],[normed,wp(W,layer,"self_attn.v_proj.weight"),vB],`vp.${layer}`);
      const kB=palloc(KV_DIM*2); // bf16 — GEMM+Cast
      launch(kKVProj,[1,Math.ceil(KV_DIM/16),1],[128,1,1],[normed,wp(W,layer,"self_attn.k_proj.weight"),kB],`kp.${layer}`);
      // q_norm → QBuf, then GPU RoPE in-place
      launch(kQNorm,[NH,1,1],[128,1,1],[qgB,wp(W,layer,"self_attn.q_norm.weight"),qBuf],`qn.${layer}`);
      launch(kRoPE,[NH,1,1],[128,1,1],[qBuf,ropeCosD,ropeSinD,step],`rope.${layer}`);
      // k_norm → kNormB, then GPU RoPE in-place
      const kNormB=palloc(KV_DIM*2);
      launch(kKNormD,[NKV,1,1],[128,1,1],[kB,wp(W,layer,"self_attn.k_norm.weight"),kNormB],`kn.${layer}`);
      launch(kRoPEK,[NKV,1,1],[128,1,1],[kNormB,ropeCosD,ropeSinD,step],`ropek.${layer}`);
      // FA2 attention (pre-rotated q from QBuf, k from kNormB, v from vB; causal
      // mask computed inside the kernel from Pos — no host transfer)
      const fa2Out=palloc(NH*HD*2);
      launch(kFA2A,[NH,1,1],[128,1,1],[qBuf,kNormB,vB,qgB,kvCacheK[layer],kvCacheV[layer],fa2Out,step],`fa2.${layer}`);
      pfree(qgB); pfree(vB); pfree(kB); pfree(kNormB);
      // Fused o_proj+Cast+Add: output = bf16(fa2Out @ W + x)
      afterAttn=palloc(H*2); // bf16 output with residual
      launch(kOProj,[1,Math.ceil(H/16),1],[128,1,1],[fa2Out,wp(W,layer,"self_attn.o_proj.weight"),afterAttn,x],`op.${layer}`);
      pfree(fa2Out); pfree(x);
    }
    
    // post_attention_layernorm (afterAttn is already bf16 from fused GEMM+Cast+Add)
    pfree(normed);
    
    // post_attention_layernorm
    const normed2=palloc(H*2);
    launch(kRms,[1,1,1],[128,1,1],[afterAttn,wp(W,layer,"post_attention_layernorm.weight"),normed2],`rms2.${layer}`);
    
    // MLP — fused gate+up dual GEMM, then SwiGLU, then fused down+Cast+Add
    const gate=palloc(INTER*4); const up=palloc(INTER*4); // f32 — dual GEMM (gate+up)
    launch(kGPUP,[1,Math.ceil(INTER/16),1],[128,1,1],[normed2,wp(W,layer,"mlp.gate_proj.weight"),gate,wp(W,layer,"mlp.up_proj.weight"),up],`gpup.${layer}`);
    pfree(normed2);
    const act=palloc(INTER*2);
    launch(kSg,[BLK1(INTER),1,1],[128,1,1],[gate,up,act],`sg.${layer}`);
    pfree(gate); pfree(up);
    // Fused down_proj+Cast+Add: output = bf16(act @ W + afterAttn)
    x=palloc(H*2); // bf16 output with residual
    launch(kDP,[1,Math.ceil(H/16),1],[128,1,1],[act,wp(W,layer,"mlp.down_proj.weight"),x,afterAttn],`dp.${layer}`);
    pfree(afterAttn); pfree(act);
  }
  
  // Final norm + lm_head
  const fn=palloc(H*2);
  launch(kRms,[1,1,1],[128,1,1],[x,W.get("model.language_model.norm.weight")!,fn],`rmsF`);
  const logits=palloc(VOCAB*4);
  launch(kLM,[1,Math.ceil(VOCAB/64),1],[128,1,1],[fn,embedW,logits],`lmHead`);

  // GPU argmax — no 1MB download, just 4 bytes (blocking cuDtoH later syncs)
  launch(kArgmax,[ARGMAX_SPLIT,1,1],[128,1,1],[logits,argmaxPV,argmaxPI],`argmax`);
  launch(kArgmaxC,[1,1,1],[128,1,1],[argmaxPV,argmaxPI,argmaxVal,argmaxIdx],`argmaxC`);
  pfree(x); pfree(fn); pfree(logits);
  const idxBuf=new Int32Array(1);
  cuDtoH(idxBuf.buffer,argmaxIdx,BigInt(4));
  const valBuf=new Float32Array(1);
  cuDtoH(valBuf.buffer,argmaxVal,BigInt(4));
  const bestId=idxBuf[0];
  const bestLog=valBuf[0];
  
  if (step >= tokenIds.length - 1) {
    // This is a generated token
    allTokens.push(bestId);
    const match = refTokens[step] !== undefined ? (bestId === refTokens[step] ? "✓" : "✗") : " ";
    console.log(`  step ${step}: token ${bestId} (logit ${bestLog.toFixed(3)}) ${match} ${step < refTokens.length ? `(ref: ${refTokens[step]})` : ""}`);
  }
  
  if (allTokens.length >= tokenIds.length + genLen) break;
  if (step < tokenIds.length - 1) continue; // still processing prompt
}

const dt=(performance.now()-fStart)/1000;
console.log(`\nGenerated ${genLen} tokens in ${dt.toFixed(1)}s (${(genLen/dt).toFixed(1)} tok/s)`);
console.log(`Pool: ${_allocCount} allocs, ${_poolHits} pool hits (hit rate: ${(_poolHits/(_allocCount+_poolHits)*100).toFixed(0)}%)`);
if (__PROF) profReport();
profGpuReport();
console.log(`\n=== Generated Text ===`);

// Decode tokens to text using the tokenizer (we need to reverse the tokenization)
// For now, just print token IDs
console.log(`Tokens: [${allTokens.join(", ")}]`);

// Try to decode using the tokenizer
// We can use the gigatoken library to decode, but it only has encode.
// Let's just compare with reference tokens.
console.log(`\nReference: [${refTokens.slice(0, allTokens.length).join(", ")}]`);
let matches=0;
for(let i=0;i<Math.min(allTokens.length,refTokens.length);i++) if(allTokens[i]===refTokens[i]) matches++;
console.log(`Match: ${matches}/${Math.min(allTokens.length,refTokens.length)} tokens`);
