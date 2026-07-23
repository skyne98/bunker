// prototypes/qwen35.ts — Qwen3.5-0.8B forward pass with GDN linear attention.
//   TOKENIZER_PATH=/tmp/tokenizer.json SAFETENSORS_PATH=/tmp/qwen35_0.8b.safetensors bun run prototypes/qwen35.ts "Hello"
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";
import { dlopen, ptr as ffiPtr } from "bun:ffi";

const H=1024, VOCAB=248320, NL=24, FAI=4, INTER=3584, QKVD=6144, ZD=2048, EPS=1e-6;
const NH=8, NKV=2, HD=256, LKH=16, LVH=16, LKD=128, LVD=128;
const KEYDIM=LKH*LKD, VALDIM=LVH*LVD; // 2048 each
const isFull = (l:number) => l % FAI === FAI-1;
const BLK1 = (n:number) => Math.ceil(n/1024);

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
  console.log(`loaded ${W.size} weights (${((data.length-ds)/1e9).toFixed(2)} GB)`);
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

// ── kernel builders ──
function buildMM(M:number,N:number,K:number) {
  const BM=Math.min(64,M),BN=Math.min(64,N),BK=Math.min(64,K);
  const b=new TTIRBuilder();
  const A=b.param("A",{ptr:"bf16"}),B=b.param("B",{ptr:"bf16"}),C=b.param("C",{ptr:"f32"});
  const pM=b.programId(0),pN=b.programId(1);
  const tpA=b.makeTensorPtr(A,[M,K],[K,1],[b.mul(pM,b.i32(BM)),b.i32(0)],[BM,BK],"bf16",[1,0]);
  const tpB=b.makeTensorPtr(B,[K,N],[1,K],[b.i32(0),b.mul(pN,b.i32(BN))],[BK,BN],"bf16",[0,1]);
  const tpC=b.makeTensorPtr(C,[M,N],[N,1],[b.mul(pM,b.i32(BM)),b.mul(pN,b.i32(BN))],[BM,BN],"f32",[1,0]);
  const a0=b.zeros([BM,BN],"f32");
  const [acc]=b.forIter(b.index(0),b.index(K),b.index(BK),[a0,tpA,tpB],(bb,_,[a,tA,tB])=>{
    const n=bb.dot(bb.load(tA),bb.load(tB),a);
    return[n,bb.advance(tA,[bb.i32(0),bb.i32(BK)]),bb.advance(tB,[bb.i32(BK),bb.i32(0)])];
  });
  b.store(tpC,acc,{boundaryCheck:[0,1]});
  return b.build("mm",4,3);
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
  let yy=b.f32(1);
  for(let i=0;i<6;i++){const y2=b.mul(yy,yy);yy=b.mul(yy,b.sub(b.f32(1.5),b.mul(b.f32(0.5),b.mul(b.add(msBc,b.f32(EPS)),y2))));}
  let y=b.mul(x,yy);
  y=b.mul(y,b.add(b.f32(1),b.fpext(b.load(tpW,{boundaryCheck:[0,1],padding:1}),"f32")));
  b.store(tpY,b.fptrunc(y,"bf16"),{boundaryCheck:[0,1]});
  return b.build("rms",4,3);
}
function buildSwiGLU(N:number) {
  const b=new TTIRBuilder();
  const G=b.param("G",{ptr:"f32"}),U=b.param("U",{ptr:"f32"}),O=b.param("O",{ptr:"bf16"});
  const row=b.programId(0);
  const BLK=Math.min(1024,N);
  const off=b.mul(row,b.i32(BLK));
  const tpG=b.makeTensorPtr(G,[1,N],[N,1],[off,b.i32(0)],[1,BLK],"f32",[1,0]);
  const tpU=b.makeTensorPtr(U,[1,N],[N,1],[off,b.i32(0)],[1,BLK],"f32",[1,0]);
  const tpO=b.makeTensorPtr(O,[1,N],[N,1],[off,b.i32(0)],[1,BLK],"bf16",[1,0]);
  const g=b.load(tpG,{boundaryCheck:[0,1],padding:1});
  const u=b.load(tpU,{boundaryCheck:[0,1],padding:1});
  const sig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(g,b.f32(-1)))));
  b.store(tpO,b.fptrunc(b.mul(b.mul(g,sig),u),"bf16"),{boundaryCheck:[0,1]});
  return b.build("sg",4,3);
}
function buildAdd(N:number) {
  const b=new TTIRBuilder();
  const A=b.param("A",{ptr:"bf16"}),B=b.param("B",{ptr:"bf16"}),O=b.param("O",{ptr:"bf16"});
  const row=b.programId(0);
  const BLK=Math.min(1024,N);
  const off=b.mul(row,b.i32(BLK));
  const tpA=b.makeTensorPtr(A,[1,N],[N,1],[off,b.i32(0)],[1,BLK],"bf16",[1,0]);
  const tpB=b.makeTensorPtr(B,[1,N],[N,1],[off,b.i32(0)],[1,BLK],"bf16",[1,0]);
  const tpO=b.makeTensorPtr(O,[1,N],[N,1],[off,b.i32(0)],[1,BLK],"bf16",[1,0]);
  const a=b.fpext(b.load(tpA,{boundaryCheck:[0,1],padding:1}),"f32");
  const bb=b.fpext(b.load(tpB,{boundaryCheck:[0,1],padding:1}),"f32");
  b.store(tpO,b.fptrunc(b.add(a,bb),"bf16"),{boundaryCheck:[0,1]});
  return b.build("ad",4,3);
}
function buildCast(N:number) {
  const b=new TTIRBuilder();
  const X=b.param("X",{ptr:"f32"}),Y=b.param("Y",{ptr:"bf16"});
  const row=b.programId(0);
  const BLK=Math.min(1024,N);
  const off=b.mul(row,b.i32(BLK));
  const tpX=b.makeTensorPtr(X,[1,N],[N,1],[off,b.i32(0)],[1,BLK],"f32",[1,0]);
  const tpY=b.makeTensorPtr(Y,[1,N],[N,1],[off,b.i32(0)],[1,BLK],"bf16",[1,0]);
  const x=b.load(tpX,{boundaryCheck:[0,1],padding:1});
  b.store(tpY,b.fptrunc(x,"bf16"),{boundaryCheck:[0,1]});
  return b.build("cs",4,3);
}

// ── GDN linear attention: single-kernel for seq=1 ──
// Does: conv1d(tap3) + split + L2norm(q,k) + delta_rule(T=1) + RMSNormGated
// Input: QKV[1, QKVD] bf16, Z[1, ZD] bf16, ConvW[QKVD, 4] bf16, ALog[LVH] f32, dtBias[LVH] bf16, aP[1, LVH] f32, bP[1, LVH] f32, NormW[LVD] f32, Out[1, ZD] bf16
// Grid: [LVH] — one program per value head (d_k=d_v=128)
function buildGDNSeq1() {
  const b=new TTIRBuilder();
  const QKV=b.param("QKV",{ptr:"bf16"}),Z=b.param("Z",{ptr:"bf16"}),ConvW=b.param("CW",{ptr:"bf16"});
  const ALog=b.param("AL",{ptr:"f32"}),dtB=b.param("DT",{ptr:"bf16"}),aP=b.param("AP",{ptr:"f32"}),bP=b.param("BP",{ptr:"f32"});
  const NormW=b.param("NW",{ptr:"f32"}),Out=b.param("O",{ptr:"bf16"});
  const head=b.programId(0); // 0..15

  // Load q, k, v for this head from QKV[1, QKVD] = [1, 3*2048]
  // q = QKV[0, head*128 : (head+1)*128]  (offset head*LKD in QKV, within first 2048)
  // k = QKV[0, 2048 + head*128 : 2048 + (head+1)*128]
  // v = QKV[0, 4096 + head*128 : 4096 + (head+1)*128]
  const qOff=b.mul(head,b.i32(LKD));
  const kOff=b.add(b.i32(KEYDIM),qOff);
  const vOff=b.add(b.i32(2*KEYDIM),qOff);
  // Load conv weight tap 3 for this head's channels: ConvW[head*128 + i, 3] for i=0..127
  // ConvW is [6144, 1, 4] → flattened [6144, 4] row-major. Tap 3 = ConvW[head*128+i, 3] = ConvW_flat[(head*128+i)*4 + 3]
  const cwBase=b.mul(b.add(b.mul(head,b.i32(LKD)),b.i32(0)),b.i32(4)); // head*128*4
  const cwBase2=b.add(b.mul(b.mul(head,b.i32(LKD)),b.i32(4)),b.i32(3));
  const cwAr2=b.mul(b.arange(0,LKD),b.i32(4));
  const cwOffs=b.add(cwAr2,cwBase2);
  const cw3=b.broadcast(b.expandDims(b.fpext(b.load(b.addptr(b.splatPtr(ConvW,LKD,"bf16"),cwOffs)),"f32"),0),[1,LKD]);
  // Apply conv1d (tap 3 only for seq=1): qkv_silu = silu(w3 * qkv)
  const tpQ=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),qOff],[1,LKD],"bf16",[1,0]);
  const tpK=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),kOff],[1,LKD],"bf16",[1,0]);
  const tpV=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),vOff],[1,LKD],"bf16",[1,0]);
  const qRaw=b.fpext(b.load(tpQ,{boundaryCheck:[0,1],padding:1}),"f32");
  const kRaw=b.fpext(b.load(tpK,{boundaryCheck:[0,1],padding:1}),"f32");
  const vRaw=b.fpext(b.load(tpV,{boundaryCheck:[0,1],padding:1}),"f32");
  // Conv1d tap 3: multiply by w[:, 3] and apply silu
  // Load conv weights: CW[head*128+i, 3] for i=0..127 → pointer arithmetic
  // Apply: q_silu = silu(w3 * q), k_silu = silu(w3 * k), v_silu = silu(w3 * v)
  // But wait — conv1d weight is per-channel: CW[head*128+i, 3] applies to channel head*128+i
  // The q/k/v for this head are at QKV[0, head*128 : head*128+128], which corresponds to channels head*128..head*128+127
  // So CW[head*128+i, 3] applies to q[i] for i=0..127 ✓
  const qSig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(b.mul(qRaw,cw3),b.f32(-1))))); // sigmoid(w3*q)
  const qConv=b.mul(b.mul(qRaw,cw3),qSig); // silu(w3*q) = w3*q*sigmoid(w3*q)
  const kSig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(b.mul(kRaw,cw3),b.f32(-1)))));
  const kConv=b.mul(b.mul(kRaw,cw3),kSig);
  const vSig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(b.mul(vRaw,cw3),b.f32(-1)))));
  const vConv=b.mul(b.mul(vRaw,cw3),vSig);
  // L2norm q and k: q = q / ||q||, k = k / ||k|| (then q *= 1/sqrt(d_k))
  const qRstd=b.broadcast(b.expandDims(b.rsqrt(b.add(b.sum(b.mul(qConv,qConv),1),b.f32(1e-6))),1),[1,LKD]); const qNorm=b.mul(qConv,qRstd);
  const kRstd=b.broadcast(b.expandDims(b.rsqrt(b.add(b.sum(b.mul(kConv,kConv),1),b.f32(1e-6))),1),[1,LKD]); const kNorm=b.mul(kConv,kRstd);
  const qScaled=b.mul(qNorm,b.f32(1/Math.sqrt(LKD)));
  // Delta rule for T=1 (S=0):
  // beta = sigmoid(bP[head])
  // delta = v * beta (since k·S = 0)
  // o = (q·k) * delta = sum(q*k) * (v * beta)
  const bVal=b.load(b.addptr(b.splatPtr(bP,1,"f32"),b.splat(head,[1],"i32")));
  const beta=b.broadcast(b.expandDims(b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(bVal,b.f32(-1))))),0),[1,LKD]);
  const qkDot=b.broadcast(b.expandDims(b.sum(b.mul(qScaled,kNorm),1),1),[1,LKD]);
  const delta=b.mul(vConv,beta); // [1, 128]
  const o=b.mul(delta,b.broadcast(b.expandDims(qkDot,1),[1,LKD])); // [1, 128]
  // RMSNormGated: norm(o) * silu(z)
  const oMs=b.divf(b.sum(b.mul(o,o),1),b.f32(LVD));
  const oRstd=b.broadcast(b.expandDims(b.rsqrt(b.add(oMs,b.f32(EPS))),1),[1,LVD]);
  const oNormed=b.mul(o,oRstd);
  // Load norm weight for this head: NormW[head*128 : head*128+128]
  const tpNW=b.makeTensorPtr(NormW,[1,LVD],[LVD,1],[b.i32(0),qOff],[1,LVD],"f32",[1,0]);
  const nw=b.load(tpNW,{boundaryCheck:[0,1],padding:1});
  const oWeighted=b.mul(oNormed,nw);
  // Load z for this head: Z[0, head*128 : head*128+128]
  const tpZ=b.makeTensorPtr(Z,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"bf16",[1,0]);
  const zVal=b.fpext(b.load(tpZ,{boundaryCheck:[0,1],padding:1}),"f32");
  const zSig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(zVal,b.f32(-1)))));
  const oGated=b.mul(oWeighted,zSig);
  // Store output: Out[0, head*128 : head*128+128]
  const tpO=b.makeTensorPtr(Out,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"bf16",[1,0]);
  b.store(tpO,b.fptrunc(oGated,"bf16"),{boundaryCheck:[0,1]});
  return b.build("gdn1",4,3);
}

function launch(k:any,g:number[],bl:number[],args:any[],label:string) {
  const rc=cuLaunch(k,[g[0],g[1],g[2]] as [number,number,number],[bl[0],bl[1],bl[2]] as [number,number,number],args);
  const sr=cuSync(); if(rc)console.log(`  [!]${label} rc=${rc}`); if(sr)console.log(`  [!]${label} sync=${sr}`);
  return rc||sr;
}

// ── main ──
const tokPath=process.env.TOKENIZER_PATH||"/tmp/tokenizer.json";
const stPath=process.env.SAFETENSORS_PATH||"/tmp/qwen35_0.8b.safetensors";
const prompt=process.argv[2]||"Hello";
console.log("=== Qwen3.5-0.8B ===");
const W=await loadWeights(stPath);
const tokenIds=tokenize(prompt,tokPath);
const seq=tokenIds.length;
console.log(`prompt: "${prompt}" → ${seq} tokens: [${tokenIds.slice(0,10).join(", ")}...]`);

console.log("compiling kernels...");
const t0=performance.now();
const mm=(M:number,N:number,K:number)=>compileAndLoad(buildMM(M,N,K),"mm",4);
const kQKV=mm(seq,QKVD,H), kZ=mm(seq,ZD,H), kA=mm(seq,LVH,H), kB=mm(seq,LVH,H);
const kOutProj=mm(seq,H,ZD);
const kGP=mm(seq,INTER,H), kUP=mm(seq,INTER,H), kDP=mm(seq,H,INTER);
const kLM=mm(seq,VOCAB,H);
const kRms=compileAndLoad(buildRMS(H),"rms",4);
const kSg=compileAndLoad(buildSwiGLU(INTER),"sg",4);
const kAd=compileAndLoad(buildAdd(H),"ad",4);
const kCs=compileAndLoad(buildCast(H),"cs",4);
const kGDN1=compileAndLoad(buildGDNSeq1(),"gdn1",4);
console.log(`compiled ${12} kernels in ${((performance.now()-t0)/1000).toFixed(1)}s`);

// embed
console.log("embedding...");
const embedW=W.get("model.language_model.embed_tokens.weight")!;
const hEmb=new Uint16Array(seq*H);
{
  const data=await Bun.file(stPath).bytes();
  const dv2=new DataView(data.buffer,data.byteOffset,data.byteLength);
  const hl2=Number(dv2.getBigUint64(0,true));
  const hdr2=JSON.parse(new TextDecoder().decode(data.subarray(8,8+hl2)));
  const ds2=8+hl2;
  const ei=hdr2["model.language_model.embed_tokens.weight"];
  const es=ds2+ei.data_offsets[0];
  for(let i=0;i<seq;i++){const id=tokenIds[i];const rs=es+id*H*2;for(let j=0;j<H;j++)hEmb[i*H+j]=data[rs+j*2]|(data[rs+j*2+1]<<8);}
}
let x=cuAlloc(BigInt(seq*H*2));cuHtoD(x,hEmb.buffer);cuSync();
console.log(`embedded: ${seq}×${H} bf16`);

// forward
console.log("forward pass...");
const fStart=performance.now();

for(let layer=0;layer<NL;layer++){
  const full=isFull(layer);

  // 1. input_layernorm
  const normed=cuAlloc(BigInt(seq*H*2));
  launch(kRms,[seq,1,1],[128,1,1],[x,wp(W,layer,"input_layernorm.weight"),normed],`rms1.L${layer}`);

  let attnF32=cuAlloc(BigInt(seq*H*4));

  if(!full){
    // Linear attention (GatedDeltaNet) — wired for seq=1
    // Projections
    const qkv=cuAlloc(BigInt(seq*QKVD*4));
    launch(kQKV,[Math.ceil(seq/64),Math.ceil(QKVD/64),1],[128,1,1],[normed,wp(W,layer,"linear_attn.in_proj_qkv.weight"),qkv],`qkv.L${layer}`);
    const z=cuAlloc(BigInt(seq*ZD*4));
    launch(kZ,[Math.ceil(seq/64),Math.ceil(ZD/64),1],[128,1,1],[normed,wp(W,layer,"linear_attn.in_proj_z.weight"),z],`z.L${layer}`);
    const aP=cuAlloc(BigInt(seq*LVH*4));
    launch(kA,[Math.ceil(seq/64),Math.ceil(LVH/64),1],[128,1,1],[normed,wp(W,layer,"linear_attn.in_proj_a.weight"),aP],`a.L${layer}`);
    const bP=cuAlloc(BigInt(seq*LVH*4));
    launch(kB,[Math.ceil(seq/64),Math.ceil(LVH/64),1],[128,1,1],[normed,wp(W,layer,"linear_attn.in_proj_b.weight"),bP],`b.L${layer}`);
    // Cast qkv, z to bf16
    const qkvB=cuAlloc(BigInt(seq*QKVD*2));
    // Need a cast for QKVD — compile one or reuse kCs with different N
    // Actually kCs was compiled for H=1024. Need a cast for QKVD=6144 and ZD=2048.
    // Let's just cast manually with a simple approach: use the GDN kernel which takes bf16 input
    // The GDN kernel needs bf16 QKV, bf16 Z, f32 ALog, bf16 dtBias, f32 aP, f32 bP, f32 NormW
    // We have f32 qkv, f32 z, f32 aP, f32 bP. Need to cast qkv and z to bf16.
    // For now: the GDN kernel can be adapted to take f32 input. Let me change the buildGDNSeq1 to take f32 QKV and Z.
    // Actually, let me just pass the f32 pointers and adjust the kernel.
    // Simpler: compile cast kernels for QKVD and ZD.
    const kCsQKV=compileAndLoad(buildCast(QKVD),"csQ",4);
    const kCsZD=compileAndLoad(buildCast(ZD),"csZ",4);
    launch(kCsQKV,[BLK1(seq*QKVD),1,1],[128,1,1],[qkv,qkvB],`csQKV.L${layer}`);
    const zB=cuAlloc(BigInt(seq*ZD*2));
    launch(kCsZD,[BLK1(seq*ZD),1,1],[128,1,1],[z,zB],`csZ.L${layer}`);
    cuFree(qkv);cuFree(z);
    // GDN kernel: takes bf16 QKV, bf16 Z, bf16 ConvW, f32 ALog, bf16 dtBias, f32 aP, f32 bP, f32 NormW → bf16 Out
    const gdnOut=cuAlloc(BigInt(seq*ZD*2));
    launch(kGDN1,[LVH,1,1],[128,1,1],[
      qkvB, zB,
      wp(W,layer,"linear_attn.conv1d.weight"),
      wp(W,layer,"linear_attn.A_log"),
      wp(W,layer,"linear_attn.dt_bias"),
      aP, bP,
      wp(W,layer,"linear_attn.norm.weight"),
      gdnOut
    ],`gdn.L${layer}`);
    cuFree(qkvB);cuFree(zB);cuFree(aP);cuFree(bP);
    // out_proj: [seq, H] = gdnOut[seq, ZD] @ W[H, ZD]
    launch(kOutProj,[Math.ceil(seq/64),Math.ceil(H/64),1],[128,1,1],[gdnOut,wp(W,layer,"linear_attn.out_proj.weight"),attnF32],`op.L${layer}`);
    cuFree(gdnOut);
    if(layer<4)console.log(`  layer ${layer} (linear): GDN+out_proj`);
  } else {
    // Full attention — still placeholder (identity)
    if(layer<8)console.log(`  layer ${layer} (full): FA2 placeholder (identity)`);
  }

  // 3. cast + residual
  const attnBf=cuAlloc(BigInt(seq*H*2));
  launch(kCs,[BLK1(seq*H),1,1],[128,1,1],[attnF32,attnBf],`cast.L${layer}`);
  const afterAttn=cuAlloc(BigInt(seq*H*2));
  launch(kAd,[BLK1(seq*H),1,1],[128,1,1],[x,attnBf,afterAttn],`add1.L${layer}`);
  cuFree(x);cuFree(attnF32);cuFree(attnBf);cuFree(normed);

  // 4. post_attention_layernorm
  const normed2=cuAlloc(BigInt(seq*H*2));
  launch(kRms,[seq,1,1],[128,1,1],[afterAttn,wp(W,layer,"post_attention_layernorm.weight"),normed2],`rms2.L${layer}`);

  // 5. MLP
  const gate=cuAlloc(BigInt(seq*INTER*4));
  launch(kGP,[Math.ceil(seq/64),Math.ceil(INTER/64),1],[128,1,1],[normed2,wp(W,layer,"mlp.gate_proj.weight"),gate],`gp.L${layer}`);
  const up=cuAlloc(BigInt(seq*INTER*4));
  launch(kUP,[Math.ceil(seq/64),Math.ceil(INTER/64),1],[128,1,1],[normed2,wp(W,layer,"mlp.up_proj.weight"),up],`up.L${layer}`);
  const act=cuAlloc(BigInt(seq*INTER*2));
  launch(kSg,[BLK1(seq*INTER),1,1],[128,1,1],[gate,up,act],`sg.L${layer}`);
  cuFree(gate);cuFree(up);cuFree(normed2);
  const mlpOut=cuAlloc(BigInt(seq*H*4));
  launch(kDP,[Math.ceil(seq/64),Math.ceil(H/64),1],[128,1,1],[act,wp(W,layer,"mlp.down_proj.weight"),mlpOut],`dp.L${layer}`);
  cuFree(act);

  // 6. cast + residual
  const mlpBf=cuAlloc(BigInt(seq*H*2));
  launch(kCs,[BLK1(seq*H),1,1],[128,1,1],[mlpOut,mlpBf],`cast2.L${layer}`);
  x=cuAlloc(BigInt(seq*H*2));
  launch(kAd,[BLK1(seq*H),1,1],[128,1,1],[afterAttn,mlpBf,x],`add2.L${layer}`);
  cuFree(afterAttn);cuFree(mlpOut);cuFree(mlpBf);
}

// final norm + lm_head
const fn=cuAlloc(BigInt(seq*H*2));
launch(kRms,[seq,1,1],[128,1,1],[x,W.get("model.language_model.norm.weight")!,fn],`rmsF`);
const logits=cuAlloc(BigInt(seq*VOCAB*4));
launch(kLM,[Math.ceil(seq/64),Math.ceil(VOCAB/64),1],[128,1,1],[fn,embedW,logits],`lmHead`);
cuSync();

// argmax
const hLog=new Float32Array(VOCAB);
cuDtoH(hLog.buffer,logits+BigInt((seq-1)*VOCAB*4));
let bestId=0,bestLog=-Infinity;
for(let i=0;i<VOCAB;i++)if(hLog[i]>bestLog){bestLog=hLog[i];bestId=i;}
const dt=(performance.now()-fStart)/1000;
console.log(`\nforward: ${dt.toFixed(2)}s (${NL} layers, GDN wired)`);
console.log(`next token: ${bestId} (logit=${bestLog.toFixed(3)})`);
console.log(`\n=== Result ===`);
console.log(`prompt: "${prompt}"`);
console.log(`next token ID: ${bestId}`);
