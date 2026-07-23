// prototypes/qwen35.ts — Qwen3.5-0.8B forward pass (block assembly).
//   TOKENIZER_PATH=/tmp/tokenizer.json SAFETENSORS_PATH=/tmp/qwen35_0.8b.safetensors bun run prototypes/qwen35.ts "Hello"
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";
import { dlopen, ptr as ffiPtr } from "bun:ffi";

const H=1024, VOCAB=248320, NL=24, FAI=4, INTER=3584, QKVD=6144, ZD=2048, EPS=1e-6;
const NH=8, NKV=2, HD=256, LKH=16, LVH=16, LKD=128, LVD=128;
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
const kOutProj=mm(seq,H,ZD); const kOP=mm(seq,H,NH*HD);
const kGP=mm(seq,INTER,H), kUP=mm(seq,INTER,H), kDP=mm(seq,H,INTER);
const kLM=mm(seq,VOCAB,H);
const kRms=compileAndLoad(buildRMS(H),"rms",4);
const kSg=compileAndLoad(buildSwiGLU(INTER),"sg",4);
const kAd=compileAndLoad(buildAdd(H),"ad",4);
const kCs=compileAndLoad(buildCast(H),"cs",4);
console.log(`compiled ${9} kernels in ${((performance.now()-t0)/1000).toFixed(1)}s`);

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

// forward pass (attention skipped — only MLP wired)
console.log("forward pass (MLP only, attention=identity)...");
const fStart=performance.now();

for(let layer=0;layer<NL;layer++){
  const full=isFull(layer);
  const lp=`model.language_model.layers.${layer}`;

  // 1. input_layernorm
  const normed=cuAlloc(BigInt(seq*H*2));
  launch(kRms,[seq,1,1],[128,1,1],[x,wp(W,layer,"input_layernorm.weight"),normed],`rms1.L${layer}`);

  // 2. attention = identity (skip entirely for now — GDN/FA2 need more work to inline)
  // Just pass normed through as the attention output
  // Cast normed (bf16) → f32 for the "attention output" path
  const attnF32=cuAlloc(BigInt(seq*H*4));
  // Leave attnF32 as zeros — we'll add it to x (no-op attention)

  // 3. cast + residual (x = x + 0 = x, since attention is zero)
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

  if(layer<4 || layer>=NL-2)console.log(`  layer ${layer} (${full?"full":"linear"}): MLP done`);
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
console.log(`\nforward: ${dt.toFixed(2)}s (${NL} layers, MLP-only)`);
console.log(`next token: ${bestId} (logit=${bestLog.toFixed(3)})`);
console.log(`\n=== Result ===`);
console.log(`prompt: "${prompt}"`);
console.log(`next token ID: ${bestId}`);
