// Debug GDN: test each stage independently
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch } from "../src/ttir";

const LKD=128, LVD=128, LVH=16, QKVD=6144, ZD=2048, KEYDIM=2048, EPS=1e-6;

function rsqrtNR(b: any, x: any): any {
  let yy=b.f32(1);
  for(let i=0;i<6;i++){const y2=b.mul(yy,yy);yy=b.mul(yy,b.sub(b.f32(1.5),b.mul(b.f32(0.5),b.mul(x,y2))));}
  return yy;
}

// Stage 1: just load and output q (raw, from QKV)
function buildLoadQ() {
  const b=new TTIRBuilder();
  const QKV=b.param("Q",{ptr:"bf16"}),Out=b.param("O",{ptr:"bf16"});
  const head=b.programId(0);
  const qOff=b.mul(head,b.i32(LKD));
  const tpQ=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),qOff],[1,LKD],"bf16",[1,0]);
  const q=b.load(tpQ,{boundaryCheck:[0,1],padding:1});
  const tpO=b.makeTensorPtr(Out,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"bf16",[1,0]);
  b.store(tpO,q,{boundaryCheck:[0,1]});
  return b.build("lq",4,3);
}

// Stage 2: conv1d output (silu(w3*q))
function buildConvOut() {
  const b=new TTIRBuilder();
  const QKV=b.param("Q",{ptr:"bf16"}),ConvW=b.param("CW",{ptr:"bf16"}),Out=b.param("O",{ptr:"bf16"});
  const head=b.programId(0);
  const qOff=b.mul(head,b.i32(LKD));
  const tpQ=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),qOff],[1,LKD],"bf16",[1,0]);
  const qRaw=b.fpext(b.load(tpQ,{boundaryCheck:[0,1],padding:1}),"f32");
  const cwOff=b.add(b.mul(b.mul(head,b.i32(LKD)),b.i32(4)),b.i32(3));
  const cwIdx=b.mul(b.arange(0,LKD),b.i32(4));
  const cwPtrs=b.addptr(b.splatPtr(ConvW,LKD,"bf16"),b.add(cwIdx,cwOff));
  const cw3=b.broadcastTo(b.fpext(b.load(cwPtrs),"f32"),[1,LKD]);
  const qW=b.mul(qRaw,cw3);
  const qConv=b.mul(qW,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(qW,b.f32(-1))))));
  const tpO=b.makeTensorPtr(Out,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"bf16",[1,0]);
  b.store(tpO,b.fptrunc(qConv,"bf16"),{boundaryCheck:[0,1]});
  return b.build("cv",4,3);
}

// Stage 3: qkDot (dot product after L2norm)
function buildQkDot() {
  const b=new TTIRBuilder();
  const QKV=b.param("Q",{ptr:"bf16"}),ConvW=b.param("CW",{ptr:"bf16"}),Out=b.param("O",{ptr:"f32"});
  const head=b.programId(0);
  const qOff=b.mul(head,b.i32(LKD));
  const kOff=b.add(b.i32(KEYDIM),qOff);
  const tpQ=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),qOff],[1,LKD],"bf16",[1,0]);
  const tpK=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),kOff],[1,LKD],"bf16",[1,0]);
  const qRaw=b.fpext(b.load(tpQ,{boundaryCheck:[0,1],padding:1}),"f32");
  const kRaw=b.fpext(b.load(tpK,{boundaryCheck:[0,1],padding:1}),"f32");
  const cwOff=b.add(b.mul(b.mul(head,b.i32(LKD)),b.i32(4)),b.i32(3));
  const cwIdx=b.mul(b.arange(0,LKD),b.i32(4));
  const cwPtrs=b.addptr(b.splatPtr(ConvW,LKD,"bf16"),b.add(cwIdx,cwOff));
  const cw3=b.broadcastTo(b.fpext(b.load(cwPtrs),"f32"),[1,LKD]);
  const qW=b.mul(qRaw,cw3);
  const qConv=b.mul(qW,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(qW,b.f32(-1))))));
  const kW=b.mul(kRaw,cw3);
  const kConv=b.mul(kW,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(kW,b.f32(-1))))));
  const qRstd=rsqrtNR(b,b.add(b.sum(b.mul(qConv,qConv),1),b.f32(1e-6)));
  const kRstd=rsqrtNR(b,b.add(b.sum(b.mul(kConv,kConv),1),b.f32(1e-6)));
  const qNorm=b.mul(qConv,b.mul(qRstd,b.f32(1/Math.sqrt(LKD))));
  const kNorm=b.mul(kConv,kRstd);
  const qkDot=b.sum(b.mul(qNorm,kNorm),1);
  // Store qkDot to Out[head]
  const tpO=b.makeTensorPtr(Out,[1,LVH],[LVH,1],[b.i32(0),head],[1,1],"f32",[1,0]);
  b.store(tpO,b.broadcastTo(qkDot,[1,1]),{boundaryCheck:[0,1]});
  return b.build("qd",4,3);
}

// bf16 → f32
function bf16ToFloat(u16: number): number {
  const buf = new ArrayBuffer(4);
  const u8 = new Uint8Array(buf);
  const f32 = new Float32Array(buf);
  u8[2] = u16 & 0xFF;
  u8[3] = u16 >> 8;
  return f32[0];
}

// Load weights
const stPath="/tmp/qwen35_0.8b.safetensors";
const data=await Bun.file(stPath).bytes();
const dv=new DataView(data.buffer,data.byteOffset,data.byteLength);
const hl=Number(dv.getBigUint64(0,true));
const hdr=JSON.parse(new TextDecoder().decode(data.subarray(8,8+hl)));
const ds=8+hl;
const base=cuAlloc(BigInt(data.length-ds));
cuHtoD(base,data.subarray(ds));cuSync();
const W=(n:string)=>{const i=hdr[n];if(!i)throw new Error("missing "+n);return base+BigInt(i.data_offsets[0]);};

// Load embedding for "Hello" (token 9419)
const embedW = W("model.language_model.embed_tokens.weight");
const hEmb=new Uint16Array(1024);
const eOff = ds + hdr["model.language_model.embed_tokens.weight"].data_offsets[0] + 9419*1024*2;
for(let j=0;j<1024;j++) hEmb[j]=data[eOff+j*2]|(data[eOff+j*2+1]<<8);
const xD=cuAlloc(BigInt(1024*2));cuHtoD(xD,hEmb.buffer);cuSync();

// RMSNorm to get normed input
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

const kRms=compileAndLoad(buildRMS(1024),"rms",4);
const normed=cuAlloc(BigInt(1024*2));
cuLaunch(kRms,[1,1,1],[128,1,1],[xD,W("model.language_model.layers.0.input_layernorm.weight"),normed]);
cuSync();
console.log("RMSNorm done");

// Project QKV
const kQKV=compileAndLoad(buildMM(1,QKVD,1024),"mm",4);
const qkvF32=cuAlloc(BigInt(QKVD*4));
cuLaunch(kQKV,[1,Math.ceil(QKVD/64),1],[128,1,1],[normed,W("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),qkvF32]);
cuSync();

// Cast to bf16
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
const kCs=compileAndLoad(buildCast(QKVD),"cs",4);
const qkvBf=cuAlloc(BigInt(QKVD*2));
cuLaunch(kCs,[Math.ceil(QKVD/1024),1,1],[128,1,1],[qkvF32,qkvBf]);
cuSync();

// Check QKV values
const hQKV=new Uint16Array(QKVD);
cuDtoH(hQKV.buffer,qkvBf,BigInt(QKVD*2));
let qMax=0,kMax=0,vMax=0;
for(let i=0;i<2048;i++){const v=Math.abs(bf16ToFloat(hQKV[i]));if(v>qMax)qMax=v;}
for(let i=2048;i<4096;i++){const v=Math.abs(bf16ToFloat(hQKV[i]));if(v>kMax)kMax=v;}
for(let i=4096;i<6144;i++){const v=Math.abs(bf16ToFloat(hQKV[i]));if(v>vMax)vMax=v;}
console.log(`QKV: qMax=${qMax.toFixed(4)}, kMax=${kMax.toFixed(4)}, vMax=${vMax.toFixed(4)}`);

// Test stage 1: load q
const kLQ=compileAndLoad(buildLoadQ(),"lq",4);
const outQ=cuAlloc(BigInt(ZD*2));
cuLaunch(kLQ,[LVH,1,1],[128,1,1],[qkvBf,outQ]);
cuSync();
const hQ=new Uint16Array(ZD);
cuDtoH(hQ.buffer,outQ,BigInt(ZD*2));
let lqNz=0,lqMax=0;
for(let i=0;i<ZD;i++){const v=bf16ToFloat(hQ[i]);if(v!==0)lqNz++;if(Math.abs(v)>lqMax)lqMax=Math.abs(v);}
console.log(`LoadQ: ${lqNz}/${ZD} nz, max=${lqMax.toFixed(6)}`);

// Test stage 2: conv1d output
const kCV=compileAndLoad(buildConvOut(),"cv",4);
const outCV=cuAlloc(BigInt(ZD*2));
cuLaunch(kCV,[LVH,1,1],[128,1,1],[qkvBf,W("model.language_model.layers.0.linear_attn.conv1d.weight"),outCV]);
cuSync();
const hCV=new Uint16Array(ZD);
cuDtoH(hCV.buffer,outCV,BigInt(ZD*2));
let cvNz=0,cvMax=0;
for(let i=0;i<ZD;i++){const v=bf16ToFloat(hCV[i]);if(v!==0)cvNz++;if(Math.abs(v)>cvMax)cvMax=Math.abs(v);}
console.log(`ConvOut: ${cvNz}/${ZD} nz, max=${cvMax.toFixed(6)}`);

// Test stage 3: qkDot
const kQD=compileAndLoad(buildQkDot(),"qd",4);
const outQD=cuAlloc(BigInt(LVH*4));
cuLaunch(kQD,[LVH,1,1],[128,1,1],[qkvBf,W("model.language_model.layers.0.linear_attn.conv1d.weight"),outQD]);
cuSync();
const hQD=new Float32Array(LVH);
cuDtoH(hQD.buffer,outQD,BigInt(LVH*4));
console.log("qkDot per head:", hQD.map(v=>v.toFixed(6)));
