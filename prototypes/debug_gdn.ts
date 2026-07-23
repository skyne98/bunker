// Debug: check if GDN produces non-zero output
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch } from "../src/ttir";
import { dlopen, ptr as ffiPtr } from "bun:ffi";

const LKD=128, LVD=128, LVH=16, QKVD=6144, ZD=2048, KEYDIM=2048, EPS=1e-6;

// Reuse the GDN kernel builder
function buildGDNSeq1() {
  const b=new TTIRBuilder();
  const QKV=b.param("QKV",{ptr:"bf16"}),Z=b.param("Z",{ptr:"bf16"}),ConvW=b.param("CW",{ptr:"bf16"});
  const ALog=b.param("AL",{ptr:"f32"}),dtB=b.param("DT",{ptr:"bf16"}),aP=b.param("AP",{ptr:"f32"}),bP=b.param("BP",{ptr:"f32"});
  const NormW=b.param("NW",{ptr:"f32"}),Out=b.param("O",{ptr:"bf16"});
  const head=b.programId(0);
  const qOff=b.mul(head,b.i32(LKD));
  const kOff=b.add(b.i32(KEYDIM),qOff);
  const vOff=b.add(b.i32(2*KEYDIM),qOff);
  const tpQ=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),qOff],[1,LKD],"bf16",[1,0]);
  const tpK=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),kOff],[1,LKD],"bf16",[1,0]);
  const tpV=b.makeTensorPtr(QKV,[1,QKVD],[QKVD,1],[b.i32(0),vOff],[1,LKD],"bf16",[1,0]);
  const qRaw=b.fpext(b.load(tpQ,{boundaryCheck:[0,1],padding:1}),"f32");
  const kRaw=b.fpext(b.load(tpK,{boundaryCheck:[0,1],padding:1}),"f32");
  const vRaw=b.fpext(b.load(tpV,{boundaryCheck:[0,1],padding:1}),"f32");
  const cwOff=b.add(b.mul(b.mul(head,b.i32(LKD)),b.i32(4)),b.i32(3));
  const cwIdx=b.mul(b.arange(0,LKD),b.i32(4));
  const cwPtrs=b.addptr(b.splatPtr(ConvW,LKD,"bf16"),b.add(cwIdx,cwOff));
  const cw3=b.broadcastTo(b.fpext(b.load(cwPtrs),"f32"),[1,LKD]);
  const qW=b.mul(qRaw,cw3);
  const qConv=b.mul(qW,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(qW,b.f32(-1))))));
  const kW=b.mul(kRaw,cw3);
  const kConv=b.mul(kW,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(kW,b.f32(-1))))));
  const vW=b.mul(vRaw,cw3);
  const vConv=b.mul(vW,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(vW,b.f32(-1))))));
  const qRstd=rsqrtNR(b,b.add(b.sum(b.mul(qConv,qConv),1),b.f32(1e-6)));
  const kRstd=rsqrtNR(b,b.add(b.sum(b.mul(kConv,kConv),1),b.f32(1e-6)));
  const qNorm=b.mul(qConv,b.mul(qRstd,b.f32(1/Math.sqrt(LKD))));
  const kNorm=b.mul(kConv,kRstd);
  const bVal=b.load(b.addptr(b.splatPtr(bP,1,"f32"),head));
  const beta=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(bVal,b.f32(-1)))));
  const qkDot=b.sum(b.mul(qNorm,kNorm),1);
  const delta=b.mul(vConv,beta);
  const o=b.mul(delta,qkDot);
  const oMs=b.divf(b.sum(b.mul(o,o),1),b.f32(LVD));
  const oRstd=rsqrtNR(b,b.add(oMs,b.f32(EPS)));
  const oNormed=b.mul(o,oRstd);
  const tpNW=b.makeTensorPtr(NormW,[1,LVD],[LVD,1],[b.i32(0),qOff],[1,LVD],"f32",[1,0]);
  const nw=b.load(tpNW,{boundaryCheck:[0,1],padding:1});
  const oWeighted=b.mul(oNormed,nw);
  const tpZ=b.makeTensorPtr(Z,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"bf16",[1,0]);
  const zVal=b.fpext(b.load(tpZ,{boundaryCheck:[0,1],padding:1}),"f32");
  const zSig=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(zVal,b.f32(-1)))));
  const oGated=b.mul(oWeighted,zSig);
  const tpO=b.makeTensorPtr(Out,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"bf16",[1,0]);
  b.store(tpO,b.fptrunc(oGated,"bf16"),{boundaryCheck:[0,1]});
  return b.build("gdn1",4,3);
}
function rsqrtNR(b: any, x: any): any {
  let yy=b.f32(1);
  for(let i=0;i<6;i++){const y2=b.mul(yy,yy);yy=b.mul(yy,b.sub(b.f32(1.5),b.mul(b.f32(0.5),b.mul(x,y2))));}
  return yy;
}

// Load weights
const stPath="/tmp/qwen35_0.8b.safetensors";
const data=await Bun.file(stPath).bytes();
const dv=new DataView(data.buffer,data.byteOffset,data.byteLength);
const hl=Number(dv.getBigUint64(0,true));
const hdr=JSON.parse(new TextDecoder().decode(data.subarray(8,8+hl)));
const ds=8+hl;
const W=new Map<string,bigint>();
for(const [n,i] of Object.entries(hdr)){if(n==="__metadata__")continue;W.set(n,BigInt(ds+(i as any).data_offsets[0]));}

// Create synthetic QKV input (all 1.0 in bf16)
const qkvBf=new Uint16Array(QKVD).fill(0x3F00); // 0.5 in bf16
const qkvD=cuAlloc(BigInt(QKVD*2));cuHtoD(qkvD,qkvBf.buffer);cuSync();

const zBf=new Uint16Array(ZD).fill(0x3F00); // 0.5
const zD=cuAlloc(BigInt(ZD*2));cuHtoD(zD,zBf.buffer);cuSync();

const aP=new Float32Array(LVH).fill(0.0); // a=0
const aPD=cuAlloc(BigInt(LVH*4));cuHtoD(aPD,aP.buffer);cuSync();

const bP=new Float32Array(LVH).fill(1.0); // b=1 → beta=sigmoid(1)=0.73
const bPD=cuAlloc(BigInt(LVH*4));cuHtoD(bPD,bP.buffer);cuSync();

const outD=cuAlloc(BigInt(ZD*2));

const kGDN=compileAndLoad(buildGDNSeq1(),"gdn1",4);
console.log("GDN compiled");

const rc=cuLaunch(kGDN,[LVH,1,1],[128,1,1],[
  qkvD, zD,
  W.get("model.language_model.layers.0.linear_attn.conv1d.weight")!,
  W.get("model.language_memory.layers.0.linear_attn.A_log") ?? W.get("model.language_model.layers.0.linear_attn.A_log")!,
  W.get("model.language_model.layers.0.linear_attn.dt_bias")!,
  aPD, bPD,
  W.get("model.language_model.layers.0.linear_attn.norm.weight")!,
  outD
]);
console.log("launch rc:",rc); const sr=cuSync(); console.log("sync rc:",sr);
cuSync();

const hOut=new Uint16Array(ZD);
cuDtoH(hOut.buffer,outD,BigInt(ZD*2));
// Convert bf16 to float
const out=new Float32Array(ZD);
const tmpBuf=new ArrayBuffer(4);
const tmpU8=new Uint8Array(tmpBuf);
const tmpU16=new Uint16Array(tmpBuf);
const tmpF32=new Float32Array(tmpBuf);
for(let i=0;i<ZD;i++){
  tmpU16[0]=hOut[i];
  // bf16 → f32: shift left 16
  tmpU8[2]=hOut[i]&0xFF;
  tmpU8[3]=hOut[i]>>8;
  out[i]=tmpF32[0];
}
let nz=0,maxV=0;
for(let i=0;i<ZD;i++){if(out[i]!==0)nz++; if(Math.abs(out[i])>maxV)maxV=Math.abs(out[i]);}
console.log(`GDN output: ${nz}/${ZD} non-zero, max abs=${maxV.toFixed(6)}`);
console.log("first 8:",out.slice(0,8).map(v=>v.toFixed(6)));
