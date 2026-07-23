// prototypes/compare.ts — Differential testing: compare our TTIR pipeline against PyTorch reference.
//   source /tmp/refenv/bin/activate && python3 prototypes/ref_dump.py --prompt "Hello" --output /tmp/ref_tensors.safetensors
//   TOKENIZER_PATH=/tmp/tokenizer.json SAFETENSORS_PATH=/tmp/qwen35_0.8b.safetensors bun run prototypes/compare.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";
import { dlopen, ptr as ffiPtr } from "bun:ffi";

const H=1024, VOCAB=248320, NL=24, FAI=4, INTER=3584, QKVD=6144, ZD=2048, EPS=1e-6;
const LKH=16, LVH=16, LKD=128, LVD=128, KEYDIM=LKH*LKD, VALDIM=LVH*LVD;
const isFull = (l:number) => l % FAI === FAI-1;
const BLK1 = (n:number) => Math.ceil(n/1024);

// ── Load reference tensors ──
async function loadRef(path:string): Promise<{tensors: Map<string,Float32Array>, shapes: Map<string,number[]>}> {
  const data = await Bun.file(path).bytes();
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const hl = Number(dv.getBigUint64(0, true));
  const hdr = JSON.parse(new TextDecoder().decode(data.subarray(8, 8+hl)));
  const ds = 8+hl;
  const tensors = new Map<string,Float32Array>();
  const shapes = new Map<string,number[]>();
  for (const [n,i] of Object.entries(hdr)) {
    if(n==="__metadata__") continue;
    const info = i as any;
    const off = ds + info.data_offsets[0];
    const nbytes = info.data_offsets[1] - info.data_offsets[0];
    const nelem = nbytes / 4; // all f32
    const arr = new Float32Array(nelem);
    for(let k=0;k<nelem;k++){
      arr[k] = dv.getFloat32(off + k*4, true);
    }
    tensors.set(n, arr);
    shapes.set(n, info.shape);
  }
  return {tensors, shapes};
}

// ── Stats comparison ──
function compare(a: Float32Array, b: Float32Array, name: string): {maxAbs: number, meanAbs: number, cosSim: number, pass: boolean} {
  const n = Math.min(a.length, b.length);
  let maxAbs = 0, sumAbs = 0, dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxAbs) maxAbs = d;
    sumAbs += d;
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const meanAbs = sumAbs / n;
  const cosSim = (normA > 0 && normB > 0) ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : (normA === normB ? 1 : 0);
  const pass = maxAbs < 0.01 || cosSim > 0.999;
  return {maxAbs, meanAbs, cosSim, pass};
}

// bf16 → f32
function bf16BufToFloat(devPtr: bigint, n: number): Float32Array {
  const raw = new Uint16Array(n);
  cuDtoH(raw.buffer, devPtr, BigInt(n*2));
  const out = new Float32Array(n);
  const tmp = new ArrayBuffer(4);
  const u8 = new Uint8Array(tmp);
  const f32 = new Float32Array(tmp);
  for (let i = 0; i < n; i++) {
    u8[2] = raw[i] & 0xFF;
    u8[3] = raw[i] >> 8;
    out[i] = f32[0];
  }
  return out;
}

function f32BufToFloat(devPtr: bigint, n: number): Float32Array {
  const out = new Float32Array(n);
  cuDtoH(out.buffer, devPtr, BigInt(n*4));
  return out;
}

// ── Load model weights ──
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

// ── Tokenizer ──
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

// ── Kernel builders (same as qwen35.ts) ──
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
  const yy=b.rsqrtHw(b.add(msBc,b.f32(EPS)));
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
  const tpG=b.makeTensorPtr(G,[1,N],[N,1],[b.i32(0),off],[1,BLK],"f32",[1,0]);
  const tpU=b.makeTensorPtr(U,[1,N],[N,1],[b.i32(0),off],[1,BLK],"f32",[1,0]);
  const tpO=b.makeTensorPtr(O,[1,N],[N,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
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
  const row=b.programId(0);
  const BLK=Math.min(1024,N);
  const off=b.mul(row,b.i32(BLK));
  const tpX=b.makeTensorPtr(X,[1,N],[N,1],[b.i32(0),off],[1,BLK],"f32",[1,0]);
  const tpY=b.makeTensorPtr(Y,[1,N],[N,1],[b.i32(0),off],[1,BLK],"bf16",[1,0]);
  const x=b.load(tpX,{boundaryCheck:[0,1],padding:1});
  b.store(tpY,b.fptrunc(x,"bf16"),{boundaryCheck:[0,1]});
  return b.build("cs",4,3);
}
function rsqrtNR(b:any,x:any):any{let yy=b.f32(1);for(let i=0;i<13;i++){const y2=b.mul(yy,yy);yy=b.mul(yy,b.sub(b.f32(1.5),b.mul(b.f32(0.5),b.mul(x,y2))));}return yy;}

function buildGDNSeq1() {
  const b=new TTIRBuilder();
  const QKV=b.param("QKV",{ptr:"bf16"}),Z=b.param("Z",{ptr:"bf16"}),ConvW=b.param("CW",{ptr:"bf16"});
  const ALog=b.param("AL",{ptr:"f32"}),dtB=b.param("DT",{ptr:"bf16"}),aP=b.param("AP",{ptr:"f32"}),bP=b.param("BP",{ptr:"f32"});
  const NormW=b.param("NW",{ptr:"f32"}),Out=b.param("O",{ptr:"bf16"}),DbgOut=b.param("DO",{ptr:"f32"});
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
  // Conv1d weights are per-channel (6144 channels). q/k/v have different channels:
  // q: channels 0..2047, k: 2048..4095, v: 4096..6143
  // Weight layout: [6144, 1, 4] flat, tap 3 = flat[c*4+3]
  const cwIdx=b.mul(b.arange(0,LKD),b.i32(4));
  const cwOffQ=b.add(b.mul(b.mul(head,b.i32(LKD)),b.i32(4)),b.i32(3));
  const cwOffK=b.add(b.mul(b.add(b.i32(KEYDIM),b.mul(head,b.i32(LKD))),b.i32(4)),b.i32(3));
  const cwOffV=b.add(b.mul(b.add(b.i32(2*KEYDIM),b.mul(head,b.i32(LKD))),b.i32(4)),b.i32(3));
  const cw3q=b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(ConvW,LKD,"bf16"),b.add(cwIdx,cwOffQ))),"f32"),[1,LKD]);
  const cw3k=b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(ConvW,LKD,"bf16"),b.add(cwIdx,cwOffK))),"f32"),[1,LKD]);
  const cw3v=b.broadcastTo(b.fpext(b.load(b.addptr(b.splatPtr(ConvW,LKD,"bf16"),b.add(cwIdx,cwOffV))),"f32"),[1,LKD]);
  const qW=b.mul(qRaw,cw3q);
  const qConv=b.mul(qW,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(qW,b.f32(-1))))));
  const kW=b.mul(kRaw,cw3k);
  const kConv=b.mul(kW,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(kW,b.f32(-1))))));
  const vW=b.mul(vRaw,cw3v);
  const vConv=b.mul(vW,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(vW,b.f32(-1))))));
  const qRstd=b.rsqrtHw(b.add(b.sum(b.mul(qConv,qConv),1),b.f32(1e-6)));
  const kRstd=b.rsqrtHw(b.add(b.sum(b.mul(kConv,kConv),1),b.f32(1e-6)));
  const qNorm=b.mul(qConv,b.mul(qRstd,b.f32(1/Math.sqrt(LKD))));
  const kNorm=b.mul(kConv,kRstd);
  const bVal=b.load(b.addptr(b.splatPtr(bP,1,"f32"),head));
  const beta=b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(bVal,b.f32(-1)))));
  const qkDot=b.sum(b.mul(qNorm,kNorm),1);
  const delta=b.mul(vConv,beta);
  const o=b.mul(delta,qkDot);
  const oMs=b.divf(b.sum(b.mul(o,o),1),b.f32(LVD));
  const oRstd=b.rsqrtHw(b.add(oMs,b.f32(EPS)));
  const oNormed=b.mul(o,oRstd);
  const tpNW=b.makeTensorPtr(NormW,[1,LVD],[LVD,1],[b.i32(0),b.i32(0)],[1,LVD],"f32",[1,0]);
  const nw=b.load(tpNW,{boundaryCheck:[0,1],padding:1});
  const oWeighted=b.mul(oNormed,nw);
  const tpZ=b.makeTensorPtr(Z,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"bf16",[1,0]);
  const zVal=b.fpext(b.load(tpZ,{boundaryCheck:[0,1],padding:1}),"f32");
  // silu(z) = z * sigmoid(z)  — NOT just sigmoid(z)!
  const zSig=b.mul(zVal,b.divf(b.f32(1),b.add(b.f32(1),b.exp(b.mul(zVal,b.f32(-1))))));
  const oGated=b.mul(oWeighted,zSig);
  // Debug: store conv1d q output (for comparison)
  const tpDO=b.makeTensorPtr(DbgOut,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"f32",[1,0]);
  b.store(tpDO,o,{boundaryCheck:[0,1]});
  const tpO=b.makeTensorPtr(Out,[1,ZD],[ZD,1],[b.i32(0),qOff],[1,LVD],"bf16",[1,0]);
  b.store(tpO,b.fptrunc(oGated,"bf16"),{boundaryCheck:[0,1]});
  return b.build("gdn1",4,4);
}

function launch(k:any,g:number[],bl:number[],args:any[],label:string) {
  const rc=cuLaunch(k,[g[0],g[1],g[2]] as [number,number,number],[bl[0],bl[1],bl[2]] as [number,number,number],args);
  const sr=cuSync(); if(rc)console.log(`  [!]${label} rc=${rc}`); if(sr)console.log(`  [!]${label} sync=${sr}`);
  return rc||sr;
}

// ── Main ──
const tokPath=process.env.TOKENIZER_PATH||"/tmp/tokenizer.json";
const stPath=process.env.SAFETENSORS_PATH||"/tmp/qwen35_0.8b.safetensors";
const prompt=process.argv[2]||"Hello";
console.log("=== Differential Test: TTIR vs PyTorch ===");

// Load reference
console.log("Loading reference tensors...");
const refPath = "/tmp/ref_tensors.safetensors";
const {tensors: ref, shapes: refShapes} = await loadRef(refPath);
console.log(`  ${ref.size} reference tensors loaded`);

// Load model weights
console.log("Loading model weights...");
const W = await loadWeights(stPath);

// Tokenize
const tokenIds = tokenize(prompt, tokPath);
const seq = tokenIds.length;
console.log(`prompt: "${prompt}" → ${seq} tokens: [${tokenIds}]`);

// Compile kernels
console.log("Compiling kernels...");
const mm=(M:number,N:number,K:number)=>compileAndLoad(buildMM(M,N,K),"mm",4);
const kQKV=mm(seq,QKVD,H), kZ=mm(seq,ZD,H), kA=mm(seq,LVH,H), kB=mm(seq,LVH,H);
const kOutProj=mm(seq,H,ZD);
const kGP=mm(seq,INTER,H), kUP=mm(seq,INTER,H), kDP=mm(seq,H,INTER);
const kLM=mm(seq,VOCAB,H);
const kRms=compileAndLoad(buildRMS(H),"rms",4);
const kSg=compileAndLoad(buildSwiGLU(INTER),"sg",4);
const kAd=compileAndLoad(buildAdd(H),"ad",4);
const kCs=compileAndLoad(buildCast(H),"cs",4);
const kCsQKV=compileAndLoad(buildCast(QKVD),"cs",4);
const kCsZD=compileAndLoad(buildCast(ZD),"cs",4);
const kGDN1=compileAndLoad(buildGDNSeq1(),"gdn1",4);
console.log("  done");

// Embed
const embedW = W.get("model.language_model.embed_tokens.weight")!;
const hEmb = new Uint16Array(seq*H);
{
  const data = await Bun.file(stPath).bytes();
  const dv2 = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const hl2 = Number(dv2.getBigUint64(0, true));
  const hdr2 = JSON.parse(new TextDecoder().decode(data.subarray(8, 8+hl2)));
  const ds2 = 8+hl2;
  const ei = hdr2["model.language_model.embed_tokens.weight"];
  const es = ds2 + ei.data_offsets[0];
  for (let i=0; i<seq; i++) {
    const id = tokenIds[i];
    const rs = es + id*H*2;
    for (let j=0; j<H; j++) hEmb[i*H+j] = data[rs+j*2] | (data[rs+j*2+1]<<8);
  }
}
let x = cuAlloc(BigInt(seq*H*2)); cuHtoD(x, hEmb.buffer); cuSync();

// Compare helper: compare our tensor (bf16 or f32) against a reference tensor name
const results: {name: string, status: string, maxAbs: number, cosSim: number}[] = [];
function check(name: string, ourData: Float32Array, refName: string) {
  const refData = ref.get(refName);
  if (!refData) {
    results.push({name, status: "NO REF", maxAbs: -1, cosSim: -1});
    console.log(`  ${name.padEnd(50)} NO REF (${refName})`);
    return;
  }
  const c = compare(ourData, refData, name);
  const status = c.pass ? "✓ PASS" : "✗ FAIL";
  results.push({name, status, maxAbs: c.maxAbs, cosSim: c.cosSim});
  console.log(`  ${name.padEnd(50)} ${status}  maxAbs=${c.maxAbs.toExponential(3)}  cosSim=${c.cosSim.toFixed(6)}`);
}

// Check embedding
{
  const embF32 = bf16BufToFloat(x, seq*H);
  // Reference: model.embed_tokens__out doesn't exist directly, but model.layers.0__in is the layer 0 input
  check("embed", embF32, "model.layers.0.input_layernorm__in");
}

// Forward pass
console.log("\nForward pass with comparison:");
for (let layer = 0; layer < NL; layer++) {
  const full = isFull(layer);
  const lp = `L${layer}`;

  // 1. input_layernorm
  const normed = cuAlloc(BigInt(seq*H*2));
  launch(kRms, [seq,1,1], [128,1,1], [x, wp(W,layer,"input_layernorm.weight"), normed], `rms1.${lp}`);
  check(`L${layer}.input_layernorm`, bf16BufToFloat(normed, seq*H), `model.layers.${layer}.input_layernorm__out`);
  if (layer === 0) {
    const ourNorm = bf16BufToFloat(normed, seq*H);
    const refNorm = ref.get(`model.layers.0.input_layernorm__out`)!;
    console.log(`    [debug] RMSNorm L0 first 5: our=[${ourNorm.slice(0,5).map(v=>v.toFixed(4)).join(",")}] ref=[${refNorm.slice(0,5).map(v=>v.toFixed(4)).join(",")}]`);
  }

  let attnF32 = cuAlloc(BigInt(seq*H*4));

  if (!full) {
    // Linear attention (GDN)
    const qkv = cuAlloc(BigInt(seq*QKVD*4));
    launch(kQKV, [Math.ceil(seq/64),Math.ceil(QKVD/64),1], [128,1,1], [normed, wp(W,layer,"linear_attn.in_proj_qkv.weight"), qkv], `qkv.${lp}`);
    check(`L${layer}.in_proj_qkv`, f32BufToFloat(qkv, seq*QKVD), `model.layers.${layer}.linear_attn.in_proj_qkv__out`);

    const z = cuAlloc(BigInt(seq*ZD*4));
    launch(kZ, [Math.ceil(seq/64),Math.ceil(ZD/64),1], [128,1,1], [normed, wp(W,layer,"linear_attn.in_proj_z.weight"), z], `z.${lp}`);
    check(`L${layer}.in_proj_z`, f32BufToFloat(z, seq*ZD), `model.layers.${layer}.linear_attn.in_proj_z__out`);

    const aP = cuAlloc(BigInt(seq*LVH*4));
    launch(kA, [Math.ceil(seq/64),Math.ceil(LVH/64),1], [128,1,1], [normed, wp(W,layer,"linear_attn.in_proj_a.weight"), aP], `a.${lp}`);
    check(`L${layer}.in_proj_a`, f32BufToFloat(aP, seq*LVH), `model.layers.${layer}.linear_attn.in_proj_a__out`);

    const bP = cuAlloc(BigInt(seq*LVH*4));
    launch(kB, [Math.ceil(seq/64),Math.ceil(LVH/64),1], [128,1,1], [normed, wp(W,layer,"linear_attn.in_proj_b.weight"), bP], `b.${lp}`);
    check(`L${layer}.in_proj_b`, f32BufToFloat(bP, seq*LVH), `model.layers.${layer}.linear_attn.in_proj_b__out`);

    // Cast qkv, z → bf16
    const qkvB = cuAlloc(BigInt(seq*QKVD*2));
    launch(kCsQKV, [BLK1(seq*QKVD),1,1], [128,1,1], [qkv, qkvB], `csQ.${lp}`);
    const zB = cuAlloc(BigInt(seq*ZD*2));
    launch(kCsZD, [BLK1(seq*ZD),1,1], [128,1,1], [z, zB], `csZ.${lp}`);
    cuFree(qkv); cuFree(z);

    // GDN kernel
    const gdnOut = cuAlloc(BigInt(seq*ZD*2));
    const gdnDbg = cuAlloc(BigInt(seq*ZD*4)); // pre-norm debug output
    launch(kGDN1, [LVH,1,1], [128,1,1], [
      qkvB, zB,
      wp(W,layer,"linear_attn.conv1d.weight"),
      wp(W,layer,"linear_attn.A_log"),
      wp(W,layer,"linear_attn.dt_bias"),
      aP, bP,
      wp(W,layer,"linear_attn.norm.weight"),
      gdnOut,
      gdnDbg
    ], `gdn.${lp}`);
    // Compare conv1d q output vs reference (compute silu(conv1d__out[:,:,0]) from ref)
    {
      const refConvOut = ref.get(`model.layers.${layer}.linear_attn.conv1d__out`);
      if (refConvOut && layer === 0) {
        // refConvOut is flat [6144*4], reshape to [6144, 4], take [:, 0] (first output for seq=1)
        const ourQ = f32BufToFloat(gdnDbg, seq*ZD); // [2048] — pre-norm o
        // Compare pre-norm o vs reference norm__in
        const refNormIn = ref.get(`model.layers.${layer}.linear_attn.norm__in`);
        if (refNormIn) {
          const c2 = compare(ourQ, refNormIn, "gdn_prenorm");
          console.log(`  ${"L0.gdn_prenorm".padEnd(50)} ${c2.pass ? "✓ PASS" : "✗ FAIL"}  maxAbs=${c2.maxAbs.toExponential(3)}  cosSim=${c2.cosSim.toFixed(6)}`);
          console.log(`    our: ${ourQ.slice(0, 5).map(v=>v.toFixed(6)).join(", ")}`);
          console.log(`    ref: ${refNormIn.slice(0, 5).map(v=>v.toFixed(6)).join(", ")}`);
        }
        // Compute reference: silu(conv1d__out[:, 0]) for q channels (0..2047)
        const refSilu = new Float32Array(2048);
        for (let i = 0; i < 2048; i++) {
          const convOut = refConvOut[i*4]; // conv1d__out[:, 0] = w[3]*x
          refSilu[i] = convOut * (1 / (1 + Math.exp(-convOut))); // silu
        }
        const c = compare(ourQ, refSilu, "conv1d_q");
        console.log(`  ${"L0.conv1d_q_silu".padEnd(50)} ${c.pass ? "✓ PASS" : "✗ FAIL"}  maxAbs=${c.maxAbs.toExponential(3)}  cosSim=${c.cosSim.toFixed(6)}`);
        // Print first 5 values for debugging
        console.log(`    our:  ${ourQ.slice(0, 5).map(v=>v.toFixed(6)).join(", ")}`);
        console.log(`    ref:  ${refSilu.slice(0, 5).map(v=>v.toFixed(6)).join(", ")}`);
        // Also check our bf16 QKV vs reference f32 QKV (cast to bf16)
        const refQkvF32 = ref.get(`model.layers.${layer}.linear_attn.in_proj_qkv__out`)!;
        const ourQkvBf = bf16BufToFloat(qkvB, seq*QKVD);
        let qkvMaxDiff = 0;
        for (let i = 0; i < Math.min(10, refQkvF32.length); i++) {
          const diff = Math.abs(ourQkvBf[i] - refQkvF32[i]);
          if (diff > qkvMaxDiff) qkvMaxDiff = diff;
        }
        console.log(`    QKV bf16 vs ref: first 10 maxDiff=${qkvMaxDiff.toExponential(3)}`);
        console.log(`    our QKV: ${ourQkvBf.slice(0, 5).map(v=>v.toFixed(6)).join(", ")}`);
        console.log(`    ref QKV: ${refQkvF32.slice(0, 5).map(v=>v.toFixed(6)).join(", ")}`);
      }
    }
    // Compare GDN output (after RMSNormGated) — ref name is linear_attn.norm__out
    check(`L${layer}.gdn_norm_out`, bf16BufToFloat(gdnOut, seq*ZD), `model.layers.${layer}.linear_attn.norm__out`);

    // out_proj
    launch(kOutProj, [Math.ceil(seq/64),Math.ceil(H/64),1], [128,1,1], [gdnOut, wp(W,layer,"linear_attn.out_proj.weight"), attnF32], `op.${lp}`);
    cuFree(qkvB); cuFree(zB); cuFree(aP); cuFree(bP); cuFree(gdnDbg);
    cuFree(gdnOut);
    check(`L${layer}.attn_out`, f32BufToFloat(attnF32, seq*H), `model.layers.${layer}.linear_attn.out_proj__out`);
  } else {
    // Full attention — placeholder (identity)
    console.log(`  L${layer} (full): SKIPPED (FA2 not wired)`);
  }

  // cast + residual
  const attnBf = cuAlloc(BigInt(seq*H*2));
  launch(kCs, [BLK1(seq*H),1,1], [128,1,1], [attnF32, attnBf], `cast.${lp}`);
  const afterAttn = cuAlloc(BigInt(seq*H*2));
  launch(kAd, [BLK1(seq*H),1,1], [128,1,1], [x, attnBf, afterAttn], `add1.${lp}`);
  cuFree(x); cuFree(attnF32); cuFree(attnBf); cuFree(normed);
  check(`L${layer}.residual_after_attn`, bf16BufToFloat(afterAttn, seq*H), `model.layers.${layer}.post_attention_layernorm__in`);

  // post_attention_layernorm
  const normed2 = cuAlloc(BigInt(seq*H*2));
  launch(kRms, [seq,1,1], [128,1,1], [afterAttn, wp(W,layer,"post_attention_layernorm.weight"), normed2], `rms2.${lp}`);
  check(`L${layer}.post_attn_norm`, bf16BufToFloat(normed2, seq*H), `model.layers.${layer}.post_attention_layernorm__out`);

  // MLP
  const gate = cuAlloc(BigInt(seq*INTER*4));
  launch(kGP, [Math.ceil(seq/64),Math.ceil(INTER/64),1], [128,1,1], [normed2, wp(W,layer,"mlp.gate_proj.weight"), gate], `gp.${lp}`);
  check(`L${layer}.gate_proj`, f32BufToFloat(gate, seq*INTER), `model.layers.${layer}.mlp.gate_proj__out`);

  const up = cuAlloc(BigInt(seq*INTER*4));
  launch(kUP, [Math.ceil(seq/64),Math.ceil(INTER/64),1], [128,1,1], [normed2, wp(W,layer,"mlp.up_proj.weight"), up], `up.${lp}`);
  check(`L${layer}.up_proj`, f32BufToFloat(up, seq*INTER), `model.layers.${layer}.mlp.up_proj__out`);

  const act = cuAlloc(BigInt(seq*INTER*2));
  launch(kSg, [BLK1(seq*INTER),1,1], [128,1,1], [gate, up, act], `sg.${lp}`);
  cuFree(gate); cuFree(up); cuFree(normed2);
  // Reference: mlp.act_fn__in is the gate_proj output (before silu), and mlp.down_proj__in is after SwiGLU
  // Our "act" is silu(gate) * up — corresponds to the input to down_proj
  check(`L${layer}.swiglu`, bf16BufToFloat(act, seq*INTER), `model.layers.${layer}.mlp.down_proj__in`);

  const mlpOut = cuAlloc(BigInt(seq*H*4));
  launch(kDP, [Math.ceil(seq/64),Math.ceil(H/64),1], [128,1,1], [act, wp(W,layer,"mlp.down_proj.weight"), mlpOut], `dp.${lp}`);
  cuFree(act);
  check(`L${layer}.mlp_out`, f32BufToFloat(mlpOut, seq*H), `model.layers.${layer}.mlp.down_proj__out`);

  // cast + residual
  const mlpBf = cuAlloc(BigInt(seq*H*2));
  launch(kCs, [BLK1(seq*H),1,1], [128,1,1], [mlpOut, mlpBf], `cast2.${lp}`);
  x = cuAlloc(BigInt(seq*H*2));
  launch(kAd, [BLK1(seq*H),1,1], [128,1,1], [afterAttn, mlpBf, x], `add2.${lp}`);
  cuFree(afterAttn); cuFree(mlpOut); cuFree(mlpBf);
  check(`L${layer}.layer_out`, bf16BufToFloat(x, seq*H), `model.layers.${layer}__out`);

  if (layer < 3 || layer >= NL-2 || !full) {
    // already printed above
  }
}

// Final norm + lm_head
const fn = cuAlloc(BigInt(seq*H*2));
launch(kRms, [seq,1,1], [128,1,1], [x, W.get("model.language_model.norm.weight")!, fn], `rmsF`);
check("final_norm", bf16BufToFloat(fn, seq*H), "model.norm__out");

const logits = cuAlloc(BigInt(seq*VOCAB*4));
launch(kLM, [Math.ceil(seq/64),Math.ceil(VOCAB/64),1], [128,1,1], [fn, embedW, logits], `lmHead`);
check("logits", f32BufToFloat(logits, seq*VOCAB), "logits");

// Argmax
const hLog = new Float32Array(VOCAB);
cuDtoH(hLog.buffer, logits+BigInt((seq-1)*VOCAB*4), BigInt(VOCAB*4));
let bestId=0, bestLog=-Infinity;
for (let i=0; i<VOCAB; i++) if (hLog[i]>bestLog) { bestLog=hLog[i]; bestId=i; }

const refLogits = ref.get("logits")!;
let refBestId=0, refBestLog=-Infinity;
for (let i=0; i<VOCAB; i++) if (refLogits[i]>refBestLog) { refBestLog=refLogits[i]; refBestId=i; }

console.log(`\n=== Summary ===`);
const passed = results.filter(r => r.status === "✓ PASS").length;
const failed = results.filter(r => r.status === "✗ FAIL").length;
const noRef = results.filter(r => r.status === "NO REF").length;
console.log(`Passed: ${passed}  Failed: ${failed}  No Ref: ${noRef}  Total: ${results.length}`);
console.log(`\nOur prediction:  token ${bestId} (logit=${bestLog.toFixed(3)})`);
console.log(`Ref prediction:  token ${refBestId} (logit=${refBestLog.toFixed(3)})`);
console.log(`Match: ${bestId === refBestId ? "YES ✓" : "NO ✗"}`);

// Print first failures
const failures = results.filter(r => r.status === "✗ FAIL");
if (failures.length > 0) {
  console.log(`\nFirst failures:`);
  for (const f of failures.slice(0, 10)) {
    console.log(`  ${f.name.padEnd(50)} maxAbs=${f.maxAbs.toExponential(3)}  cosSim=${f.cosSim.toFixed(6)}`);
  }
}
