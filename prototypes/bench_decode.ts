// Measure per-component time
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const H=1024, VOCAB=248320, NL=24, FAI=4, INTER=3584, QKVD=6144, ZD=2048, EPS=1e-6;
const NH=8, NKV=2, HD=256, LKH=16, LVH=16, LKD=128, LVD=128, KEYDIM=LKH*LKD, VALDIM=LVH*LVD;
const QGATE=NH*HD*2, KV_DIM=NKV*HD;
const isFull = (l:number) => l % FAI === FAI-1;
const BLK1 = (n:number) => Math.ceil(n/1024);

const stPath = "/tmp/qwen35_0.8b.safetensors";
const data = await Bun.file(stPath).bytes();
const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
const hl = Number(dv.getBigUint64(0, true));
const hdr = JSON.parse(new TextDecoder().decode(data.subarray(8, 8+hl)));
const ds = 8+hl;
const base = cuAlloc(BigInt(data.length-ds));
cuHtoD(base, data.subarray(ds)); cuSync();
const W = new Map<string,bigint>();
for (const [n,i] of Object.entries(hdr)) { if(n==="__metadata__")continue; W.set(n, base+BigInt((i as any).data_offsets[0])); }
const wp = (l:number,n:string) => { const v=W.get(`model.language_model.layers.${l}.${n}`); if(!v) throw new Error(`missing: layers.${l}.${n}`); return v; };

// Build and compile a GEMM
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
const mm=(M:number,N:number,K:number)=>compileAndLoad(buildMM(M,N,K),"mm",4);
const kLM=mm(1,VOCAB,H);
const kQKV=mm(1,QKVD,H);
const kOutProj=mm(1,H,ZD);
const kGP=mm(1,INTER,H), kUP=mm(1,INTER,H), kDP=mm(1,H,INTER);

// Allocate dummy buffers
const x = cuAlloc(BigInt(H*2));
const normed = cuAlloc(BigInt(H*2));
const out = cuAlloc(BigInt(H*4));

// Benchmark individual operations
function bench(label:string, fn:()=>void, iters:number=100) {
  // warmup
  for(let i=0;i<5;i++) fn();
  cuSync();
  const t0=performance.now();
  for(let i=0;i<iters;i++) fn();
  cuSync();
  const dt=(performance.now()-t0)/iters;
  console.log(`  ${label.padEnd(35)} ${dt.toFixed(3)} ms`);
  return dt;
}

console.log("=== Per-operation timing (100 iters) ===");

// GEMMs
bench("GEMM qkv [1,6144,1024]", ()=>cuLaunch(kQKV,[1,Math.ceil(QKVD/64),1],[128,1,1],[normed,wp(0,"linear_attn.in_proj_qkv.weight"),out]));
bench("GEMM out_proj [1,1024,2048]", ()=>cuLaunch(kOutProj,[1,Math.ceil(H/64),1],[128,1,1],[x,wp(0,"linear_attn.out_proj.weight"),out]));
bench("GEMM gate_proj [1,3584,1024]", ()=>cuLaunch(kGP,[1,Math.ceil(INTER/64),1],[128,1,1],[normed,wp(0,"mlp.gate_proj.weight"),out]));
bench("GEMM up_proj [1,3584,1024]", ()=>cuLaunch(kUP,[1,Math.ceil(INTER/64),1],[128,1,1],[normed,wp(0,"mlp.up_proj.weight"),out]));
bench("GEMM down_proj [1,1024,3584]", ()=>cuLaunch(kDP,[1,Math.ceil(H/64),1],[128,1,1],[x,wp(0,"mlp.down_proj.weight"),out]));
bench("GEMM lm_head [1,248320,1024]", ()=>cuLaunch(kLM,[1,Math.ceil(VOCAB/64),1],[128,1,1],[x,W.get("model.language_model.embed_tokens.weight")!,out]));

// cuSync overhead
bench("cuSync only", ()=>cuSync());

// CPU RoPE overhead (simulated)
const ropeBuf = new Uint16Array(NH*HD);
const ropeCos = new Float32Array(32);
const ropeSin = new Float32Array(32);
const ropeD = cuAlloc(BigInt(NH*HD*2));
bench("CPU RoPE (DtoH+compute+HtoD)", ()=>{
  cuDtoH(ropeBuf.buffer, ropeD, BigInt(NH*HD*2));
  // Simulate RoPE
  for(let h=0;h<NH;h++) for(let i=0;i<32;i++) { /* skip actual compute */ }
  cuHtoD(ropeD, ropeBuf.buffer); cuSync();
});
