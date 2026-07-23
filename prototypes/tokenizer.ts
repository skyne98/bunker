// prototypes/tokenizer.ts — multi-threaded BPE tokenizer (postMessage + SharedArrayBuffer).
// bun run prototypes/tokenizer.ts
import { BPETokenizer } from "./tokenizer_lib";
const S = 400;
const mI = new Uint32Array(400*S), mR = new Uint32Array(400*S).fill(0xFFFFFFFF), bI = new Int32Array(256);
for (let i=0;i<256;i++) bI[i]=i; mR[116*S+104]=1; mI[116*S+104]=256;
const ms=new SharedArrayBuffer(mI.byteLength), rs=new SharedArrayBuffer(mR.byteLength), bs=new SharedArrayBuffer(bI.byteLength);
new Uint32Array(ms).set(mI); new Uint32Array(rs).set(mR); new Int32Array(bs).set(bI);
const st = new BPETokenizer(mI, mR, S, bI);
const NUM = 16;
const workers: Worker[] = [];
for (let i=0;i<NUM;i++) workers.push(new Worker(new URL("./tokenizer_worker.ts", import.meta.url)));
await Promise.all(workers.map((w,i) => new Promise<void>(r => {
  const h = (e:MessageEvent) => { if (e.data.ready) { w.removeEventListener("message",h); r(); } };
  w.addEventListener("message", h);
  w.postMessage({setup:true, mergeSab:ms, mergeRanksSab:rs, byteToIdSab:bs, stride:S});
})));
console.log(`${NUM} workers ready (postMessage + SAB)`);

let _sab: SharedArrayBuffer | null = null;
async function encodeMT(bytes: Uint8Array): Promise<number[]> {
  const textLen = bytes.length;
  const outOff = (textLen + 3) & ~3;
  const chunkMax = Math.ceil((textLen / NUM) * 2 * 4) + 64;
  const sabSize = outOff + NUM * chunkMax;
  if (!_sab || _sab.byteLength < sabSize) _sab = new SharedArrayBuffer(sabSize);
  const sab = _sab;
  new Uint8Array(sab, 0, textLen).set(bytes);
  const cs = Math.ceil(textLen / NUM);
  const promises: Promise<number>[] = [];
  const chunkInfo: {outOff:number}[] = [];
  for (let i = 0; i < NUM; i++) {
    let s=i*cs, e=Math.min((i+1)*cs, textLen);
    if (i > 0) while (s < e && bytes[s] === 32) s++;
    if (i < NUM-1) while (e < textLen && bytes[e] !== 32) e++;
    const len = e - s;
    const outBase = outOff + i * chunkMax;
    chunkInfo.push({outOff: outBase});
    const idx = i;
    promises.push(new Promise<number>(resolve => {
      const h = (e: MessageEvent) => { if (e.data.done) { workers[idx].removeEventListener("message", h); resolve(e.data.count); } };
      workers[idx].addEventListener("message", h);
      workers[idx].postMessage({encode:true, sab, inputOff:s, inputLen:len, outputOff:outBase, outputLen:Math.ceil(len*2)});
    }));
  }
  const counts = await Promise.all(promises);
  const ids: number[] = [];
  for (let i = 0; i < NUM; i++) {
    const c = counts[i];
    if (c > 0) { const arr = new Int32Array(sab, chunkInfo[i].outOff, c); for (let j=0;j<c;j++) ids.push(arr[j]); }
  }
  return ids;
}

const words = ["the","quick","brown","fox","jumps","over","lazy","dog","hello","world","this","is","a","test","of","the","tokenizer","running","at","high","speed","with","many","words","and","some","numbers","like","123","and","456","plus","punctuation!","The","END"];
let text = "";
while (text.length < 40_000_000) text += words[Math.floor(Math.random()*words.length)] + " ";
text = text.substring(0, 40_000_000);
const textBytes = new TextEncoder().encode(text);
console.log(`text: ${(textBytes.length/1e6).toFixed(1)} MB`);

await encodeMT(textBytes.subarray(0, 100000));
st.encode(text.substring(0, 100000));

{ const t0=performance.now(),it=3; for(let i=0;i<it;i++) st.encode(text); const dt=(performance.now()-t0)/1000/it; console.log(`ST: ${(dt*1e3).toFixed(0)}ms ${(text.length/dt/1e9).toFixed(2)}GB/s`); }
{ const t0=performance.now(),it=5; for(let i=0;i<it;i++) await encodeMT(textBytes); const dt=(performance.now()-t0)/1000/it; const toks=(await encodeMT(textBytes)).length; console.log(`MT(${NUM}w,SAB): ${(dt*1e3).toFixed(0)}ms ${(text.length/dt/1e9).toFixed(2)}GB/s ${toks} tokens`); }
for(const w of workers) w.terminate();
