// prototypes/tokenizer.ts — multi-threaded BPE tokenizer (persistent Bun Workers).
//
// bun run prototypes/tokenizer.ts
import { BPETokenizer } from "./tokenizer_lib";

const STRIDE = 400;
function makeTestVocab() {
  const mergeIds = new Uint32Array(400 * STRIDE);
  const mergeRanks = new Uint32Array(400 * STRIDE).fill(0xFFFFFFFF);
  const byteToId = new Int32Array(256);
  for (let i = 0; i < 256; i++) byteToId[i] = i;
  const pairs: [string, string][] = [
    ["t","h"],["h","e"],["i","n"],["e","r"],["a","n"],["r","e"],["o","n"],["a","t"],["e","n"],["n","d"],
    ["t","i"],["e","s"],["o","r"],["t","e"],["o","f"],["e","d"],["i","s"],["i","t"],["a","l"],["a","r"],
    ["s","t"],["t","o"],["i","n","g"],["a","n","d"],["t","h","e"],["f","o","r"],["a","r","e"],["b","u","t"],
    ["n","o","t"],["y","o","u"],["a","l","l"],["c","a","n"],["h","e","r"],["w","a","s"],["o","n","e"],
    ["o","u","r"],["o","u","t"],["d","a","y"],["g","e","t"],["h","a","s"],["h","i","m"],["h","i","s"],
    ["h","o","w"],["i","t","s"],["m","a","y"],["n","e","w"],["n","o","w"],["o","l","d"],["s","e","e"],
    ["w","a","y"],["w","h","o"],["d","i","d"],["g","o","t"],["l","e","t"],["s","a","y"],["s","h","e"],
    ["t","o","o"],["u","s","e"],["e","r","e"],["i","o","n"],["e","n","t"],["t","e","r"],["e","s","t"],
    ["a","t","i"],["h","a","t"],["c","o","n"],["t","e","d"],["d","e","r"],["p","e","r"],
    ["t","h","r"],["t","r","a"],["n","c","e"],["m","e","n"],["i","v","e"],["p","r","e"],["o","v","e"],
  ];
  let nextId = 256, rank = 1;
  for (const pair of pairs) {
    let ids: number[] = [];
    for (const ch of pair) ids.push(ch.charCodeAt(0));
    for (let step = 0; step < pair.length - 1; step++) {
      const key = ids[0] * STRIDE + ids[1];
      mergeIds[key] = nextId; mergeRanks[key] = rank++; ids = [nextId++, ...ids.slice(2)];
    }
  }
  return { mergeIds, mergeRanks, byteToId };
}

const { mergeIds, mergeRanks, byteToId } = makeTestVocab();
const stTokenizer = new BPETokenizer(mergeIds, mergeRanks, STRIDE, byteToId);

// ── persistent worker pool ────────────────────────────────────────────
const NUM_WORKERS = 32;
const workers: Worker[] = [];
const workerBusy: boolean[] = new Array(NUM_WORKERS).fill(false);

async function initWorkers() {
  const readyPromises: Promise<void>[] = [];
  for (let i = 0; i < NUM_WORKERS; i++) {
    const w = new Worker(new URL("./tokenizer_worker.ts", import.meta.url));
    workers.push(w);
    readyPromises.push(new Promise<void>((resolve) => {
      const h = (e: MessageEvent) => { if (e.data.ready) { w.removeEventListener("message", h); resolve(); } };
      w.addEventListener("message", h);
      // copy merge tables (each worker gets its own copy)
      w.postMessage({
        setup: true,
        mergeIds: new Uint32Array(mergeIds),
        mergeRanks: new Uint32Array(mergeRanks),
        byteToId: new Int32Array(byteToId),
        stride: STRIDE,
      });
    }));
  }
  await Promise.all(readyPromises);
}

function splitChunks(bytes: Uint8Array, n: number): Uint8Array[] {
  const cs = Math.ceil(bytes.length / n);
  const chunks: Uint8Array[] = [];
  for (let i = 0; i < n; i++) {
    let start = i * cs, end = Math.min((i + 1) * cs, bytes.length);
    if (i > 0) while (start < end && bytes[start] === 32) start++;
    if (i < n - 1) while (end < bytes.length && bytes[end] !== 32) end++;
    chunks.push(bytes.subarray(start, end));
  }
  return chunks;
}

async function encodeMT(bytes: Uint8Array): Promise<number[]> {
  const chunks = splitChunks(bytes, NUM_WORKERS);
  const results: Int32Array[] = new Array(NUM_WORKERS);
  const promises: Promise<void>[] = [];
  for (let i = 0; i < NUM_WORKERS; i++) {
    const idx = i;
    promises.push(new Promise<void>((resolve) => {
      const h = (e: MessageEvent) => {
        if (e.data.ids) { workers[idx].removeEventListener("message", h); results[idx] = e.data.ids; resolve(); }
      };
      workers[idx].addEventListener("message", h);
      const buf = new Uint8Array(chunks[idx].length);
      buf.set(chunks[idx]);
      workers[idx].postMessage({ work: true, bytes: buf }, [buf.buffer]);
    }));
  }
  await Promise.all(promises);
  const allIds: number[] = [];
  for (const r of results) for (let i = 0; i < r.length; i++) allIds.push(r[i]);
  return allIds;
}

// ── generate text ────────────────────────────────────────────────────
const words = ["the","quick","brown","fox","jumps","over","lazy","dog","hello","world","this","is","a","test","of","the","tokenizer","running","at","high","speed","with","many","words","and","some","numbers","like","123","and","456","plus","punctuation!","The","END"];
let text = "";
while (text.length < 40_000_000) text += words[Math.floor(Math.random() * words.length)] + " ";
text = text.substring(0, 40_000_000);
const textBytes = new TextEncoder().encode(text);
console.log(`text: ${(textBytes.length / 1e6).toFixed(1)} MB, ${NUM_WORKERS} workers`);

// ── init workers ─────────────────────────────────────────────────────
console.log("initializing workers...");
await initWorkers();
console.log("workers ready");

// correctness
const stIds = stTokenizer.encode("the quick brown fox");
console.log(`ST: ${stIds.length} tokens`);
const mtIds = await encodeMT(new TextEncoder().encode("the quick brown fox"));
console.log(`MT: ${mtIds.length} tokens`);

// warmup
stTokenizer.encode(text.substring(0, 100000));
await encodeMT(textBytes.subarray(0, 100000));

// ST bench
{
  const t0 = performance.now(); const it = 3;
  for (let i = 0; i < it; i++) stTokenizer.encode(text);
  const dt = (performance.now() - t0) / 1000 / it;
  console.log(`single-threaded: ${(dt*1e3).toFixed(0)} ms, ${(text.length/dt/1e9).toFixed(2)} GB/s`);
}
// MT bench
{
  const t0 = performance.now(); const it = 3;
  for (let i = 0; i < it; i++) await encodeMT(textBytes);
  const dt = (performance.now() - t0) / 1000 / it;
  const toks = (await encodeMT(textBytes)).length;
  console.log(`multi-threaded (${NUM_WORKERS} workers): ${(dt*1e3).toFixed(0)} ms, ${(text.length/dt/1e9).toFixed(2)} GB/s, ${toks} tokens`);
}

// terminate
for (const w of workers) w.terminate();
