// prototypes/tokenizer.ts — simdjson-inspired BPE tokenizer (flat arrays, no Map in hot path).
//
// SIMD-inspired: 256-byte classification table, flat Uint32Array for merges
// (array indexing = ~1ns vs Map.get ~2ns). No string allocations.
//
//   bun run prototypes/tokenizer.ts

// ── 256-byte character classification (L1-resident) ──────────────────
const CLS = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  if (i === 32 || i === 9 || i === 10 || i === 13 || i === 11 || i === 12) CLS[i] = 0;
  else if (i >= 48 && i <= 57) CLS[i] = 1;
  else if ((i >= 65 && i <= 90) || (i >= 97 && i <= 122)) CLS[i] = 2;
  else if ((i >= 33 && i <= 47) || (i >= 58 && i <= 64) || (i >= 91 && i <= 96) || (i >= 123 && i <= 126)) CLS[i] = 3;
  else CLS[i] = 4;
}
for (let i = 0xC0; i < 0x100; i++) CLS[i] = 2;

// ── pre-tokenizer: scan bytes, yield [start, len, ...] on class change ─
function preTokenize(bytes: Uint8Array): number[] {
  const out: number[] = [];
  const n = bytes.length;
  let start = 0, prev = CLS[bytes[0]];
  for (let i = 1; i < n; i++) {
    const c = CLS[bytes[i]];
    if (c !== prev) {
      if (prev !== 0) out.push(start, i - start);
      start = i; prev = c;
    }
  }
  if (prev !== 0) out.push(start, n - start);
  return out;
}

// ── BPE tokenizer (flat Uint32Array merges, no Map in hot path) ────────
const NO_RANK = 0xFFFFFFFF;
export class BPETokenizer {
  mergeIds: Uint32Array;     // pair_key → merged_id (0 = none)
  mergeRanks: Uint32Array;   // pair_key → rank (NO_RANK = none)
  stride: number;
  byteToId: Int32Array;
  buf: Int32Array;
  next: Uint8Array; prev: Int8Array; ranks: Uint32Array;

  constructor(mergeIds: Uint32Array, mergeRanks: Uint32Array, stride: number, byteToId: Int32Array) {
    this.mergeIds = mergeIds; this.mergeRanks = mergeRanks;
    this.stride = stride; this.byteToId = byteToId;
    this.buf = new Int32Array(2048);
    this.next = new Uint8Array(2048); this.prev = new Int8Array(2048); this.ranks = new Uint32Array(2048);
  }

  encodeWord(word: Uint8Array, wordLen: number, out: number[]): void {
    if (wordLen <= 1) { if (wordLen === 1) out.push(this.byteToId[word[0]]); return; }
    const buf = this.buf, stride = this.stride, mIds = this.mergeIds, mRanks = this.mergeRanks;
    const n = Math.min(wordLen, buf.length);
    for (let i = 0; i < n; i++) buf[i] = this.byteToId[word[i]];
    let len = n;
    while (len > 1) {
      let bestRank = NO_RANK, bestIdx = -1, bestKey = 0;
      for (let i = 0; i < len - 1; i++) {
        const key = buf[i] * stride + buf[i + 1];
        const rank = mRanks[key];
        if (rank < bestRank) { bestRank = rank; bestIdx = i; bestKey = key; }
      }
      if (bestIdx === -1) break;
      buf[bestIdx] = mIds[bestKey];
      len--;
      for (let i = bestIdx + 1; i < len; i++) buf[i] = buf[i + 1];
    }
    for (let i = 0; i < len; i++) out.push(buf[i]);
  }

  encode(text: string): number[] {
    const bytes = new TextEncoder().encode(text);
    const words = preTokenize(bytes);
    const ids: number[] = [];
    for (let w = 0; w < words.length; w += 2)
      this.encodeWord(bytes.subarray(words[w], words[w] + words[w + 1]), words[w + 1], ids);
    return ids;
  }
}

// ── synthetic test vocab ──────────────────────────────────────────────
function makeTestVocab() {
  const STRIDE = 400;
  const maxId = 400;
  const mergeIds = new Uint32Array(maxId * STRIDE);
  const mergeRanks = new Uint32Array(maxId * STRIDE).fill(NO_RANK);
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
      const merged = nextId++;
      mergeIds[key] = merged;
      mergeRanks[key] = rank++;
      ids = [merged, ...ids.slice(2)];
    }
  }
  return { mergeIds, mergeRanks, STRIDE, byteToId };
}

// ── benchmark ──────────────────────────────────────────────────────────
const { mergeIds, mergeRanks, STRIDE, byteToId } = makeTestVocab();
const tokenizer = new BPETokenizer(mergeIds, mergeRanks, STRIDE, byteToId);
console.log(`tokenizer loaded (merges=${mergeRanks.filter(x=>x!==NO_RANK).length}, STRIDE=${STRIDE})`);

const testIds = tokenizer.encode("the quick brown fox");
console.log(`encode("the quick brown fox") → [${testIds}] (${testIds.length} tokens)`);

// generate 2MB text
const words = ["the","quick","brown","fox","jumps","over","lazy","dog","hello","world","this","is","a","test","of","the","tokenizer","running","at","high","speed","with","many","words","and","some","numbers","like","123","and","456","plus","punctuation!","The","END"];
let text = "";
while (text.length < 2_000_000) text += words[Math.floor(Math.random() * words.length)] + " ";
text = text.substring(0, 2_000_000);
const bytes = new TextEncoder().encode(text);

// bench pre-tokenize
{ for (let i=0;i<5;i++) preTokenize(bytes); const t0=performance.now(),it=100;
  for(let i=0;i<it;i++) preTokenize(bytes);
  const dt=(performance.now()-t0)/1000/it;
  console.log(`pre-tokenize: ${(dt*1e3).toFixed(2)} ms, ${(bytes.length/dt/1e9).toFixed(1)} GB/s`); }
// bench full encode
{ for(let i=0;i<3;i++) tokenizer.encode(text); const t0=performance.now(),it=20;
  for(let i=0;i<it;i++) tokenizer.encode(text);
  const dt=(performance.now()-t0)/1000/it;
  const toks=tokenizer.encode(text).length;
  console.log(`full encode: ${(dt*1e3).toFixed(1)} ms, ${(text.length/dt/1e9).toFixed(2)} GB/s, ${toks} tokens (${(toks/dt/1e6).toFixed(1)}M tok/s)`); }

// realistic text (short words, avg 5 chars)
{
  const shortWords = ["the","a","to","of","in","is","it","on","he","be","at","as","by","an","or","so","if","no","we","me","my","up","do","go","us","am","an","do","go","hi","ok","run","set","get","put","let","say","see","try","use","way","yes","new","old","big","low","end","top","cut","add","fit","fix","own","red","sum","ten","two","one","six","bad","mad","sad","fun","sun","run","gun","can","man","pan","fan","van","ran","tan","ban","win","sin","tin","pin","bin","kin","din","fin","gin","hip","nip","rip","sip","tip","zip","dip","kip","lip","log","dog","fog","hog","jog","cog","bog","cog","hog","jog","log","fog","dog"];
  let st = "";
  while (st.length < 2_000_000) st += shortWords[Math.floor(Math.random() * shortWords.length)] + " ";
  st = st.substring(0, 2_000_000);
  for (let i = 0; i < 3; i++) tokenizer.encode(st);
  const t0 = performance.now(), it = 20;
  for (let i = 0; i < it; i++) tokenizer.encode(st);
  const dt = (performance.now() - t0) / 1000 / it;
  const toks = tokenizer.encode(st).length;
  console.log(`realistic (short words): ${(dt*1e3).toFixed(1)} ms, ${(st.length/dt/1e9).toFixed(2)} GB/s, ${toks} tokens (${(toks/dt/1e6).toFixed(1)}M tok/s)`);
}
