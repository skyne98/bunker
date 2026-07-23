// prototypes/tokenizer_worker.ts — Bun Worker for parallel BPE tokenization.
// Receives merge tables (setup) + text chunks (work), returns token IDs.
//
// Protocol:
//   { setup: true, mergeIds, mergeRanks, byteToId, stride }  → init
//   { work: true, bytes: Uint8Array }                          → encode, return { ids: Int32Array }

// ── 256-byte character classification ────────────────────────────────
const CLS = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  if (i === 32 || i === 9 || i === 10 || i === 13 || i === 11 || i === 12) CLS[i] = 0;
  else if (i >= 48 && i <= 57) CLS[i] = 1;
  else if ((i >= 65 && i <= 90) || (i >= 97 && i <= 122)) CLS[i] = 2;
  else if ((i >= 33 && i <= 47) || (i >= 58 && i <= 64) || (i >= 91 && i <= 96) || (i >= 123 && i <= 126)) CLS[i] = 3;
  else CLS[i] = 4;
}
for (let i = 0xC0; i < 0x100; i++) CLS[i] = 2;

// ── pre-tokenizer ────────────────────────────────────────────────────
function preTokenize(bytes: Uint8Array, out: Int32Array): number {
  const n = bytes.length;
  let start = 0, prev = CLS[bytes[0]], count = 0;
  for (let i = 1; i < n; i++) {
    const c = CLS[bytes[i]];
    if (c !== prev) {
      if (prev !== 0) { out[count++] = start; out[count++] = i - start; }
      start = i; prev = c;
    }
  }
  if (prev !== 0) { out[count++] = start; out[count++] = n - start; }
  return count;
}

// ── BPE tokenizer (gigatoken-style linked-list merge) ────────────────
const MAX = 0xFFFFFFFF;
let mergeIds: Uint32Array, mergeRanks: Uint32Array, byteToId: Int32Array, stride: number;
const buf = new Int32Array(2048);
const next = new Uint8Array(2048), prev = new Int8Array(2048), ranks = new Uint32Array(2048);
const wordsOut = new Int32Array(65536);
const idsOut = new Int32Array(1048576); // 1M tokens max per chunk

function encodeChunk(bytes: Uint8Array): Int32Array {
  const wordCount = preTokenize(bytes, wordsOut);
  let idCount = 0;
  for (let w = 0; w < wordCount; w += 2) {
    const start = wordsOut[w], len = wordsOut[w + 1];
    if (len <= 1) { if (len === 1) idsOut[idCount++] = byteToId[bytes[start]]; continue; }
    const n = Math.min(len, 2048);
    for (let i = 0; i < n; i++) buf[i] = byteToId[bytes[start + i]];
    for (let i = 0; i < n; i++) { next[i] = i + 1; prev[i] = i - 1; }
    for (let i = 0; i < n - 1; i++) ranks[i] = mergeRanks[buf[i] * stride + buf[i + 1]];
    ranks[n - 1] = MAX;
    let nn = n;
    while (nn > 1) {
      let best = MAX, bestI = 0;
      for (let i = 0; i < nn - 1; i++) { if (ranks[i] < best) { best = ranks[i]; bestI = i; } }
      if (best === MAX) break;
      buf[bestI] = mergeIds[buf[bestI] * stride + buf[next[bestI]]];
      const dead = next[bestI];
      const newRight = next[dead];
      next[bestI] = newRight;
      ranks[dead] = MAX;
      if (newRight < nn) { prev[newRight] = bestI; ranks[bestI] = mergeRanks[buf[bestI] * stride + buf[newRight]]; } else ranks[bestI] = MAX;
      const left = prev[bestI];
      if (left >= 0) ranks[left] = mergeRanks[buf[left] * stride + buf[bestI]];
      nn--;
    }
    let write = 0, i = 0;
    while (i < n) { buf[write++] = buf[i]; i = next[i]; }
    for (let j = 0; j < write; j++) idsOut[idCount++] = buf[j];
  }
  return idsOut.subarray(0, idCount);
}

// ── message handler ──────────────────────────────────────────────────
self.onmessage = (e: MessageEvent) => {
  const d = e.data;
  if (d.setup) {
    mergeIds = d.mergeIds; mergeRanks = d.mergeRanks; byteToId = d.byteToId; stride = d.stride;
    (self as any).postMessage({ ready: true });
    return;
  }
  if (d.work) {
    const bytes: Uint8Array = d.bytes;
    const ids = encodeChunk(bytes);
    // transfer the result back (zero-copy)
    (self as any).postMessage({ ids: new Int32Array(ids) }, [ids.buffer]);
    return;
  }
};
