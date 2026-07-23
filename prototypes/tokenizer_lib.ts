// prototypes/tokenizer_lib.ts — shared BPE tokenizer library (single-threaded).
// Used by tokenizer.ts for ST comparison. The Worker has its own inline copy.

const CLS = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  if (i === 32 || i === 9 || i === 10 || i === 13 || i === 11 || i === 12) CLS[i] = 0;
  else if (i >= 48 && i <= 57) CLS[i] = 1;
  else if ((i >= 65 && i <= 90) || (i >= 97 && i <= 122)) CLS[i] = 2;
  else if ((i >= 33 && i <= 47) || (i >= 58 && i <= 64) || (i >= 91 && i <= 96) || (i >= 123 && i <= 126)) CLS[i] = 3;
  else CLS[i] = 4;
}
for (let i = 0xC0; i < 0x100; i++) CLS[i] = 2;

function preTokenize(bytes: Uint8Array): number[] {
  const out: number[] = [];
  const n = bytes.length;
  let start = 0, prev = CLS[bytes[0]];
  for (let i = 1; i < n; i++) {
    const c = CLS[bytes[i]];
    if (c !== prev) { if (prev !== 0) out.push(start, i - start); start = i; prev = c; }
  }
  if (prev !== 0) out.push(start, n - start);
  return out;
}

const MAX = 0xFFFFFFFF;
export class BPETokenizer {
  mergeIds: Uint32Array; mergeRanks: Uint32Array; stride: number; byteToId: Int32Array;
  buf = new Int32Array(2048);
  next = new Uint8Array(2048); prev = new Int8Array(2048); ranks = new Uint32Array(2048);
  constructor(mergeIds: Uint32Array, mergeRanks: Uint32Array, stride: number, byteToId: Int32Array) {
    this.mergeIds = mergeIds; this.mergeRanks = mergeRanks; this.stride = stride; this.byteToId = byteToId;
  }
  encode(text: string): number[] {
    const bytes = new TextEncoder().encode(text);
    const ws = preTokenize(bytes);
    const ids: number[] = [];
    for (let w = 0; w < ws.length; w += 2) {
      const start = ws[w], len = ws[w + 1];
      if (len <= 1) { if (len === 1) ids.push(this.byteToId[bytes[start]]); continue; }
      const n = Math.min(len, 2048);
      for (let i = 0; i < n; i++) this.buf[i] = this.byteToId[bytes[start + i]];
      for (let i = 0; i < n; i++) { this.next[i] = i + 1; this.prev[i] = i - 1; }
      for (let i = 0; i < n - 1; i++) this.ranks[i] = this.mergeRanks[this.buf[i] * this.stride + this.buf[i + 1]];
      this.ranks[n - 1] = MAX;
      let nn = n;
      while (nn > 1) {
        let best = MAX, bestI = 0;
        for (let i = 0; i < nn - 1; i++) { if (this.ranks[i] < best) { best = this.ranks[i]; bestI = i; } }
        if (best === MAX) break;
        this.buf[bestI] = this.mergeIds[this.buf[bestI] * this.stride + this.buf[this.next[bestI]]];
        const dead = this.next[bestI]; const newRight = this.next[dead];
        this.next[bestI] = newRight; this.ranks[dead] = MAX;
        if (newRight < nn) { this.prev[newRight] = bestI; this.ranks[bestI] = this.mergeRanks[this.buf[bestI] * this.stride + this.buf[newRight]]; } else this.ranks[bestI] = MAX;
        const left = this.prev[bestI];
        if (left >= 0) this.ranks[left] = this.mergeRanks[this.buf[left] * this.stride + this.buf[bestI]];
        nn--;
      }
      let write = 0, i = 0;
      while (i < n) { this.buf[write++] = this.buf[i]; i = this.next[i]; }
      for (let j = 0; j < write; j++) ids.push(this.buf[j]);
    }
    return ids;
  }
}
