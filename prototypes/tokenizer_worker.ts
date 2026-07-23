// Worker: postMessage-based, reads from SharedArrayBuffer (zero-copy data).
const CLS = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  if (i === 32 || i === 9 || i === 10 || i === 13 || i === 11 || i === 12) CLS[i] = 0;
  else if (i >= 48 && i <= 57) CLS[i] = 1;
  else if ((i >= 65 && i <= 90) || (i >= 97 && i <= 122)) CLS[i] = 2;
  else if ((i >= 33 && i <= 47) || (i >= 58 && i <= 64) || (i >= 91 && i <= 96) || (i >= 123 && i <= 126)) CLS[i] = 3;
  else CLS[i] = 4;
}
for (let i = 0xC0; i < 0x100; i++) CLS[i] = 2;
const MAX = 0xFFFFFFFF;
let mIds: Uint32Array, mRanks: Uint32Array, bId: Int32Array, stride: number;
const buf = new Int32Array(2048), next = new Uint8Array(2048), prev = new Int8Array(2048), ranks = new Uint32Array(2048), wOut = new Int32Array(65536);

self.onmessage = (e: MessageEvent) => {
  const d = e.data;
  if (d.setup) {
    mIds = new Uint32Array(d.mergeSab); mRanks = new Uint32Array(d.mergeRanksSab);
    bId = new Int32Array(d.byteToIdSab); stride = d.stride;
    (self as any).postMessage({ ready: true });
  } else if (d.encode) {
    const bytes = new Uint8Array(d.sab, d.inputOff, d.inputLen);
    const output = new Int32Array(d.sab, d.outputOff, d.outputLen);
    const n = bytes.length; let wc = 0, st = 0, pc = CLS[bytes[0]];
    for (let i = 1; i < n; i++) { const c = CLS[bytes[i]]; if (c !== pc) { if (pc !== 0) { wOut[wc++] = st; wOut[wc++] = i - st; } st = i; pc = c; } }
    if (pc !== 0) { wOut[wc++] = st; wOut[wc++] = n - st; }
    let ic = 0;
    for (let w = 0; w < wc; w += 2) {
      const ws = wOut[w], wl = wOut[w + 1];
      if (wl <= 1) { if (wl === 1) output[ic++] = bId[bytes[ws]]; continue; }
      const wn = Math.min(wl, 2048);
      for (let i = 0; i < wn; i++) buf[i] = bId[bytes[ws + i]];
      for (let i = 0; i < wn; i++) { next[i] = i + 1; prev[i] = i - 1; }
      for (let i = 0; i < wn - 1; i++) ranks[i] = mRanks[buf[i] * stride + buf[i + 1]];
      ranks[wn - 1] = MAX; let nn = wn;
      while (nn > 1) { let best = MAX, bi = 0; for (let i = 0; i < nn - 1; i++) { if (ranks[i] < best) { best = ranks[i]; bi = i; } } if (best === MAX) break;
      buf[bi] = mIds[buf[bi] * stride + buf[next[bi]]]; const dd = next[bi], nr = next[dd]; next[bi] = nr; ranks[dd] = MAX;
      if (nr < nn) { prev[nr] = bi; ranks[bi] = mRanks[buf[bi] * stride + buf[nr]]; } else ranks[bi] = MAX;
      const lp = prev[bi]; if (lp >= 0) ranks[lp] = mRanks[buf[lp] * stride + buf[bi]]; nn--; }
      let wr = 0, i = 0; while (i < wn) { buf[wr++] = buf[i]; i = next[i]; } for (let j = 0; j < wr; j++) output[ic++] = buf[j];
    }
    (self as any).postMessage({ done: true, count: ic });
  }
};
