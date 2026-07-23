// prototypes/gigatoken_bun.ts — gigatoken via Bun FFI (multi-threaded).
//
//   TOKENIZER_PATH=/tmp/tokenizer.json bun run prototypes/gigatoken_bun.ts
import { dlopen, ptr as ffiPtr } from "bun:ffi";

const lib = dlopen("./shim/libgigatoken_wrapper.so", {
  gt_init:      { args: ["ptr"], returns: "ptr" },
  gt_encode:    { args: ["ptr", "ptr", "u64", "ptr", "u64"], returns: "i32" },
  gt_encode_mt: { args: ["ptr", "ptr", "u64", "ptr", "u64"], returns: "i32" },
  gt_free:      { args: ["ptr"], returns: "void" },
}).symbols;

export class GigaTokenizer {
  handle: number;
  constructor(path: string) {
    const buf = Buffer.from(path + "\0");
    this.handle = lib.gt_init(ffiPtr(buf)) as number;
    if (!this.handle) throw new Error(`gt_init failed`);
  }
  encode(text: string): Uint32Array {
    return this._enc(text, false);
  }
  encodeMT(text: string): Uint32Array {
    return this._enc(text, true);
  }
  private _enc(text: string, mt: boolean): Uint32Array {
    const bytes = new TextEncoder().encode(text);
    const maxTok = bytes.length + 64;
    const out = new Uint32Array(maxTok);
    const fn = mt ? lib.gt_encode_mt : lib.gt_encode;
    const rc = fn(this.handle, ffiPtr(bytes), BigInt(bytes.length), ffiPtr(out), BigInt(maxTok));
    if (rc < 0) throw new Error(`encode failed (rc=${rc})`);
    return out.subarray(0, rc);
  }
  free() { if (this.handle) { lib.gt_free(this.handle); this.handle = 0; } }
}

// ── benchmark ────────────────────────────────────────────────────────
const words = ["the","quick","brown","fox","jumps","over","lazy","dog","hello","world","this","is","a","test","of","the","tokenizer","running","at","high","speed","with","many","words","and","some","numbers","like","123","and","456","plus","punctuation!","The","END"];
let text = "";
while (text.length < 40_000_000) text += words[Math.floor(Math.random()*words.length)] + " ";
text = text.substring(0, 40_000_000);
console.log(`text: ${(text.length/1e6).toFixed(1)} MB`);

const path = process.env.TOKENIZER_PATH || "/tmp/tokenizer.json";
const tok = new GigaTokenizer(path);
console.log("gigatoken initialized");

// correctness
const ids = tok.encode("the quick brown fox");
console.log(`encode("the quick brown fox") → ${ids.length} tokens: [${ids.join(", ")}]`);

// warmup
for (let i = 0; i < 3; i++) { tok.encode(text.substring(0, 1_000_000)); tok.encodeMT(text.substring(0, 1_000_000)); }

// single-threaded bench
{
  const t0 = performance.now(); const it = 5;
  for (let i = 0; i < it; i++) tok.encode(text);
  const dt = (performance.now() - t0) / 1000 / it;
  const toks = tok.encode(text).length;
  console.log(`ST:  ${(dt*1e3).toFixed(0)} ms, ${(text.length/dt/1e9).toFixed(2)} GB/s, ${toks} tokens (${(toks/dt/1e6).toFixed(1)}M tok/s)`);
}
// multi-threaded bench
{
  const t0 = performance.now(); const it = 5;
  for (let i = 0; i < it; i++) tok.encodeMT(text);
  const dt = (performance.now() - t0) / 1000 / it;
  const toks = tok.encodeMT(text).length;
  console.log(`MT:  ${(dt*1e3).toFixed(0)} ms, ${(text.length/dt/1e9).toFixed(2)} GB/s, ${toks} tokens (${(toks/dt/1e6).toFixed(1)}M tok/s)`);
}
tok.free();
