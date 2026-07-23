// prototypes/gigatoken_bun.ts — gigatoken via Bun FFI (15+ GB/s tokenizer).
//
//   bun run prototypes/gigatoken_bun.ts
import { dlopen, ptr as ffiPtr } from "bun:ffi";

const lib = dlopen("./shim/libgigatoken_wrapper.so", {
  gt_init:   { args: ["ptr"], returns: "ptr" },
  gt_encode: { args: ["ptr", "ptr", "u64", "ptr", "u64"], returns: "i32" },
  gt_free:   { args: ["ptr"], returns: "void" },
}).symbols;

export class GigaTokenizer {
  handle: number;  // raw pointer (Bun FFI returns number for ptr)

  constructor(tokenizerJsonPath: string) {
    const pathBuf = Buffer.from(tokenizerJsonPath + "\0");
    this.handle = lib.gt_init(ffiPtr(pathBuf)) as number;
    if (!this.handle) throw new Error(`gt_init failed for ${tokenizerJsonPath}`);
  }

  encode(text: string): Uint32Array {
    const bytes = new TextEncoder().encode(text);
    const maxTokens = bytes.length + 64;
    const outBuf = new Uint32Array(maxTokens);
    const rc = lib.gt_encode(
      this.handle,         // ptr (number)
      ffiPtr(bytes),       // ptr to text
      BigInt(bytes.length), // u64
      ffiPtr(outBuf),       // ptr to output
      BigInt(maxTokens),    // u64
    );
    if (rc < 0) throw new Error(`gt_encode failed (rc=${rc})`);
    return outBuf.subarray(0, rc);
  }

  free() { if (this.handle) { lib.gt_free(this.handle); this.handle = 0; } }
}

// ── benchmark ────────────────────────────────────────────────────────
const words = ["the","quick","brown","fox","jumps","over","lazy","dog","hello","world","this","is","a","test","of","the","tokenizer","running","at","high","speed","with","many","words","and","some","numbers","like","123","and","456","plus","punctuation!","The","END"];
let text = "";
while (text.length < 10_000_000) text += words[Math.floor(Math.random() * words.length)] + " ";
text = text.substring(0, 10_000_000);
console.log(`text: ${(text.length / 1e6).toFixed(1)} MB`);

const tokenizerPath = process.env.TOKENIZER_PATH || "/tmp/tokenizer.json";
const tok = new GigaTokenizer(tokenizerPath);
console.log("gigatoken initialized!");

const ids = tok.encode("the quick brown fox");
console.log(`encode("the quick brown fox") → ${ids.length} tokens: [${ids.slice(0, 20).join(", ")}...]`);

// warmup
for (let i = 0; i < 3; i++) tok.encode(text);
// bench
const t0 = performance.now(); const it = 10;
for (let i = 0; i < it; i++) tok.encode(text);
const dt = (performance.now() - t0) / 1000 / it;
const tokens = tok.encode(text).length;
console.log(`gigatoken: ${(dt * 1e3).toFixed(1)} ms, ${(text.length / dt / 1e9).toFixed(2)} GB/s, ${tokens} tokens (${(tokens / dt / 1e6).toFixed(1)}M tok/s)`);

tok.free();


