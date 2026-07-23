// prototypes/safetensors_loader.ts — safetensors loader for Qwen3.5 weights.
//
// Format: [u64 LE header_len][JSON header][raw tensor data]
// Parses the header, then uploads each tensor to GPU via the pinned-async
// pipeline from gpu_upload.ts. Returns a Map<name, {devPtr, shape, dtype}>.
//
//   bun run prototypes/safetensors_loader.ts
import { cuAlloc, cuHtoD, cuFree, cuSync } from "../src/ttir";
import { readFileSync, openSync, readSync, fstatSync, closeSync } from "fs";

export type DType = "BF16" | "F16" | "F32" | "I64" | "I32" | "I8" | "U8" | "BOOL" | "F64";

export interface Tensor {
  name: string;
  devPtr: bigint;
  shape: number[];
  dtype: DType;
  nbytes: number;
}

export interface SafeTensorsFile {
  tensors: Map<string, Tensor>;
  free: () => void;
}

const DTYPE_SIZE: Record<DType, number> = {
  BF16: 2, F16: 2, F32: 4, F64: 8, I64: 8, I32: 4, I8: 1, U8: 1, BOOL: 1,
};

// ── parse the safetensors header (8-byte length + JSON) ───────────────
export function parseSafeTensorsHeader(data: Buffer) {
  const headerLen = Number(data.readBigUInt64LE(0));
  const headerJson = data.subarray(8, 8 + headerLen).toString("utf8");
  const header = JSON.parse(headerJson);
  const dataStart = 8 + headerLen;
  const tensors: { name: string; dtype: DType; shape: number[]; offset: number; nbytes: number }[] = [];
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const { dtype, shape, data_offsets } = info as any;
    const [start, end] = data_offsets;
    tensors.push({ name, dtype, shape, offset: dataStart + start, nbytes: end - start });
  }
  return { tensors, dataStart, headerLen };
}

// ── load: read file, parse header, upload each tensor to GPU ──────────
export function loadSafeTensors(path: string): SafeTensorsFile {
  const fd = openSync(path, "r");
  const stat = fstatSync(fd);
  const fileSize = stat.size;
  // read header (first 8 bytes + JSON)
  const headerBuf = Buffer.alloc(8);
  readSync(fd, headerBuf, 0, 8, 0);
  const headerLen = Number(headerBuf.readBigUInt64LE(0));
  const jsonBuf = Buffer.alloc(headerLen);
  readSync(fd, jsonBuf, 0, headerLen, 8);
  const header = JSON.parse(jsonBuf.toString("utf8"));
  const dataStart = 8 + headerLen;

  const tensors = new Map<string, Tensor>();
  const allPtrs: bigint[] = [];

  console.log(`safetensors: ${path} (${(fileSize / 1e9).toFixed(2)} GB, ${Object.keys(header).length - 1} tensors)`);

  // Group tensors by offset to read sequentially (minimizes disk seeks)
  const entries: { name: string; dtype: DType; shape: number[]; offset: number; nbytes: number }[] = [];
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const { dtype, shape, data_offsets } = info as any;
    entries.push({ name, dtype, shape, offset: dataStart + data_offsets[0], nbytes: data_offsets[1] - data_offsets[0] });
  }
  entries.sort((a, b) => a.offset - b.offset);

  // Read + upload each tensor
  const t0 = performance.now();
  let totalBytes = 0;
  for (const entry of entries) {
    const buf = Buffer.alloc(entry.nbytes);
    readSync(fd, buf, 0, entry.nbytes, entry.offset);
    const devPtr = cuAlloc(BigInt(entry.nbytes));
    cuHtoD(devPtr, buf);
    allPtrs.push(devPtr);
    tensors.set(entry.name, { name: entry.name, devPtr, shape: entry.shape, dtype: entry.dtype, nbytes: entry.nbytes });
    totalBytes += entry.nbytes;
  }
  cuSync();
  closeSync(fd);
  const dt = (performance.now() - t0) / 1000;
  console.log(`loaded ${tensors.size} tensors, ${(totalBytes / 1e9).toFixed(2)} GB in ${dt.toFixed(2)}s (${(totalBytes / dt / 1e9).toFixed(1)} GB/s)`);

  return {
    tensors,
    free: () => { for (const p of allPtrs) cuFree(p); },
  };
}

// ── benchmark ─────────────────────────────────────────────────────────
const path = process.env.SAFETENSORS_PATH || "/tmp/qwen35_0.8b.safetensors";
const sf = loadSafeTensors(path);

// Print tensor summary
const byDtype = new Map<string, { count: number; bytes: number }>();
for (const t of sf.tensors.values()) {
  const e = byDtype.get(t.dtype) || { count: 0, bytes: 0 };
  e.count++; e.bytes += t.nbytes;
  byDtype.set(t.dtype, e);
}
console.log("\ntensor summary by dtype:");
for (const [dtype, info] of byDtype) {
  console.log(`  ${dtype.padEnd(5)}: ${String(info.count).padStart(4)} tensors, ${(info.bytes / 1e6).toFixed(0).padStart(5)} MB`);
}

// Print some key tensor shapes
const keys = ["model.language_model.embed_tokens.weight", "model.language_model.layers.0.self_attn.q_proj.weight"];
for (const k of keys) {
  const t = sf.tensors.get(k);
  if (t) console.log(`  ${k}: shape=[${t.shape}] dtype=${t.dtype} (${(t.nbytes / 1e6).toFixed(1)} MB)`);
  else {
    // try to find similar
    const similar = [...sf.tensors.keys()].filter(n => n.includes("embed_tokens") || n.includes("layers.0")).slice(0, 5);
    if (similar.length > 0) console.log(`  (not found: ${k}) similar: ${similar.join(", ")}`);
  }
}

// Print first 10 tensor names
console.log("\nfirst 10 tensors:");
let i = 0;
for (const [name, t] of sf.tensors) {
  if (i++ >= 10) break;
  console.log(`  ${name}: shape=[${t.shape}] dtype=${t.dtype}`);
}

sf.free();

// Print all layer-0 tensor names + shapes (the blueprint for block assembly)
console.log("\nlayer 0 tensors:");
for (const [name, t] of sf.tensors) {
  if (name.includes("layers.0.")) console.log(`  ${name.replace("model.language_model.layers.0.", "")}: [${t.shape}] ${t.dtype}`);
}
// Also print non-layer tensors
console.log("\nnon-layer tensors:");
for (const [name, t] of sf.tensors) {
  if (!name.includes("layers.")) console.log(`  ${name}: [${t.shape}] ${t.dtype}`);
}
