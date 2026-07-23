// prototypes/safetensors_loader.ts — safetensors loader at SSD max speed.
// Parallel 8-chunk disk read + single GPU buffer + 8 batched uploads.
//   bun run prototypes/safetensors_loader.ts
import { cuAlloc, cuHtoD, cuFree, cuSync } from "../src/ttir";
import { execSync } from "child_process";

export type DType = "BF16" | "F16" | "F32" | "I64" | "I32" | "I8" | "U8" | "BOOL" | "F64";
export interface Tensor { name: string; devPtr: bigint; shape: number[]; dtype: DType; nbytes: number; }
export interface SafeTensorsFile { tensors: Map<string, Tensor>; free: () => void; }

const N_CHUNKS = 8;

export async function loadSafeTensors(path: string): Promise<SafeTensorsFile> {
  const t0 = performance.now();
  const file = Bun.file(path);
  const fileSize = file.size;
  const chunkSize = Math.ceil(fileSize / N_CHUNKS);

  // Step 1: parallel chunked read (8 slices → 6.7 GB/s SSD)
  const readStart = performance.now();
  const chunks = await Promise.all(
    Array.from({ length: N_CHUNKS }, (_, i) => {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, fileSize);
      return file.slice(start, end).bytes();
    })
  );
  const readMs = performance.now() - readStart;

  // Step 2: parse header from chunk 0 (first 8 bytes + JSON)
  const c0 = chunks[0];
  const dv = new DataView(c0.buffer, c0.byteOffset, c0.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(c0.subarray(8, 8 + headerLen)));
  const dataStart = 8 + headerLen;
  const tensorBytes = fileSize - dataStart;

  // Step 3: allocate one GPU buffer for all tensor data
  const uploadStart = performance.now();
  const baseDev = cuAlloc(BigInt(tensorBytes));

  // Step 4: upload each chunk to the right GPU offset (8 cuHtoD calls)
  for (let i = 0; i < N_CHUNKS; i++) {
    const chunkFileOff = i * chunkSize;       // offset in file
    const chunkDevOff = chunkFileOff - dataStart;  // offset in tensor data region
    if (chunkDevOff < 0) {
      // chunk 0 may start before dataStart (header is in it)
      const skip = dataStart - chunkFileOff;
      const uploadBytes = chunks[i].length - skip;
      if (uploadBytes > 0) {
        cuHtoD(baseDev, chunks[i].subarray(skip));
      }
    } else {
      cuHtoD(baseDev + BigInt(chunkDevOff), chunks[i]);
    }
  }
  cuSync();
  const uploadMs = performance.now() - uploadStart;

  // Step 5: create tensor views (devPtr = baseDev + offset, zero per-tensor copy)
  const tensors = new Map<string, Tensor>();
  let totalBytes = 0;
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const { dtype, shape, data_offsets } = info as any;
    const off = data_offsets[0];
    const nbytes = data_offsets[1] - off;
    tensors.set(name, { name, devPtr: baseDev + BigInt(off), shape, dtype, nbytes });
    totalBytes += nbytes;
  }

  const dt = (performance.now() - t0) / 1000;
  console.log(`safetensors: ${path} (${(fileSize/1e9).toFixed(2)} GB, ${tensors.size} tensors)`);
  console.log(`  disk read (${N_CHUNKS} parallel): ${(readMs/1000).toFixed(3)}s  ${(fileSize/readMs*1000/1e9).toFixed(1)} GB/s`);
  console.log(`  GPU upload (${N_CHUNKS} copies):   ${(uploadMs/1000).toFixed(3)}s  ${(tensorBytes/uploadMs*1000/1e9).toFixed(1)} GB/s`);
  console.log(`  total:                          ${dt.toFixed(3)}s  ${(totalBytes/dt/1e9).toFixed(1)} GB/s effective`);

  return { tensors, free: () => cuFree(baseDev) };
}

const path = process.env.SAFETENSORS_PATH || "/tmp/qwen35_0.8b.safetensors";
try { execSync("sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches'"); } catch {}
console.log("=== cold ===");
const sf1 = await loadSafeTensors(path); sf1.free();
console.log("\n=== warm ===");
const sf = await loadSafeTensors(path);
console.log(`\n${sf.tensors.size} tensors on GPU`);
sf.free();
