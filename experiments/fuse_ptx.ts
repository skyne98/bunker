// fuse_ptx.ts — Post-process TTIR kernel PTX: replace A tile loads with Q4_K dequant
// Strategy: take the working TTIR INT8 PTX, find A tile global→shared loads,
// and replace them with Q4_K block loads + nibble extraction.
// The shared memory layout, PRMT packing, and mma.sync stay unchanged.

import { readFileSync, writeFileSync } from "fs";
import { int8MatmulTTIR, compileTTIR } from "../src/kernel.ts";

// Compile the TTIR INT8 kernel
const cfg = { BM: 32, BN: 32, BK: 1024, numWarps: 4 };
const { ptx, shmem } = compileTTIR(int8MatmulTTIR(cfg), 4);
writeFileSync("/tmp/base_tc.ptx", ptx);
console.log(`Base PTX: ${(ptx.match(/mma\.sync/g)||[]).length} mma.sync, ${ptx.length} bytes`);

// The TTIR kernel loads A tiles from global memory and stores to shared memory.
// We need to replace the A tile global→shared path with Q4_K dequant.
// 
// In the PTX, A tile loading follows this pattern:
// 1. Compute global address: A_base + row*1024 + col + thread_offset
// 2. ld.global.b8 (per byte) 
// 3. st.shared (store to shared memory for warp access)
//
// We need to instead:
// 1. Compute Q4_K block address: Q_base + block_idx * 20
// 2. Load 16 bytes from Q4_K block (2×ld.global.u64)
// 3. Dequant: extract nibbles, subtract 8 → INT8
// 4. Store INT8 to shared memory
//
// The B tile loading stays unchanged.
//
// The challenge: identifying which global→shared loads are for A vs B.
// In the TTIR kernel:
// - A tile shared base is at %rd1093 (from global_smem base)
// - B tile shared base is at %rd1095 (from global_smem + offset)
// - A tile global loads go to addresses computed from %rd521 (A param_0)
// - B tile global loads go to addresses computed from %rd522 (B param_1)
//
// We can identify A loads by tracing back to matmul_param_0.

console.log("\nAnalyzing PTX structure...");
console.log("Params:", (ptx.match(/matmul_param_\d/g) || []).join(", "));

// Count shared memory stores
const shStores = (ptx.match(/st\.shared/g) || []).length;
console.log("st.shared count:", shStores);

// Count global loads  
const glStores = (ptx.match(/ld\.global/g) || []).length;
console.log("ld.global count:", glStores);

// The key insight: rather than PTX surgery, we can build a new kernel
// that uses the SAME shared memory layout and fragment packing as TTIR,
// but loads Q4_K data instead of INT8 for the A tile.
// 
// For now, this serves as the foundation for the fused kernel.
// The full implementation requires:
// 1. Understanding the TTIR shared memory layout for A and B tiles
// 2. Replacing A tile global→shared loads with Q4_K dequant→shared stores
// 3. Keeping B tile loading, PRMT packing, and mma.sync unchanged
// 
// This is best done as a LLVM IR generator (already started in fused_kernel.ts)
// that produces a correct kernel using either:
//   a) The exact fragment layout from the PTX ISA spec
//   b) The ldmatrix instruction (simpler but requires shared memory)

console.log("\n=== Fused kernel approach ===");
console.log("The TTIR PTX uses ~105 shared memory operations for A/B tile management.");
console.log("Replacing A tile global→shared with Q4_K dequant would create the fused kernel.");
console.log("This requires understanding the shared memory layout (8KB A-tile + 8KB B-tile).");
console.log("\nAlternative: use ldmatrix instruction to load from shared memory");
console.log("with automatic mma.sync fragment packing. Requires shared memory-only dequant.");

// For a practical approach: the dequant writes to a buffer, then the TC matmul
// reads from that buffer. The intermediate copy is in global memory, not shared.
// To fuse: dequant writes to shared memory directly, then ldmatrix + mma.sync.
// This eliminates the global INT8 buffer entirely and runs in ONE kernel.
