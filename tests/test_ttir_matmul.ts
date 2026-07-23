// test_ttir_matmul.ts — tiled-pointer FP16 matmul via the fluent TTIRBuilder.
// Verifies makeTensorPtr + load + dot + store compile through the shim and run.
import { dlopen, ptr as ffiPtr, CString } from "bun:ffi";
import { TTIRBuilder, compileTTIR } from "../src/ttir";

const M = 512, N = 512, K = 512;
const BM = 64, BN = 64, BK = 64;

const b = new TTIRBuilder();
const A = b.param("A", { ptr: "f16" });
const Bp = b.param("B", { ptr: "f16" });
const C = b.param("C", { ptr: "f32" });

const pidM = b.programId(0);
const pidN = b.programId(1);
const cM = b.i32(BM), cN = b.i32(BN), cK = b.i32(BK);
const offM = b.mul(pidM, cM);
const offN = b.mul(pidN, cN);

// Tiled pointers to A[pidM*BM : , :], B[:, pidN*BN], C[pidM*BM, pidN*BN]
const tpA = b.makeTensorPtr(A, [M, K], [K, 1], [offM, b.i32(0)], [BM, BK], "f16", [1, 0]);
const tpB = b.makeTensorPtr(Bp, [K, N], [N, 1], [b.i32(0), offN], [BK, BN], "f16", [1, 0]);
const tpC = b.makeTensorPtr(C, [M, N], [N, 1], [offM, offN], [BM, BN], "f32", [1, 0]);

const acc = b.zeros([BM, BN], "f32");
// Single-tile dot (no K-loop yet): A_tile[BM,BK] * B_tile[BK,BN]
const aTile = b.load(tpA);
const bTile = b.load(tpB);
const out = b.dot(aTile, bTile, acc);
b.store(tpC, out, { boundaryCheck: [0, 1] });

const ttir = b.build("matmul", 4);
console.log("=== TTIR ===");
console.log(ttir);

let ptx: string;
try {
  const res = compileTTIR(ttir, 4);
  ptx = res.ptx;
  console.log(`PTX: ${ptx.length} bytes, shmem=${res.shmem}`);
} catch (e: any) {
  console.error("Compile failed:\n", e.message);
  process.exit(1);
}
