// prototypes/gdn_solve_tril.ts — EXACT port of fla's merge_16x16_to_64x64_inverse_kernel.
//
// Computes (I + A)^{-1} where A is a strictly-lower-triangular 64×64 matrix, using
// fla's blocked 16×16 approach: forward-substitute 4 diagonal blocks (14-step
// where+sum loop each), then compute 6 off-diagonal blocks via chained matmuls.
// Grid [nChunks, B·H]. Input A [BH·nC, C, C] (the kkt), output Ai [BH·nC, C, C].
//
//   bun run prototypes/gdn_solve_tril.ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const C = 64, B16 = 16;   // 4 × 16 = 64

function buildSolveTril64(BH: number, nC: number) {
  const b = new TTIRBuilder();
  const A = b.param("A", { ptr: "f32" }), Ai = b.param("Ai", { ptr: "f32" });
  const pidC = b.programId(0), pidBH = b.programId(1);
  const base = b.mul(b.add(b.mul(pidBH, b.i32(nC)), pidC), b.i32(C));   // row base in [BH*nC*C, C]

  // ── helper: forward-substitute a 16×16 diagonal block at (br*16, br*16) ──
  // b_Ai starts as -strict_lower(loaded block), then row-by-row forward-subst, +I.
  // Returns the inverted [16,16] block.
  const o_i = b.arange(0, B16);
  const m_A = b.gt(b.broadcast(b.expandDims(o_i, 1), [B16, B16]), b.broadcast(b.expandDims(o_i, 0), [B16, B16]));
  const m_I = b.eq(b.broadcast(b.expandDims(o_i, 1), [B16, B16]), b.broadcast(b.expandDims(o_i, 0), [B16, B16]));
  const zero16 = b.splat(b.f32(0), [B16, B16], "f32");
  const zeroVec = b.splat(b.f32(0), [B16], "f32");

  function fwdSubstDiag(br: number): any {
    const blkRow = b.add(base, b.i32(br * B16));
    const tpBlk = b.makeTensorPtr(A, [BH * nC * C, C], [C, 1], [blkRow, b.i32(br * B16)], [B16, B16], "f32", [1, 0]);
    const blk = b.load(tpBlk, { boundaryCheck: [0, 1] });
    const bAi0 = b.select(m_A, b.mul(blk, b.f32(-1)), zero16);   // -strict_lower(block)

    const [bAi] = b.forIter(b.index(2), b.index(B16), b.index(1), [bAi0], (bb, iv, [bA]) => {
      const i = bb.indexCast(iv, "i32");
      // extract row i from pre-loaded blk: sum(select(row==i, blk, 0), 0) -> [16], negate
      const rMask = bb.broadcast(bb.expandDims(bb.eq(o_i, i), 1), [B16, B16]);
      const baRow = bb.select(rMask, blk, zero16);             // row i of blk, rest 0  [16,16]
      const ba = bb.mul(bb.sum(baRow, 0), b.f32(-1));          // [16] extract + negate
      const baMasked = bb.select(bb.lt(o_i, i), ba, zeroVec);  // where(o_i < i, ba, 0)
      const contrib = bb.sum(bb.mul(bb.broadcast(bb.expandDims(baMasked, 1), [B16, B16]), bA), 0);
      const baNew = bb.add(baMasked, contrib);
      const baBc = bb.broadcast(bb.expandDims(baNew, 0), [B16, B16]);
      const updMask = bb.broadcast(bb.expandDims(bb.eq(o_i, i), 1), [B16, B16]);
      return [bb.select(updMask, baBc, bA)];
    });
    return b.add(bAi, b.select(m_I, b.splat(b.f32(1), [B16, B16], "f32"), zero16));  // +I
  }

  // ── 4 diagonal blocks ──
  const Ai11 = fwdSubstDiag(0);
  const Ai22 = fwdSubstDiag(1);
  const Ai33 = fwdSubstDiag(2);
  const Ai44 = fwdSubstDiag(3);

  // ── load off-diagonal blocks from input A ──
  function loadBlock(br: number, bc: number): any {
    const tp = b.makeTensorPtr(A, [BH * nC * C, C], [C, 1], [b.add(base, b.i32(br * B16)), b.i32(bc * B16)], [B16, B16], "f32", [1, 0]);
    return b.load(tp, { boundaryCheck: [0, 1] });
  }
  const A21 = loadBlock(1, 0), A31 = loadBlock(2, 0), A32 = loadBlock(2, 1);
  const A41 = loadBlock(3, 0), A42 = loadBlock(3, 1), A43 = loadBlock(3, 2);

  // ── off-diagonal inverse blocks (chained matmuls) ──
  const Ai21 = b.mul(b.dot(b.dot(Ai22, A21), Ai11), b.f32(-1));
  const Ai32 = b.mul(b.dot(b.dot(Ai33, A32), Ai22), b.f32(-1));
  const Ai43 = b.mul(b.dot(b.dot(Ai44, A43), Ai33), b.f32(-1));
  const Ai31 = b.mul(b.dot(Ai33, b.add(b.dot(A31, Ai11), b.dot(A32, Ai21))), b.f32(-1));
  const Ai42 = b.mul(b.dot(Ai44, b.add(b.dot(A42, Ai22), b.dot(A43, Ai32))), b.f32(-1));
  const Ai41 = b.mul(b.dot(Ai44, b.add(b.add(b.dot(A41, Ai11), b.dot(A42, Ai21)), b.dot(A43, Ai31))), b.f32(-1));

  // ── store 10 blocks ──
  function storeBlock(br: number, bc: number, val: any) {
    const tp = b.makeTensorPtr(Ai, [BH * nC * C, C], [C, 1], [b.add(base, b.i32(br * B16)), b.i32(bc * B16)], [B16, B16], "f32", [1, 0]);
    b.store(tp, val, { boundaryCheck: [0, 1] });
  }
  storeBlock(0, 0, Ai11); storeBlock(1, 1, Ai22); storeBlock(2, 2, Ai33); storeBlock(3, 3, Ai44);
  storeBlock(1, 0, Ai21); storeBlock(2, 0, Ai31); storeBlock(2, 1, Ai32);
  storeBlock(3, 0, Ai41); storeBlock(3, 1, Ai42); storeBlock(3, 2, Ai43);
  return b.build("solve_tril64", 4);
}

// ── host reference: (I + A)^{-1} where A is strict-lower ──
function refSolveTril(A: Float32Array, N: number) {
  const Ai = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    Ai[i * N + i] = 1;   // identity
    for (let j = 0; j < i; j++) {
      // forward-subst: Ai[i,j] = -A[i,j] + sum_{k=j}^{i-1} (-A[i,k]) * Ai[k,j]
      let s = -A[i * N + j];
      for (let k = j + 1; k < i; k++) s += -A[i * N + k] * Ai[k * N + j];
      Ai[i * N + j] = s;
    }
  }
  return Ai;
}

const BH = 4, nC = 4;
const _t = buildSolveTril64(BH, nC); if (process.env.DUMP) { console.log(_t); process.exit(0); } const k = compileAndLoad(_t, "solve_tril64", 4);
console.log(`solve_tril64 loaded (BH=${BH} nC=${nC} C=${C})`);

// test: random strict-lower A, compute (I+A)^{-1}, verify vs reference
const N = BH * nC;
const AIn = new Float32Array(N * C * C);
for (let n = 0; n < N; n++) for (let i = 0; i < C; i++) for (let j = 0; j < C; j++)
  AIn[n * C * C + i * C + j] = j < i ? (Math.random() * 2 - 1) * 0.1 : 0;   // strict lower
const AOut = new Float32Array(N * C * C);
const dA = cuAlloc(AIn.byteLength), dAi = cuAlloc(AOut.byteLength);
cuHtoD(dA, AIn.buffer);
cuLaunch(k, [nC, BH, 1], [128, 1, 1], [dA, dAi]); cuSync(); cuDtoH(AOut.buffer, dAi);

let maxErr = 0, ok = true;
for (let n = 0; n < N && ok; n++) {
  const ref = refSolveTril(AIn.slice(n * C * C, (n + 1) * C * C), C);
  for (let i = 0; i < C * C; i++) {
    const d = Math.abs(AOut[n * C * C + i] - ref[i]);
    if (d > maxErr) maxErr = d;
    if (d > 1e-3) { ok = false; console.log(`mismatch [n=${n}, i=${i}]: got ${AOut[n*C*C+i].toFixed(6)} ref ${ref[i].toFixed(6)}`); break; }
  }
  if (n === 0) for(let j=0;j<5;j++) console.log(`row${j}: ${AOut.slice(j*C,j*C+5).map(x=>x.toFixed(4)).join(" ")}`);
}
console.log(ok ? `✓ solve_tril64 correct (max err ${maxErr.toExponential(2)})` : "✗ FAILED");
cuFree(dA); cuFree(dAi);
