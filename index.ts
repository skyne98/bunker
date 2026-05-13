// Q4_K × Q8_1 matmul with struct definitions
import { kernel, ptr, f32, i32, i8, i16, struct, array, sizeof } from "./dsl";

// Define block formats
const Q8_1 = struct({
  q: array(i8, 32),
  d: i16,
  s: i16,
}, "Q8_1");

const Q4_K = struct({
  q: array(i8, 16),
  d: i16,
  dmin: i16,
}, "Q4_K");

console.log("sizeof(Q8_1) =", sizeof(Q8_1));  // 36
console.log("sizeof(Q4_K) =", sizeof(Q4_K));  // 20

const qmm = kernel((A: ptr<Q8_1>, B: ptr<Q4_K>, C: ptr<f32>, M: i32, N: i32, K: i32) => {
  const row = blockIdx.y * 16 + threadIdx.y;
  const col = blockIdx.x * 16 + threadIdx.x;
  if (row >= M || col >= N) return;

  const nb = K / 32;
  let sum = 0;

  for (let b = 0; b < nb; b++) {
    const aBlk = A[row * nb + b];  // Q8_1 block (auto-sized from ptr<Q8_1>)
    const bBlk = B[col * nb + b];  // Q4_K block (auto-sized from ptr<Q4_K>)

    const d8 = half_to_float(aBlk.d);
    const d4 = half_to_float(bBlk.d);
    const dm = half_to_float(bBlk.dmin);

    let s1 = 0, s2 = 0;
    for (let i = 0; i < 16; i++) {
      const q4byte = bBlk.q[i];
      const lo = q4byte & 15;
      const hi = (q4byte >> 4) & 15;
      s1 += lo * aBlk.q[i * 2] + hi * aBlk.q[i * 2 + 1];
      s2 += aBlk.q[i * 2] + aBlk.q[i * 2 + 1];
    }

    sum += d4 * d8 * s1 + dm * d8 * s2;
  }

  C[row * N + col] = sum;
});

console.log("── LLVM IR ──\n" + qmm.source);
