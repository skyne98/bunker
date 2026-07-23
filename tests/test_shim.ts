// test_shim.ts — Baseline: raw TTIR text → shim → PTX → GPU (sanity check).
// Uses the robust compileAndLoad (driver JIT OPT=4 + ptxas fallback) + cu helpers.
import { ffiPtr } from "bun:ffi";
import { compileTTIR, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const N = 4096;
const BLOCK = 256;
const grid = Math.ceil(N / BLOCK);

// Raw TTIR for a vector-add kernel.
const ttir = `module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %pid = tt.get_program_id x : i32
    %c256 = arith.constant ${BLOCK} : i32
    %base = arith.muli %pid, %c256 : i32
    %range = tt.make_range {end = ${BLOCK} : i32, start = 0 : i32} : tensor<${BLOCK}xi32>
    %offsets = tt.splat %base : i32 -> tensor<${BLOCK}xi32>
    %idx = arith.addi %offsets, %range : tensor<${BLOCK}xi32>
    %Nv = tt.splat %arg3 : i32 -> tensor<${BLOCK}xi32>
    %mask = arith.cmpi slt, %idx, %Nv : tensor<${BLOCK}xi32>
    %zero = arith.constant 0.000000e+00 : f32
    %other = tt.splat %zero : f32 -> tensor<${BLOCK}xf32>
    %A = tt.splat %arg0 : !tt.ptr<f32> -> tensor<${BLOCK}x!tt.ptr<f32>>
    %aptr = tt.addptr %A, %idx : tensor<${BLOCK}x!tt.ptr<f32>>, tensor<${BLOCK}xi32>
    %B = tt.splat %arg1 : !tt.ptr<f32> -> tensor<${BLOCK}x!tt.ptr<f32>>
    %bptr = tt.addptr %B, %idx : tensor<${BLOCK}x!tt.ptr<f32>>, tensor<${BLOCK}xi32>
    %a = tt.load %aptr, %mask, %other : tensor<${BLOCK}x!tt.ptr<f32>>
    %b = tt.load %bptr, %mask, %other : tensor<${BLOCK}x!tt.ptr<f32>>
    %c = arith.addf %a, %b : tensor<${BLOCK}xf32>
    %C = tt.splat %arg2 : !tt.ptr<f32> -> tensor<${BLOCK}x!tt.ptr<f32>>
    %cptr = tt.addptr %C, %idx : tensor<${BLOCK}x!tt.ptr<f32>>, tensor<${BLOCK}xi32>
    tt.store %cptr, %c, %mask : tensor<${BLOCK}x!tt.ptr<f32>>
    tt.return
  }
}
`;

const { ptx } = compileTTIR(ttir, 4);
console.log(`PTX compiled: ${ptx.length} bytes`);
const k = compileAndLoad(ttir, "add_kernel", 4);

const hA = new Float32Array(N);
const hB = new Float32Array(N);
const hC = new Float32Array(N);
for (let i = 0; i < N; i++) { hA[i] = i; hB[i] = i * 2; }
const dA = cuAlloc(hA.byteLength), dB = cuAlloc(hB.byteLength), dC = cuAlloc(hC.byteLength);
cuHtoD(dA, hA.buffer); cuHtoD(dB, hB.buffer);
cuLaunch(k, [grid, 1, 1], [128, 1, 1], [dA, dB, dC, N]);
cuSync();
cuDtoH(hC.buffer, dC);

let ok = true;
for (let i = 0; i < N; i++) {
  if (Math.abs(hC[i] - (hA[i] + hB[i])) > 0.001) { ok = false; break; }
}
console.log(ok ? `✓ All ${N} elements correct!` : "✗ FAILED");
cuFree(dA); cuFree(dB); cuFree(dC);
void ffiPtr;
