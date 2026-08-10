# bunker — Python-free GPU kernels from TypeScript (bun kernels)

Write GPU kernels in TypeScript. They lower to **TTIR** (Triton's MLIR dialect),
compile to PTX via a single C shim, and run on the GPU — **zero Python anywhere**
in the pipeline. The TTIR is backend-agnostic; the shim selects NVIDIA (PTX) or
AMD (AMDGCN) at build/runtime.

```
TypeScript kernel  →  TTIR text  →  libtriton_shim.so (MLIR passes)  →  PTX/AMDGCN  →  GPU
```

Targeted at an **RTX 3090** (`sm_86`) by default; portable to any Triton backend
(see [AGENTS.md](AGENTS.md) "Design principles").

## Quick start

```bash
# Run the test suite (14 tests, all pass on a 3090)
bun run tests/test_kernel_ttir.ts          # arrow-function lift: vector add
bun run tests/test_ttir_softmax.ts         # softmax: reduce + math.exp + elementwise
bun run tests/test_ttir_matmul_kloop.ts    # FP16 matmul with a K-loop (scf.for + tt.advance)
```

## Two ways to write kernels

### 1. Arrow-function lift (pleasant — like Triton Python)

Write a plain TypeScript arrow function with typed params. `kernel_ttir` parses
it (via the TypeScript compiler API) and lowers it to TTIR. Captured numeric
constants (`const BLOCK = 256`) are resolved from the caller's source file.

```ts
import { kernel_ttir, ptr, i32, f32, programId, arange, load, store,
         compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const N = 4096;
const BLOCK = 256;

const add = kernel_ttir((A: ptr<f32>, B: ptr<f32>, C: ptr<f32>, n: i32) => {
  const pid = programId(0);
  const offs = arange(0, BLOCK) + pid * BLOCK;   // captured const + ptr+tensor math
  const mask = offs < n;
  const a = load(A + offs, { mask });            // Triton-Python-style pointer arithmetic
  const b = load(B + offs, { mask });
  store(C + offs, a + b, { mask });
});

const k = compileAndLoad(add.ttir, "kernel", 4);  // TTIR → PTX → load into CUDA module
const dA = cuAlloc(N * 4), dB = cuAlloc(N * 4), dC = cuAlloc(N * 4);
cuHtoD(dA, hA.buffer); cuHtoD(dB, hB.buffer);
cuLaunch(k, [Math.ceil(N / BLOCK), 1, 1], [128, 1, 1], [dA, dB, dC, N]);
cuSync();
cuDtoH(hC.buffer, dC);
```

The arrow body is **parsed, never executed** — `programId`/`arange`/`load`/`store`
are stubs that exist only so the source type-checks and the editor offers
completion. Supports: arithmetic, comparisons, `if`/`for`/`while`, array
indexing, math (`exp`), captured consts, pointer arithmetic.

### 2. Fluent, type-tracking `TTIRBuilder` (for tiled / complex kernels)

Each `Value` carries its tensor type, so ops infer result types and chain
fluently. Scalars auto-splat to tensors. Use this for matmuls and anything
with tiled pointers / K-loops.

```ts
import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuFree, cuSync, cuLaunch } from "../src/ttir";

const M = 512, N = 512, K = 512, BM = 64, BN = 64, BK = 64;

const b = new TTIRBuilder();
const A = b.param("A", { ptr: "f16" });
const Bp = b.param("B", { ptr: "f16" });
const C = b.param("C", { ptr: "f32" });
const pidM = b.programId(0), pidN = b.programId(1);
const offM = b.mul(pidM, b.i32(BM)), offN = b.mul(pidN, b.i32(BN));

const tpA0 = b.makeTensorPtr(A, [M, K], [K, 1], [offM, b.i32(0)], [BM, BK], "f16", [1, 0]);
const tpB0 = b.makeTensorPtr(Bp, [K, N], [N, 1], [b.i32(0), offN], [BK, BN], "f16", [1, 0]);
const tpC  = b.makeTensorPtr(C, [M, N], [N, 1], [offM, offN], [BM, BN], "f32", [1, 0]);

// K-loop: thread pointers + accumulator through scf.for iter-args.
const acc0 = b.zeros([BM, BN], "f32");
const [acc] = b.forIter(b.index(0), b.index(K), b.index(BK), [acc0, tpA0, tpB0],
  (bb, k, [a, tpA, tpB]) => {
    const next = bb.dot(bb.load(tpA), bb.load(tpB), a);   // tt.dot → mma.sync (tensor cores)
    const nextA = bb.advance(tpA, [bb.i32(0), bb.i32(BK)]);
    const nextB = bb.advance(tpB, [bb.i32(BK), bb.i32(0)]);
    return [next, nextA, nextB];
  });
b.store(tpC, acc, { boundaryCheck: [0, 1] });

const k = compileAndLoad(b.build("matmul", 4), "matmul", 4);
```

## Flagship example: softmax (one block per row-tile)

```ts
const x = b.load(tpA);
const rowMax = b.max(x, 1);                                        // tt.reduce (max)
const maxBc = b.broadcast(b.expandDims(rowMax, 1), [BM, BN]);
const shifted = b.sub(x, maxBc);
const expX = b.exp(shifted);                                        // math.exp
const denom = b.sum(expX, 1);                                       // tt.reduce (sum)
const denomBc = b.broadcast(b.expandDims(denom, 1), [BM, BN]);
b.store(tpC, b.divf(expX, denomBc));
```
512×256, max err **4.9e-9** vs reference.

## Feature coverage (portable TTIR — all verified on the 3090)

| Category | Ops |
|---|---|
| **Elementwise** | `add/sub/mul/div`, `divi/divf/remi`, `minimum/maximum`, `shl/lshr/ashr/and/or/xor`, `neg`, `abs` (type-aware int/float) |
| **Casts** | `trunc/ext/zext/sitofp/fptosi/fpext/fptrunc/bitcast/indexCast` |
| **Compare** | `eq/ne/lt/gt/le/ge` (type-aware → i1), `select` |
| **Math** | `exp` (inlines). `sqrt/rsqrt/log/sin/cos/tanh/floor/ceil/round` need libdevice (see Limitations) |
| **Reductions** | `sum/max/min` (via `tt.reduce`, region form) |
| **Matmul** | `dot` (single-tile, chained, `scf.for` K-loop with iter-arg accumulator → `mma.sync` tensor cores) |
| **Memory** | `makeTensorPtr` + `advance` (tiled pointers), `splatPtr` + `addptr` (pointer-tensors), `load`/`store` (both forms, boundaryCheck/mask/other) |
| **Atomics** | `atomicAdd/Max/Min/Xchg` (`tt.atomic_rmw`, float uses `fadd`, tiled + pointer-tensor) |
| **Layout** | `trans`, `reshape`, `broadcast`, `expandDims` |
| **Control flow** | `scf.if`, `scf.for` (void + iter-args), `scf.while` |
| **Program** | `programId`, `numPrograms` |

## Pleasant CUDA runtime

Boilerplate (alloc/copy/free, module load, launch, sync) is hidden behind
helpers that share one context:

```ts
compileAndLoad(ttir, funcName, numWarps)   // TTIR → PTX → loaded CUDA module + fn handle
cuAlloc(bytes) → bigint                    cuHtoD(devPtr, buf)     cuDtoH(buf, devPtr)
cuLaunch(kernel, [gx,gy,gz], [bx,by,bz], args[])   cuSync()    cuFree(devPtr)
```

## Limitations (honest)

- **`math.sqrt`/`rsqrt`/`log`/`sin`/`cos`/`tanh`** lower to libdevice calls
  (`__nv_*`) the shim doesn't link. Only `math.exp` inlines. **Fix:** link
  `libdevice.10.bc` (or lower `math.*` to native PTX) — a shim rebuild.
- **AMDGCN backend** is structurally ready in `triton_shim.c`
  (`triton_compile_targeted`, `amdgcn-amd-amdhsa` triple) but gated on linking
  Triton's AMD backend (`triton/third_party/amd`) at build time.

## Building the shim (zero Python)

```bash
shim/build_shim.sh   # → shim/libtriton_shim.so (TTIR → PTX via Triton MLIR)
```

The build is fully de-"`/tmp`-ized": Triton source and all objects live in the
persistent, git-ignored `third_party/` dir (`third_party/triton-src` at commit
`76e268973`, pinned to LLVM `ac5dc54d5`), so GCs/reboots no longer break it.
Triton sources compile in parallel (`xargs -P`), then link against the Nix
triton-llvm archives. Verifies with `bun run tests/test_ttir_matmul_run.ts`.

## Project layout

```
src/            core source (mutually independent, no inter-imports)
  dsl.ts          legacy NVIDIA-only LLVM-IR DSL (asm() inline PTX). Not used for
                  new portable work — the TTIR layer is the path forward.
  ttir.ts         the portable authoring layer: TTIRBuilder, kernel_ttir lift,
                  compileTTIR/compileAndLoad/loadPTX (driver-JIT OPT=4 + ptxas
                  fallback), shared cu() runtime, setTarget/compileTTIRTargeted.
                  Loads the shim from ../shim/libtriton_shim.so via __dirname.
  kernel.ts       INT8/quantized matmul kernels + bun:ffi runtime (33 TFLOPS).
shim/           TTIR → PTX via Triton MLIR passes
  triton_shim.c   target-parameterized (triton_compile_targeted);
                  triton_compile is the 3090-default wrapper.
  build_shim.sh   builds libtriton_shim.so (SCRIPT_DIR-relative output).
  libtriton_shim.so / libtriton_shim_dbg.so   build artifacts.
tests/          test_*.ts — 14 end-to-end tests (all pass on the 3090).
bench/          gemm.ts, bench.ts, tc_bench.ts, tc_gemm.ts, autotune.ts.
experiments/    q4k_*, fused_kernel.ts, fuse_ptx.ts, qkernel.ts, index.ts
                  (standalone / superseded quantized-matmul experiments).
prototypes/    self-contained exploratory prototypes (not part of the GPU pipeline).
                  ssd_saturate.ts — Bun parallel disk-read bandwidth benchmark.
```
- `AGENTS.md` — rules & design principles. `LOG.md` — full development log.
- `shell.nix` — builds triton-llvm (dev/build environment).

## How loading works (the gotcha that took a while)

Plain `cuModuleLoadData(ptx)` uses the driver's default PTX JIT, which is
**nondeterministic** — it intermittently rejects valid PTX with
`CUDA_ERROR_INVALID_PTX` (rc=218). `loadPTX` instead uses
`cuModuleLoadDataEx` with `CU_JIT_OPTIMIZATION_LEVEL=4`, and falls back to
`ptxas`→cubin→`cuModuleLoadData` when the driver JIT still rejects. This makes
loading 5/5 reliable.
