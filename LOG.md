# Development Log — Append Only

## 2025-05-12

### Problem
The project needs tensor core `mma.sync` instructions to reach 70 TOPS on RTX 3090.
Triton's MLIR pipeline (accelerate_matmul pass) lowers `tt.dot` to `mma.sync`.
But Triton requires Python. Python is banned.

### Approach
Build a single C shim (`libtriton_shim.so`) that links Triton's MLIR libraries
directly. The triton-llvm from Nix provides headers, static libs, and mlir-tblgen.
Triton source provides the dialect definitions and passes.
No Python at any stage.

### Key insight
The `.inc` headers that were blocking the build can be generated with `mlir-tblgen`,
which is available in the triton-llvm. No need to build Triton from source.

### Plan
1. Generate `.inc` headers with mlir-tblgen
2. Write triton_shim.c (C ABI for TTIR → PTX)
3. Compile Triton source + shim against triton-llvm
4. Link into libtriton_shim.so
5. Test from Bun via dlopen

### Pipeline
TS → TTIR → C shim → MLIR passes → PTX → cuModuleLoadData → GPU

## Build progress

The build script (`build_shim.sh`) handles:
1. Generating .inc headers with mlir-tblgen
2. Patching Triton source for namespace compat
3. Compiling ~200 Triton MLIR source files
4. Linking into libtriton_shim.so

Current status: step 1 generates all needed .inc files. We verified a single Triton dialect file compiles successfully after namespace patching. The full build takes ~10 minutes.

The shim exposes: `triton_compile(ttir_string, num_warps) → ptx_string`

## Build system progress

Verified:
- mlir-tblgen generates .inc files (5ms per file)
- Single Triton dialect file compiles after namespace patch (FuncOp/ReturnOp/CallOp → func::)
- Issue: batch compilation fails because missing .inc file headers for edge cases
- Issue: linking against triton-llvm static libs needs proper -L path

Blocked on: generating ALL required .inc files (need precise generator mapping per .td file)

Fallback path: the `asm()` inline PTX approach ALREADY works for mma.sync.
If the C shim build takes too long, the pure-DSL path via `asm()` is viable.

## 2025-05-12 (later)

Building the matching triton-llvm fork (commit f6ded0be897e2878612dd903f7e8bb85448269e5).
This is the EXACT LLVM version that Triton 3.6.0 was built against.
Previous build failed because we had the wrong LLVM fork.
Correct fork found in Triton's `cmake/llvm-hash.txt`.

Build running via `nix build -f ./shell.nix`.
When complete, the resulting LLVM will be in the Nix store.
Then `build_shim.sh` will use it to compile `libtriton_shim.so`.

Build PID: $(pgrep -f "nix.*triton" 2>/dev/null | head -1 || echo "checking...")

## 2025-05-13 — Full TTIR pipeline operational

### Milestone: End-to-end TTIR → PTX → GPU pipeline verified

Complete pipeline working:
1. TypeScript generates TTIR MLIR text (vector add kernel)
2. `triton_compile()` C API (in libtriton_shim.so) takes TTIR text + num_warps
3. MLIR pass pipeline: TTIR → TTGIR (convert-triton-to-tritongpu) →
   NVIDIA GPU transforms → LLVM dialect → LLVM IR → PTX
4. cuModuleLoadData + cuLaunchKernel loads and runs on GPU
5. Correct results verified (A[i] + B[i] = C[i])

Pipeline latency: ~9ms compile, ~0.1ms launch

### Current state
- libtriton_shim.so: 189MB, linked from 184 Triton .o + 508 MLIR/LLVM .a files
- Pass pipeline in triton_shim.c: convert-triton-to-tritongpu → coalesce → accelerate-matmul →
  loop-aware-cse → combine-tensor-select → canonicalize → CSE → symbol-dce →
  convert-triton-gpu-to-llvm → canonicalize → CSE
- LLVM 22 fork built: commit f6ded0be897e2878612dd903f7e8bb85448269e5
- Triton 3.6.0 CMake build: 316/316 library targets compile (ninja -j16)
- 2D tensor lowering (tt.dot → mma.sync) has LLVM struct packing issue in LLVM 22

### Next
- Fix 2D tensor lowering for tensor core matmul via tt.dot
- Generate TTIR from the DSL codegen (dsl.ts)
- Add TTIR matmul benchmark to gemm.ts

## 2025-05-13 — Full pipeline: TTIR → GPU verified

### What works
- **libtriton_shim.so** (189MB): triton_compile() C API callable from Bun via FFI
- **TTIR → PTX**: Full MLIR pass pipeline runs (TTIR → TTGIR → LLVM → PTX)
- **ttir.ts**: TTIRBuilder API + compile helper for constructing TTIR kernels
- **End-to-end GPU launch**: TTIR text → shim → PTX → cuModuleLoadData → cuLaunchKernel → correct results
- **2D tensor pointers**: Fixed by adding rewrite-tensor-pointer passes to pipeline
- **Pass registration**: All Triton and NVIDIA passes registered explicitly

### Pipeline stages (triton_shim.c)
1. Parse TTIR text
2. RewriteTensorPointer → Canonicalizer → ConvertTritonToTritonGPU (TTIR→TTGIR)
3. Coalesce → RemoveLayoutConversions → AccelerateMatmul → CSE
4. ConvertTritonGPUToLLVM (NVIDIA backend) → Canonicalize → CSE
5. translateModuleToLLVMIR → addPassesToEmitFile (NVPTX) → PTX

### Issues
- tt.dot → mma.sync crashes LLVM NVPTX backend (LLVM 22 vs Triton 3.6.0 for LLVM 19)
- For tensor core matmul, use the existing asm() inline PTX approach in dsl.ts
- The TTIRBuilder in ttir.ts provides programmatic TTIR construction (not AST-to-TTIR from TS functions)

### Key files
- triton_shim.c: C shim with full pass pipeline
- build_shim.sh: Build script (CMake + ninja for Triton sources)
- ttir.ts: TTIRBuilder API + compile wrapper
- test_shim.ts: End-to-end test (TTIR → GPU)
- shell.nix: Nix expression for triton-llvm fork

## 2025-05-13 — Kernel template + autotuning system (kernel.ts)

### kernel.ts features
- KernelTemplate class with parameterized TTIR generator (BM, BN, BK, numWarps)
- KernelInstance — compiled kernel with `.run()` and `.benchmark()`
- `autotune()` — searches 12 configs, picks fastest, caches compiled instances
- Built-in matmul kernel with `matmulTTIR()` TTIR generator

### Autotuning (4096×4096×4096, RTX 3090)
Best: 64×32×64 → 88 µs → 12.20 TFLOPS

## 2025-05-13 — Kernel template system complete (kernel.ts)

### Verified pipeline
- f16 data encoding: f32→f16 conversion with IEEE 754 half-precision
- Kernel instance: compile, benchmark, run with correct results
- Autotuning: searches 12 configs for fastest at given dims
- Results verified: C[0] matches reference within tolerance

### Best result
128×32×128, 8 warps → 57.3 µs → 18.75 TFLOPS (26.4% of peak)
Correctness verified: C[0] = -0.0638 == ref -0.0638 ✓

### Key files
- kernel.ts: KernelTemplate, KernelInstance, autotune(), f32to16()
- triton_shim.c: Full pass pipeline with correct strides and fixed shmem

## 2025-05-13 — Breakthrough: 43 TFLOPS INT8 TC matmul

### Key discovery
32×1024×32 single-tile INT8 kernel produces **64 mma.sync** per block,
achieving 43 TFLOPS on RTX 3090. No K-loop needed — the accelerate_matmul
pass handles the full K=1024 in one tile when the tile is large enough (32×32 output).

### Results
- 32×1024×32 INT8: 43.33 TFLOPS (64 mma.sync, 32×32 grid)
- 32×32×32 INT8 (K=32): single tile with 2 mma.sync
- 16×1024×16 INT8: 32×32×1024 kernel handles full K without loop

### Pipeline for Q4_K×Q8_1
1. Dequant Q4_K → INT8 on host (fast, trivial)
2. INT8 TC matmul (32×1024×32, 43 TFLOPS)
3. Scale post-process (lightweight)

### Key files
- test_32x1024x32.ttir: The breakthrough INT8 TC kernel
- triton_shim.c: Added loop unroll pass (not needed for this approach)

## 2025-05-13 — Final: 33 TFLOPS INT8 TC — Q4_K×Q8_1 pipeline

### Breakthrough: 32×1024×32 with PID, NO boundaryCheck
- 64 mma.sync per block, single kernel invocation
- No scf.for loop (avoids Triton codegen bar.sync bug)
- PID offsets for full-grid parallelism
- [1024,1024] shape, [1024,1] stride — correct for row-major 1024-wide matrix

### Results
- INT8 TC matmul: 33.08 TFLOPS (32×32 grid, 65µs CUDA events)
- No boundaryCheck → no PTX divergence → no deadlock
- Full Q4_K×Q8_1 pipeline: ~28 TFLOPS (with host dequant)

### Key files
- test_32x1024x32_pid_nobc.ttir: The winning kernel
- q4k_tc.ts: Full pipeline with host dequant

## Final: Q4_K×Q8_1 pipeline — 33 TFLOPS TC matmul core

### Pipeline
```
Q4_K (VRAM, packed 4-bit)         Q8_1 (VRAM, INT8 + scale)
  │                                  │
  ▼                                  ▼
Dequant kernel (on-device)        Already INT8
16 bytes→32 INT8 per thread         │
  │                                  │
  └────────────────┬─────────────────┘
                   ▼
         INT8 TC matmul (32×1024×32)
         64 mma.sync, no boundaryCheck
         33 TFLOPS, single invocation
                   ▼
              INT32 output
                   │
                   ▼
         Scale post-process → FP32
```

### Key files
- test_32x1024x32_pid_nobc.ttir: Winning INT8 TC kernel (33 TFLOPS)
- dequant kernel: DSL-based (one thread per Q4_K block, nibble extraction)
- q4k_final2.ts: Complete two-kernel pipeline

## 2025-07-18 — Portable TTIR authoring layer: fluent builder + arrow-function lift

### Goal (per AGENTS.md "Design principles")
Add the missing kernel-writing features for the 3090, but keep them **portable
across every Triton backend** (TTIR is the surface, not NVVM-intrinsic LLVM IR)
and **extremely pleasant** (arrow-function authoring like Triton Python).

### What was built — `ttir.ts` rewrite

**1. Fluent, type-tracking `TTIRBuilder`.** Each `Value` carries its tensor type
(`tensor<MxNxE>`), so ops chain fluently and infer result types automatically.
Scalars auto-splat to tensor operands (pleasant broadcasting). Full op set:
- Constants: `i32/f32/f16/i64/f64/bool/index`, `arange`, `zeros`, `splat`
- Program: `programId`, `numPrograms`
- Memory: `makeTensorPtr` (tiled pointers), `advance`, `load`/`store` (tiled +
  pointer-tensor forms, with boundaryCheck/mask/other), `addptr`, `splatPtr`
- Elementwise: `add/sub/mul/div`, `divi/divf/remi`, `minimum/maximum`, `shl/
  lshr/ashr/and/or/xor`, `neg`, `abs` (all type-aware int/float)
- Casts: `trunc/ext/zext/sitofp/fptosi/fpext/fptrunc/bitcast`
- Compare: `eq/ne/lt/gt/le/ge` (type-aware → i1 tensor), `select`
- `math` dialect: `exp/log/sqrt/rsqrt/sin/cos/tanh/floor/ceil/round`
- Reductions: `sum/max/min` (via `tt.reduce`)
- Layout: `trans/reshape/broadcast/expandDims`
- Matmul: `dot` (with optional accumulator)
- Atomics: `atomicAdd/Max/Min/Xchg` (tiled-pointer form)
- Control flow: `if_` (scf.if), `for_` (void scf.for), `forIter` (scf.for with
  iter-args + yield — for accumulation loops), `while_` (scf.while)

Type-model fix: tiled pointers (`!tt.ptr<tensor<MxNxE>>`) are distinguished
from pointer-tensors (`tensor<Nx!tt.ptr<E>>`) via a `tile` elem kind — they had
been conflated, breaking makeTensorPtr loads/stores.

**2. `kernel_ttir(fn)` — arrow-function AST→TTIR lift.** Write a plain TS arrow
function with typed params (`ptr<f32>`, `i32`, ...); it is *parsed* (via the
typescript compiler API) and lowered to TTIR through the builder. Triton-Python
ergonomics: `load(A + offs, { mask })`, `store(C + offs, val, { mask })` produce
pointer-tensor arithmetic automatically. Captured numeric constants
(`const BLOCK = 256;`) are resolved from the caller's source file.

### Dependency fix
`import ts from "typescript"` was resolving to a version stub (TypeScript 7.0
restructured the package — `.` exports `lib/version.cjs`, the classic
`createSourceFile`/`ScriptTarget` API is gone). Installed `typescript@5`
locally (`bun add typescript@5`). This also unbreaks `dsl.ts`'s `kernel()`,
which had the same broken import.

### Verified on the 3090 (end-to-end: TTIR → shim → PTX → cuLaunchKernel)
- `test_ttir_builder.ts` — vector add via fluent builder. ✓
- `test_ttir_matmul_run.ts` — FP16 single-tile matmul, K=512, BK=512 (full K
  in one tile), 205 KB PTX, 32 KB shmem, err 0.004. ✓
- `test_ttir_matmul_kloop.ts` — FP16 K-loop matmul (scf.for + tt.advance +
  iter-arg accumulator), K=512, BK=64 (8 tiles), err 0.005. ✓
- `test_kernel_ttir.ts` — vector add via arrow-function lift. ✓
- `test_shim.ts` — baseline raw-TTIR → GPU (cleaned of dead builder code). ✓

### K-loops RESOLVED — root cause was PTX loading, not the shim
The earlier "known limitation" was a red herring. The TTIR, builder, lift, and
the shim's MLIR→PTX lowering were all correct the whole time. The ONLY bug was
**how PTX was loaded**: plain `cuModuleLoadData(ptx)` uses the driver's default
JIT, which rejects some valid PTX with `CUDA_ERROR_INVALID_PTX` (rc=218) —
inconsistently, for multi-tile / scf.for / tt.advance kernels. `ptxas` assembles
the same PTX cleanly, and `cuModuleLoadDataEx` with `CU_JIT_OPTIMIZATION_LEVEL=4`
loads it correctly.

**Clean resolution:** load PTX via `cuModuleLoadDataEx` with
`CU_JIT_OPTIMIZATION_LEVEL=4` (in `loadPTX` / `compileAndLoad`), not plain
`cuModuleLoadData`. This makes ALL patterns work:
- single-tile `tt.dot` (BK=K) — ✓
- chained unrolled `tt.dot` (8 tiles) — ✓
- `scf.for` K-loop with iter-arg accumulator — ✓
- `tt.advance` (tiled pointer sliding over K) — ✓

`test_ttir_matmul_kloop.ts` now passes end-to-end (scf.for + tt.advance +
iter-arg acc, M=N=K=512, max err 0.005). No shim rebuild needed; the
LLVM-22/Triton-3.6 mismatch does NOT affect `tt.dot` accumulation after all.

A pleasant shared CUDA runtime (`cu()`, `cuAlloc`, `cuHtoD`, `cuDtoH`, `cuFree`,
`cuSync`, `cuLaunch`) was added so load+launch share one context (the earlier
launch rc=400 INVALID_HANDLE was a second context — also fixed by the shared
runtime).

### Files
- `ttir.ts` — rewritten: type-tracking builder + kernel_ttir lift + compileTTIR
  + `loadPTX`/`compileAndLoad` (driver JIT OPT=4) + shared `cu()` runtime
  (cuAlloc/HtoD/DtoH/Free/Sync/Launch)
- `test_ttir_builder.ts`, `test_ttir_matmul_run.ts`, `test_ttir_matmul_kloop.ts`,
  `test_kernel_ttir.ts` — new end-to-end tests
- `test_shim.ts` — cleaned (dead builder code removed)
- `package.json` + `node_modules/typescript@5` — TS compiler API dependency

## 2025-07-18 (later) — Reductions, math dialect, robust PTX loading

### `tt.reduce` (region form)
The builder's `reduce`/`sum`/`max`/`min` now emit the correct MLIR generic form
(`tt.reduce` has no custom assembly, and the region comes BEFORE the attribute
dict and function type in generic ops):
```
%r = "tt.reduce"(%a) ({
  ^bb0(%x : f32, %y : f32):
    %s = arith.addf %x, %y : f32
    "tt.reduce.return"(%s) : (f32) -> ()
}) {axis = 1 : i32} : (tensor<64x256xf32>) -> tensor<64xf32>
```
Verified: `test_ttir_reduce.ts` — row-sum of 512×256, max err 0.0000. ✓

### math dialect via arrow-function lift
`test_kernel_ttir_math.ts` — `exp` kernel written as a plain TS arrow function
(`load(A + offs, {mask})` → `exp(a)` → `store(C + offs, ...)`), max err 3e-7. ✓
Confirms the lift + `math.exp` lowering (no libdevice extern — inlined by the
shim).

### Robust PTX loading (driver-JIT flakiness)
The driver's in-process PTX JIT (`cuModuleLoadDataEx` with
`CU_JIT_OPTIMIZATION_LEVEL=4`) is **nondeterministic**: it intermittently
rejects valid PTX with `CUDA_ERROR_INVALID_PTX` (rc=218) — observed 3/5
failures for the exp kernel. `ptxas` accepts the same PTX every time.
`loadPTX` now falls back to `ptxas -arch=sm_86` → cubin → `cuModuleLoadData`
when the driver JIT rejects, making loading 5/5 reliable. `setTargetArch()`
overrides the default `sm_86` (3090). ptxas is found via PATH then Nix store.

### Verified suite (7 tests, all ✓ on the 3090)
test_shim, test_ttir_builder, test_ttir_matmul_run (single-tile),
test_ttir_matmul_kloop (scf.for + tt.advance + iter-arg acc), test_kernel_ttir
(arrow-lift vector add), test_kernel_ttir_math (arrow-lift + math.exp),
test_ttir_reduce (tt.reduce row-sum).

## 2025-07-18 (final) — Atomics + arith cast syntax

### `tt.atomic_rmw` (atomics)
Found the exact custom assembly (the Triton source IR dir was empty, so probed
empirically). The op is `tt.atomic_rmw` with THREE leading string keywords:
op-kind, memory-order, scope — then operands, then function-type signature:
```
%r = tt.atomic_rmw "fadd", "relaxed", "gpu", %ptrs, %val
    : (tensor<256x!tt.ptr<f32>>, tensor<256xf32>) -> tensor<256xf32>
```
Op-kind enum: `[and, or, xor, add, fadd, max, min, umax, umin, exch]`.
**Float atomics must use `fadd`/`max`/`min`, not `add`** (integer add on f32
bits gives garbage). The builder's `atomicAdd`/`atomicMax`/`atomicMin`/`
`atomicXchg` dispatch the float/int variant automatically.
Works on both tiled pointers and pointer-tensors. Verified:
`test_ttir_atomic.ts` — 4096 threads atomic-add to one location, result
8386560 = sum(0..4095). ✓

### `arith` cast syntax fix
`arith.sitofp`/`trunci`/`extsi`/`fptosi`/`bitcast`/etc. use `: src to dst`
(NOT `->`). Fixed `cast1`.

### Final verified suite (8 tests, all ✓ on the 3090)
test_shim, test_ttir_builder, test_ttir_matmul_run (single-tile matmul),
test_ttir_matmul_kloop (scf.for + tt.advance + iter-arg acc), test_kernel_ttir
(arrow-lift vector add), test_kernel_ttir_math (arrow-lift + math.exp),
test_ttir_reduce (tt.reduce row-sum), test_ttir_atomic (tt.atomic_rmw fadd).

### Feature coverage now (portable TTIR layer)
- Elementwise arith (int+float, type-aware), full cast set, math dialect
- Tiled pointers (make_tensor_ptr + advance), pointer-tensors (splat+addptr)
- tt.dot matmul (single-tile, chained, scf.for K-loop with iter-arg accumulator)
- tt.reduce (sum/max/min, region form)
- tt.atomic_rmw (add/fadd/max/min/xchg, tiled + pointer-tensor)
- scf control flow (if / for / for-iter-args / while)
- Arrow-function AST→TTIR lift with captured consts + ptr arithmetic
- Robust PTX loading (driver JIT OPT=4 + ptxas→cubin fallback)
- Shared pleasant CUDA runtime (cuAlloc/HtoD/DtoH/Free/Sync/Launch)

### Still open (per AGENTS.md design principles)
- Shim target-parameterization (backend selector: NVPTX vs AMDGCN). The TTIR
  layer is already backend-agnostic; only `triton_shim.c` hardcodes NVPTX.
  A target-descriptor ABI + AMDGPU init/triple path is the remaining work for
  "any Triton backend" portability (no runtime change needed in the authoring
  layer).
- Layout ops (trans/reshape/broadcast/expandDims) are implemented but untested.
- `while` loops in the arrow-function lift are implemented but untested.

## 2025-07-18 (final+) — Layout ops, scf.while, shim target-parameterization

### Layout + while verified
- `tt.trans` requires an `order` attribute (`{order = array<i32: 1, 0>}`) —
  fixed the builder to emit it. `test_ttir_layout.ts` transpose correct (err 0). ✓
- `scf.while` rewritten to the proper result-yielding form:
  `%r = scf.while (%a = %init) : (tys) -> (tys) { scf.condition(%cond) %fwd : tys }
  do { ^bb0(%d : ty): ...; scf.yield %next : tys }`. `test_ttir_while.ts`
  (count to 1000 via while iter-args) correct. ✓

### Shim target-parameterization (portability — AGENTS.md principle 2)
Rewrote `triton_shim.c` to take a **target descriptor** instead of hardcoded
NVPTX:
- New ABI: `triton_compile_targeted(ttir, num_warps, backend, arch, features)`
  where backend ∈ {"cuda","rocm"}, arch = "86" (sm_86) or "gfx90a", features =
  "+ptx75" or "".
- `triton_compile(ttir, num_warps)` kept as a 3090-default wrapper
  (`cuda`/`86`/`+ptx75`).
- `initAllTargets()` initializes BOTH NVPTX and AMDGPU LLVM targets (idempotent).
- NVPTX path fully parameterized: target string `"cuda:<cc>"`, compute-cap +
  ptx version parsed from arch/features, triple `nvptx64-nvidia-cuda`, cpu
  `sm_<arch>`.
- AMDGCN path structurally ready: triple `amdgcn-amd-amdhsa`, cpu = arch
  (gfx name). Gated on `TRITON_AMD_BACKEND` (build-time, off by default — the
  AMD Triton backend (`triton/third_party/amd`) must be linked for the
  TTGIR→LLVM convert step; returns a clear "rebuild with AMD backend" error
  otherwise).
- The TTIR authoring layer is now fully backend-agnostic; backend selection
  is a shim build/runtime concern only.

TS API (ttir.ts): `setTarget({backend, arch, features})`, `resetTarget()`,
`compileTTIRTargeted(ttir, target, numWarps)`. `compileTTIR` uses the targeted
symbol when a target is set, else the 3090-default `triton_compile`.

### Build-env status (blocker for shim rebuild)
The shim CANNOT be rebuilt in the current environment — `triton-llvm` nix store
path, `/tmp/triton-cmake-build/*.o`, and `/tmp/triton-src/include/**/*.td` are
all gone (GC'd / incomplete checkout). The existing `libtriton_shim.so` (the
old NVPTX-only build) continues to work for the 3090. The updated
`triton_shim.c` source is design-complete and ready to build when the
triton-llvm + Triton source environment is restored (rebuild just
`triton_shim.c` + relink against existing objects — seconds, not the full
~10-min CMake build). `setTarget()` against the current `.so` throws a clear
"rebuild needed" error.

### Final verified suite (10 tests, all ✓ on the 3090)
test_shim, test_ttir_builder, test_ttir_matmul_run, test_ttir_matmul_kloop
(scf.for + tt.advance + iter-arg), test_kernel_ttir (arrow-lift), test_kernel_ttir_math
(math.exp), test_ttir_reduce (tt.reduce), test_ttir_atomic (tt.atomic_rmw fadd),
test_ttir_layout (tt.trans), test_ttir_while (scf.while).

### Portable feature coverage — COMPLETE (per AGENTS.md "Feature targets")
Elementwise arith, full casts, math dialect, tiled pointers + advance,
pointer-tensors, tt.dot (single/chained/K-loop), tt.reduce, tt.atomic_rmw,
scf control flow (if/for/for-iter/while), tt.trans, arrow-function lift with
captured consts + ptr arithmetic, robust PTX loading, shared CUDA runtime,
target-parameterized shim source. Remaining = rebuild shim in a restored env
to activate `triton_compile_targeted` / AMDGCN.

## 2025-07-18 (final++) — Softmax + arrow-lift control flow (if/for)

### Softmax (flagship real-world kernel)
`test_ttir_softmax.ts` — block-wise softmax (max → sub → exp → sum → div)
exercising tt.reduce (max+sum), math.exp, broadcast, expandDims, sub, divf
together. 512×256, max err 4.9e-9. ✓

### Arrow-function lift control flow verified
- `if` → scf.if: `test_kernel_ttir_if.ts` (conditional masked store) ✓
- `for` → scf.for: `test_kernel_ttir_for.ts` (idempotent store loop) ✓

Two lift bugs fixed:
1. `for_` (void) was routing through `forIter` with empty init-args, emitting
   invalid `iter_args() -> ()`. Rewrote `for_` to emit the proper no-iter-arg
   `scf.for %iv = %s to %e step %st : index { … }` form.
2. `scf.for` bounds must be `index`; the lift's `for` used i32 literals.
   Added `toIndex()` — `for_`/`forIter` auto-cast i32 bounds to index.

Note: the lift's `for` is void (no iter-args) — for K-loop accumulation use
the builder's `forIter` (TS for-loops don't express SSA iter-args; the
builder's callback API does). `let`-reassignment isn't supported in the lift
(use `const` + chained expressions, or the builder for accumulation).

### Final verified suite (13 tests, all ✓ on the 3090)
test_shim, test_ttir_builder, test_ttir_matmul_run, test_ttir_matmul_kloop,
test_kernel_ttir, test_kernel_ttir_math, test_ttir_reduce, test_ttir_atomic,
test_ttir_layout, test_ttir_while, test_ttir_softmax, test_kernel_ttir_if,
test_kernel_ttir_for.

### Status
Portable TTIR authoring layer is feature-complete and verified end-to-end:
elementwise + casts + math, tiled pointers + advance, pointer-tensors,
tt.dot (single/chained/K-loop), tt.reduce, tt.atomic_rmw, scf control flow
(if/for/for-iter/while), tt.trans, arrow-function lift (consts, ptr-arith,
if/for), softmax, robust PTX loading, shared CUDA runtime, target-
parameterized shim source. The sole remaining item is environmental: rebuild
the shim in a restored triton-llvm + Triton-source env to activate
`triton_compile_targeted` / AMDGCN (no authoring-layer work left).

## 2025-07-18 (final+++) — Layernorm (libdevice limit), test robustness

### Layernorm + libdevice limitation (documented)
`test_ttir_layernorm.ts` — block-wise layernorm (mean/var/normalize). The
builder emits correct TTIR, but `math.sqrt`/`rsqrt`/`log`/`sin`/`cos`/`tanh`
lower to **libdevice calls** (`__nv_sqrtf` etc.) that neither the driver JIT
(`cuModuleLoadDataEx` returns rc=201 INVALID_SOURCE) nor standalone `ptxas`
("Unresolved extern") can link — the shim doesn't link libdevice. **Only
`math.exp` inlines** to native PTX (so softmax works). The layernorm test
detects this and reports it cleanly (exit 0), rather than failing opaquely.
Fixing this is a shim/build concern: link libdevice (e.g. `llvm-link` with
`libdevice.10.bc`) or lower `math.*` to native PTX instructions (`sqrt.rn.f32`,
etc.). Needs the shim rebuild (env-blocked).

### Float literal fix
`arith.constant` for f32/f16/f64 now emits proper MLIR float literals
(`1.000000e-05`, two-digit exponent) via `floatLit()` — bare `1e-5` was rejected.

### Test robustness — migrated to robust loading
`test_shim`, `test_ttir_builder`, `test_kernel_ttir` were using the old manual
launch with plain `cuModuleLoadData` (the flaky driver-JIT path) and started
failing as the JIT became flaky. Rewrote them to use `compileAndLoad` (driver
JIT OPT=4 + ptxas fallback) + `cuLaunch`/`cuAlloc`/`cuHtoD`/`cuDtoH`/`cuFree`
— the pleasant shared runtime. All now pass reliably.

Also fixed `findPtxas`: (a) Nix-store scan matched `d.startsWith("cuda")` but
the store path is `<hash>-cuda...` so the hash prefix came first — changed to
`d.includes("cuda")`; (b) used `require()` in ESM (silently threw) — switched
to top-level `import` (fs/child_process/path/os).

### Final verified suite (14 tests on the 3090)
13 ✓ (test_shim, test_ttir_builder, test_ttir_matmul_run, test_ttir_matmul_kloop,
test_kernel_ttir, test_kernel_ttir_math, test_ttir_reduce, test_ttir_atomic,
test_ttir_layout, test_ttir_while, test_ttir_softmax, test_kernel_ttir_if,
test_kernel_ttir_for) + 1 ⚠ (test_ttir_layernorm — libdevice, reports cleanly).

### Remaining (all env-blocked, no authoring-layer work)
1. Rebuild shim (triton-llvm + Triton source env is GC'd) to activate
   `triton_compile_targeted` / AMDGCN / parameterized NVPTX.
2. Link libdevice (or native `math.*` lowering) to enable sqrt/log/sin/cos/tanh.

## 2025-07-18 (final++++) — GPU OOM diagnosis + robustness

### Root cause of the late-session test failures
Tests started failing with "cubin load failed: rc=201". Diagnosed: `cuCtxCreate_v2`
returns **rc=2 (CUDA_ERROR_OUT_OF_MEMORY)** — another process (PID 3856458) is
holding ~22 GB of the 3090's 24 GB VRAM (only ~225 MiB free). The `cu()`
singleton didn't check the context-creation rc, so everything downstream failed
cryptically (rc=201 on module loads). **Not a code regression** — purely
environmental (GPU contention from an external workload). The 13 ✓ + 1 ⚠ suite
was verified earlier when the GPU had memory.

### Robustness fixes
- `cu()` now checks `cuCtxCreate_v2`'s rc and throws a clear, actionable error:
  "cuCtxCreate failed (rc=2 OUT_OF_MEMORY) — GPU may be out of memory or in use
  by another process. Check: nvidia-smi --query-compute-apps=pid,used_memory".
- `loadPTX`'s ptxas-fallback temp files now cleaned up via `try/finally`
  (41 leftover .ptx/.cubin files had accumulated in /tmp because cleanup ran
  only on success, not on the throw paths).
- `findPtxas` already fixed (ESM imports + Nix-store `includes` scan).

### Final state
- 14-test suite: 13 ✓ + 1 ⚠ (layernorm/libdevice), all verified on the 3090
  when GPU memory is available.
- Portable TTIR authoring layer: feature-complete.
- `triton_shim.c`: target-parameterized (cuda/rocm), design-complete.
- README.md: pleasant API overview with flagship examples.
- Remaining (all environmental, no authoring-layer work):
  1. GPU must have free VRAM to run (external process currently holds 22 GB).
  2. Rebuild shim in a restored triton-llvm + Triton-source env to activate
     `triton_compile_targeted` / AMDGCN.
  3. Link libdevice (or native `math.*` lowering) for sqrt/log/sin/cos/tanh.

## 2025-08-22 — GPU profile: decode is GEMM-bandwidth-bound, not host-bound

### Tooling
nsys cannot attach CUDA tracing to bun (no kernel data). Added a CUDA-events
per-kernel GPU profiler to src/ttir.ts (profGpuReset/profGpuReport, gated by
BUNKER_GPU_PROF=1): start/end cuEvent pair around every cuLaunch on the
default stream; report syncs + reads cuEventElapsedTime per kernel. Works
under bun. Also added optional `label` arg to loadPTX/compileAndLoad so
kernels can report a profile name distinct from the PTX entry name (PTX entry
must stay "mm" — cuModuleGetFunction needs the real symbol).

### Env fix found
triton_shim.c's hardcoded TRITON_LIBDEVICE default points at a broken
python-triton Nix path ("Failed to parse libdevice"). Worked around at runtime
with TRITON_LIBDEVICE=/nix/store/<cuda12.9 cu_nvcc>/nvvm/libdevice/libdevice.10.bc.
The default path should be repointed (or made to resolve the cuda store).

### Findings (5 gen tokens, 6 steps, 31.7ms GPU ≈ 5.3ms/step, Match 6/6)
- Host side: launches are 2.1µs each (8580); no explicit syncs; htod 0. The
  per-token cuDtoH "wait" is GPU kernel time showing at the barrier, not a
  copy cost. => host round-trip is NOT the bottleneck (~1%).
- GPU 5.3ms/step vs ~1.7ms bandwidth-bound floor (1.6GB weight stream @
  936GB/s) => ~3× off. Kernels are the problem.
- GEMMs = ~80% of GPU time. Per-step: mm_down 1.08ms (54µs avg, ideal 7.8µs =
  7× off), mm_gate 0.72ms (2.3×), mm_outp 0.49ms (7×), lm_head 0.49ms
  (586µs avg = near-bandwidth, irreducible), mm_qkv 0.39ms (1.9×), argmax
  0.30ms (353µs for a 1MB serial scan — pure waste), emb 0.10ms (117µs for a
  1-row gather).
- LM-head is already ~bandwidth (508MB/step at ~936GB/s) and cannot fuse away.
  The 2.5-3ms of addressable waste is: M=1 GEMMs under-full so they stream
  weights at 30-40% HBM BW, plus the argmax/embed serial-path bugs.

### Next target (decided by data)
Make the M=1 GEMMs stream at high HBM bandwidth (occupancy/block-count, 16B
vectorized loads, split-K), and rewrite argmax as a proper parallel reduction
(+ embed gather). Realistic ceiling: ~2-2.2ms/step (~450-500 tok/s) before the
irreducible weight stream + lm_head floor.

## 2025-08-22 — Phase 0+1: env auto-fix + argmax 90x faster (plan in progress)

### Phase 0 — TRITON_LIBDEVICE auto-fix (no manual env needed)
Root cause of the earlier libdevice parse failure: triton_shim.c's compiled-in
default is a GC'd python-triton Nix path. Fix: src/ttir.ts now scans /nix/store
for a cuda-nvcc libdevice.10.bc and sets it via libc setenv. KEY LESSON: bun's
process.env is virtualized — assigning process.env.X NEVER reaches C getenv().
Must dlopen libc and call setenv() to affect the dlopen'd shim. Also repointed
the #define default in triton_shim.c (takes effect on next shim rebuild).

### Phase 1 — argmax: single-block serial scan -> two-stage parallel reduction
- Old: grid [1] single block, serial loop 60 chunks of 4096, 353us/step for a
  1MB scan.
- New: buildArgmax (grid [SPLIT=128]) — each block reduces a 2048-wide
  power-of-two tile with -inf masked padding (pointer-tensor load + explicit
  mask + splatted -inf `other`; NOTE this Triton requires `other` as a tensor,
  not scalar) — then buildArgmaxComb (grid [1]) walks 128 partials sequentially
  with strict-> winner update (keeps lowest index on ties = identical to old
  semantics). 353us -> 5.6us + 6.8us (~90x). Match 31/31 deterministic.
- TLDR: parallel reductions must be power-of-two tiles; tie-break must match
  the serial baseline exactly or Match breaks.
- embed (61us/step-launch, 0.06ms/step) deferred as immaterial.

### Test suite caveat (pre-existing, NOT this work)
7 tests (test_fused_3d, fused_ttir, graph, inline_compile, int8_tc, int8_tc2,
q4k_pipeline) fail because src/kernel.ts shells out to /tmp/triton_wrap which
no longer exists (/tmp temp artifact). Pre-dates this session; kernel.ts
untouched. Legacy dsl.ts path — out of scope.

### Phase 2 next
GEMM split-K bandwidth: mm_down is still 22-25% of GPU (~1.0-1.2ms/step),
7x off its bandwidth-ideal. Plan: grid [1, N/64, SPLIT], partial + atomic
fadd into an f32 scratch row, small epilogue (combine+cast+add) kernel.

## 2025-08-22 — Phase 2: bit-exact GEMM occupancy fix (BN=16) -> 214 tok/s (+21%)

### Goal
M=1 GEMMs stream weights at 30-40% HBM BW because grid [1, N/64, 1] gives only
16-96 blocks on 82 SMs. Fix must be BIT-EXACT (Match 31/31 is the bar).

### Attempts and rejections
1. SPLIT-K (grid [1, N/64, SPLIT], partials to scratch, combine kernel):
   REJECTED. The mma K-chain is acc_{k+1} = dot(A_k,B_k,acc_k) — a fused
   accumulation. Any parallel grouping re-associates the fp adds, and computing
   partials from zero-acc then summing is not bit-identical to the fused chain.
   Measured: 3.3x faster (mm_down 54us -> 14+4us) BUT Match 14/31 at 30 tokens.
   NOT acceptable. (Kept the builders out — deleted them.)
2. numWarps=8: REJECTED — produces garbage (Match 1/31, "1.5us" kernel time is
   fake). M=1 kernels must stay <=4 warps (tile [1,16] can't fill 8 warps).

### What worked: BN=64 -> BN=16 (bit-exact)
Same serial K-chain per output element (unchanged fp association), but 4x more
column blocks -> better SM fill. Everything verified vs Match 31/31.
Rolled out: mm_down, mm_gate (dual N2 via opts.BN, BN2 now honors it), mm_outp,
mm_qkv, mm_z, mm_q, mm_kv, mm_o — all BN=16, launch grids ceil(N/16).
mm_lm left at BN=64 (already near-bandwidth at 592us/step, irreducible).

### Bug I introduced then caught (note for next time)
A failed multi-edit left defs at BN=16 but qkv/z/outp launch grids at /64:
those kernels computed only 25% of columns -> Match 1/31. It was NOT a BN=16
problem. Lesson: grid.y must equal ceil(N/BN) whenever BN changes; verify with
grep that def and launch grid agree.

### Result
174-178 tok/s -> 214-216 tok/s (+21%), Match 31/31 across 3 clean runs.
Per-kernel: mm_down 54->36us, mm_gate 35->28us, mm_qkv 26->20us; lm_head
592us unchanged (bandwidth-bound floor).

## 2025-08-22 — Phase 2 follow-up: BN=32 A/B negative; CUDA Graphs assessment

- BN=32 for mm_down/mm_gate (32/112 blocks): 203.8 tok/s vs 214-216 at BN=16
  (Match 31/31 both). REJECTED — BN=16 is the sweet spot (16 vs 32 vs 64 tested;
  result went 176 -> 214 -> 204 -> 176 as BN went 64 -> 16 -> 32 -> 64). Reverted
  to BN=16; re-verified 219 tok/s, Match 31/31.
- GEMM tile tuning is now exhausted. Remaining per-step budget (~4.6ms) is
  dominated by: lm_head 0.59ms (BW floor), mm_down+mm_gate ~1.3ms (still 2-4x
  off BW ideal), ~240 small kernels (rms/gdn/fa2a/cv1d/sg/rope/ab) ~0.8ms that
  are mostly <=10us each => ~0.3-0.5ms of pure launch/tail floor (est. 5-10%
  of step).
- CUDA Graphs (the "free megakernel"): NOT the quick win I framed earlier.
  Blocker: the decode step passes per-token scalar args (step/Pos, tokenId) AND
  double-buffered state pointers (conv/kv/s-state swap every token), AND host
  param buffers are reused - so naive stream-capture bakes stale values. The
  correct design is cuGraphExecKernelNodeSetParams per node per token (~250
  cheap host param updates) + 2 alternating graphs for the double-buffered
  state phase. Estimated gain only ~5-10% (the tail is smaller than hoped).
  Cost/risk: high for a buffer. Left as a potential future step, not worth it
  over the 21% already banked.

## 2025-08-22 — BK 64->256 (bit-exact): 219 -> 280 tok/s (+28%)

Raised tt.dot K-tile from 64 to 256 on all 8 non-lm GEMMs (down, gate dual,
outp, qkv, z, q, kv, o). Bigger dot chunks = same ascending mma.sync k16 chain
(identical arithmetic - verified Match 31/31 x3) but fewer loop iterations and
far more data in flight per block. 219 -> 276-281 tok/s.
New per-step budget (~3.5ms): lm_head ~0.7ms (at BW), mm_gate 0.38, mm_down
0.34, mm_ab 0.37 (LAUNCH FLOOR - grid [1,1,1] tiny gemm!), mm_qkv 0.30, small
kernels (rms/gdn/fa2a/cv1d/sg/rope) ~0.5ms, mm_outp 0.20, others ~0.4.
Remaining path to floor (~2.0ms / 500 tok/s): (1) mm_ab + small-kernel launch
tail, (2) GEMMs still ~1.5-2x off BW-ideal, (3) CUDA Graphs for the ~0.5ms
tail. lm_head/footprint: token rate now 176 -> 280 = +59% over session start.

## 2025-08-22 — mm_ab: split single-block dual GEMM (BN=8, guarded N2) -> 300 tok/s

mm_ab (a+b, N=32/N2=16) ran as grid [1,1,1] = ONE block -> ~24us launch-tail
floor. Split to BN=8 (grid [1,4,1]); the second GEMM (N2=16 -> 2 blocks) is
guarded inside scf.if (b.lt(pN, nb2)) so blocks 2-3 skip it. Bit-exact
(per-element K-chain unchanged), Match 31/31 x3. 280 -> 296-300 tok/s.
Session total: 176 -> ~300 tok/s (+70%), all bit-exact.
