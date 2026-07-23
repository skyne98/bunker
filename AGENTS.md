# Rules

## NO PYTHON — EVER
Python is forbidden in this project. Not at runtime, not as a build tool, not for one-time generation. Zero Python. If Python is detected anywhere in the pipeline, it must be removed.

## Allowed pipeline: TS → TTIR → C shim → MLIR passes → PTX → GPU

The ONLY allowed path to GPU kernels:

1. **TypeScript DSL** (`src/dsl.ts`) generates TTIR MLIR text (TritonIR dialect with `tt.dot`, `tt.load`, `tt.store`, etc.)
2. **C shim** (`shim/triton_shim.c`) — a single C file linked against Triton's MLIR libraries — takes TTIR text and runs the MLIR pass pipeline (including `accelerate_matmul` for tensor core `mma.sync` lowering)
3. Output is PTX, loaded via `cuModuleLoadData`, launched via `cuLaunchKernel`

## Build process
- `shim/triton_shim.c` is compiled to `shim/libtriton_shim.so` using `shim/build_shim.sh`
- The build script:
  1. Fetches Triton source (via git or tarball)
  2. Runs `mlir-tblgen` (from triton-llvm) to generate dialect `.inc` files
  3. Compiles Triton MLIR source files against triton-llvm headers + generated `.inc` files
  4. Links everything into `libtriton_shim.so`
- No Python anywhere in this process

## What exists
- `src/dsl.ts` — Core DSL: LLVM IR generation, `asm()` inline PTX, struct types, half_to_float, shared memory, memcpy, barriers
- `bench/gemm.ts` — Pure-DSL INT8 matmul benchmarks (2.2 TOPS, no external deps)
- `bench/bench.ts` — Original Q4_K × Q8_1 quantized matmul benchmark
- `shim/triton_shim.c` — C shim for TTIR → PTX compilation
- TTIR generation from the DSL is the next capability to add

## Design principles (active work)

The DSL is being expanded to close the feature gaps for writing real GPU kernels.
These principles are hard constraints on that work — every change must satisfy all of them.

### 1. Target the RTX 3090 (sm_86) first, but never hardcode it
- The development and test GPU is an RTX 3090, compute capability 8.6.
- `sm_86` / `+ptx75` are the default target, but they must be parameters, not literals.
- No `nvptx64-nvidia-cuda`, `sm_86`, `+ptx75`, or `@llvm.nvvm.*` intrinsic may be
  baked into a code path that claims to be portable. NVIDIA-specific code must be
  isolated and clearly labeled, and must be reachable through a backend selector,
  not the default.

### 2. Portable across every Triton backend
- The portable surface is **TTIR** (Triton's MLIR dialect), not NVVM-intrinsic LLVM IR.
  TTIR is backend-agnostic; the shim's pass pipeline selects the backend.
- New kernel features must be expressible in TTIR so they lower through Triton's
  backend-agnostic pass pipeline (`accelerate_matmul`, coalesce, etc.) to either
  PTX (NVIDIA) or AMDGCN/HSACO (AMD) depending on the selected target.
- The NVVM-intrinsic LLVM-IR path in `src/dsl.ts` is a legacy/NVIDIA-only fallback.
  Do NOT extend it for new portable features. New work goes into the TTIR layer
  (`src/ttir.ts` + the `kernel_ttir(fn)` AST→TTIR lift).
- The shim (`shim/triton_shim.c`) must accept a target descriptor (backend, arch, features)
  at compile time. `triton_compile(ttir, num_warps)` remains as a 3090-default
  convenience wrapper, but a target-parameterized entry point is the real API.
- A single DSL kernel must be able to emit PTX on NVIDIA or AMDGCN on AMD based on
  the detected GPU — same kernel source, different backend. Inline-assembly
  builtins (`asm()`) must be target-dispatchable, not NVPTX-only.

### 3. Extremely pleasant and easy to use
- Writing a kernel should feel like writing Triton Python, not assembling SSA by hand.
- The primary authoring surface is the arrow-function lift: write a plain TypeScript
  arrow function with typed params (`ptr<f32>`, `i32`, ...); get a compiled, runnable
  GPU kernel. The user must never manage SSA `%v0` names or pass tensor shapes/types
  to every call.
- The underlying TTIR builder is type-tracking: each value carries its tensor type
  (`tensor<MxNxf16>`), so elementwise ops, reductions, and casts infer result types
  automatically and chain fluently.
- Boilerplate (memory alloc/copy/free, pointer-array packing, module load, launch,
  sync) must be hidden behind helpers. A kernel + a launch + a correctness check
  should be a few lines.
- Errors from the shim (parse failures, pass failures) must surface as actionable
  messages, not `null`/generic throws.

### 4. Feature targets (what "missing kernel features" means)
Concrete capabilities to add to the TTIR authoring layer, all portable:
- `tt.reduce` — sum/max/min/argmin/argmax along an axis
- `tt.dot` — matmul (already in templates; expose in the builder + AST lift)
- `tt.trans` / reshape / broadcast / expand_dims / permute / join / split
- `tt.atomic_*` — atomic_add / max / min / xchg / cas
- Full `arith` set: int/float div, rem, min/max, shifts, bitwise, select, casts
  (trunc/ext/zext/sitofp/fptosi/fpext/fptrunc/bitcast)
- `math` dialect: exp/log/sqrt/rsqrt/sin/cos/tanh/floor/ceil/round
- Control flow: `scf.if` / `scf.for` (with yield) / `scf.while` — needed for
  K-loop matmul, conditional stores, iterative algorithms
- Tiled pointers: `tt.make_tensor_ptr` + `tt.advance` for sliding K-tiles
- `tt.num_programs` (grid dimensions), complementing `tt.get_program_id`
- `gridDim`-equivalent and `while` loops in the AST lift (currently only `for`)

### 5. Non-goals (do not pursue as portable features)
- AMD runtime loading (`libhsa-runtime`, `libamdhip`) — out of scope for now; the
  goal is that the *generated code* is portable, not that this repo ships an AMD
  runtime. The shim's AMDGCN path is design-complete when rebuilt, not a runtime
  requirement today.
- A general-purpose autodiff or graph-mode system.
- Replacing the existing working 33 TFLOPS INT8 kernels — extend, don't regress.
