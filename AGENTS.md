# Rules

## NO PYTHON — EVER
Python is forbidden in this project. Not at runtime, not as a build tool, not for one-time generation. Zero Python. If Python is detected anywhere in the pipeline, it must be removed.

## Allowed pipeline: TS → TTIR → C shim → MLIR passes → PTX → GPU

The ONLY allowed path to GPU kernels:

1. **TypeScript DSL** (`dsl.ts`) generates TTIR MLIR text (TritonIR dialect with `tt.dot`, `tt.load`, `tt.store`, etc.)
2. **C shim** (`triton_shim.c`) — a single C file linked against Triton's MLIR libraries — takes TTIR text and runs the MLIR pass pipeline (including `accelerate_matmul` for tensor core `mma.sync` lowering)
3. Output is PTX, loaded via `cuModuleLoadData`, launched via `cuLaunchKernel`

## Build process
- `triton_shim.c` is compiled to `libtriton_shim.so` using a build script
- The build script:
  1. Fetches Triton source (via git or tarball)
  2. Runs `mlir-tblgen` (from triton-llvm) to generate dialect `.inc` files
  3. Compiles Triton MLIR source files against triton-llvm headers + generated `.inc` files
  4. Links everything into `libtriton_shim.so`
- No Python anywhere in this process

## What exists
- `dsl.ts` — Core DSL: LLVM IR generation, `asm()` inline PTX, struct types, half_to_float, shared memory, memcpy, barriers
- `gemm.ts` — Pure-DSL INT8 matmul benchmarks (2.2 TOPS, no external deps)
- `bench.ts` — Original Q4_K × Q8_1 quantized matmul benchmark
- `triton_shim.c` — C shim for TTIR → PTX compilation
- TTIR generation from the DSL is the next capability to add
