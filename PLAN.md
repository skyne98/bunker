# Build libtriton_shim.so — TTIR → PTX via Triton MLIR (zero Python)

## Goal
A single C shim (`triton_shim.c`) that takes TTIR MLIR text and returns PTX,
using Triton's MLIR passes (accelerate_matmul → mma.sync tensor cores).
Link it against triton-llvm + Triton source. No Python anywhere.

## Prerequisites (already satisfied)
- [x] triton-llvm available at `/nix/store/9nijn...triton-llvm-22.0.0` (headers, static libs, mlir-tblgen)
- [x] Triton source cloned at `/tmp/triton-src` (v3.6.0)
- [x] `mlir-tblgen` available for generating `.inc` headers

## Build steps

### Step 1: Generate `.inc` headers with mlir-tblgen
- [ ] Run `mlir-tblgen` on Triton's `.td` files to produce `Dialect.h.inc`, `Ops.h.inc`, etc.
- [ ] Output to a build directory (`/tmp/triton-build/include/`)

### Step 2: Write triton_shim.c
- [ ] C ABI: `triton_compile(const char* ttir, int num_warps) → const char* ptx`
- [ ] Initializes MLIR context, registers Triton dialects
- [ ] Parses TTIR string
- [ ] Runs pass pipeline: make_ttir → make_ttgir (accelerate_matmul) → make_llir
- [ ] Emits PTX via LLVM NVPTX backend (same as `llvmIRtoPTX` in `dsl.ts`)
- [ ] `triton_free(const char*)` cleanup

### Step 3: Compile shim
- [ ] Compile Triton MLIR source files with generated `.inc` headers
- [ ] Compile `triton_shim.c` 
- [ ] Link against triton-llvm static libs (`-lMLIR -lLLVM`)
- [ ] Output: `libtriton_shim.so`

### Step 4: Test
- [ ] Write minimal TTIR matmul string
- [ ] Call `triton_compile()` from Bun via `dlopen`
- [ ] Verify PTX contains `mma.sync`
- [ ] Load PTX with `cuModuleLoadData`, run on GPU

### Step 5: DSL integration
- [ ] Add TTIR generation to `dsl.ts` (emit `tt.dot`, `tt.load`, `tt.store`)
- [ ] Full pipeline: TypeScript → TTIR → C shim → PTX → GPU

## Success criteria
- `libtriton_shim.so` compiles standalone, no Python deps
- `triton_compile()` returns valid PTX with `mma.sync` instructions
- Tensor core matmul runs on RTX 3090 at >4 TOPS
- Entirely TypeScript + C, zero Python
