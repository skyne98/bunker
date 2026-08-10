#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
# build_shim.sh — Build libtriton_shim.so (TTIR → PTX via full MLIR pipeline)
#
# NO PYTHON. The shim is a single C++ translation unit (triton_shim.c) linked
# against:
#   - Triton's own .o files (compiled from source — full fusion passes)
#   - MLIR + LLVM static archives (from the Nix triton-llvm package)
#
# ════════════════════════════════════════════════════════════════════════
# VERSION MATCHING — THE CRITICAL STEP
# ════════════════════════════════════════════════════════════════════════
#
# Triton and LLVM do NOT have stable APIs. You CANNOT compile an arbitrary
# Triton commit against an arbitrary LLVM — they must match exactly.
#
# The Nix package provides:
#   triton-llvm-23.0.0-unstable-2026-01-29
#     → LLVM commit ac5dc54d509169d387fcfd495d71853d81c46484 (2026-01-29)
#
# To find the matching Triton commit:
#   1. Look at cmake/llvm-hash.txt (or cmake/llvm-info.json) in the Triton repo
#   2. Search git history:
#        git log --oneline -- cmake/llvm-hash.txt
#   3. Find the commit whose hash matches the Nix LLVM:
#        76e268973  →  LLVM ac5dc54d5091  ← THIS IS THE ONE
#
# Triton 76e268973 (v3.6.0) pins to exactly LLVM ac5dc54d5091.
# Newer Triton commits (6ea516a6e+) use PropertyRef and other APIs that
# don't exist in LLVM 23.0.0 — they will NOT compile.
#
# ════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Persistent build workspace (git-ignored, survives GC/reboots — never /tmp)
BUILD_ROOT="${BUILD_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/third_party}"

# ─── Paths ───────────────────────────────────────────────────────────
TRITON_LLVM="/nix/store/bmx2wv1b1rkhp7r4wz8zpr6zs92vgps6-triton-llvm-23.0.0-unstable-2026-01-29"
TRITON_SRC="${TRITON_SRC:-$BUILD_ROOT/triton-src}"
OBJ_DIR="${OBJ_DIR:-$BUILD_ROOT/triton-objs}"
SHIM_OBJ="$BUILD_ROOT/triton_shim.o"
OUTPUT="$SCRIPT_DIR/libtriton_shim.so"
mkdir -p "$BUILD_ROOT"

# ─── Verify Triton source is at the correct commit ───────────────────
TRITON_COMMIT="76e268973"
EXPECTED_LLVM_HASH="ac5dc54d509169d387fcfd495d71853d81c46484"

if [ ! -d "$TRITON_SRC" ]; then
  echo "Cloning Triton at commit $TRITON_COMMIT (persistent)..."
  git clone --filter=blob:none https://github.com/triton-lang/triton "$TRITON_SRC"
  cd "$TRITON_SRC"
  git checkout "$TRITON_COMMIT"
else
  cd "$TRITON_SRC"
  CURRENT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  if [ "$CURRENT_COMMIT" != "$TRITON_COMMIT" ]; then
    echo "WARNING: Triton source is at $CURRENT_COMMIT, expected $TRITON_COMMIT"
    echo "  Run: cd $TRITON_SRC && git checkout $TRITON_COMMIT"
    exit 1
  fi
fi

# Verify the LLVM hash matches
ACTUAL_LLVM_HASH=$(cat "$TRITON_SRC/cmake/llvm-hash.txt" 2>/dev/null || \
  jq -r '.llvm_hash' "$TRITON_SRC/cmake/llvm-info.json" 2>/dev/null || echo "")
if [ "$ACTUAL_LLVM_HASH" != "$EXPECTED_LLVM_HASH" ]; then
  echo "ERROR: Triton LLVM hash mismatch!"
  echo "  Expected: $EXPECTED_LLVM_HASH (matches Nix triton-llvm-23.0.0)"
  echo "  Got:      $ACTUAL_LLVM_HASH"
  echo "  Checkout commit $TRITON_COMMIT which pins to the correct LLVM."
  exit 1
fi
echo "✓ Triton $TRITON_COMMIT pins to LLVM $EXPECTED_LLVM_HASH (matches Nix package)"

# ─── Library paths (from mlir-tblgen's dependencies) ─────────────────
MLIR_TBLGEN="$TRITON_LLVM/bin/mlir-tblgen"
LDD_DEPS=$(ldd "$MLIR_TBLGEN" 2>/dev/null)
get_lib_dir() {
  local libname="$1"
  local path
  # Try ldd first (runtime deps)
  path=$(echo "$LDD_DEPS" | grep "$libname" | head -1 | sed 's/.*=> //;s/ (.*//')
  # Fallback: search nix store
  [ -z "$path" ] && path=$(find /nix/store -maxdepth 3 -name "$libname" 2>/dev/null | head -1)
  [ -n "$path" ] && dirname "$path"
}
Z_DIR=$(get_lib_dir "libz.so")
GLIBC_DIR=$(get_lib_dir "libc.so")
GCC_DIR=$(get_lib_dir "libstdc++.so")
XML2_DIR=$(get_lib_dir "libxml2.so")

# ─── Step 1: Generate .inc files via mlir-tblgen ─────────────────────
echo ""
echo "=== Step 1: Generate dialect .inc files ==="

# A helper that parses a CMakeLists.txt and runs mlir-tblgen for each
# mlir_tablegen() call with the correct -gen-* flag and extra args.
generate_inc_from_cmake() {
  local dir="$1"
  local cmake="$dir/CMakeLists.txt"
  [ -f "$cmake" ] || return 0
  local td="" output gentype extra_flags
  while IFS= read -r line; do
    if echo "$line" | grep -q "LLVM_TARGET_DEFINITIONS"; then
      td=$(echo "$line" | sed 's/.*LLVM_TARGET_DEFINITIONS //;s/)$//;s/ //')
    fi
    if echo "$line" | grep -q "mlir_tablegen"; then
      output=$(echo "$line" | sed 's/mlir_tablegen(//;s/ .*//;s/)//')
      gentype=$(echo "$line" | sed 's/.*-gen-//;s/ .*//;s/)//')
      # guard: grep -o below returns 1 if no -flag=value present (normal)
      # Capture extra flags like -dialect=ttg or -typedefs-dialect=ttg
      # grep -o returns 1 when nothing matches — must not abort under set -e
      extra_flags="$(echo "$line" | sed 's/mlir_tablegen([^)]*//' | \
        grep -o '\(-[a-z]*=[a-z]*\)' || true)"
      local tdfile="$dir/$td"
      [ -f "$tdfile" ] || continue
      "$MLIR_TBLGEN" -gen-"$gentype" $extra_flags "$tdfile" \
        -I "$TRITON_SRC/include" \
        -I "$TRITON_LLVM/include" \
        -I "$TRITON_SRC" \
        -I "$TRITON_SRC/third_party" \
        -I "$dir" \
        -o "$dir/$output" 2>/dev/null || true
    fi
  done < "$cmake"
}

# Generate for ALL directories containing .td files
for dir in $(find "$TRITON_SRC/include" "$TRITON_SRC/lib" \
             "$TRITON_SRC/third_party" \
             -name CMakeLists.txt -exec dirname {} \; 2>/dev/null | sort -u); do
  generate_inc_from_cmake "$dir"
done

# The automatic parser misses some .inc files because CMakeLists flags like
# -typedefs-dialect= or relative include paths aren't captured perfectly.
# Generate the known-missing ones explicitly.
MLIR="$MLIR_TBLGEN"
INC="$TRITON_SRC/include"
GPU="$INC/triton/Dialect/TritonGPU/IR"
NV="$INC/triton/Dialect/TritonNvidiaGPU/IR"
NVGPU="$TRITON_SRC/third_party/nvidia/include/Dialect/NVGPU/IR"
NVWS="$TRITON_SRC/third_party/nvidia/include/Dialect/NVWS/IR"
COMMON="-I $INC -I $TRITON_LLVM/include -I $TRITON_SRC -I $TRITON_SRC/third_party"

# TritonGPU
$MLIR -gen-typedef-decls -typedefs-dialect=ttg $GPU/TritonGPUTypes.td $COMMON -I $GPU -o $GPU/Types.h.inc 2>/dev/null || true
$MLIR -gen-typedef-defs -typedefs-dialect=ttg $GPU/TritonGPUTypes.td $COMMON -I $GPU -o $GPU/Types.cpp.inc 2>/dev/null || true
$MLIR -gen-dialect-decls -dialect=ttg $GPU/TritonGPUDialect.td $COMMON -I $GPU -o $GPU/Dialect.h.inc 2>/dev/null || true
$MLIR -gen-dialect-defs -dialect=ttg $GPU/TritonGPUDialect.td $COMMON -I $GPU -o $GPU/Dialect.cpp.inc 2>/dev/null || true
$MLIR -gen-attr-interface-decls $GPU/TritonGPUAttrDefs.td $COMMON -I $GPU -o $GPU/AttrInterfaces.h.inc 2>/dev/null || true
$MLIR -gen-attr-interface-defs $GPU/TritonGPUAttrDefs.td $COMMON -I $GPU -o $GPU/AttrInterfaces.cpp.inc 2>/dev/null || true
$MLIR -gen-attrdef-decls $GPU/TritonGPUAttrDefs.td $COMMON -I $GPU -o $GPU/AttrDefs.h.inc 2>/dev/null || true
$MLIR -gen-attrdef-defs $GPU/TritonGPUAttrImpls.td $COMMON -I $GPU -o $GPU/AttrDefs.cpp.inc 2>/dev/null || true
$MLIR -gen-enum-decls $GPU/TritonGPUEnums.td $COMMON -I $GPU -o $GPU/OpsEnums.h.inc 2>/dev/null || true
$MLIR -gen-enum-defs $GPU/TritonGPUEnums.td $COMMON -I $GPU -o $GPU/OpsEnums.cpp.inc 2>/dev/null || true
$MLIR -gen-attrdef-decls $GPU/CGAEncodingAttr.td $COMMON -I $GPU -o $GPU/CGAEncodingAttr.h.inc 2>/dev/null || true

# TritonNvidiaGPU
$MLIR -gen-typedef-decls $NV/TritonNvidiaGPUTypes.td $COMMON -I $NV -o $NV/Types.h.inc 2>/dev/null || true
$MLIR -gen-typedef-defs $NV/TritonNvidiaGPUTypes.td $COMMON -I $NV -o $NV/Types.cpp.inc 2>/dev/null || true
$MLIR -gen-dialect-decls -dialect=ttng $NV/TritonNvidiaGPUDialect.td $COMMON -I $NV -o $NV/Dialect.h.inc 2>/dev/null || true
$MLIR -gen-dialect-defs -dialect=ttng $NV/TritonNvidiaGPUDialect.td $COMMON -I $NV -o $NV/Dialect.cpp.inc 2>/dev/null || true
$MLIR -gen-attrdef-decls $NV/TritonNvidiaGPUAttrDefs.td $COMMON -I $NV -o $NV/TritonNvidiaGPUAttrDefs.h.inc 2>/dev/null || true
$MLIR -gen-attrdef-defs $NV/TritonNvidiaGPUAttrDefs.td $COMMON -I $NV -o $NV/TritonNvidiaGPUAttrDefs.cpp.inc 2>/dev/null || true
$MLIR -gen-enum-decls $NV/TritonNvidiaGPUAttrDefs.td $COMMON -I $NV -o $NV/OpsEnums.h.inc 2>/dev/null || true
$MLIR -gen-enum-defs $NV/TritonNvidiaGPUAttrDefs.td $COMMON -I $NV -o $NV/OpsEnums.cpp.inc 2>/dev/null || true

# NVGPU (third_party/nvidia — dialect name is "nvg")
$MLIR -gen-dialect-decls -dialect=nvg $NVGPU/NVGPUDialect.td $COMMON -I $NVGPU -o $NVGPU/Dialect.h.inc 2>/dev/null || true
$MLIR -gen-dialect-defs -dialect=nvg $NVGPU/NVGPUDialect.td $COMMON -I $NVGPU -o $NVGPU/Dialect.cpp.inc 2>/dev/null || true
$MLIR -gen-llvmir-conversions $NVGPU/NVGPUOps.td $COMMON -I $NVGPU -o $NVGPU/OpsConversions.inc 2>/dev/null || true
$MLIR -gen-op-decls $NVGPU/NVGPUOps.td $COMMON -I $NVGPU -o $NVGPU/Ops.h.inc 2>/dev/null || true
$MLIR -gen-op-defs $NVGPU/NVGPUOps.td $COMMON -I $NVGPU -o $NVGPU/Ops.cpp.inc 2>/dev/null || true
$MLIR -gen-enum-decls $NVGPU/NVGPUOps.td $COMMON -I $NVGPU -o $NVGPU/OpsEnums.h.inc 2>/dev/null || true
$MLIR -gen-enum-defs $NVGPU/NVGPUOps.td $COMMON -I $NVGPU -o $NVGPU/OpsEnums.cpp.inc 2>/dev/null || true
$MLIR -gen-attrdef-decls $NVGPU/NVGPUAttrDefs.td $COMMON -I $NVGPU -o $NVGPU/NVGPUAttrDefs.h.inc 2>/dev/null || true
$MLIR -gen-attrdef-defs $NVGPU/NVGPUAttrDefs.td $COMMON -I $NVGPU -o $NVGPU/NVGPUAttrDefs.cpp.inc 2>/dev/null || true

# NVWS (third_party/nvidia — dialect name is "nvws")
$MLIR -gen-dialect-decls -dialect=nvws $NVWS/NVWSDialect.td $COMMON -I $NVWS -o $NVWS/Dialect.h.inc 2>/dev/null || true
$MLIR -gen-dialect-defs -dialect=nvws $NVWS/NVWSDialect.td $COMMON -I $NVWS -o $NVWS/Dialect.cpp.inc 2>/dev/null || true
$MLIR -gen-op-decls $NVWS/NVWSOps.td $COMMON -I $NVWS -o $NVWS/Ops.h.inc 2>/dev/null || true
$MLIR -gen-op-defs $NVWS/NVWSOps.td $COMMON -I $NVWS -o $NVWS/Ops.cpp.inc 2>/dev/null || true
$MLIR -gen-typedef-decls -typedefs-dialect=nvws $NVWS/NVWSTypes.td $COMMON -I $NVWS -o $NVWS/Types.h.inc 2>/dev/null || true
$MLIR -gen-typedef-defs -typedefs-dialect=nvws $NVWS/NVWSTypes.td $COMMON -I $NVWS -o $NVWS/Types.cpp.inc 2>/dev/null || true
$MLIR -gen-attrdef-decls $NVWS/NVWSAttrDefs.td $COMMON -I $NVWS -o $NVWS/NVWSAttrDefs.h.inc 2>/dev/null || true
$MLIR -gen-attrdef-defs $NVWS/NVWSAttrDefs.td $COMMON -I $NVWS -o $NVWS/NVWSAttrDefs.cpp.inc 2>/dev/null || true
$MLIR -gen-enum-decls $NVWS/NVWSAttrDefs.td $COMMON -I $NVWS -o $NVWS/NVWSAttrEnums.h.inc 2>/dev/null || true
$MLIR -gen-enum-defs $NVWS/NVWSAttrDefs.td $COMMON -I $NVWS -o $NVWS/NVWSAttrEnums.cpp.inc 2>/dev/null || true
$MLIR -gen-op-interface-decls $NVWS/NVWSOpInterfaces.td $COMMON -I $NVWS -o $NVWS/NVWSOpInterfaces.h.inc 2>/dev/null || true
$MLIR -gen-op-interface-defs $NVWS/NVWSOpInterfaces.td $COMMON -I $NVWS -o $NVWS/NVWSOpInterfaces.cpp.inc 2>/dev/null || true

INC_COUNT=$(find "$TRITON_SRC/include" "$TRITON_SRC/third_party" -name '*.inc' | wc -l)
echo "  Generated $INC_COUNT .inc files"

# Fix: disable GEN_PASS_REGISTRATION in Transform Passes.h — they define
# registerPasses() in the SAME mlir::triton namespace producing redefinitions
# when included together. IMPORTANT: this must NOT touch the CONVERSION pass
# headers (TritonGPUToLLVM etc.) — their GEN_PASS_REGISTRATION populates the
# op-conversion patterns (e.g. ttg.warp_id -> LLVM) the shim's explicit
# pipeline relies on; neutering it leaves ops unconverted and the final
# translateModuleToLLVMIR fails with "missing LLVMTranslationDialectInterface".
find "$TRITON_SRC/include/triton/Dialect" "$TRITON_SRC/third_party/nvidia/include/Dialect" \
  -name "Passes.h" | while read f; do
  sed -i 's/^#define GEN_PASS_REGISTRATION/\/\/ #define GEN_PASS_REGISTRATION/' "$f"
done

# ─── Step 2: Compile all Triton .cpp files (PARALLEL) ─────────────
echo ""
echo "=== Step 2: Compile Triton source (.cpp → .o) [parallel] ==="

INC_FLAGS="-I$TRITON_SRC/include -I$TRITON_SRC/. -I$TRITON_SRC/third_party -I$TRITON_SRC/third_party/nvidia/include -I$TRITON_SRC/third_party/nvidia/lib -I$TRITON_SRC/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM -I$TRITON_LLVM/include"

rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"

# Compile up to $JOBS files in parallel (default: all cores).
JOBS="${JOBS:-$(nproc)}"
find "$TRITON_SRC/lib" "$TRITON_SRC/third_party/nvidia/lib" "$TRITON_SRC/third_party/f2reduce" \
  -name "*.cpp" -not -name "*test*" 2>/dev/null > /tmp/triton_cpp_list.txt

# Parallel compile with xargs -P
cat /tmp/triton_cpp_list.txt | \
xargs -P "$JOBS" -n 1 -I{} bash -c '
  cpp="$1"
  src_root="$2"
  objdir="$3"
  flags="$4"
  objname=$(echo "$cpp" | sed "s|$src_root/||;s|/|_|g;s|\.cpp|.o|")
  if g++ -std=c++17 -fPIC -O2 -fno-rtti -c $flags "$cpp" -o "$objdir/$objname" 2>/dev/null; then
    echo "OK $objname"
  else
    echo "FAIL $(basename "$cpp")"
  fi
' _ {} "$TRITON_SRC" "$OBJ_DIR" "$INC_FLAGS" > /tmp/triton_compile_jobs.log 2>&1

count=$(grep -c '^OK ' /tmp/triton_compile_jobs.log || true)
fail=$(grep -c '^FAIL ' /tmp/triton_compile_jobs.log || true)
echo "  Compiled: $count OK, $fail failed"
# Show first few failures for diagnostics
if [ "$fail" -gt 0 ]; then
  grep '^FAIL' /tmp/triton_compile_jobs.log | head -10
  echo "ERROR: $fail files failed to compile — check .inc generation"
  exit 1
fi

# ─── Step 3: Compile the shim itself ──────────────────────────────────
echo ""
echo "=== Step 3: Compile triton_shim.c ==="
g++ -std=c++17 -fPIC -O2 -fvisibility=default -fno-rtti \
  -I$TRITON_SRC/include \
  -I$TRITON_SRC/. \
  -I$TRITON_SRC/third_party \
  -I$TRITON_SRC/third_party/nvidia/include \
  -I$TRITON_SRC/third_party/nvidia/lib \
  -I$TRITON_SRC/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM \
  -I$TRITON_LLVM/include \
  -c "$SCRIPT_DIR/triton_shim.c" \
  -o "$SHIM_OBJ"
echo "  Done"

# ─── Step 4: Link everything ─────────────────────────────────────────
echo ""
echo "=== Step 4: Link libtriton_shim.so ==="

TRITON_OBJS=$(find "$OBJ_DIR" -name "*.o" | tr '\n' ' ')
MLIR_ARCHIVES=$(find "$TRITON_LLVM/lib" -name "libMLIR*.a" | sort)
LLVM_ARCHIVES=$(find "$TRITON_LLVM/lib" -name "libLLVM*.a" | sort)
# LLVM is linked via -l (partial membership), NOT whole-archive: whole-archive
# pulls in LLVM's line-editor/REPL code that references libedit symbols
# (history, el_*) which no compatible system libedit provides here. The
# working dbg build (May 13) linked LLVM this way with no libedit references.
LLVM_LIBS=$(for a in $LLVM_ARCHIVES; do echo "-l$(basename "$a" .a | sed 's/^lib//')"; done | tr '\n' ' ')

XML2_LIB=""
[ -n "$XML2_DIR" ] && XML2_LIB="-L$XML2_DIR -lxml2"

# libedit provides el_*/history referenced by MLIR internals; pick a 64-bit
# ELF one (some nix store copies are 32-bit EM_386 — ld rejects those).
EDIT_DIR=""
for cand in $(find /nix/store -maxdepth 3 -name "libedit.so" 2>/dev/null); do
  if od -An -j 18 -N 2 -t u2 "$cand" 2>/dev/null | grep -q "62"; then  # EM_X86_64 = 62
    EDIT_DIR=$(dirname "$cand"); break
  fi
  true
done
EDIT_LIB=""
[ -n "$EDIT_DIR" ] && EDIT_LIB="-L$EDIT_DIR -ledit -Wl,-rpath,$EDIT_DIR"

g++ -shared -o "$OUTPUT" \
  "$SHIM_OBJ" \
  $TRITON_OBJS \
  -Wl,--whole-archive \
  $MLIR_ARCHIVES \
  -Wl,--no-whole-archive \
  -L"$TRITON_LLVM/lib" \
  -Wl,--start-group $LLVM_LIBS -Wl,--end-group \
  -L"$Z_DIR" -L"$GLIBC_DIR" -L"$GCC_DIR" \
  $XML2_LIB $EDIT_LIB -lz -lpthread -ldl -lrt -lc -lm \
  -Wl,-rpath,"$Z_DIR" -Wl,-rpath,"$GCC_DIR" \
  $([ -n "$XML2_DIR" ] && echo "-Wl,-rpath,$XML2_DIR") \
  -Wl,--allow-multiple-definition \
  2>&1 | tail -5

echo ""
echo "=== Done ==="
ls -lah "$OUTPUT"
echo ""
echo "Exported symbols:"
nm -D "$OUTPUT" 2>/dev/null | grep "triton_compile\|triton_free\|triton_get_shared"
