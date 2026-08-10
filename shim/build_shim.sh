#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
# build_shim.sh — Build libtriton_shim.so (TTIR → PTX via full MLIR pipeline)
#
# NO PYTHON. The shim is a single C++ translation unit (triton_shim.c) linked
# against:
#   - Triton's own .o files (compiled from source — full fusion passes)
#   - MLIR + LLVM static archives (from the Nix triton-llvm package)
#
# Phases (each is a discrete, resumable step):
#   ensure-source   materialize Triton source (tarball, else clone) if absent
#   incgen          mlir-tblgen → dialect .inc headers
#   compile         Triton .cpp → .o (parallel)
#   shim            triton_shim.c → .o
#   link            all objs + MLIR/LLVM archives → libtriton_shim.so
#   verify          run the matmul end-to-end test
#
# Usage:
#   shim/build_shim.sh               full build + link
#   shim/build_shim.sh --verify      build then run the matmul test
#   shim/build_shim.sh --phase=incgen   run a single phase
#   shim/build_shim.sh --just-link     link only (objs must exist)
#   JOBS=16 shim/build_shim.sh       limit parallel compile jobs
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
#   2. Search git history:  git log --oneline -- cmake/llvm-hash.txt
#   3. Find the commit whose hash matches the Nix LLVM:
#        76e268973  →  LLVM ac5dc54d5091  ← THIS IS THE ONE
#
# Triton 76e268973 (v3.6.0) pins to exactly LLVM ac5dc54d5091.
# Newer Triton commits (6ea516a6e+) use PropertyRef and other APIs that
# don't exist in LLVM 23.0.0 — they will NOT compile.
# ════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Persistent build workspace (git-ignored, survives GC/reboots — never /tmp)
BUILD_ROOT="${BUILD_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/third_party}"
mkdir -p "$BUILD_ROOT"

# ─── Paths ───────────────────────────────────────────────────────────
TRITON_LLVM="/nix/store/bmx2wv1b1rkhp7r4wz8zpr6zs92vgps6-triton-llvm-23.0.0-unstable-2026-01-29"
TRITON_SRC="${TRITON_SRC:-$BUILD_ROOT/triton-src}"
OBJ_DIR="${OBJ_DIR:-$BUILD_ROOT/triton-objs}"
SHIM_OBJ="$BUILD_ROOT/triton_shim.o"
OUTPUT="$SCRIPT_DIR/libtriton_shim.so"
TMP_LIST=/tmp/triton_cpp_list.txt
TMP_JOBS=/tmp/triton_compile_jobs.log

# ─── Pinned versions ─────────────────────────────────────────────────
TRITON_COMMIT="76e268973"
EXPECTED_LLVM_HASH="ac5dc54d509169d387fcfd495d71853d81c46484"
# Hermetic source snapshot (git-ignored): pure source, no .git, so the build
# is reproducible and immune to promisor-metadata / partial-clone loss.
TRITON_TARBALL="${TRITON_TARBALL:-$BUILD_ROOT/triton-src-$TRITON_COMMIT.tar.zst}"

# Shared include set for mlir-tblgen and g++ (all backend dirs needed).
INC_FLAGS="-I$TRITON_SRC/include -I$TRITON_SRC/. -I$TRITON_SRC/third_party \
 -I$TRITON_SRC/third_party/nvidia/include -I$TRITON_SRC/third_party/nvidia/lib \
 -I$TRITON_SRC/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM -I$TRITON_LLVM/include"

# ════════════════════════════════════════════════════════════════════════
# ensure-source — make $TRITON_SRC complete at the pinned commit
# ════════════════════════════════════════════════════════════════════════
# Source is "complete" only if it has real content (the old failure mode was
# an empty dir passing the `-d` check and stalling the build).
source_incomplete() {
  [ ! -d "$TRITON_SRC" ] && return 0
  [ ! -f "$TRITON_SRC/cmake/llvm-hash.txt" ] && return 0
  ! find "$TRITON_SRC/lib" -name '*.cpp' 2>/dev/null | grep -q . && return 0
  return 1
}

ensure_source() {
  if ! source_incomplete; then
    echo "✓ Triton source present at $TRITON_SRC"
    return 0
  fi
  if [ -f "$TRITON_TARBALL" ]; then
    echo "Materializing Triton $TRITON_COMMIT from $TRITON_TARBALL..."
    rm -rf "$TRITON_SRC"
    mkdir -p "$TRITON_SRC"
    tar --zstd -xf "$TRITON_TARBALL" -C "$TRITON_SRC"
  else
    echo "Cloning Triton at commit $TRITON_COMMIT (persistent)..."
    git clone --filter=blob:none https://github.com/triton-lang/triton "$TRITON_SRC"
    git -C "$TRITON_SRC" checkout "$TRITON_COMMIT"
  fi

  # Verify the LLVM hash matches (committed source has no .git; hash +
  # completeness is the ground truth).
  local actual
  actual=$(cat "$TRITON_SRC/cmake/llvm-hash.txt" 2>/dev/null || \
    jq -r '.llvm_hash' "$TRITON_SRC/cmake/llvm-info.json" 2>/dev/null || echo "")
  if [ "$actual" != "$EXPECTED_LLVM_HASH" ]; then
    echo "ERROR: Triton LLVM hash mismatch!"
    echo "  Expected: $EXPECTED_LLVM_HASH (matches Nix triton-llvm-23.0.0)"
    echo "  Got:      $actual"
    echo "  Checkout commit $TRITON_COMMIT which pins to the correct LLVM."
    exit 1
  fi
  echo "✓ Triton $TRITON_COMMIT pins to LLVM $EXPECTED_LLVM_HASH (matches Nix package)"
}

# ════════════════════════════════════════════════════════════════════════
# resolve_libs — locate runtime/archive deps from mlir-tblgen's dependencies
# ════════════════════════════════════════════════════════════════════════
MLIR_TBLGEN="$TRITON_LLVM/bin/mlir-tblgen"
LDD_DEPS=$(ldd "$MLIR_TBLGEN" 2>/dev/null)
Z_DIR=""; GLIBC_DIR=""; GCC_DIR=""; XML2_DIR=""; EDIT_DIR=""

resolve_libs() {
  local libname path
  LDD_DEPS=$(ldd "$MLIR_TBLGEN" 2>/dev/null || true)
  for libname in libz.so libc.so libstdc++.so libxml2.so; do
    # grep returns 1 on no match — under set -e a failed $(...) assignment
    # aborts, so the probe must be `|| true`-guarded.
    path=$(echo "$LDD_DEPS" | grep "$libname" | head -1 | sed 's/.*=> //;s/ (.*//' || true)
    [ -z "$path" ] && path=$(find /nix/store -maxdepth 3 -name "$libname" 2>/dev/null | head -1 || true)
    if [ -n "$path" ]; then
      case "$libname" in
        libz.so) Z_DIR=$(dirname "$path") ;;
        libc.so) GLIBC_DIR=$(dirname "$path") ;;
        libstdc++.so) GCC_DIR=$(dirname "$path") ;;
        libxml2.so) XML2_DIR=$(dirname "$path") ;;
      esac
    fi
  done
  # libedit provides el_*/history referenced by MLIR internals; pick a 64-bit
  # ELF one (some nix store copies are 32-bit EM_386 — ld rejects those).
  local cand
  for cand in $(find /nix/store -maxdepth 3 -name "libedit.so" 2>/dev/null || true); do
    if od -An -j 18 -N 2 -t u2 "$cand" 2>/dev/null | grep -q "62"; then  # EM_X86_64
      EDIT_DIR=$(dirname "$cand"); break
    fi
  done
}

# ════════════════════════════════════════════════════════════════════════
# incgen — mlir-tblgen → dialect .inc headers
# ════════════════════════════════════════════════════════════════════════
# Parse each CMakeLists.txt and run mlir-tblgen for every mlir_tablegen()
# with the right -gen-* flag.
generate_inc_from_cmake() {
  local dir="$1" cmake="$1/CMakeLists.txt"
  [ -f "$cmake" ] || return 0
  local td="" output gentype extra_flags tdfile
  while IFS= read -r line; do
    if echo "$line" | grep -q "LLVM_TARGET_DEFINITIONS"; then
      td=$(echo "$line" | sed 's/.*LLVM_TARGET_DEFINITIONS //;s/)$//;s/ //')
    fi
    if echo "$line" | grep -q "mlir_tablegen"; then
      output=$(echo "$line" | sed 's/mlir_tablegen(//;s/ .*//;s/)//')
      gentype=$(echo "$line" | sed 's/.*-gen-//;s/ .*//;s/)//')
      # grep -o returns 1 when nothing matches — must not abort under set -e
      extra_flags="$(echo "$line" | sed 's/mlir_tablegen([^)]*//' | \
        grep -o '\(-[a-z]*=[a-z]*\)' || true)"
      tdfile="$dir/$td"
      [ -f "$tdfile" ] || continue
      "$MLIR_TBLGEN" -gen-"$gentype" $extra_flags "$tdfile" \
        -I "$TRITON_SRC/include" -I "$TRITON_LLVM/include" -I "$TRITON_SRC" \
        -I "$TRITON_SRC/third_party" -I "$dir" \
        -o "$dir/$output" 2>/dev/null || true
    fi
  done < "$cmake"
}

# The CMake parser misses some .inc (e.g. -typedefs-dialect= flags); generate
# the known-missing ones explicitly per dialect.
incgen_explicit() {
  local INC="$TRITON_SRC/include"
  local GPU="$INC/triton/Dialect/TritonGPU/IR"
  local NV="$INC/triton/Dialect/TritonNvidiaGPU/IR"
  local NVGPU="$TRITON_SRC/third_party/nvidia/include/Dialect/NVGPU/IR"
  local NVWS="$TRITON_SRC/third_party/nvidia/include/Dialect/NVWS/IR"
  local COMMON="-I $INC -I $TRITON_LLVM/include -I $TRITON_SRC -I $TRITON_SRC/third_party"

  # TritonGPU
  $MLIR_TBLGEN -gen-typedef-decls -typedefs-dialect=ttg $GPU/TritonGPUTypes.td $COMMON -I $GPU -o $GPU/Types.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-typedef-defs -typedefs-dialect=ttg $GPU/TritonGPUTypes.td $COMMON -I $GPU -o $GPU/Types.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-dialect-decls -dialect=ttg $GPU/TritonGPUDialect.td $COMMON -I $GPU -o $GPU/Dialect.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-dialect-defs -dialect=ttg $GPU/TritonGPUDialect.td $COMMON -I $GPU -o $GPU/Dialect.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attr-interface-decls $GPU/TritonGPUAttrDefs.td $COMMON -I $GPU -o $GPU/AttrInterfaces.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attr-interface-defs $GPU/TritonGPUAttrDefs.td $COMMON -I $GPU -o $GPU/AttrInterfaces.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attrdef-decls $GPU/TritonGPUAttrDefs.td $COMMON -I $GPU -o $GPU/AttrDefs.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attrdef-defs $GPU/TritonGPUAttrImpls.td $COMMON -I $GPU -o $GPU/AttrDefs.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-enum-decls $GPU/TritonGPUEnums.td $COMMON -I $GPU -o $GPU/OpsEnums.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-enum-defs $GPU/TritonGPUEnums.td $COMMON -I $GPU -o $GPU/OpsEnums.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attrdef-decls $GPU/CGAEncodingAttr.td $COMMON -I $GPU -o $GPU/CGAEncodingAttr.h.inc 2>/dev/null || true

  # TritonNvidiaGPU
  $MLIR_TBLGEN -gen-typedef-decls $NV/TritonNvidiaGPUTypes.td $COMMON -I $NV -o $NV/Types.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-typedef-defs $NV/TritonNvidiaGPUTypes.td $COMMON -I $NV -o $NV/Types.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-dialect-decls -dialect=ttng $NV/TritonNvidiaGPUDialect.td $COMMON -I $NV -o $NV/Dialect.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-dialect-defs -dialect=ttng $NV/TritonNvidiaGPUDialect.td $COMMON -I $NV -o $NV/Dialect.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attrdef-decls $NV/TritonNvidiaGPUAttrDefs.td $COMMON -I $NV -o $NV/TritonNvidiaGPUAttrDefs.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attrdef-defs $NV/TritonNvidiaGPUAttrDefs.td $COMMON -I $NV -o $NV/TritonNvidiaGPUAttrDefs.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-enum-decls $NV/TritonNvidiaGPUAttrDefs.td $COMMON -I $NV -o $NV/OpsEnums.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-enum-defs $NV/TritonNvidiaGPUAttrDefs.td $COMMON -I $NV -o $NV/OpsEnums.cpp.inc 2>/dev/null || true

  # NVGPU (third_party/nvidia — dialect name is "nvg")
  $MLIR_TBLGEN -gen-dialect-decls -dialect=nvg $NVGPU/NVGPUDialect.td $COMMON -I $NVGPU -o $NVGPU/Dialect.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-dialect-defs -dialect=nvg $NVGPU/NVGPUDialect.td $COMMON -I $NVGPU -o $NVGPU/Dialect.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-llvmir-conversions $NVGPU/NVGPUOps.td $COMMON -I $NVGPU -o $NVGPU/OpsConversions.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-op-decls $NVGPU/NVGPUOps.td $COMMON -I $NVGPU -o $NVGPU/Ops.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-op-defs $NVGPU/NVGPUOps.td $COMMON -I $NVGPU -o $NVGPU/Ops.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-enum-decls $NVGPU/NVGPUOps.td $COMMON -I $NVGPU -o $NVGPU/OpsEnums.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-enum-defs $NVGPU/NVGPUOps.td $COMMON -I $NVGPU -o $NVGPU/OpsEnums.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attrdef-decls $NVGPU/NVGPUAttrDefs.td $COMMON -I $NVGPU -o $NVGPU/NVGPUAttrDefs.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attrdef-defs $NVGPU/NVGPUAttrDefs.td $COMMON -I $NVGPU -o $NVGPU/NVGPUAttrDefs.cpp.inc 2>/dev/null || true

  # NVWS (third_party/nvidia — dialect name is "nvws")
  $MLIR_TBLGEN -gen-dialect-decls -dialect=nvws $NVWS/NVWSDialect.td $COMMON -I $NVWS -o $NVWS/Dialect.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-dialect-defs -dialect=nvws $NVWS/NVWSDialect.td $COMMON -I $NVWS -o $NVWS/Dialect.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-op-decls $NVWS/NVWSOps.td $COMMON -I $NVWS -o $NVWS/Ops.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-op-defs $NVWS/NVWSOps.td $COMMON -I $NVWS -o $NVWS/Ops.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-typedef-decls -typedefs-dialect=nvws $NVWS/NVWSTypes.td $COMMON -I $NVWS -o $NVWS/Types.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-typedef-defs -typedefs-dialect=nvws $NVWS/NVWSTypes.td $COMMON -I $NVWS -o $NVWS/Types.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attrdef-decls $NVWS/NVWSAttrDefs.td $COMMON -I $NVWS -o $NVWS/NVWSAttrDefs.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-attrdef-defs $NVWS/NVWSAttrDefs.td $COMMON -I $NVWS -o $NVWS/NVWSAttrDefs.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-enum-decls $NVWS/NVWSAttrDefs.td $COMMON -I $NVWS -o $NVWS/NVWSAttrEnums.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-enum-defs $NVWS/NVWSAttrDefs.td $COMMON -I $NVWS -o $NVWS/NVWSAttrEnums.cpp.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-op-interface-decls $NVWS/NVWSOpInterfaces.td $COMMON -I $NVWS -o $NVWS/NVWSOpInterfaces.h.inc 2>/dev/null || true
  $MLIR_TBLGEN -gen-op-interface-defs $NVWS/NVWSOpInterfaces.td $COMMON -I $NVWS -o $NVWS/NVWSOpInterfaces.cpp.inc 2>/dev/null || true
}

phase_incgen() {
  echo "=== Phase: incgen (mlir-tblgen → .inc) ==="
  local dir
  for dir in $(find "$TRITON_SRC/include" "$TRITON_SRC/lib" "$TRITON_SRC/third_party" \
               -name CMakeLists.txt -exec dirname {} \; 2>/dev/null | sort -u); do
    generate_inc_from_cmake "$dir"
  done
  incgen_explicit

  echo "  Generated $(find "$TRITON_SRC/include" "$TRITON_SRC/third_party" -name '*.inc' | wc -l) .inc files"

  # Disable GEN_PASS_REGISTRATION in Transform Passes.h — they define
  # registerPasses() in the SAME mlir::triton namespace, causing redefinitions
  # when included together. MUST NOT touch the CONVERSION pass headers — their
  # GEN_PASS_REGISTRATION populates the op-conversion patterns (e.g. ttg.warp_id
  # → LLVM) the shim's pipeline relies on; neutering it leaves ops unconverted
  # and translateModuleToLLVMIR fails with "missing LLVMTranslationDialectInterface".
  local f
  while read -r f; do
    sed -i 's|^#define GEN_PASS_REGISTRATION|// #define GEN_PASS_REGISTRATION|' "$f"
  done < <(find "$TRITON_SRC/include/triton/Dialect" \
                "$TRITON_SRC/third_party/nvidia/include/Dialect" \
                -name Passes.h 2>/dev/null)
}

# ════════════════════════════════════════════════════════════════════════
# phase_compile — Triton .cpp → .o (parallel)
# ════════════════════════════════════════════════════════════════════════
phase_compile() {
  echo "=== Phase: compile Triton sources (parallel) ==="
  rm -rf "$OBJ_DIR"
  mkdir -p "$OBJ_DIR"
  local jobs="${JOBS:-$(nproc)}"

  find "$TRITON_SRC/lib" "$TRITON_SRC/third_party/nvidia/lib" \
       "$TRITON_SRC/third_party/f2reduce" \
       -name '*.cpp' -not -name '*test*' 2>/dev/null > "$TMP_LIST"

  cat "$TMP_LIST" | xargs -P "$jobs" -n 1 -I{} bash -c '
    cpp="$1"; src_root="$2"; objdir="$3"; flags="$4"
    objname=$(echo "$cpp" | sed "s|$src_root/||;s|/|_|g;s|\.cpp|.o|")
    if g++ -std=c++17 -fPIC -O2 -fno-rtti -c $flags "$cpp" -o "$objdir/$objname" 2>/dev/null; then
      echo "OK $objname"
    else
      echo "FAIL $(basename "$cpp")"
    fi
  ' _ {} "$TRITON_SRC" "$OBJ_DIR" "$INC_FLAGS" > "$TMP_JOBS" 2>&1

  local count fail
  count=$(grep -c '^OK ' "$TMP_JOBS" || true)
  fail=$(grep -c '^FAIL ' "$TMP_JOBS" || true)
  echo "  Compiled: $count OK, $fail failed"
  if [ "$fail" -gt 0 ]; then
    grep '^FAIL' "$TMP_JOBS" | head -10
    echo "ERROR: $fail files failed to compile — check .inc generation"
    exit 1
  fi
}

# ════════════════════════════════════════════════════════════════════════
# phase_shim — triton_shim.c → .o
# ════════════════════════════════════════════════════════════════════════
phase_shim() {
  echo "=== Phase: compile triton_shim.c ==="
  g++ -std=c++17 -fPIC -O2 -fvisibility=default -fno-rtti \
    $INC_FLAGS \
    -c "$SCRIPT_DIR/triton_shim.c" -o "$SHIM_OBJ"
  echo "  Done"
}

# ════════════════════════════════════════════════════════════════════════
# phase_link — link libtriton_shim.so
# ════════════════════════════════════════════════════════════════════════
phase_link() {
  echo "=== Phase: link libtriton_shim.so ==="
  local triton_objs mlir_archives llvm_archives llvm_libs xml2_lib edit_lib
  triton_objs=$(find "$OBJ_DIR" -name '*.o' | tr '\n' ' ')
  mlir_archives=$(find "$TRITON_LLVM/lib" -name 'libMLIR*.a' | sort)
  llvm_archives=$(find "$TRITON_LLVM/lib" -name 'libLLVM*.a' | sort)
  # LLVM is linked via -l (partial membership), NOT whole-archive: whole-archive
  # pulls in LLVM's line-editor/REPL code referencing libedit symbols no
  # compatible system libedit provides. The working dbg build linked this way.
  llvm_libs=$(for a in $llvm_archives; do echo "-l$(basename "$a" .a | sed 's/^lib//')"; done | tr '\n' ' ')

  xml2_lib=""
  [ -n "$XML2_DIR" ] && xml2_lib="-L$XML2_DIR -lxml2"
  edit_lib=""
  [ -n "$EDIT_DIR" ] && edit_lib="-L$EDIT_DIR -ledit -Wl,-rpath,$EDIT_DIR"

  g++ -shared -o "$OUTPUT" \
    "$SHIM_OBJ" \
    $triton_objs \
    -Wl,--whole-archive $mlir_archives -Wl,--no-whole-archive \
    -L"$TRITON_LLVM/lib" \
    -Wl,--start-group $llvm_libs -Wl,--end-group \
    -L"$Z_DIR" -L"$GLIBC_DIR" -L"$GCC_DIR" \
    $xml2_lib $edit_lib -lz -lpthread -ldl -lrt -lc -lm \
    -Wl,-rpath,"$Z_DIR" -Wl,-rpath,"$GCC_DIR" \
    $([ -n "$XML2_DIR" ] && echo "-Wl,-rpath,$XML2_DIR") \
    -Wl,--allow-multiple-definition \
    2>&1 | tail -5

  echo "  Done"
  ls -lah "$OUTPUT"
  nm -D "$OUTPUT" 2>/dev/null | grep "triton_compile\|triton_free\|triton_get_shared"
}

# ════════════════════════════════════════════════════════════════════════
# phase_verify — run the end-to-end matmul correctness test
# ════════════════════════════════════════════════════════════════════════
phase_verify() {
  echo "=== Phase: verify (matmul end-to-end) ==="
  (cd "$SCRIPT_DIR/.." && timeout 100 bun run tests/test_ttir_matmul_run.ts)
}

# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════
PHASE=""
VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY=1 ;;
    --phase=*) PHASE="${arg#--phase=}" ;;
    --just-link) PHASE="link" ;;
    *) echo "Unknown arg: $arg"; exit 2 ;;
  esac
done

resolve_libs
ensure_source

run_phase() {
  case "$1" in
    incgen) phase_incgen ;;
    compile) phase_compile ;;
    shim) phase_shim ;;
    link) phase_link ;;
    verify) phase_verify ;;
    *) echo "Unknown phase: $1 (incgen|compile|shim|link|verify)"; exit 2 ;;
  esac
}

if [ -n "$PHASE" ]; then
  run_phase "$PHASE"
else
  phase_incgen
  phase_compile
  phase_shim
  phase_link
fi

if [ "$VERIFY" = "1" ]; then
  phase_verify
fi

echo ""
echo "✓ Build complete"
