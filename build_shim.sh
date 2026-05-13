#!/usr/bin/env bash
# Build libtriton_shim.so from CMake-built objects + shim
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRITON_LLVM="/nix/store/9k8wp0i929q44248fjgd55x80jjawcfv-triton-llvm-22.0.0-triton"
TRITON_LLVM_DEV=""
TRITON_SRC="/tmp/triton-src"
BUILD_DIR="/tmp/triton-cmake-build"
LLVM_LIB="$TRITON_LLVM/lib"
OUTPUT="$SCRIPT_DIR/libtriton_shim.so"

echo "=== Build libtriton_shim.so ==="

# Get library paths from mlir-tblgen's ldd
MLIR_TBLGEN="$TRITON_LLVM/bin/mlir-tblgen"
LDD_DEPS=$(ldd "$MLIR_TBLGEN" 2>/dev/null)
Z_DIR=$(echo "$LDD_DEPS" | grep "libz.so" | head -1 | sed 's/.*=> //;s/ (.*//' | xargs dirname)
GLIBC_DIR=$(echo "$LDD_DEPS" | grep "libc.so" | head -1 | sed 's/.*=> //;s/ (.*//' | xargs dirname)
GCC_DIR=$(echo "$LDD_DEPS" | grep "libstdc++.so" | head -1 | sed 's/.*=> //;s/ (.*//' | xargs dirname)
XML2_DIR=$(echo "$LDD_DEPS" | grep "libxml2" | head -1 | sed 's/.*=> //;s/ (.*//' | xargs dirname)
echo "XML2_DIR=$XML2_DIR"

# Compile the shim as C++
echo "Compiling triton_shim.c..."
g++ -std=c++17 -fPIC -O2 \
  -fvisibility=default \
  -I$TRITON_SRC/include \
  -I$TRITON_SRC/. \
  -I$TRITON_LLVM/include \
  -I$TRITON_LLVM_DEV/include \
  -I$TRITON_SRC/third_party \
  -I$BUILD_DIR/include \
  -I$BUILD_DIR/third_party \
  -c "$SCRIPT_DIR/triton_shim.c" \
  -o /tmp/triton_shim.o 2>&1 | tail -5
echo "  Done"

# Find all triton .o files from the CMake build
cd "$BUILD_DIR"
TRITON_OBJS=$(find . -name "*.o" -not -path "*/test/*" -not -path "*/bin/*" | tr '\n' ' ')

# MLIR/LLVM static archives
MLIR_ARCHIVES=$(find "$LLVM_LIB" -name "libMLIR*.a" | sort | tr '\n' ' ')
LLVM_ARCHIVES=$(find "$LLVM_LIB" -name "libLLVM*.a" | sort | tr '\n' ' ')

echo "Linking..."
XML2_LIB=""
[ -n "$XML2_DIR" ] && XML2_LIB="-L$XML2_DIR -lxml2"

g++ -shared -o "$OUTPUT" \
  /tmp/triton_shim.o \
  $TRITON_OBJS \
  -Wl,--whole-archive \
  $MLIR_ARCHIVES $LLVM_ARCHIVES \
  -Wl,--no-whole-archive \
  -L"$Z_DIR" -L"$GLIBC_DIR" -L"$GCC_DIR" \
  $XML2_LIB -lz -lpthread -ldl -lrt -lc \
  -Wl,--gc-sections \
  2>&1 | tail -5

echo ""
echo "=== Done ==="
ls -lah "$OUTPUT"
echo ""
nm -D "$OUTPUT" 2>/dev/null | grep "triton_compile" || echo "(checking symbols...)"
nm -D "$OUTPUT" 2>/dev/null | grep "triton_compile"