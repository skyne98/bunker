{ pkgs ? import <nixpkgs> {} }:

let
  llvm-src = builtins.fetchTarball {
    url = "https://github.com/triton-lang/llvm-project/archive/f6ded0be897e2878612dd903f7e8bb85448269e5.tar.gz";
  };
in pkgs.stdenv.mkDerivation {
  pname = "triton-llvm";
  version = "22.0.0-triton";

  src = llvm-src;
  sourceRoot = "source/llvm";

  nativeBuildInputs = with pkgs; [ cmake ninja python3 ];
  buildInputs = with pkgs; [ zlib libxml2 ncurses ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DLLVM_ENABLE_PROJECTS=mlir;llvm"
    "-DLLVM_TARGETS_TO_BUILD=NVPTX;X86"
    "-DMLIR_ENABLE_CUDA_RUNNER=OFF"
    "-DMLIR_ENABLE_ROCM_RUNNER=OFF"
    "-DLLVM_ENABLE_ASSERTIONS=OFF"
    "-DLLVM_ENABLE_TERMINFO=OFF"
    "-DLLVM_ENABLE_ZLIB=FORCE_ON"
    "-DLLVM_PARALLEL_LINK_JOBS=16"
    "-DLLVM_PARALLEL_COMPILE_JOBS=16"
  ];

  enableParallelBuilding = true;



  meta = {
    description = "Triton-forked LLVM+MLIR";
    platforms = pkgs.lib.platforms.linux;
  };
}
