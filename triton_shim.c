// triton_shim.c — TTIR → PTX via MLIR pass pipeline string
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

// Shared memory size from the last compilation
static int64_t g_last_shmem_size = 0;

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/InitAllPasses.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/TargetParser/Triple.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* triton_compile(const char* ttir_mlir, int num_warps) {
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  mlir::registerAllPasses();
  mlir::triton::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::triton::nvidia_gpu::registerTritonNvidiaGPUPasses();
  mlir::triton::instrument::registerTritonInstrumentPasses();
  mlir::triton::registerTritonToTritonGPUPasses();
  mlir::triton::registerTritonNVIDIAGPUToLLVMPasses();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::triton::registerNVWSTransformsPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect,
                  mlir::triton::gpu::TritonGPUDialect,
                  mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                  mlir::func::FuncDialect,
                  mlir::arith::ArithDialect,
                  mlir::math::MathDialect,
                  mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect,
                   mlir::tensor::TensorDialect>();

  // Register index → LLVM conversion interface
  mlir::index::registerConvertIndexToLLVMInterface(registry);

  // Register LLVM IR translations before MLIRContext creation
  mlir::registerAllToLLVMIRTranslations(registry);

  mlir::MLIRContext ctx(registry);

  auto module = mlir::parseSourceString<mlir::ModuleOp>(ttir_mlir, &ctx);
  if (!module) {
    return strdup("ERROR: Failed to parse TTIR MLIR");
  }

  // Full TTGIR pipeline
  {
    mlir::PassManager pm(&ctx);
    pm.enableVerifier(true);

    // Rewrite tensor pointers (make_tensor_ptr → explicit ptr arithmetic)
    pm.addPass(mlir::triton::createTritonRewriteTensorPointer());
    pm.addPass(mlir::triton::createTritonRewriteTensorDescriptorToPointer());
    pm.addPass(mlir::createCanonicalizerPass());

    // Unroll loops so accelerate_matmul can match every tt.dot
    pm.addPass(mlir::triton::createTritonLoopUnroll());
    pm.addPass(mlir::createCanonicalizerPass());

    // TTIR → TTGIR
    mlir::triton::ConvertTritonToTritonGPUOptions opts;
    opts.target = "cuda:86";
    opts.numWarps = num_warps;
    opts.threadsPerWarp = 32;
    opts.numCTAs = 1;
    pm.addPass(mlir::triton::createConvertTritonToTritonGPU(opts));

    // TTGIR optimizations
    pm.addPass(mlir::triton::gpu::createTritonGPUCoalesce());
    pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());
    pm.addPass(mlir::triton::createTritonLoopAwareCSE());
    pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());

    if (pm.run(module.get()).failed()) {
      return strdup("ERROR: TTGIR pipeline failed");
    }
  }

  // TTGIR → LLVM
  {
    mlir::PassManager pm(&ctx);
    pm.enableVerifier(true);

    // Required preparatory passes
    pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(mlir::triton::gpu::createTritonGPUAllocateWarpGroups());
    pm.addPass(mlir::triton::gpu::createAllocateSharedMemory());
    pm.addPass(mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());

    // Lower scf.for and index types before GPU → LLVM conversion
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());

    // NVIDIA convert-triton-gpu-to-llvm handles all lowering internally
    pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(86, 75));

    // Remaining cleanup passes
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());

    if (pm.run(module.get()).failed()) {
      return strdup("ERROR: LLVM pipeline failed");
    }
  }

  // Extract shared memory size from module attribute
  g_last_shmem_size = 0;
  if (auto shmemAttr = module->getOperation()->getAttrOfType<mlir::IntegerAttr>("ttg.shared"))
    g_last_shmem_size = shmemAttr.getInt();

  // Debug: dump the module after passes
  std::string debugStr;
  llvm::raw_string_ostream debugStream(debugStr);
  module.get()->print(debugStream);
  debugStream.flush();
  // Check if tt.func is still present
  if (debugStr.find("tt.func") != std::string::npos) {
    return strdup(("ERROR: tt.func still present after pipeline:\n" + debugStr).c_str());
  }

  // Translate to LLVM IR
  llvm::LLVMContext llvmCtx;
  auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmCtx);
  if (!llvmModule) {
    return strdup("ERROR: Failed to translate to LLVM IR");
  }

  llvm::Triple targetTriple("nvptx64-nvidia-cuda");
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    return strdup(("ERROR: " + error).c_str());
  }

  auto targetMachine = target->createTargetMachine(
      targetTriple, "sm_86", "+ptx75", llvm::TargetOptions(),
      llvm::Reloc::PIC_, llvm::CodeModel::Small,
      llvm::CodeGenOptLevel::Aggressive);

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  llvm::SmallVector<char, 0> ptxBuf;
  llvm::raw_svector_ostream ptxStream(ptxBuf);
  llvm::legacy::PassManager codegenPM;
  if (targetMachine->addPassesToEmitFile(codegenPM, ptxStream, nullptr,
                                          llvm::CodeGenFileType::AssemblyFile)) {
    return strdup("ERROR: TargetMachine can't emit PTX");
  }
  codegenPM.run(*llvmModule);
  std::string ptxStr(ptxBuf.data(), ptxBuf.size());

  return strdup(ptxStr.c_str());
}

void triton_free(const char* p) {
  free(const_cast<char*>(p));
}

int64_t triton_get_shared_mem_size() {
  return g_last_shmem_size;
}

#ifdef __cplusplus
}
#endif
