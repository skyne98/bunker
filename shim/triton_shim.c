// triton_shim.c — TTIR → PTX via MLIR pass pipeline string
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

// Shared memory size from the last compilation
static int64_t g_last_shmem_size = 0;

#include <unordered_set>

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
#include "llvm/Linker/Linker.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/TargetParser/Triple.h"

#ifdef __cplusplus
extern "C" {
#endif

// ════════════════════════════════════════════════════════════════════
// Target descriptor (portable: same TTIR → PTX on NVIDIA or AMDGCN on AMD)
//
//   backend  : "cuda" | "rocm"
//   arch     : compute capability string, e.g. "86" (sm_86) for cuda,
//              or a gfx name e.g. "gfx90a" for rocm
//   features : target features, e.g. "+ptx75" for cuda, "" for rocm
//
// The TTIR text produced by ttir.ts is identical regardless of backend;
// the selected target only affects the shim's MLIR pass pipeline (TTGIR→LLVM)
// and the final ISA emission (nvptx64-nvidia-cuda vs amdgcn-amd-amdhsa).
// ════════════════════════════════════════════════════════════════════

static bool g_targets_initialized = false;
static void initAllTargets() {
  if (g_targets_initialized) return;
  // NVIDIA / NVPTX
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  // AMD / AMDGCN (always available in LLVM; used when backend=="rocm")
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  g_targets_initialized = true;
}

// True when the Triton AMD backend was linked into the shim at build time.
// (The default build links only the NVIDIA backend; rebuilding with
// triton/third_party/amd enables this via -DTRITON_AMD_BACKEND=1.)
#ifndef TRITON_AMD_BACKEND
#define TRITON_AMD_BACKEND 0
#endif

// Path to NVIDIA's libdevice.10.bc (device math: __nv_sqrtf, __nv_logf, ...).
// Defaults to the Triton-vendored copy (ABI-matched to this build's LLVM).
// Override with TRITON_LIBDEVICE env var at runtime.
#ifndef TRITON_LIBDEVICE_PATH
#define TRITON_LIBDEVICE_PATH \
  "/nix/store/5lxy0mn9vj7s6xr97jvdgk2b2vxm743j-python3.14-triton-3.7.0/lib/python3.14/site-packages/triton/backends/nvidia/lib/libdevice.10.bc"
#endif

// Link NVIDIA's libdevice bitcode into the LLVM module so calls to __nv_*
// (sqrt/log/sin/cos/tanh/...) resolve before PTX emission. Mirrors Triton's
// llvm.link_extern_libs exactly (LinkOnlyNeeded + mark internal).
static bool link_libdevice(llvm::Module* mod, std::string& err) {
  const char* path = getenv("TRITON_LIBDEVICE");
  if (!path || !*path) path = TRITON_LIBDEVICE_PATH;
  if (!path || !*path) return true;  // no libdevice configured — leave as-is

  llvm::SMDiagnostic diag;
  auto lib = llvm::parseIRFile(path, diag, mod->getContext());
  if (!lib) {
    err = "Failed to parse libdevice at " + std::string(path);
    return false;
  }
  lib->setTargetTriple(mod->getTargetTriple());
  lib->setDataLayout(mod->getDataLayout());

  std::unordered_set<std::string> externalFns;
  for (llvm::Function& fn : lib->functions())
    if (!fn.isDeclaration()) externalFns.insert(fn.getName().str());

  llvm::Linker linker(*mod);
  if (linker.linkInModule(std::move(lib), llvm::Linker::Flags::LinkOnlyNeeded)) {
    err = "Failed to link libdevice at " + std::string(path);
    return false;
  }
  for (llvm::Function& fn : mod->functions())
    if (externalFns.count(fn.getName().str()))
      fn.setLinkage(llvm::GlobalValue::InternalLinkage);
  return true;
}

const char* triton_compile_targeted(const char* ttir_mlir, int num_warps,
                                     const char* backend, const char* arch,
                                     const char* features) {
  initAllTargets();
  const bool isCuda = (backend == nullptr || std::string(backend) == "cuda");
  const bool isRocm = (backend != nullptr && std::string(backend) == "rocm");
  if (!isCuda && !isRocm)
    return strdup("ERROR: unknown backend (use 'cuda' or 'rocm')");
  if (isRocm && !TRITON_AMD_BACKEND)
    return strdup("ERROR: rocm backend not linked — rebuild the shim with "
                  "triton/third_party/amd (-DTRITON_AMD_BACKEND=1)");
  if (arch == nullptr) arch = isCuda ? "86" : "gfx90a";
  if (features == nullptr) features = isCuda ? "+ptx75" : "";

  mlir::registerAllPasses();

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
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::triton::createTritonRewriteTensorPointer());
    pm.addPass(mlir::triton::createTritonRewriteTensorDescriptorToPointer());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::triton::createTritonCombineOps());
    pm.addPass(mlir::triton::createTritonReorderBroadcast());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());

    // Unroll loops so accelerate_matmul can match every tt.dot
    pm.addPass(mlir::triton::createTritonLoopUnroll());
    pm.addPass(mlir::createCanonicalizerPass());

    // TTIR → TTGIR. target string: "cuda:86" or "hip".
    mlir::triton::ConvertTritonToTritonGPUOptions opts;
    opts.target = isCuda ? ("cuda:" + std::string(arch)) : "hip";
    opts.numWarps = num_warps;
    opts.threadsPerWarp = 32;
    opts.numCTAs = 1;
    pm.addPass(mlir::triton::createConvertTritonToTritonGPU(opts));

    // TTGIR optimizations
    pm.addPass(mlir::triton::gpu::createTritonGPUCoalesce());
    pm.addPass(mlir::triton::gpu::createTritonGPUF32DotTC());
    pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
    pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());
    pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands());
    pm.addPass(mlir::triton::createTritonLoopAwareCSE());
    pm.addPass(mlir::triton::gpu::createTritonGPUFuseNestedLoops());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::triton::createTritonLoopInvariantCodeMotion());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(mlir::triton::gpu::createTritonGPUAssignLatencies());
    pm.addPass(mlir::triton::gpu::createTritonGPUScheduleLoops());
    pm.addPass(mlir::triton::gpu::createTritonGPUPipeline());
    pm.addPass(mlir::triton::gpu::createTritonGPUFuseNestedLoops());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::triton::createTritonLoopInvariantCodeMotion());
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

    // TTGIR → LLVM. The backend-specific convert pass lowers tt.dot/mma etc.
    // NVIDIA: createConvertTritonGPUToLLVMPass(compute_cap, ptx_version).
    // AMD:    the AMD convert pass (registered when TRITON_AMD_BACKEND=1).
    if (isCuda) {
      int cc = std::atoi(arch);
      int ptx = 75;
      // parse "+ptxNN" from features if present
      const char* p = features ? std::strstr(features, "+ptx") : nullptr;
      if (p) ptx = std::atoi(p + 4);
      pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(cc, ptx));
      // NVIDIA warp-id / TMEM ops are intentionally left for the NVGPUToLLVM
      // pass (see TritonGPUToLLVM.cpp: "We handle the warp ID op during
      // NVGPUToLLVM"). Without it ttg.warp_id survives and the final
      // translateModuleToLLVMIR fails with "missing LLVMTranslationDialectInterface".
      pm.addPass(mlir::triton::createConvertNVGPUToLLVM());
    } else {
      // AMD path: the AMD GPU → LLVM conversion pass (requires AMD backend).
      // When TRITON_AMD_BACKEND is enabled, the AMD pass is registered and
      // invoked here. The pass is backend-aware via the module target attr.
      pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(0, 0));
    }

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

  // Sanity check: tt.func must be fully lowered by now (it survives only on
  // a broken pipeline). The full module dump is gated behind TRITON_SHIM_LOG
  // so the hot path never prints the whole IR.
  if (module->getOperation()->walk([&](mlir::triton::FuncOp op) {
        return mlir::WalkResult::interrupt();
      }).wasInterrupted()) {
    if (getenv("TRITON_SHIM_LOG")) {
      std::string debugStr;
      llvm::raw_string_ostream debugStream(debugStr);
      module.get()->print(debugStream);
      debugStream.flush();
      return strdup(("ERROR: tt.func still present after pipeline:\n" + debugStr).c_str());
    }
    return strdup("ERROR: tt.func still present after pipeline (set TRITON_SHIM_LOG=1 for IR)");
  }

  // Translate to LLVM IR
  llvm::LLVMContext llvmCtx;
  auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmCtx);
  if (!llvmModule) {
    return strdup("ERROR: Failed to translate to LLVM IR");
  }

  // Link libdevice so __nv_* calls (sqrt/log/sin/...) resolve before PTX.
  if (isCuda) {
    std::string linkErr;
    if (!link_libdevice(llvmModule.get(), linkErr)) {
      return strdup(("ERROR: " + linkErr).c_str());
    }
  }

  // LLVM IR → ISA (PTX for cuda, AMDGCN for rocm).
  llvm::Triple targetTriple(isCuda ? "nvptx64-nvidia-cuda"
                                   : "amdgcn-amd-amdhsa");
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    return strdup(("ERROR: " + error).c_str());
  }

  std::string cpu = isCuda ? ("sm_" + std::string(arch)) : std::string(arch);
  auto targetMachine = target->createTargetMachine(
      targetTriple, cpu, features, llvm::TargetOptions(),
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
  std::string outStr(ptxBuf.data(), ptxBuf.size());

  return strdup(outStr.c_str());
}

// 3090-default convenience wrapper (backward-compatible ABI).
const char* triton_compile(const char* ttir_mlir, int num_warps) {
  return triton_compile_targeted(ttir_mlir, num_warps, "cuda", "86", "+ptx75");
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
