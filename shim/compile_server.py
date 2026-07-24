#!/usr/bin/env python3
"""
compile_server.py — Compile TTIR to PTX using the full Triton pipeline.
Usage: python3 compile_server.py <input.ttir> <output.ptx> [num_warps] [capability] [ptx_version]
"""
import sys, os
from triton._C.libtriton import ir, passes, llvm, nvidia

def compile_ttir(ttir_path: str, num_warps: int = 4, capability: int = 86, ptx_version: int = 75) -> str:
    ctx = ir.context()
    ir.load_dialects(ctx)
    nvidia.load_dialects(ctx)
    mod = ir.parse_mlir_module(ttir_path, ctx)

    # Stage 1: make_ttir (add_combine, add_inliner, add_reorder_broadcast)
    pm = ir.pass_manager(ctx)
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod, 'make_ttir')

    # Stage 2: make_ttgir (add_fuse_nested_loops, add_pipeline, add_optimize_dot_operands, etc.)
    pm2 = ir.pass_manager(ctx)
    passes.ttir.add_convert_to_ttgpuir(pm2, f'cuda:{capability}', num_warps, 32, 1)
    passes.ttgpuir.add_coalesce(pm2)
    passes.ttgpuir.add_f32_dot_tc(pm2, False)
    nvidia.passes.ttnvgpuir.add_plan_cta(pm2)
    passes.ttgpuir.add_remove_layout_conversions(pm2)
    passes.ttgpuir.add_optimize_thread_locality(pm2)
    passes.ttgpuir.add_accelerate_matmul(pm2)
    passes.ttgpuir.add_remove_layout_conversions(pm2)
    passes.ttgpuir.add_optimize_dot_operands(pm2, capability >= 80)
    nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm2)
    passes.ttir.add_loop_aware_cse(pm2)
    passes.ttgpuir.add_fuse_nested_loops(pm2)
    passes.common.add_canonicalizer(pm2)
    passes.ttir.add_triton_licm(pm2)
    passes.common.add_canonicalizer(pm2)
    passes.ttgpuir.add_combine_tensor_select_and_if(pm2)
    passes.ttgpuir.add_assign_latencies(pm2, 3)
    passes.ttgpuir.add_schedule_loops(pm2)
    passes.ttgpuir.add_pipeline(pm2, 3, False)
    passes.ttgpuir.add_fuse_nested_loops(pm2)
    passes.common.add_canonicalizer(pm2)
    passes.ttir.add_triton_licm(pm2)
    pm2.run(mod, 'make_ttgir')

    # Stage 3: make_llir (NV-specific shared memory + LLVM conversion + cleanup)
    pm3 = ir.pass_manager(ctx)
    passes.ttgpuir.add_combine_tensor_select_and_if(pm3)
    passes.ttgpuir.add_allocate_warp_groups(pm3)
    nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm3, capability, ptx_version)
    nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm3)
    nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm3)
    passes.ttgpuir.add_allocate_global_scratch_memory(pm3)
    nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm3, capability)
    nvidia.passes.ttgpuir.add_to_llvmir(pm3, capability, ptx_version)
    passes.common.add_canonicalizer(pm3)
    passes.common.add_cse(pm3)
    nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm3)
    nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm3)
    passes.common.add_canonicalizer(pm3)
    passes.common.add_cse(pm3)
    passes.common.add_symbol_dce(pm3)
    passes.convert.add_nvvm_to_llvm(pm3)
    pm3.run(mod, 'make_llir')

    # Stage 4: LLVM → PTX
    llvm.init_targets()
    nvidia.set_short_ptr()
    triple = 'nvptx64-nvidia-cuda'
    proc = f'sm_{capability}'
    features = f'+ptx{ptx_version}'
    flags = ['nvptx-mad-wide-opt']
    context = llvm.context()
    llvm_mod = llvm.to_module(mod, context)
    llvm.attach_datalayout(llvm_mod, triple, proc, features)
    nvidia.set_nvvm_reflect_ftz(llvm_mod)
    llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
    llvm_ir = str(llvm_mod)
    ptx = llvm.translate_to_asm(llvm_ir, triple, proc, features, flags, False, False)
    return ptx

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.ttir> <output.ptx> [num_warps] [capability] [ptx_version]", file=sys.stderr)
        sys.exit(1)
    ttir_path = sys.argv[1]
    ptx_path = sys.argv[2]
    num_warps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    capability = int(sys.argv[4]) if len(sys.argv) > 4 else 86
    ptx_version = int(sys.argv[5]) if len(sys.argv) > 5 else 75

    try:
        ptx = compile_ttir(ttir_path, num_warps, capability, ptx_version)
        with open(ptx_path, 'w') as f:
            f.write(ptx)
        print(f"OK: {len(ptx)} chars PTX → {ptx_path}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
