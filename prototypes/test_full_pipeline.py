import tempfile, os
from triton._C.libtriton import ir, passes, llvm, nvidia

ttir = '''module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mm(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
    %pid0 = tt.get_program_id x : i32
    %pid1 = tt.get_program_id y : i32
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %e3 = arith.muli %pid0, %c1 : i32
    %e5 = arith.muli %pid1, %c64 : i32
    %c6 = arith.constant 1 : i64
    %c7 = arith.constant 64 : i64
    %c8 = arith.constant 64 : i64
    %c9 = arith.constant 1 : i64
    %tp10 = tt.make_tensor_ptr %arg0, [%c6, %c7], [%c8, %c9], [%e3, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<1x64xbf16>>
    %c11 = arith.constant 0 : i32
    %c12 = arith.constant 64 : i32
    %e13 = arith.muli %pid1, %c12 : i32
    %c14 = arith.constant 64 : i64
    %c15 = arith.constant 1 : i64
    %c16 = arith.constant 1 : i64
    %c17 = arith.constant 64 : i64
    %tp18 = tt.make_tensor_ptr %arg1, [%c14, %c15], [%c16, %c17], [%c0, %e13] {order = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
    %c19 = arith.constant 1 : i64
    %c20 = arith.constant 64 : i64
    %c21 = arith.constant 64 : i64
    %c22 = arith.constant 1 : i64
    %tp23 = tt.make_tensor_ptr %arg2, [%c19, %c20], [%c21, %c22], [%e3, %e5] {order = array<i32: 1, 0>} : !tt.ptr<tensor<1x64xf32>>
    %z24 = arith.constant dense<0.000000e+00> : tensor<1x64xf32>
    %tp25 = tt.make_tensor_ptr %arg0, [%c6, %c7], [%c8, %c9], [%e3, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<1x64xbf16>>
    %tp26 = tt.make_tensor_ptr %arg1, [%c14, %c15], [%c16, %c17], [%c0, %e13] {order = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
    %ld27 = tt.load %tp25 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<1x64xbf16>>
    %ld28 = tt.load %tp26 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
    %d29 = "tt.dot"(%ld27, %ld28, %z24) : (tensor<1x64xbf16>, tensor<64x64xbf16>, tensor<1x64xf32>) -> tensor<1x64xf32>
    tt.store %tp23, %d29 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<1x64xf32>>
    tt.return
  }
}
'''
f = tempfile.NamedTemporaryFile(mode='w', suffix='.ttir', delete=False)
f.write(ttir); f.close()

ctx = ir.context()
ir.load_dialects(ctx)
nvidia.load_dialects(ctx)
mod = ir.parse_mlir_module(f.name, ctx)
os.unlink(f.name)

# Stage 1: make_ttir
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
print('1. make_ttir OK')

# Stage 2: make_ttgir
pm2 = ir.pass_manager(ctx)
passes.ttir.add_convert_to_ttgpuir(pm2, 'cuda:86', 4, 32, 1)
passes.ttgpuir.add_coalesce(pm2)
passes.ttgpuir.add_f32_dot_tc(pm2, False)
nvidia.passes.ttnvgpuir.add_plan_cta(pm2)
passes.ttgpuir.add_remove_layout_conversions(pm2)
passes.ttgpuir.add_optimize_thread_locality(pm2)
passes.ttgpuir.add_accelerate_matmul(pm2)
passes.ttgpuir.add_remove_layout_conversions(pm2)
passes.ttgpuir.add_optimize_dot_operands(pm2, True)
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
print('2. make_ttgir OK')

# Stage 3: make_llir
pm3 = ir.pass_manager(ctx)
passes.ttgpuir.add_combine_tensor_select_and_if(pm3)
passes.ttgpuir.add_allocate_warp_groups(pm3)
nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm3, 86, 75)
nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm3)
nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm3)
passes.ttgpuir.add_allocate_global_scratch_memory(pm3)
nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm3, 86)
nvidia.passes.ttgpuir.add_to_llvmir(pm3, 86, 75)
passes.common.add_canonicalizer(pm3)
passes.common.add_cse(pm3)
nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm3)
nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm3)
passes.common.add_canonicalizer(pm3)
passes.common.add_cse(pm3)
passes.common.add_symbol_dce(pm3)
passes.convert.add_nvvm_to_llvm(pm3)
pm3.run(mod, 'make_llir')
print('3. make_llir OK')

# Stage 4: LLVM → PTX
llvm.init_targets()
nvidia.set_short_ptr()
triple = 'nvptx64-nvidia-cuda'
proc = 'sm_86'
features = '+ptx75'
flags = ['nvptx-mad-wide-opt']
context = llvm.context()
llvm_mod = llvm.to_module(mod, context)
llvm.attach_datalayout(llvm_mod, triple, proc, features)
nvidia.set_nvvm_reflect_ftz(llvm_mod)
llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
llvm_ir = str(llvm_mod)
ptx = llvm.translate_to_asm(llvm_ir, triple, proc, features, flags, False, False)
print(f'4. PTX: {len(ptx)} chars')
print(ptx[:300])
print('SUCCESS — full Triton pipeline with ALL fusion passes!')
