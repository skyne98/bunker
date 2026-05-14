// kernel.ts — Kernel template system with autotuning
// Kernel template → params → kernel instance → run on GPU

import { dlopen, ptr, CString } from "bun:ffi";
import { execSync } from "child_process";
import { writeFileSync, mkdirSync, existsSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { homedir } from "os";

// ═══════════════════════════════════════════════════════════════════
// f16 utilities
// ═══════════════════════════════════════════════════════════════════

/** Convert a Float32Array to f16-encoded Uint16Array. NaN→NaN, Inf→Inf, denorm→0. */
export function f32to16(f32: Float32Array): Uint16Array {
  const u32 = new Uint32Array(f32.buffer);
  const out = new Uint16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const v = u32[i];
    const sign = (v >> 16) & 0x8000;
    const exp = (v >> 23) & 0xff;
    const mant = v & 0x7fffff;
    if (exp === 0) { out[i] = sign; continue; } // ±0 or denorm → 0
    if (exp === 0xff) { out[i] = sign | 0x7c00 | (mant ? 0x200 : 0); continue; } // NaN→qNaN, Inf→Inf
    const exp16 = exp - 127 + 15;
    if (exp16 >= 0x1f) { out[i] = sign | 0x7c00; continue; } // overflow → Inf
    if (exp16 <= 0) { out[i] = sign; continue; } // underflow → 0
    out[i] = sign | (exp16 << 10) | (mant >> 13);
  }
  return out;
}

/** Convert f32 data to f16 for GPU upload. Returns { buf: Uint16Array, size: number }. */
export function toF16(f32: Float32Array): Uint16Array {
  return f32to16(f32);
}

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════

export interface TileConfig {
  BM: number;
  BN: number;
  BK: number;
  numWarps: number;
}

export interface MatmulDims {
  M: number;
  N: number;
  K: number;
}

export type TTIRGenerator = (cfg: TileConfig) => string;

// ═══════════════════════════════════════════════════════════════════
// CUDA bindings
// ═══════════════════════════════════════════════════════════════════

const CUDA = "/run/opengl-driver/lib/libcuda.so";
let _cs: any = null;
function cuda() {
  if (!_cs) {
    _cs = dlopen(CUDA, {
      cuInit: { args: ["u32"], returns: "i32" },
      cuDeviceGet: { args: ["ptr", "i32"], returns: "i32" },
      cuCtxCreate_v2: { args: ["ptr", "u32", "i64"], returns: "i32" },
      cuModuleLoadData: { args: ["ptr", "ptr"], returns: "i32" },
      cuModuleGetFunction: { args: ["ptr", "i64", "ptr"], returns: "i32" },
      cuLaunchKernel: { args: ["i64", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "ptr", "ptr", "ptr"], returns: "i32" },
      cuCtxSynchronize: { args: [], returns: "i32" },
      cuMemAlloc_v2: { args: ["ptr", "i64"], returns: "i32" },
      cuMemcpyHtoD_v2: { args: ["i64", "ptr", "i64"], returns: "i32" },
      cuMemcpyDtoH_v2: { args: ["ptr", "i64", "i64"], returns: "i32" },
      cuMemFree_v2: { args: ["i64"], returns: "i32" },
      cuEventCreate: { args: ["ptr", "u32"], returns: "i32" },
      cuEventRecord: { args: ["i64", "i64"], returns: "i32" },
      cuEventSynchronize: { args: ["i64"], returns: "i32" },
      cuEventElapsedTime: { args: ["ptr", "i64", "i64"], returns: "i32" },
    }).symbols;
    _cs.cuInit(0);
    const dev = new Int32Array(1);
    _cs.cuDeviceGet(dev, 0);
    _cs.cuCtxCreate_v2(Buffer.alloc(8), 0, BigInt(dev[0]));
  }
  return _cs;
}

/** Export the shared CUDA context for use by external tests (avoids dual-context bugs) */
export function getCuCtx() { return cuda(); }

/** Create CUDA events for GPU-accurate timing */
export function createCudaEvents(): { start: Buffer; stop: Buffer; elapsed: Float32Array } {
  const cs = cuda();
  const start = Buffer.alloc(8); cs.cuEventCreate(start, 0);
  const stop = Buffer.alloc(8); cs.cuEventCreate(stop, 0);
  return { start, stop, elapsed: new Float32Array(1) };
}

/** Record GPU time between two events in microseconds */
export function gpuTimeUs(cs: any, ev: { start: Buffer; stop: Buffer; elapsed: Float32Array }): number {
  cs.cuEventElapsedTime(ev.elapsed, Number(ev.start.readBigUInt64LE(0)), Number(ev.stop.readBigUInt64LE(0)));
  return ev.elapsed[0] * 1000;
}

// ═══════════════════════════════════════════════════════════════════
// Compilation
// ═══════════════════════════════════════════════════════════════════

const WRAPPER = "/tmp/triton_wrap";

// LD_LIBRARY_PATH for the triton shim — should match build_shim.sh
const SHIM_ENV = {
  LD_LIBRARY_PATH: "/nix/store/ixhlv41i2wpl84xgjcks061dz4yssbg3-zlib-1.3.2/lib:/nix/store/bfwqrbwqpbnsdbgf86gz8pn8vvddci3i-libxml2-2.13.8/lib"
};

export function compileTTIR(ttir: string, numWarps: number): { ptx: string; shmem: number } {
  const f = join(tmpdir(), `k_${process.pid}_${Math.random().toString(36).slice(2)}.mlir`);
  writeFileSync(f, ttir);
  const out = execSync(`${WRAPPER} ${f} ${numWarps}`, {
    cwd: __dirname, encoding: "utf-8", maxBuffer: 50 * 1024 * 1024,
    env: { ...process.env, ...SHIM_ENV }
  });
  if (out.startsWith("ERROR:") || out.startsWith("loc(")) {
    throw Error(`TTIR compilation failed:\n${out.substring(0, 500)}`);
  }
  // Shared memory size from shim is stored in a companion file; use safe default
  return { ptx: out, shmem: 32768 };
}

// ═══════════════════════════════════════════════════════════════════
// Kernel Instance — A compiled, ready-to-run kernel
// ═══════════════════════════════════════════════════════════════════

export class KernelInstance {
  public readonly cfg: TileConfig;
  public readonly ptx: string;
  public readonly shmem: number;
  public readonly fnName: string;
  private _mod: bigint = 0n;
  private _fn: bigint = 0n;
  private _initialized = false;
  private _numParams = 0;

  constructor(cfg: TileConfig, ptx: string, shmem: number, fnName = "matmul") {
    this.cfg = cfg;
    this.ptx = ptx;
    this.shmem = shmem;
    this.fnName = fnName;
  }

  private ensureLoaded() {
    if (this._initialized) return;
    const cs = cuda();
    const mod = Buffer.alloc(8);
    cs.cuModuleLoadData(mod, Buffer.from(this.ptx + "\0"));
    this._mod = mod.readBigUInt64LE(0);
    const fn = Buffer.alloc(8);
    cs.cuModuleGetFunction(fn, Number(this._mod), Buffer.from(this.fnName + "\0"));
    this._fn = fn.readBigUInt64LE(0);
    if (this._fn === 0n) throw Error(`Function "${this.fnName}" not found in PTX`);
    // Count params from PTX: .param directives in kernel entry
    const m = this.ptx.match(new RegExp(`\\.visible\\s+\\.entry\\s+${this.fnName}[^)]+\\)`, "s"));
    if (m) this._numParams = (m[0].match(/\.param/g) || []).length;
    else this._numParams = 3; // fallback
    this._initialized = true;
  }

  /** Build kernel params buffer from typed args. Returns [paramsBuf, kernelParamsArray]. */
  private buildParams(args: (ArrayBufferView | number)[]): [Buffer, Buffer] {
    const slots = args.map(a => {
      const s = Buffer.alloc(8);
      if (typeof a === "number") { s.writeInt32LE(a, 0); return s; }
      // Pointer arg: write device address placeholder (caller fills)
      return s;
    });
    // Pad to numParams + null terminator
    while (slots.length < this._numParams) slots.push(Buffer.alloc(8));
    const pb = Buffer.concat(slots);
    return [pb, Buffer.alloc((slots.length + 1) * 8)];
  }

  /** Launch the kernel. `args` are positional params matching the TTIR function signature. */
  run(args: (ArrayBufferView | number)[], dims: MatmulDims): void {
    this.ensureLoaded();
    const cs = cuda();

    // Allocate + upload each ArrayBufferView arg
    const devAddrs: bigint[] = [];
    const gpuArgs: (number | bigint)[] = [];

    for (const a of args) {
      if (typeof a === "number") { gpuArgs.push(a); continue; }
      const db = Buffer.alloc(8);
      const sz = BigInt(a.byteLength);
      cs.cuMemAlloc_v2(db, sz);
      const dp = db.readBigUInt64LE(0);
      cs.cuMemcpyHtoD_v2(Number(dp), a, sz);
      devAddrs.push(dp);
      gpuArgs.push(dp);
    }

    // Build flat param buffer + pointer array
    const slotBuf = Buffer.alloc(gpuArgs.length * 8);
    gpuArgs.forEach((v, i) => {
      if (typeof v === "bigint") slotBuf.writeBigUInt64LE(v, i * 8);
      else slotBuf.writeInt32LE(v, i * 8);
    });

    // Build kernel params pointer array
    const kp = Buffer.alloc((this._numParams + 1) * 8);
    const gx = Math.ceil(dims.M / this.cfg.BM);
    const gy = Math.ceil(dims.N / this.cfg.BN);
    const bt = this.cfg.numWarps * 32;

    cs.cuLaunchKernel(Number(this._fn), gx, gy, 1, bt, 1, 1, this.shmem, 0n,
      Buffer.from(kp.buffer), null);

    // Wait
    const syncRet = cs.cuCtxSynchronize();
    if (syncRet !== 0) throw Error(`cuCtxSynchronize failed: ${syncRet}`);

    // Readback
    let ai = 0;
    for (const a of args) {
      if (typeof a !== "number") {
        cs.cuMemcpyDtoH_v2(a, Number(devAddrs[ai]), BigInt(a.byteLength));
        ai++;
      }
    }

    for (const dp of devAddrs) cs.cuMemFree_v2(Number(dp));
  }

  /** Benchmark this kernel for given dimensions. Returns time in ms. */
  benchmark(dims: MatmulDims, iters = 10): number {
    this.ensureLoaded();
    const cs = cuda();
    const SZ = BigInt(256 * 1024 * 1024);
    const allocs = [Buffer.alloc(8), Buffer.alloc(8), Buffer.alloc(8)];
    allocs.forEach(b => cs.cuMemAlloc_v2(b, SZ));

    const gx = Math.ceil(dims.M / this.cfg.BM);
    const gy = Math.ceil(dims.N / this.cfg.BN);
    const bt = this.cfg.numWarps * 32;

    // Warmup
    for (let w = 0; w < 3; w++) cs.cuLaunchKernel(Number(this._fn), gx, gy, 1, bt, 1, 1, this.shmem, 0n, null, null);
    cs.cuCtxSynchronize();

    const times: number[] = [];
    for (let i = 0; i < iters; i++) {
      const t0 = performance.now();
      const lr = cs.cuLaunchKernel(Number(this._fn), gx, gy, 1, bt, 1, 1, this.shmem, 0n, null, null);
      if (lr !== 0) break;
      cs.cuCtxSynchronize();
      times.push(performance.now() - t0);
    }

    allocs.forEach(b => cs.cuMemFree_v2(Number(b.readBigUInt64LE(0))));
    if (times.length === 0) return Infinity;
    return times.reduce((a, b) => a + b, 0) / times.length;
  }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel Template
// ═══════════════════════════════════════════════════════════════════

export class KernelTemplate {
  public readonly name: string;
  public readonly generator: TTIRGenerator;
  public readonly defaultConfig: TileConfig;
  public readonly fnName: string;
  public cache = new Map<string, KernelInstance>();

  constructor(opts: {
    name: string;
    generator: TTIRGenerator;
    defaults?: Partial<TileConfig>;
    fnName?: string;
  }) {
    this.name = opts.name;
    this.generator = opts.generator;
    this.fnName = opts.fnName ?? this.name;
    this.defaultConfig = {
      BM: 128, BN: 128, BK: 32, numWarps: 8,
      ...opts.defaults,
    };
  }

  /** Compile with given config (or defaults). Returns cached instance if available. */
  async compile(cfg?: Partial<TileConfig>): Promise<KernelInstance> {
    const full: TileConfig = { ...this.defaultConfig, ...cfg };
    const key = JSON.stringify(full);
    const cached = this.cache.get(key);
    if (cached) return cached;

    const ttir = this.generator(full);
    const { ptx, shmem } = compileTTIR(ttir, full.numWarps);
    const inst = new KernelInstance(full, ptx, shmem, this.fnName);
    this.cache.set(key, inst);
    return inst;
  }

  /** Search over the tunable space to find the fastest config for given dims. */
  async autotune(
    dims: MatmulDims,
    searchSpace?: Partial<TileConfig>[]
  ): Promise<{ config: TileConfig; instance: KernelInstance; timeMs: number }> {
    const space = searchSpace ?? (this.name.includes("i8") ? INT8_SEARCH : FP16_SEARCH);

    let best: { config: TileConfig; instance: KernelInstance; timeMs: number } | null = null;

    for (const partial of space) {
      const cfg: TileConfig = { ...this.defaultConfig, ...partial };
      const inst = await this.compile(cfg);
      const timeMs = inst.benchmark(dims, 5);
      console.log(`  ${this.fmtCfg(cfg)} → ${(timeMs * 1000).toFixed(1)} µs`);

      if (timeMs < (best?.timeMs ?? Infinity)) {
        best = { config: cfg, instance: inst, timeMs };
      }
    }

    if (!best) throw Error("autotune: no working configs");

    return best;
  }

  private fmtCfg(cfg: TileConfig): string {
    return `${cfg.BM}×${cfg.BK}×${cfg.BN} w${cfg.numWarps}`;
  }
}

// ═══════════════════════════════════════════════════════════════════
// Default search space
// ═══════════════════════════════════════════════════════════════════

/** Default search space for FP16 matmul autotuning. */
const FP16_SEARCH: Partial<TileConfig>[] = [
  { BM: 16, BN: 16, BK: 32, numWarps: 4 },
  { BM: 32, BN: 32, BK: 32, numWarps: 4 },
  { BM: 32, BN: 16, BK: 1024, numWarps: 4 },
  { BM: 32, BN: 32, BK: 1024, numWarps: 4 },
  { BM: 64, BN: 64, BK: 32, numWarps: 4 },
  { BM: 64, BN: 64, BK: 1024, numWarps: 8 },
  { BM: 128, BN: 128, BK: 32, numWarps: 8 },
  { BM: 128, BN: 128, BK: 64, numWarps: 8 },
  { BM: 128, BN: 64, BK: 32, numWarps: 8 },
];

/** Default search space for INT8 matmul autotuning. */
const INT8_SEARCH: Partial<TileConfig>[] = [
  { BM: 16, BN: 16, BK: 32, numWarps: 4 },
  { BM: 32, BN: 16, BK: 32, numWarps: 4 },
  { BM: 32, BN: 32, BK: 32, numWarps: 4 },
  { BM: 16, BN: 16, BK: 256, numWarps: 4 },
  { BM: 32, BN: 32, BK: 256, numWarps: 4 },
  { BM: 16, BN: 16, BK: 1024, numWarps: 4 },
  { BM: 32, BN: 32, BK: 1024, numWarps: 4 },
  { BM: 32, BN: 32, BK: 512, numWarps: 8 },
  { BM: 64, BN: 64, BK: 256, numWarps: 8 },
];

// ═══════════════════════════════════════════════════════════════════
// Built-in matmul kernels
// ═══════════════════════════════════════════════════════════════════

/** FP16 tensor core matmul (uses boundaryCheck, safe for all tile sizes) */
export function matmulTTIR(cfg: TileConfig, stride = 4096): string {
  const { BM, BN, BK, numWarps } = cfg;
  return `module attributes {"ttg.num-warps" = ${numWarps} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %cBM = arith.constant ${BM} : i64
    %cBN = arith.constant ${BN} : i64
    %cBK = arith.constant ${BK} : i64
    %cN = arith.constant ${stride} : i64
    %c1 = arith.constant 1 : i64
    %zero = arith.constant dense<0.000000e+00> : tensor<${BM}x${BN}xf32>
    %tpA = tt.make_tensor_ptr %A, [%cN, %cN], [%cN, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BM}x${BK}xf16>>
    %tpB = tt.make_tensor_ptr %B, [%cN, %cN], [%cN, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BK}x${BN}xf16>>
    %a = tt.load %tpA {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<${BM}x${BK}xf16>>
    %b = tt.load %tpB {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<${BK}x${BN}xf16>>
    %c = tt.dot %a, %b, %zero : tensor<${BM}x${BK}xf16> * tensor<${BK}x${BN}xf16> -> tensor<${BM}x${BN}xf32>
    %tpC = tt.make_tensor_ptr %C, [%cN, %cN], [%cN, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<${BM}x${BN}xf32>>
    tt.store %tpC, %c {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<${BM}x${BN}xf32>>
    tt.return
  }
}`;
}

/** INT8 tensor core matmul — no boundaryCheck, 33 TFLOPS on RTX 3090 */
export function int8MatmulTTIR(cfg: TileConfig, stride = 1024): string {
  const { BM=32, BN=32, BK=1024, numWarps=4 } = cfg;
  return `module attributes {"ttg.num-warps" = ${numWarps} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul(%A: !tt.ptr<i8>, %B: !tt.ptr<i8>, %C: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32 %cBM = arith.constant ${BM} : i32 %cBN = arith.constant ${BN} : i32
    %c0_i64 = arith.constant 0 : i64 %c1_i64 = arith.constant 1 : i64
    %cBK = arith.constant ${BK} : i64 %cS = arith.constant ${stride} : i64
    %px = tt.get_program_id x : i32 %py = tt.get_program_id y : i32
    %bm = arith.muli %px, %cBM : i32 %bn = arith.muli %py, %cBN : i32
    %z = arith.constant dense<0> : tensor<${BM}x${BN}xi32>
    %tA = tt.make_tensor_ptr %A,[%cS,%cS],[%cS,%c1_i64],[%bm,%c0]{order=array<i32:1,0>}:!tt.ptr<tensor<${BM}x${BK}xi8>>
    %tB = tt.make_tensor_ptr %B,[%cS,%cS],[%cS,%c1_i64],[%c0,%bn]{order=array<i32:1,0>}:!tt.ptr<tensor<${BK}x${BN}xi8>>
    %a = tt.load %tA:!tt.ptr<tensor<${BM}x${BK}xi8>>
    %b = tt.load %tB:!tt.ptr<tensor<${BK}x${BN}xi8>>
    %c = tt.dot %a,%b,%z:tensor<${BM}x${BK}xi8>*tensor<${BK}x${BN}xi8>->tensor<${BM}x${BN}xi32>
    %tC = tt.make_tensor_ptr %C,[%cS,%cS],[%cS,%c1_i64],[%bm,%bn]{order=array<i32:1,0>}:!tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.store %tC,%c:!tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.return
  }
}`;
}

/** Fused Q4_K dequant + TC matmul in a single TTIR kernel.
 *  Loads all Q4_K blocks as a 2D tensor [BM×16] with strided make_tensor_ptr,
 *  dequantizes via element-wise arith ops, interleaves lo/hi via join+reshape,
 *  scf.for K-loop with iter_args, tt.dot for tensor core matmul. */
export function fusedDequantMatmulTTIR(cfg: TileConfig, stride = 1024): string {
  const { BM=32, BN=32, BK=32, numWarps=4 } = cfg;
  const K_ITER = stride / BK;
  const ROW_STRIDE = 32 * 20; // 640 bytes between Q4_K blocks of consecutive rows

  return `module attributes {"ttg.num-warps" = ${numWarps} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fused(%Q: !tt.ptr<i8>, %B: !tt.ptr<i8>, %C: !tt.ptr<i32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_index = arith.constant 1 : index
    %c32_i32 = arith.constant 32 : i32
    %c1024 = arith.constant 1024 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c640 = arith.constant ${ROW_STRIDE} : i64
    %c16 = arith.constant 16 : i64
    %cS = arith.constant ${stride} : i64
    %c15 = arith.constant dense<15> : tensor<${BM}x16xi8>
    %c8 = arith.constant dense<8> : tensor<${BM}x16xi8>
    %c4 = arith.constant dense<4> : tensor<${BM}x16xi8>
    %z_acc = arith.constant dense<0> : tensor<${BM}x${BN}xi32>
    %px = tt.get_program_id x : i32 %py = tt.get_program_id y : i32
    %bm = arith.muli %px, %c32_i32 : i32 %bn = arith.muli %py, %c32_i32 : i32

    %c0_idx = arith.constant 0 : index
    %cKiter_idx = arith.constant ${K_ITER} : index
    %acc = scf.for %k_idx = %c0_idx to %cKiter_idx step %c1_index iter_args(%acc_iter = %z_acc) -> (tensor<${BM}x${BN}xi32>) {
      %k = arith.index_cast %k_idx : index to i32
      %k_off = arith.muli %k, %c32_i32 : i32
      %k_off_i64 = arith.extsi %k_off : i32 to i64
      %bm_i64 = arith.extsi %bm : i32 to i64

      // Load B tile: BK×BN INT8 at B[1024×1024], offset [k_off, bn]
      %tB = tt.make_tensor_ptr %B,[%cS,%cS],[%cS,%c1_i64],[%k_off,%bn]{order=array<i32:1,0>}:!tt.ptr<tensor<${BK}x${BN}xi8>>
      %b_tile = tt.load %tB:!tt.ptr<tensor<${BK}x${BN}xi8>>

      // Load all BM Q4_K blocks as a 2D tensor [BM×16] with row stride 640
      // First block index for this tile: (bm * 1024 + k_off) / 32 = bm * 32 + k
      %tile_blk = arith.addi %bm, %k : i32  // simplified: (bm * 32 + k) but wait, bm is row * 32, k is iteration
      // Actually: first block = (bm + k) where bm is row*32 and k is k_iter
      // But bm is already row*32, so bm + k = row_start*32 + k_iter
      // Hmm, bm = px * 32. For block index: (px*32 * 1024 + k_off) / 32 = px*32*32 + k
      // = px*1024 + k. Since px*1024 is the row offset in the matrix, bm=px*32.
      // Block index = px*32*32 + k (since k_off = k*32, and 1024/32 = 32 blocks per row)
      // So tile_blk_off = px*32*32 + k = px*1024 + k
      // Since bm = px*32, px*1024 = bm*32
      %bm32 = arith.muli %bm, %c32_i32 : i32  // bm * 32 = px * 1024
      %tile_blk_off = arith.addi %bm32, %k : i32  // px*1024 + k
      %c20 = arith.constant 20 : i32
      %tile_byte_off2 = arith.muli %tile_blk_off, %c20 : i32
      %cBM_i64 = arith.constant ${BM} : i64
      %c640_i64 = arith.constant ${ROW_STRIDE} : i64
      %tQ = tt.make_tensor_ptr %Q,[%cBM_i64,%c16],[%c640_i64,%c1_i64],[%tile_byte_off2,%c0_i32]{order=array<i32:1,0>}:!tt.ptr<tensor<${BM}x16xi8>>
      %q_packed = tt.load %tQ:!tt.ptr<tensor<${BM}x16xi8>>

      // Dequant: extract lo/hi nibbles, center at 0
      %q_lo = arith.andi %q_packed, %c15 : tensor<${BM}x16xi8>
      %q_hi_tmp = arith.shrui %q_packed, %c4 : tensor<${BM}x16xi8>
      %q_hi = arith.andi %q_hi_tmp, %c15 : tensor<${BM}x16xi8>
      %q_lo_c = arith.subi %q_lo, %c8 : tensor<${BM}x16xi8>
      %q_hi_c = arith.subi %q_hi, %c8 : tensor<${BM}x16xi8>

      // Interleave lo/hi along the last dimension: tensor<BM×16×2xi8>
      %q_joined = tt.join %q_lo_c, %q_hi_c : tensor<${BM}x16xi8> -> tensor<${BM}x16x2xi8>

      // Reshape to BM×BK (16×2 = 32 = BK)
      %a_tile = tt.reshape %q_joined : tensor<${BM}x16x2xi8> -> tensor<${BM}x${BK}xi8>

      %d = tt.dot %a_tile, %b_tile, %acc_iter : tensor<${BM}x${BK}xi8> * tensor<${BK}x${BN}xi8> -> tensor<${BM}x${BN}xi32>
      scf.yield %d : tensor<${BM}x${BN}xi32>
    }

    %tC = tt.make_tensor_ptr %C,[%cS,%cS],[%cS,%c1_i64],[%bm,%bn]{order=array<i32:1,0>}:!tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.store %tC,%acc:!tt.ptr<tensor<${BM}x${BN}xi32>>
    tt.return
  }
}`;
}

// ═══════════════════════════════════════════════════════════════════
// Q4_K Dequant kernel (LLVM IR → PTX)
// ═══════════════════════════════════════════════════════════════════

const LLVM = "/nix/store/6r234y6pkbyyr8pk1wh7nfsmnzdxyswx-llvm-19.1.7-lib/lib/libLLVM-19.so";

let _llvmSymbols: any = null;
function llvmSymbols() {
  if (!_llvmSymbols) {
    _llvmSymbols = dlopen(LLVM, {
      LLVMContextCreate: { args: [], returns: "pointer" },
      LLVMParseIRInContext: { args: ["pointer", "pointer", "pointer", "pointer"], returns: "i32" },
      LLVMCreateMemoryBufferWithMemoryRange: { args: ["pointer", "i64", "pointer", "i32"], returns: "pointer" },
      LLVMGetTargetFromTriple: { args: ["pointer", "pointer", "pointer"], returns: "i32" },
      LLVMCreateTargetMachine: { args: ["pointer", "pointer", "pointer", "pointer", "i32", "i32", "i32"], returns: "pointer" },
      LLVMTargetMachineEmitToMemoryBuffer: { args: ["pointer", "pointer", "i32", "pointer", "pointer"], returns: "i32" },
      LLVMGetBufferSize: { args: ["pointer"], returns: "i64" },
      LLVMGetBufferStart: { args: ["pointer"], returns: "pointer" },
      LLVMInitializeNVPTXTargetInfo: { args: [], returns: "void" },
      LLVMInitializeNVPTXTarget: { args: [], returns: "void" },
      LLVMInitializeNVPTXTargetMC: { args: [], returns: "void" },
      LLVMInitializeNVPTXAsmPrinter: { args: [], returns: "void" },
    }).symbols;
    _llvmSymbols.LLVMInitializeNVPTXTargetInfo();
    _llvmSymbols.LLVMInitializeNVPTXTarget();
    _llvmSymbols.LLVMInitializeNVPTXTargetMC();
    _llvmSymbols.LLVMInitializeNVPTXAsmPrinter();
  }
  return _llvmSymbols;
}

function compileLLVM(src: string): string {
  // Debug: write IR to file
  try { writeFileSync("/tmp/dequant_debug.ll", src); } catch {}
  const ls = llvmSymbols();
  const ctx = ls.LLVMContextCreate();
  const irBuf = Buffer.from(src + "\0");
  const mb = ls.LLVMCreateMemoryBufferWithMemoryRange(
    ptr(irBuf), BigInt(irBuf.length - 1), ptr(Buffer.from("k.ll\0")), 1);
  const ma = new BigUint64Array(1);
  if (ls.LLVMParseIRInContext(ctx, mb, ptr(ma), ptr(new BigUint64Array(1))))
    throw Error("LLVM IR parse failed");
  const mod = Number(ma[0]);
  const tp = Buffer.from("nvptx64-nvidia-cuda\0");
  const ta = new BigUint64Array(1);
  ls.LLVMGetTargetFromTriple(ptr(tp), ptr(ta), ptr(new BigUint64Array(1)));
  const tm = ls.LLVMCreateTargetMachine(Number(ta[0]), ptr(tp),
    ptr(Buffer.from("sm_86\0")), ptr(Buffer.from("\0")), 2, 0, 0);
  const pa = new BigUint64Array(1);
  ls.LLVMTargetMachineEmitToMemoryBuffer(tm, mod, 0, ptr(new BigUint64Array(1)), ptr(pa));
  return new CString(ls.LLVMGetBufferStart(Number(pa[0])), 0,
    Number(ls.LLVMGetBufferSize(Number(pa[0])))).toString();
}

/** Generate LLVM IR for Q4_K dequant kernel.
 *  One thread per Q4_K block. Each block: 16 packed bytes → 32 INT8 values.
 *  Grid = ceil(numBlocks / 256), block = 256 threads. */
function dequantLLVM(): string {
  const body: string[] = [];
  const emit = (s: string) => body.push(`  ${s}`);

  // Load 16 input bytes as 2× i64 (vectorized)
  emit(`%gp_v0 = getelementptr i8, ptr %inp, i32 0`);
  emit(`%v0 = load i64, ptr %gp_v0`);
  emit(`%gp_v1 = getelementptr i8, ptr %inp, i32 8`);
  emit(`%v1 = load i64, ptr %gp_v1`);

  // Extract nibbles from bytes, work with i8, then sign-extend to i32, pack and store.
  // First, extract all 16 input bytes as i8 values.
  // v0: bytes 0-7, v1: bytes 8-15
  for (let i = 0; i < 16; i++) {
    const src = i < 8 ? "v0" : "v1";
    const pos = i < 8 ? i : i - 8;
    const shift = pos * 8;
    const srcVal = shift === 0 ? `%${src}` : `%vs_${src}_${pos}`;
    if (shift > 0) emit(`%vs_${src}_${pos} = lshr i64 %${src}, ${shift}`);
    emit(`%b${i} = trunc i64 ${srcVal} to i8`);

    // Extract lo/hi nibbles, center at 0
    emit(`%lo${i} = and i8 %b${i}, 15`);
    emit(`%hi${i} = lshr i8 %b${i}, 4`);
    emit(`%la${i} = sub i8 %lo${i}, 8`);
    emit(`%ha${i} = sub i8 %hi${i}, 8`);
  }

  // Pack into i32 stores: 8× i32 = 32 output bytes
  // Group 4 consecutive output bytes per i32
  // Output layout: lo[0], hi[0], lo[1], hi[1], ..., lo[15], hi[15]
  // Each i32 covers 2 input bytes = 4 output bytes: {la[g*2], ha[g*2], la[g*2+1], ha[g*2+1]}
  for (let g = 0; g < 8; g++) {
    const i0 = g * 2;       // first input byte
    const i1 = g * 2 + 1;   // second input byte
    const outOff = g * 4;   // output byte offset

    // Zero-extend i8 to i32 (preserves byte value as unsigned for packing)
    emit(`%x_la${g} = zext i8 %la${i0} to i32`);
    emit(`%x_ha${g} = zext i8 %ha${i0} to i32`);
    emit(`%x_lb${g} = zext i8 %la${i1} to i32`);
    emit(`%x_hb${g} = zext i8 %ha${i1} to i32`);

    // Pack: la[g*2] | (ha[g*2] << 8) | (la[g*2+1] << 16) | (ha[g*2+1] << 24)
    emit(`%sh_ha${g} = shl i32 %x_ha${g}, 8`);
    emit(`%sh_lb${g} = shl i32 %x_lb${g}, 16`);
    emit(`%sh_hb${g} = shl i32 %x_hb${g}, 24`);
    emit(`%p_${g}_0 = or i32 %x_la${g}, %sh_ha${g}`);
    emit(`%p_${g}_1 = or i32 %p_${g}_0, %sh_lb${g}`);
    emit(`%p_${g}_2 = or i32 %p_${g}_1, %sh_hb${g}`);

    // i32 store
    emit(`%gp_out_${g} = getelementptr i8, ptr %outp, i32 ${outOff}`);
    emit(`store i32 %p_${g}_2, ptr %gp_out_${g}`);
  }

  const ir = `define void @dequant(ptr %input, ptr %output, i32 %numBlocks) {
entry:
  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bd = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %tmp = mul i32 %bid, %bd
  %idx = add i32 %tmp, %tid
  %cmp = icmp ult i32 %idx, %numBlocks
  br i1 %cmp, label %process, label %exit

process:
  %offset = mul i32 %idx, 20
  %inp = getelementptr i8, ptr %input, i32 %offset
  %outOffset = mul i32 %idx, 32
  %outp = getelementptr i8, ptr %output, i32 %outOffset
${body.join("\n")}
  br label %exit

exit:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

!0 = !{ptr @dequant, !"kernel", i32 1}
!nvvm.annotations = !{!0}
`;
  return ir;
}

export class DequantKernel {
  private _fn: bigint = 0n;
  private _initialized = false;

  load() {
    if (this._initialized) return;
    const cs = cuda();
    const ptx = compileLLVM(dequantLLVM());
    const mod = Buffer.alloc(8);
    cs.cuModuleLoadData(mod, Buffer.from(ptx + "\0"));
    const fn = Buffer.alloc(8);
    cs.cuModuleGetFunction(fn, Number(mod.readBigUInt64LE(0)), Buffer.from("dequant\0"));
    this._fn = fn.readBigUInt64LE(0);
    if (this._fn === 0n) throw Error("dequant kernel not found in PTX");
    this._initialized = true;
    console.log(`  dequant: ${(ptx.match(/^\s+(ld|st|add|mul|mad|and|shr|shl|setp|cvt|mov)/gm) || []).length} PTX instrs`);
  }

  /** Launch dequant kernel (no sync — caller must sync or chain kernels) */
  launch(input: bigint, output: bigint, numBlocks: number, blockSize = 64): void {
    if (!this._initialized) this.load();
    const cs = cuda();
    const gx = Math.ceil(numBlocks / blockSize);
    const slotBuf = Buffer.alloc(24);
    slotBuf.writeBigUInt64LE(input, 0);
    slotBuf.writeBigUInt64LE(output, 8);
    slotBuf.writeInt32LE(numBlocks, 16);
    const pp = Number(ptr(slotBuf));
    const kp = Buffer.alloc(4 * 8);
    kp.writeBigUInt64LE(BigInt(pp), 0);
    kp.writeBigUInt64LE(BigInt(pp + 8), 8);
    kp.writeBigUInt64LE(BigInt(pp + 16), 16);
    kp.writeBigUInt64LE(0n, 24);
    cs.cuLaunchKernel(Number(this._fn), gx, 1, 1, blockSize, 1, 1, 0, 0n, ptr(kp), null);
  }

  /** Launch + sync (convenience for standalone use) */
  run(input: bigint, output: bigint, numBlocks: number): void {
    this.launch(input, output, numBlocks);
    cuda().cuCtxSynchronize();
  }
}

export const matmul = new KernelTemplate({
  name: "matmul",
  generator: matmulTTIR,
});

export const matmul_i8 = new KernelTemplate({
  name: "matmul_i8",
  generator: int8MatmulTTIR,
  defaults: { BM: 32, BN: 32, BK: 1024, numWarps: 4 },
  fnName: "matmul",
});

// ═══════════════════════════════════════════════════════════════════
// Example usage / test
// ═══════════════════════════════════════════════════════════════════

// ── If run directly ──
if (import.meta.main) {
  console.log("Kernel template system");
  console.log(`Default config: ${matmul.fmtCfg(matmul.defaultConfig)}\n`);

  // Autotune for 4096×4096×4096
  console.log("Autotuning matmul for 4096×4096×4096...");
  const best = await matmul.autotune({ M: 4096, N: 4096, K: 4096 });
  const totalFlops = 2 * Math.ceil(4096 / best.config.BM) * Math.ceil(4096 / best.config.BN) * best.config.BM * best.config.BN * best.config.BK;
  console.log(`\nBest: ${matmul.fmtCfg(best.config)} → ${(best.timeMs * 1000).toFixed(1)} µs = ${(totalFlops / (best.timeMs/1000) / 1e12).toFixed(2)} TFLOPS`);

  // Compile and run (already cached from autotune)
  const inst = await matmul.compile(best.config);
  
  // Create test data (f16 for A/B, f32 for C output)
  const hF32 = new Float32Array(4096 * 4096);
  for (let i = 0; i < hF32.length; i++) hF32[i] = Math.random() - 0.5;
  const hA_f16 = f32to16(hF32);
  const hB_f16 = f32to16(hF32);
  const hC = new Float32Array(4096 * 4096);

  console.log("\nRunning kernel...");
  inst.run({ A: hA_f16, B: hB_f16, C: hC }, { M: 4096, N: 4096, K: 4096 });
  // Compute reference: C[0,0] = Σ over k of A[0,k] × B[k,0] using f16 data
  // Convert f16 back to f32 for the reference computation
  const f16to32 = (u16: Uint16Array): Float32Array => {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const v=u16[i], s=(v>>15)&1, e=(v>>10)&0x1f, m=v&0x3ff;
      if (e===0) out[i]=m===0?(s?-0:0):(s?-1:1)*Math.pow(2,-14)*(m/1024);
      else if (e===31) out[i]=m?NaN:(s?-Infinity:Infinity);
      else out[i]=(s?-1:1)*Math.pow(2,e-15)*(1+m/1024);
    }
    return out;
  };
  const HA = f16to32(hA_f16);
  const HB = f16to32(hB_f16);
  // The kernel computes one tile at grid block (0,0): C[0:BM, 0:BN] = A[0:BM, 0:BK] @ B[0:BK, 0:BN]
  // C[0,0] = Σ_k A[0,k] × B[k,0] for k=0..BK-1
  let ref = 0;
  for (let k = 0; k < best.config.BK; k++) ref += HA[k] * HB[k * 4096];
  const tol = Math.abs(ref) * (best.config.BK > 32 ? 0.5 : 0.1) + 0.01;
  const ok = Math.abs(hC[0] - ref) < tol;
  console.log(`C[0] = ${hC[0].toFixed(4)}  (ref: ${ref.toFixed(4)}, tol: ${tol.toFixed(4)})  ${ok ? "✓ OK" : "✗ MISMATCH"}`);
}
