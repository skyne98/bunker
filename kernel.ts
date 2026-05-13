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
    }).symbols;
    _cs.cuInit(0);
    const dev = new Int32Array(1);
    _cs.cuDeviceGet(dev, 0);
    _cs.cuCtxCreate_v2(Buffer.alloc(8), 0, BigInt(dev[0]));
  }
  return _cs;
}

// ═══════════════════════════════════════════════════════════════════
// Compilation
// ═══════════════════════════════════════════════════════════════════

const WRAPPER = "/tmp/triton_wrap";

function compileTTIR(ttir: string, numWarps: number): { ptx: string; shmem: number } {
  const f = join(tmpdir(), `k_${process.pid}_${Math.random().toString(36).slice(2)}.mlir`);
  writeFileSync(f, ttir);
  const out = execSync(
    `LD_LIBRARY_PATH=/nix/store/ixhlv41i2wpl84xgjcks061dz4yssbg3-zlib-1.3.2/lib:/nix/store/bfwqrbwqpbnsdbgf86gz8pn8vvddci3i-libxml2-2.13.8/lib ${WRAPPER} ${f} ${numWarps}`,
    { cwd: __dirname, encoding: "utf-8", maxBuffer: 50 * 1024 * 1024 }
  );
  // Estimate shmem from tile size: roughly BM*BK*2 + BK*BN*2 bytes for f16 tiles
  return { ptx: out, shmem: 32768 };
}

// ═══════════════════════════════════════════════════════════════════
// Kernel Instance — A compiled, ready-to-run kernel
// ═══════════════════════════════════════════════════════════════════

export class KernelInstance {
  public readonly cfg: TileConfig;
  public readonly ptx: string;
  public readonly shmem: number;
  private _mod: bigint = 0n;
  private _fn: bigint = 0n;
  private _initialized = false;

  constructor(cfg: TileConfig, ptx: string, shmem: number) {
    this.cfg = cfg;
    this.ptx = ptx;
    this.shmem = shmem;
  }

  private ensureLoaded() {
    if (this._initialized) return;
    const cs = cuda();
    const mod = Buffer.alloc(8);
    cs.cuModuleLoadData(mod, ptr(Buffer.from(this.ptx)));
    this._mod = mod.readBigUInt64LE(0);
    const fn = Buffer.alloc(8);
    cs.cuModuleGetFunction(fn, Number(this._mod), ptr(Buffer.from("matmul\0")));
    this._fn = fn.readBigUInt64LE(0);
    this._initialized = true;
  }

  /** Launch the kernel. `bufs` maps argument names to typed arrays (uploaded to GPU) or numbers (passed directly). */
  run(bufs: Record<string, ArrayBufferView | number>, dims: MatmulDims): void {
    this.ensureLoaded();
    const cs = cuda();

    // Allocate GPU memory for each buffer arg, upload data
    const devPtrs: Map<string, bigint> = new Map();
    const devAllocs: bigint[] = [];
    const argSlots: Buffer[] = [];

    for (const [name, val] of Object.entries(bufs)) {
      if (typeof val === "number") {
        const slot = Buffer.alloc(8);
        slot.writeInt32LE(val, 0);
        argSlots.push(slot);
      } else {
        const db = Buffer.alloc(8);
        const sz = BigInt(val.byteLength);
        cs.cuMemAlloc_v2(db, sz);
        const dp = db.readBigUInt64LE(0);
        cs.cuMemcpyHtoD_v2(Number(dp), val, sz);
        devPtrs.set(name, dp);
        devAllocs.push(dp);
        const slot = Buffer.alloc(8);
        slot.writeBigUInt64LE(dp, 0);
        argSlots.push(slot);
      }
    }

    // Build kernel params array
    const pb = Buffer.concat(argSlots.map(s => {
      const buf = Buffer.alloc(8);
      // Each param is an 8-byte slot with the pointer or value
      return s;
    }));
    
    // Pad to 5 params + null (CUDA kernel expects this many)
    while (argSlots.length < 5) {
      const empty = Buffer.alloc(8);
      argSlots.push(empty);
    }

    const pp = Number(ptr(pb));
    const kp = Buffer.alloc((argSlots.length + 1) * 8);
    for (let i = 0; i < argSlots.length; i++) {
      kp.writeBigUInt64LE(pp > 0 ? BigInt(pp + i * 8) : 0n, i * 8);
    }
    kp.writeBigUInt64LE(0n, argSlots.length * 8);

    // Compute grid dimensions
    const gx = Math.ceil(dims.M / this.cfg.BM);
    const gy = Math.ceil(dims.N / this.cfg.BN);
    const blockThreads = this.cfg.numWarps * 32;

    // Launch
    cs.cuLaunchKernel(
      Number(this._fn), gx, gy, 1, blockThreads, 1, 1, this.shmem, 0n, ptr(kp), null
    );
    cs.cuCtxSynchronize();

    // Readback results for ArrayBuffer outputs
    for (const [name, val] of Object.entries(bufs)) {
      if (typeof val !== "number") {
        const dp = devPtrs.get(name)!;
        cs.cuMemcpyDtoH_v2(ptr(Buffer.from((val as ArrayBufferView).buffer)), Number(dp), BigInt(val.byteLength));
      }
    }

    // Free device memory
    for (const dp of devAllocs) cs.cuMemFree_v2(Number(dp));
  }

  /** Benchmark this kernel for given dimensions. Returns time in ms. */
  benchmark(dims: MatmulDims, iters = 10): number {
    this.ensureLoaded();
    const cs = cuda();

    // Allocate dummy data
    const sz = BigInt(128 * 1024 * 1024);
    const dA = Buffer.alloc(8); cs.cuMemAlloc_v2(dA, sz);
    const dB = Buffer.alloc(8); cs.cuMemAlloc_v2(dB, sz);
    const dC = Buffer.alloc(8); cs.cuMemAlloc_v2(dC, sz);

    const pb = Buffer.alloc(5 * 8);
    pb.writeBigUInt64LE(dA.readBigUInt64LE(0), 0);
    pb.writeBigUInt64LE(dB.readBigUInt64LE(0), 8);
    pb.writeBigUInt64LE(dC.readBigUInt64LE(0), 16);
    const kp = Buffer.alloc(6 * 8);
    const pp = Number(ptr(pb));
    for (let i = 0; i < 5; i++) kp.writeBigUInt64LE(BigInt(pp + i * 8), i * 8);
    kp.writeBigUInt64LE(0n, 40);

    const gx = Math.ceil(dims.M / this.cfg.BM);
    const gy = Math.ceil(dims.N / this.cfg.BN);
    const blockThreads = this.cfg.numWarps * 32;
    const totalFlops = BigInt(2) * BigInt(gx * gy) * BigInt(this.cfg.BM) * BigInt(this.cfg.BN) * BigInt(this.cfg.BK);

    // Warmup
    for (let w = 0; w < 3; w++) cs.cuLaunchKernel(Number(this._fn), gx, gy, 1, blockThreads, 1, 1, this.shmem, 0n, ptr(kp), null);
    cs.cuCtxSynchronize();

    const times: number[] = [];
    for (let i = 0; i < iters; i++) {
      const t0 = performance.now();
      const lr = cs.cuLaunchKernel(Number(this._fn), gx, gy, 1, blockThreads, 1, 1, this.shmem, 0n, ptr(kp), null);
      if (lr !== 0) break;
      cs.cuCtxSynchronize();
      times.push(performance.now() - t0);
    }

    cs.cuMemFree_v2(Number(dA.readBigUInt64LE(0)));
    cs.cuMemFree_v2(Number(dB.readBigUInt64LE(0)));
    cs.cuMemFree_v2(Number(dC.readBigUInt64LE(0)));

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
  public cache = new Map<string, KernelInstance>();

  constructor(opts: {
    name: string;
    generator: TTIRGenerator;
    defaults?: Partial<TileConfig>;
  }) {
    this.name = opts.name;
    this.generator = opts.generator;
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
    const inst = new KernelInstance(full, ptx, shmem);
    this.cache.set(key, inst);
    return inst;
  }

  /** Search over the tunable space to find the fastest config for given dims. */
  async autotune(
    dims: MatmulDims,
    searchSpace?: Partial<TileConfig>[]
  ): Promise<{ config: TileConfig; instance: KernelInstance; timeMs: number }> {
    const space = searchSpace ?? DEFAULT_SEARCH_SPACE;

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

const DEFAULT_SEARCH_SPACE: Partial<TileConfig>[] = [
  { BM: 16, BN: 16, BK: 32, numWarps: 4 },
  { BM: 32, BN: 16, BK: 32, numWarps: 4 },
  { BM: 32, BN: 32, BK: 32, numWarps: 4 },
  { BM: 16, BN: 16, BK: 1024, numWarps: 4 },
  { BM: 32, BN: 16, BK: 1024, numWarps: 4 },
  { BM: 32, BN: 32, BK: 1024, numWarps: 4 },
  { BM: 64, BN: 64, BK: 32, numWarps: 4 },
  { BM: 64, BN: 64, BK: 1024, numWarps: 8 },
  { BM: 128, BN: 128, BK: 32, numWarps: 8 },
  { BM: 128, BN: 128, BK: 64, numWarps: 8 },
  { BM: 64, BN: 128, BK: 32, numWarps: 8 },
  { BM: 128, BN: 64, BK: 32, numWarps: 8 },
];

// ═══════════════════════════════════════════════════════════════════
// Built-in matmul kernel
// ═══════════════════════════════════════════════════════════════════

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

export const matmul = new KernelTemplate({
  name: "matmul",
  generator: matmulTTIR,
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
