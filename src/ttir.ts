// ttir.ts — Pleasant, type-tracking TTIR (Triton MLIR) kernel builder.
//
// Design (see AGENTS.md "Design principles"):
//   - TTIR is the portable surface. It is backend-agnostic; the shim's pass
//     pipeline selects PTX (NVIDIA) or AMDGCN (AMD). Nothing here is NVPTX-specific.
//   - Each Value carries its tensor type, so elementwise ops / casts / reductions
//     infer result types automatically and chain fluently.
//   - The user never manages SSA names or passes shapes to every call.
//
// Two layers are exported:
//   1. `TTIRBuilder`  — the fluent, type-tracking builder (used directly or by the lift)
//   2. `kernel_ttir`  — the arrow-function AST→TTIR lift (pleasant authoring surface)
//
// The shim ABI is `triton_compile(ttir, num_warps)` (3090 default). A target-
// parameterized entry point lives in triton_shim.c once rebuilt; the TTIR text
// produced here is identical regardless of backend.

import ts from "typescript";
import { readFileSync, readdirSync, existsSync, writeFileSync, unlinkSync } from "fs";
import { execSync } from "child_process";
import { join } from "path";
import { tmpdir } from "os";
import { dlopen, ptr as ffiPtr, CString } from "bun:ffi";

// ═══════════════════════════════════════════════════════════════════
// Type model
// ═══════════════════════════════════════════════════════════════════

export type ScalarElem =
  | "i1" | "i8" | "i16" | "i32" | "i64"
  | "f16" | "bf16" | "f32" | "f64";

/** Element type of a tensor: a scalar, a pointer to a scalar, or a tiled pointer. */
export type Elem =
  | ScalarElem
  | { ptr: ScalarElem }                       // !tt.ptr<E>  (kernel param / pointer-tensor element)
  | { tile: { shape: number[]; elem: ScalarElem } }; // !tt.ptr<tensor<MxNxE>>  (tiled pointer)

export interface TensorType {
  /** [] means a 0-d scalar. */
  shape: number[];
  elem: Elem;
}

export function isPtr(e: Elem): e is { ptr: ScalarElem } | { tile: { shape: number[]; elem: ScalarElem } } {
  return typeof e === "object" && e !== null && ("ptr" in (e as any) || "tile" in (e as any));
}
export function isTiledPtr(e: Elem): e is { tile: { shape: number[]; elem: ScalarElem } } {
  return typeof e === "object" && e !== null && "tile" in (e as any);
}
export function isFloat(e: Elem): boolean {
  return e === "f16" || e === "bf16" || e === "f32" || e === "f64";
}
export function isInt(e: Elem): boolean {
  return e === "i1" || e === "i8" || e === "i16" || e === "i32" || e === "i64";
}

/** Render an element type to TTIR text, e.g. `f32` or `!tt.ptr<i8>`. */
function elemText(e: Elem): string {
  if (isTiledPtr(e)) return `!tt.ptr<${typeText({ shape: e.tile.shape, elem: e.tile.elem })}>`;
  if (isPtr(e)) return `!tt.ptr<${e.ptr}>`;
  return e as string;
}

/** Render a tensor type, e.g. `tensor<128x128xf32>` or `i32` (scalar). */
export function typeText(t: TensorType): string {
  if (t.shape.length === 0) return elemText(t.elem);
  const shapeStr = t.shape.join("x");
  const elemStr = elemText(t.elem);
  return `tensor<${shapeStr}x${elemStr}>`;
}

/** A typed SSA value produced by the builder. */
export class Value {
  constructor(
    public readonly name: string,
    public readonly type: TensorType,
  ) {}
  get rank(): number { return this.type.shape.length; }
  get isScalar(): boolean { return this.type.shape.length === 0; }
  /** Element type as a scalar (panics for pointer-tensor values). */
  get elem(): ScalarElem {
    if (isPtr(this.type.elem) || isTiledPtr(this.type.elem))
      throw new Error(`elem: value is pointer-typed, not scalar`);
    return this.type.elem as ScalarElem;
  }
}

function scalar(name: string, e: ScalarElem): Value {
  return new Value(name, { shape: [], elem: e });
}

/** Format a JS number as an MLIR float literal: decimal mantissa, 2-digit exponent. */
function floatLit(v: number): string {
  if (v === 0) return "0.000000e+00";
  if (!isFinite(v)) return v > 0 ? "0x7FF0000000000000" : "0xFFF0000000000000";  // ±inf
  let s = v.toExponential(6);                 // "1.000000e-5", "-1.500000e+0"
  s = s.replace(/e([+-])(\d)$/, "e$1" + "0" + "$2");  // zero-pad exponent to 2 digits
  return s;
} function tensor(name: string, shape: number[], elem: Elem): Value {
  return new Value(name, { shape, elem });
}

// ═══════════════════════════════════════════════════════════════════
// Shim binding
// ═══════════════════════════════════════════════════════════════════

let _shimLib: any = null;
function getShim() {
  if (!_shimLib) {
    _shimLib = dlopen(`${__dirname}/../shim/libtriton_shim.so`, {
      triton_compile: { args: ["ptr", "i32"], returns: "ptr" },
      triton_free: { args: ["ptr"], returns: "void" },
      triton_get_shared_mem_size: { args: [], returns: "i64" },
    }).symbols;
  }
  return _shimLib;
}

export interface CompileResult { ptx: string; shmem: number; }

/** Compile a TTIR module string to PTX via the shim. Throws on shim error. */
export function compileTTIR(ttir: string, numWarps = 4): CompileResult {
  const shim = getShim();
  // If a target is set and the rebuilt shim exposes triton_compile_targeted,
  // use it; otherwise fall back to the 3090-default triton_compile.
  if (_target && _shimTargeted) {
    const buf = Buffer.from(ttir + "\0", "utf-8");
    const b = _target;
    const rp = _shimTargeted.triton_compile_targeted(
      ffiPtr(buf), numWarps, ffiPtr(Buffer.from(b.backend + "\0")),
      ffiPtr(Buffer.from(b.arch + "\0")), ffiPtr(Buffer.from(b.features + "\0")));
    return finishShimResult(rp, shim);
  }
  const buf = Buffer.from(ttir + "\0", "utf-8");
  const rp = shim.triton_compile(ffiPtr(buf), numWarps);
  return finishShimResult(rp, shim);
}

function finishShimResult(rp: number, shim: any): CompileResult {
  if (!rp) throw new Error("triton_compile returned null (shim crash?)");
  const result = new CString(rp);
  shim.triton_free(rp);
  const str = result.toString();
  if (str.startsWith("ERROR:")) {
    const msg = str.length > 4000 ? str.slice(0, 4000) + "\n…(truncated)" : str;
    throw new Error(msg);
  }
  const shmem = Number(shim.triton_get_shared_mem_size());
  return { ptx: str, shmem };
}

/** Backend target descriptor for `triton_compile_targeted`. */
export interface TritonTarget {
  backend: "cuda" | "rocm";
  arch: string;       // "86" (sm_86) for cuda; "gfx90a" etc. for rocm
  features: string;   // "+ptx75" for cuda; "" for rocm
}

let _target: TritonTarget | null = null;
let _shimTargeted: any = null;
/** Default target (RTX 3090). Override for other NVIDIA GPUs or AMD. */
export function setTarget(t: TritonTarget): void {
  _target = t;
  if (!_shimTargeted) {
    try {
      _shimTargeted = dlopen(`${__dirname}/../shim/libtriton_shim.so`, {
        triton_compile_targeted: { args: ["ptr", "i32", "ptr", "ptr", "ptr"], returns: "ptr" },
      }).symbols;
    } catch {
      throw new Error("setTarget: shim lacks triton_compile_targeted — " +
        "rebuild libtriton_shim.so from the updated triton_shim.c (see build_shim.sh)");
    }
  }
}
/** Reset to the 3090 default (uses triton_compile). */
export function resetTarget(): void { _target = null; }

/** Compile with an explicit target (one-shot; does not change the global). */
export function compileTTIRTargeted(ttir: string, t: TritonTarget, numWarps = 4): CompileResult {
  const prev = _target; setTarget(t); try { return compileTTIR(ttir, numWarps); } finally { _target = prev; }
}


// ═══════════════════════════════════════════════════════════════════
// CUDA driver helpers — the pleasant launch surface hides these.
// ═══════════════════════════════════════════════════════════════════
//
// KEY FINDING: the driver's plain cuModuleLoadData(PTX) JIT rejects some
// valid PTX with CUDA_ERROR_INVALID_PTX (rc=218), but cuModuleLoadDataEx
// with CU_JIT_OPTIMIZATION_LEVEL=4 accepts and compiles the SAME PTX
// correctly. So always load PTX via loadPTX() (which uses the Ex form).
// This makes scf.for K-loops, tt.advance, and chained tt.dot all work.

let _cuda: any = null;
/** Shared CUDA driver binding with one context (the pleasant launch surface). */
export function cu() {
  if (!_cuda) {
    _cuda = dlopen("/run/opengl-driver/lib/libcuda.so", {
      cuInit: { args: ["u32"], returns: "i32" },
      cuDeviceGet: { args: ["ptr", "i32"], returns: "i32" },
      cuCtxCreate_v2: { args: ["ptr", "u32", "i64"], returns: "i32" },
      cuModuleLoadData: { args: ["ptr", "ptr"], returns: "i32" },
      cuModuleLoadDataEx: { args: ["ptr", "ptr", "u32", "ptr", "ptr"], returns: "i32" },
      cuModuleGetFunction: { args: ["ptr", "i64", "ptr"], returns: "i32" },
      cuFuncSetAttribute: { args: ["i64", "i32", "i32"], returns: "i32" },
      cuMemAlloc_v2: { args: ["ptr", "i64"], returns: "i32" },
      cuMemcpyHtoD_v2: { args: ["i64", "ptr", "i64"], returns: "i32" },
      cuMemcpyDtoH_v2: { args: ["ptr", "i64", "i64"], returns: "i32" },
      cuLaunchKernel: { args: ["i64", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "ptr", "ptr", "ptr"], returns: "i32" },
      cuCtxSynchronize: { args: [], returns: "i32" },
      cuMemFree_v2: { args: ["i64"], returns: "i32" },
    }).symbols;
    _cuda.cuInit(0);
    const dev = new Int32Array(1); _cuda.cuDeviceGet(dev, 0);
    const ctxBuf = Buffer.alloc(8);
    const ctxRc = _cuda.cuCtxCreate_v2(ctxBuf, 0, BigInt(dev[0]));
    if (ctxRc !== 0) {
      const rcName: Record<number, string> = { 2: "OUT_OF_MEMORY", 100: "NO_DEVICE", 101: "INVALID_DEVICE", 201: "INVALID_SOURCE", 999: "UNKNOWN" };
      throw new Error(`cuCtxCreate failed (rc=${ctxRc} ${rcName[ctxRc] ?? ""}) — ` +
        `GPU may be out of memory or in use by another process. ` +
        `Check: nvidia-smi --query-compute-apps=pid,used_memory --format=csv`);
    }
  }
  return _cuda;
}

function cuda() { return cu(); }

/** Allocate a device buffer; returns the device pointer as a BigInt. */
export function cuAlloc(bytes: number | bigint): bigint {
  const cs = cu();
  const b = Buffer.alloc(8);
  if (cs.cuMemAlloc_v2(b, BigInt(bytes)) !== 0) throw new Error(`cuMemAlloc failed (${bytes} bytes)`);
  return b.readBigUInt64LE(0);
}
/** Copy host → device. */
export function cuHtoD(devPtr: bigint, hostBuf: Buffer | ArrayBuffer, bytes?: number | bigint): void {
  const buf = hostBuf instanceof ArrayBuffer ? Buffer.from(hostBuf) : hostBuf;
  if (cu().cuMemcpyHtoD_v2(devPtr, buf, BigInt(bytes ?? buf.byteLength)) !== 0) throw new Error("cuMemcpyHtoD failed");
}
/** Copy device → host. */
export function cuDtoH(hostBuf: Buffer | ArrayBuffer, devPtr: bigint, bytes?: number | bigint): void {
  const buf = hostBuf instanceof ArrayBuffer ? Buffer.from(hostBuf) : hostBuf;
  if (cu().cuMemcpyDtoH_v2(buf, devPtr, BigInt(bytes ?? buf.byteLength)) !== 0) throw new Error("cuMemcpyDtoH failed");
}
/** Free a device buffer. */
export function cuFree(devPtr: bigint): void { cu().cuMemFree_v2(devPtr); }
/** Synchronize the context. */
export function cuSync(): number { return cu().cuCtxSynchronize(); }

/** Launch a loaded kernel. `args` is an array of BigInt device pointers and numbers. */
export function cuLaunch(k: LoadedKernel, grid: [number,number,number], block: [number,number,number], args: (bigint | number)[]): number {
  const cs = cu();
  // Pack scalar args into a flat buffer (8 bytes each) and build a pointer array.
  const n = args.length;
  const slot = Buffer.alloc(n * 8);
  for (let i = 0; i < n; i++) {
    const a = args[i];
    if (typeof a === "bigint") slot.writeBigUInt64LE(a, i * 8);
    else slot.writeBigInt64LE(BigInt(a), i * 8);
  }
  // The shim's kernels may take extra trailing slots (global scratch); pad to be safe.
  const padded = Buffer.alloc((n + 3) * 8);
  slot.copy(padded);
  const pp = Number(ffiPtr(padded));
  const kp = Buffer.alloc((n + 4) * 8);
  for (let i = 0; i < n + 3; i++) kp.writeBigUInt64LE(BigInt(pp + i * 8), i * 8);
  return cs.cuLaunchKernel(k.fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], k.shmem, 0n, ffiPtr(kp), null);
}

export interface LoadedKernel { module: number; fn: number; shmem: number; }

/** Find ptxas on the system (PATH, then common Nix store locations). */
function findPtxas(): string | null {
  try { const p = execSync("command -v ptxas", { stdio: ["ignore","pipe","ignore"] }).toString().trim(); if (p) return p; } catch {}
  // Nix store fallback (dir name contains both 'cuda' and 'nvcc').
  try {
    for (const d of readdirSync("/nix/store")) {
      if (d.includes("cuda") && d.includes("nvcc")) {
        const p = `/nix/store/${d}/bin/ptxas`;
        if (existsSync(p)) return p;
      }
    }
  } catch {}
  return null;
}

let _defaultArch = "sm_86";   // RTX 3090 = compute 8.6. Set via setTargetArch().
/** Override the default ptxas arch (e.g. "sm_80", "sm_90"). */
export function setTargetArch(arch: string): void { _defaultArch = arch; }

/**
 * Load PTX text into a CUDA module. The driver's in-process PTX JIT
 * (`cuModuleLoadDataEx` with OPTIMIZATION_LEVEL=4) is nondeterministic for
 * some valid PTX (it intermittently returns INVALID_PTX, rc=218), so this
 * falls back to assembling via `ptxas` → cubin and loading the cubin, which
 * is always reliable. Throws if both paths fail.
 */
export function loadPTX(ptx: string, funcName: string, shmem: number): LoadedKernel {
  const cs = cuda();
  const img = Buffer.from(ptx + "\0", "utf-8");
  const mod = Buffer.alloc(8);
  // CU_JIT_OPTIMIZATION_LEVEL = 7, value = pointer to uint32(4).
  const optLevel = new Uint32Array([4]);
  const opts = new Int32Array([7]);
  const vals = new BigUint64Array([BigInt(ffiPtr(optLevel))]);
  let rc = cs.cuModuleLoadDataEx(ffiPtr(mod), ffiPtr(img), 1, ffiPtr(opts), ffiPtr(vals));
  if (rc !== 0) {
    // Fallback: assemble PTX → cubin via ptxas (always reliable).
    const ptxas = findPtxas();
    if (!ptxas) throw new Error(`cuModuleLoadDataEx failed (rc=${rc}) and ptxas not found`);
    const ptxPath = join(tmpdir(), `k${process.pid}_${Math.random().toString(36).slice(2)}.ptx`);
    const cubinPath = ptxPath + ".cubin";
    writeFileSync(ptxPath, ptx);
    try {
      try {
        execSync(`"${ptxas}" -arch=${_defaultArch} -o "${cubinPath}" "${ptxPath}"`, { stdio: ["ignore","pipe","pipe"] });
      } catch (e: any) {
        throw new Error(`cuModuleLoadDataEx rc=${rc}; ptxas fallback failed: ${e.stderr?.toString() ?? e.message}`);
      }
      const cubin = readFileSync(cubinPath);
      const mod2 = Buffer.alloc(8);
      const rc2 = cs.cuModuleLoadData(ffiPtr(mod2), ffiPtr(cubin));
      if (rc2 !== 0) throw new Error(`cubin load failed: rc=${rc2}`);
      mod.writeBigUInt64LE(mod2.readBigUInt64LE(0), 0);
    } finally {
      try { unlinkSync(ptxPath); } catch {}
      try { unlinkSync(cubinPath); } catch {}
    }
  }
  const fn = Buffer.alloc(8);
  const gfr = cs.cuModuleGetFunction(fn, Number(mod.readBigUInt64LE(0)), ffiPtr(Buffer.from(funcName + "\0")));
  if (gfr !== 0) throw new Error(`cuModuleGetFunction(${funcName}) failed: rc=${gfr}`);
  // Allow the kernel's dynamic shared memory allocation.
  if (shmem > 0) cs.cuFuncSetAttribute(Number(fn.readBigUInt64LE(0)), 8, shmem);
  return { module: Number(mod.readBigUInt64LE(0)), fn: Number(fn.readBigUInt64LE(0)), shmem };
}

/** One-shot: compile TTIR → PTX → load into a CUDA module + function handle. */
export function compileAndLoad(ttir: string, funcName = "kernel", numWarps = 4): LoadedKernel {
  const { ptx, shmem } = compileTTIR(ttir, numWarps);
  return loadPTX(ptx, funcName, shmem);
}

// ═══════════════════════════════════════════════════════════════════
// TTIRBuilder — fluent, type-tracking
// ═══════════════════════════════════════════════════════════════════

/** A kernel parameter descriptor. */
export interface Param {
  name: string;
  elem: ScalarElem | { ptr: ScalarElem };
}

/** Internal: param rendered as a TTIR function argument type. */
function paramTypeText(p: Param): string {
  if (typeof p.elem === "object") return `!tt.ptr<${p.elem.ptr}>`;
  return p.elem;
}

export class TTIRBuilder {
  private lines: string[] = [];
  private uid = 0;
  private params: Param[] = [];

  /** Reset for a fresh kernel. */
  reset() { this.lines = []; this.uid = 0; this.params = []; return this; }

  private fresh(prefix = "t"): string { return `${prefix}${this.uid++}`; }
  private emit(line: string): void { this.lines.push("    " + line); }

  /** Declare a kernel parameter; returns its SSA value. */
  param(name: string, elem: ScalarElem | { ptr: ScalarElem }): Value {
    this.params.push({ name, elem });
    const idx = this.params.length - 1;
    return new Value(`%arg${idx}`, { shape: [], elem });
  }

  // ───────────────────────────────────────────────────────────────
  // Constants & program info
  // ───────────────────────────────────────────────────────────────

  /** Scalar i32 constant. */
  i32(v: number): Value { const r = this.fresh("c"); this.emit(`%${r} = arith.constant ${v} : i32`); return scalar(`%${r}`, "i32"); }
  /** Scalar f32 constant. */
  f32(v: number): Value { const r = this.fresh("c"); this.emit(`%${r} = arith.constant ${floatLit(v)} : f32`); return scalar(`%${r}`, "f32"); }
  /** Scalar f16 constant. */
  f16(v: number): Value { const r = this.fresh("c"); this.emit(`%${r} = arith.constant ${floatLit(v)} : f16`); return scalar(`%${r}`, "f16"); }
  bf16(v: number): Value { const r = this.fresh("c"); this.emit(`%${r} = arith.constant ${floatLit(v)} : bf16`); return scalar(`%${r}`, "bf16"); }
  /** Scalar i64 constant. */
  i64(v: number): Value { const r = this.fresh("c"); this.emit(`%${r} = arith.constant ${v} : i64`); return scalar(`%${r}`, "i64"); }
  /** Scalar `index`-type constant (for scf.for bounds/IV). */
  index(v: number): Value { const r = this.fresh("c"); this.emit(`%${r} = arith.constant ${v} : index`); return new Value(`%${r}`, { shape: [], elem: "index" as any }); }
  /** Scalar f64 constant. */
  f64(v: number): Value { const r = this.fresh("c"); this.emit(`%${r} = arith.constant ${floatLit(v)} : f64`); return scalar(`%${r}`, "f64"); }
  /** Scalar boolean i1. */
  bool(v: boolean): Value { const r = this.fresh("c"); this.emit(`%${r} = arith.constant ${v ? "true" : "false"} : i1`); return scalar(`%${r}`, "i1"); }

  /** `tt.make_range {start, end}` → tensor<(end-start)xi32>. */
  arange(start: number, end: number): Value {
    const n = end - start;
    const r = this.fresh("r");
    this.emit(`%${r} = tt.make_range {end = ${end} : i32, start = ${start} : i32} : tensor<${n}xi32>`);
    return tensor(`%${r}`, [n], "i32");
  }

  /** Dense-zero tensor of the given shape/element. */
  zeros(shape: number[], elem: ScalarElem = "f32"): Value {
    const r = this.fresh("z");
    const zero = isFloat(elem) ? "0.000000e+00" : "0";
    this.emit(`%${r} = arith.constant dense<${zero}> : ${typeText({ shape, elem })}`);
    return tensor(`%${r}`, shape, elem);
  }

  /** Broadcast a scalar to a tensor of `shape`. */
  splat(s: Value, shape: number[], elem?: ScalarElem): Value {
    if (!s.isScalar) throw new Error(`splat: expected scalar, got rank ${s.rank}`);
    const outElem = elem ?? (s.elem as ScalarElem);
    const r = this.fresh("s");
    this.emit(`%${r} = tt.splat ${s.name} : ${typeText(s.type)} -> ${typeText({ shape, elem: outElem })}`);
    return tensor(`%${r}`, shape, outElem);
  }

  /**
   * Broadcast a value (scalar or any tensor) to `targetShape`.
   * Handles: scalar → splat, [128]→reshape[1,128]→broadcast, [1]→[1,1]→broadcast, etc.
   */
  broadcastTo(a: Value, targetShape: number[]): Value {
    if (a.type.shape.join("x") === targetShape.join("x")) return a;
    // Scalar → splat directly
    if (a.isScalar) return this.splat(a, targetShape, a.elem as ScalarElem);
    // Pad with leading 1s to match target rank
    if (a.rank < targetShape.length) {
      const padShape = new Array(targetShape.length - a.rank).fill(1).concat(a.type.shape);
      a = this.reshape(a, padShape);
    }
    // Check broadcastable: each dim must be 1 or equal
    const canBroadcast = a.type.shape.every((d, i) => d === 1 || d === targetShape[i]);
    if (!canBroadcast) throw new Error(`broadcastTo: cannot broadcast ${a.type.shape} to ${targetShape}`);
    return this.broadcast(a, targetShape);
  }

  /** `tt.get_program_id` along axis (0=x, 1=y, 2=z). */
  programId(axis: 0 | 1 | 2 = 0): Value {
    const dim = axis === 0 ? "x" : axis === 1 ? "y" : "z";
    const r = this.fresh("pid");
    this.emit(`%${r} = tt.get_program_id ${dim} : i32`);
    return scalar(`%${r}`, "i32");
  }

  /** `tt.num_programs` along axis. */
  numPrograms(axis: 0 | 1 | 2 = 0): Value {
    const dim = axis === 0 ? "x" : axis === 1 ? "y" : "z";
    const r = this.fresh("np");
    this.emit(`%${r} = tt.num_programs ${dim} : i32`);
    return scalar(`%${r}`, "i32");
  }

  // ───────────────────────────────────────────────────────────────
  // Tiled pointers & memory
  // ───────────────────────────────────────────────────────────────

  /**
   * `tt.make_tensor_ptr` — create a tiled pointer to a tensor block.
   *   base:    scalar !tt.ptr<E> kernel argument
   *   shape:   logical shape of the full tensor [D0, D1, ...]
   *   strides: row strides per dim (in elements)
   *   offsets: starting offsets per dim (i32 tensors or scalars)
   *   elem:    element type of the tile
   *   tileShape: shape of the tile to load (e.g. [128, 128])
   *   order:   dimension order (default row-major [1, 0])
   */
  makeTensorPtr(
    base: Value,
    shape: number[],
    strides: number[],
    offsets: Value[],
    tileShape: number[],
    elem: ScalarElem,
    order: number[] = tileShape.map((_, i) => tileShape.length - 1 - i),
  ): Value {
    if (!base.isScalar || !isPtr(base.type.elem)) throw new Error(`makeTensorPtr: base must be !tt.ptr<E> scalar`);
    const shapeConsts = shape.map(s => this.i64(s).name).join(", ");
    const strideConsts = strides.map(s => this.i64(s).name).join(", ");
    const offsetVals = offsets.map(o => o.name).join(", ");
    const r = this.fresh("tp");
    this.emit(`%${r} = tt.make_tensor_ptr ${base.name}, [${shapeConsts}], [${strideConsts}], [${offsetVals}] {order = array<i32: ${order.join(", ")}>} : ${elemText({ tile: { shape: tileShape, elem } })}`);
    // A tiled pointer is a scalar SSA of type !tt.ptr<tensor<MxNxE>>.
    return new Value(`%${r}`, { shape: [], elem: { tile: { shape: tileShape, elem } } });
  }

  /** `tt.advance` — move a tiled pointer by per-dim offsets. */
  /** `tt.advance` — move a tiled pointer by per-dim offsets. */
  advance(tp: Value, offsets: Value[]): Value {
    if (!isTiledPtr(tp.type.elem)) throw new Error(`advance: expected tiled pointer`);
    const r = this.fresh("adv");
    this.emit(`%${r} = tt.advance ${tp.name}, [${offsets.map(o => o.name).join(", ")}] : ${elemText(tp.type.elem)}`);
    return new Value(`%${r}`, tp.type);
  }

  /**
   * `tt.load` on a tiled pointer or pointer-tensor. Returns a tensor value.
   *   boundaryCheck: dims to bounds-check (tiled pointer only, default none)
   *   padding: padding value code (0 = zero, 1 = NaN, 2 = rounding)
   *   mask/other: for pointer-tensor form
   */
  load(tp: Value, opts: { boundaryCheck?: number[]; padding?: 0 | 1 | 2; mask?: Value; other?: Value } = {}): Value {
    if (!isPtr(tp.type.elem) && !isTiledPtr(tp.type.elem))
      throw new Error(`load: expected tiled pointer or pointer tensor`);
    const r = this.fresh("ld");
    if (isTiledPtr(tp.type.elem)) {
      const tile = tp.type.elem.tile;
      const attrs: string[] = [];
      if (opts.boundaryCheck && opts.boundaryCheck.length) attrs.push(`boundaryCheck = array<i32: ${opts.boundaryCheck.join(", ")}>`);
      if (opts.padding !== undefined) attrs.push(`padding = ${opts.padding} : i32`);
      const attrStr = attrs.length ? ` {${attrs.join(", ")}}` : "";
      this.emit(`%${r} = tt.load ${tp.name}${attrStr} : ${elemText(tp.type.elem)}`);
      return tensor(`%${r}`, tile.shape, tile.elem);
    }
    // pointer-tensor form: tt.load %ptrs, %mask, %other : tensor<Nx!tt.ptr<E>>
    const n = tp.type.shape[0];
    const e = (tp.type.elem as { ptr: ScalarElem }).ptr;
    const extra = [opts.mask, opts.other].filter(Boolean).map(v => v!.name).join(", ");
    const sep = extra ? ", " + extra : "";
    this.emit(`%${r} = tt.load ${tp.name}${sep} : ${typeText(tp.type)}`);
    return tensor(`%${r}`, [n], e);
  }

  /** `tt.store` on a tiled pointer or pointer-tensor. */
  store(tp: Value, val: Value, opts: { boundaryCheck?: number[]; mask?: Value } = {}): void {
    if (isTiledPtr(tp.type.elem)) {
      // tiled pointer form: tt.store %tp {boundaryCheck}, %val : !tt.ptr<tensor<...>>
      const attrs: string[] = [];
      if (opts.boundaryCheck && opts.boundaryCheck.length) attrs.push(`boundaryCheck = array<i32: ${opts.boundaryCheck.join(", ")}>`);
      const attrStr = attrs.length ? ` {${attrs.join(", ")}}` : "";
      this.emit(`tt.store ${tp.name}, ${val.name}${attrStr} : ${elemText(tp.type.elem)}`);
      return;
    }
    if (isPtr(tp.type.elem)) {
      // pointer-tensor form: tt.store %ptrs, %val, %mask : tensor<Nx!tt.ptr<E>>
      const maskPart = opts.mask ? `, ${opts.mask.name}` : "";
      this.emit(`tt.store ${tp.name}, ${val.name}${maskPart} : ${typeText(tp.type)}`);
      return;
    }
    throw new Error(`store: expected tiled pointer or pointer tensor`);
  }

  /** `tt.addptr` — elementwise pointer+offset for pointer-tensor style. */
  addptr(ptrs: Value, offsets: Value): Value {
    if (!isPtr(ptrs.type.elem)) throw new Error(`addptr: expected pointer tensor`);
    // Auto-broadcast scalar or [1] offsets to match pointer-tensor shape
    if (offsets.isScalar) offsets = this.splat(offsets, ptrs.type.shape, "i32");
    else if (offsets.type.shape.join("x") !== ptrs.type.shape.join("x"))
      offsets = this.broadcastTo(offsets, ptrs.type.shape);
    const r = this.fresh("ap");
    this.emit(`%${r} = tt.addptr ${ptrs.name}, ${offsets.name} : ${typeText(ptrs.type)}, ${typeText(offsets.type)}`);
    return new Value(`%${r}`, ptrs.type);
  }

  /** Splat a base pointer to a pointer-tensor of length N. */
  splatPtr(base: Value, n: number, elem: ScalarElem = "f32"): Value {
    if (!base.isScalar || !isPtr(base.type.elem)) throw new Error(`splatPtr: base must be !tt.ptr<E>`);
    const r = this.fresh("sp");
    this.emit(`%${r} = tt.splat ${base.name} : !tt.ptr<${elem}> -> ${typeText({ shape: [n], elem: { ptr: elem } })}`);
    return tensor(`%${r}`, [n], { ptr: elem });
  }

  // ───────────────────────────────────────────────────────────────
  // Elementwise arithmetic (type-aware: float vs int)
  // ───────────────────────────────────────────────────────────────

  private elemwise2(a: Value, b: Value, op: string, floatOp: string): Value {
    // Auto-broadcast: scalar or singleton ([1], [1,1], …) operands match any shape.
    if (a.type.shape.join("x") !== b.type.shape.join("x")) {
      const aSingleton = a.isScalar || a.type.shape.every(d => d === 1);
      const bSingleton = b.isScalar || b.type.shape.every(d => d === 1);
      if (aSingleton && bSingleton) {
        // Both singletons — broadcast the lower-rank one up
        if (a.rank <= b.rank) a = this.broadcastTo(a, b.type.shape);
        else b = this.broadcastTo(b, a.type.shape);
      } else if (aSingleton) a = this.broadcastTo(a, b.type.shape);
      else if (bSingleton) b = this.broadcastTo(b, a.type.shape);
    }
    if (a.type.shape.join("x") !== b.type.shape.join("x"))
      throw new Error(`${op}: shape mismatch ${a.type.shape} vs ${b.type.shape}`);
    const e = a.elem;
    const isF = isFloat(e);
    const r = this.fresh("e");
    this.emit(`%${r} = arith.${isF ? floatOp : op} ${a.name}, ${b.name} : ${typeText(a.type)}`);
    return tensor(`%${r}`, a.type.shape, e);
  }

  add(a: Value, b: Value): Value { return this.elemwise2(a, b, "addi", "addf"); }
  sub(a: Value, b: Value): Value { return this.elemwise2(a, b, "subi", "subf"); }
  mul(a: Value, b: Value): Value { return this.elemwise2(a, b, "muli", "mulf"); }

  /** Integer division (signed). */
  divi(a: Value, b: Value): Value { return this.elemwise2(a, b, "divsi", "divsi"); }
  /** Float division. */
  divf(a: Value, b: Value): Value { return this.elemwise2(a, b, "divsi", "divf"); }
  /** Signed remainder. */
  remi(a: Value, b: Value): Value { return this.elemwise2(a, b, "remsi", "remsi"); }

  /** Minimum (type-aware). */
  minimum(a: Value, b: Value): Value { return this.elemwise2(a, b, "minsi", "minimumf"); }
  /** Maximum (type-aware). */
  maximum(a: Value, b: Value): Value { return this.elemwise2(a, b, "maxsi", "maximumf"); }

  // Bitwise (integer only)
  shl(a: Value, b: Value): Value { return this.bitwise(a, b, "shl"); }
  lshr(a: Value, b: Value): Value { return this.bitwise(a, b, "lshr"); }
  ashr(a: Value, b: Value): Value { return this.bitwise(a, b, "ashr"); }
  and(a: Value, b: Value): Value { return this.bitwise(a, b, "andi"); }
  or(a: Value, b: Value): Value { return this.bitwise(a, b, "ori"); }
  xor(a: Value, b: Value): Value { return this.bitwise(a, b, "xori"); }
  private bitwise(a: Value, b: Value, op: string): Value {
    if (!isInt(a.elem)) throw new Error(`${op}: requires integer, got ${a.elem}`);
    const r = this.fresh("b");
    this.emit(`%${r} = arith.${op} ${a.name}, ${b.name} : ${typeText(a.type)}`);
    return tensor(`%${r}`, a.type.shape, a.elem);
  }

  /** Unary negation (type-aware). */
  neg(a: Value): Value {
    const r = this.fresh("n");
    if (isFloat(a.elem)) {
      this.emit(`%${r} = arith.negf ${a.name} : ${typeText(a.type)}`);
    } else {
      const z = this.zeros(a.type.shape, a.elem);
      this.emit(`%${r} = arith.subi ${z.name}, ${a.name} : ${typeText(a.type)}`);
    }
    return tensor(`%${r}`, a.type.shape, a.elem);
  }

  /** Absolute value (type-aware). */
  abs(a: Value): Value {
    const r = this.fresh("abs");
    this.emit(`%${r} = ${isFloat(a.elem) ? "arith.absf" : "arith.absi"} ${a.name} : ${typeText(a.type)}`);
    return tensor(`%${r}`, a.type.shape, a.elem);
  }

  // ───────────────────────────────────────────────────────────────
  // Comparisons (type-aware) → i1 tensor
  // ───────────────────────────────────────────────────────────────

  private cmp(a: Value, b: Value, intPred: string, floatPred: string): Value {
    if (a.type.shape.join("x") !== b.type.shape.join("x")) {
      const aSingleton = a.isScalar || a.type.shape.every(d => d === 1);
      const bSingleton = b.isScalar || b.type.shape.every(d => d === 1);
      if (aSingleton && bSingleton) {
        if (a.rank <= b.rank) a = this.broadcastTo(a, b.type.shape);
        else b = this.broadcastTo(b, a.type.shape);
      } else if (aSingleton) a = this.broadcastTo(a, b.type.shape);
      else if (bSingleton) b = this.broadcastTo(b, a.type.shape);
    }
    const isF = isFloat(a.elem);
    const pred = isF ? floatPred : intPred;
    const r = this.fresh("cmp");
    this.emit(`%${r} = ${isF ? "arith.cmpf" : "arith.cmpi"} ${pred}, ${a.name}, ${b.name} : ${typeText(a.type)}`);
    return tensor(`%${r}`, a.type.shape, "i1");
  }
  eq(a: Value, b: Value): Value { return this.cmp(a, b, "eq", "oeq"); }
  ne(a: Value, b: Value): Value { return this.cmp(a, b, "ne", "une"); }
  lt(a: Value, b: Value): Value { return this.cmp(a, b, "slt", "olt"); }
  gt(a: Value, b: Value): Value { return this.cmp(a, b, "sgt", "ogt"); }
  le(a: Value, b: Value): Value { return this.cmp(a, b, "sle", "ole"); }
  ge(a: Value, b: Value): Value { return this.cmp(a, b, "sge", "oge"); }

  /** `arith.select` — elementwise ternary. */
  select(cond: Value, a: Value, b: Value): Value {
    if (cond.elem !== "i1") throw new Error(`select: cond must be i1, got ${cond.elem}`);
    const r = this.fresh("sel");
    this.emit(`%${r} = "arith.select"(${cond.name}, ${a.name}, ${b.name}) : (${typeText(cond.type)}, ${typeText(a.type)}, ${typeText(b.type)}) -> ${typeText(a.type)}`);
    return tensor(`%${r}`, a.type.shape, a.elem);
  }

  // ───────────────────────────────────────────────────────────────
  // Casts & conversions
  // ───────────────────────────────────────────────────────────────

  private cast1(a: Value, op: string, outElem: ScalarElem): Value {
    const r = this.fresh("c");
    this.emit(`%${r} = arith.${op} ${a.name} : ${typeText(a.type)} to ${typeText({ shape: a.type.shape, elem: outElem })}`);
    return tensor(`%${r}`, a.type.shape, outElem);
  }
  /** Truncate i32 → i8/i16. */
  trunc(a: Value, to: "i8" | "i16"): Value { return this.cast1(a, "trunci", to); }
  /** Sign-extend i8/i16 → i32/i64. */
  ext(a: Value, to: "i32" | "i64"): Value { return this.cast1(a, "sexti", to); }
  /** Zero-extend. */
  zext(a: Value, to: "i32" | "i64"): Value { return this.cast1(a, "zexti", to); }
  /** Signed int → float. */
  sitofp(a: Value, to: "f16" | "bf16" | "f32" | "f64"): Value { return this.cast1(a, "sitofp", to); }
  /** Float → signed int. */
  fptosi(a: Value, to: "i8" | "i16" | "i32" | "i64"): Value { return this.cast1(a, "fptosi", to); }
  /** Float → float (widening/narrowing). */
  fpext(a: Value, to: "f32" | "f64"): Value { return this.cast1(a, "extf", to); }
  fptrunc(a: Value, to: "f16" | "bf16" | "f32"): Value { return this.cast1(a, "truncf", to); }
  /** Reinterpret bits. */
  bitcast(a: Value, to: ScalarElem): Value { return this.cast1(a, "bitcast", to); }
  /** Cast between `index` and integer types (needed for scf.for IV → i32 offsets). */
  indexCast(a: Value, to: "i32" | "i64" | "index"): Value {
    const r = this.fresh("ic");
    const toTy: any = to === "index" ? "index" : to;
    this.emit(`%${r} = arith.index_cast ${a.name} : ${typeText(a.type)} to ${to === "index" ? "index" : toTy}`);
    return new Value(`%${r}`, { shape: a.type.shape, elem: toTy });
  }

  // ───────────────────────────────────────────────────────────────
  // math dialect
  // ───────────────────────────────────────────────────────────────

  private math1(a: Value, op: string): Value {
    const r = this.fresh("m");
    this.emit(`%${r} = math.${op} ${a.name} : ${typeText(a.type)}`);
    return tensor(`%${r}`, a.type.shape, a.elem);
  }
  exp(a: Value): Value { return this.math1(a, "exp"); }
  log(a: Value): Value { return this.math1(a, "log"); }
  sqrt(a: Value): Value { return this.math1(a, "sqrt"); }
  rsqrt(a: Value): Value { return this.math1(a, "rsqrt"); }

  /**
   * `tt.elementwise_inline_asm` — emit inline PTX assembly.
   * Avoids libdevice for ops like rsqrt.approx.f32, sqrt.approx.f32.
   * `asmStr` should end with `;`, constraints follow LLVM inline asm conventions.
   */
  inlineAsm(asmStr: string, constraints: string, args: Value[], outShape: number[], outElem: ScalarElem, isPure = true, pack = 1): Value {
    if (!asmStr.endsWith(";")) asmStr += ";";
    const r = this.fresh("asm");
    const argNames = args.map(a => a.name).join(", ");
    const argTypes = args.map(a => typeText(a.type)).join(", ");
    const outType = typeText({ shape: outShape, elem: outElem });
    this.emit(`%${r} = "tt.elementwise_inline_asm"(${argNames}) <{asm_string = "${asmStr}", constraints = "${constraints}", pure = ${isPure}, packed_element = ${pack} : i32}> : (${argTypes}) -> ${outType}`);
    return tensor(`%${r}`, outShape, outElem);
  }

  /** Hardware rsqrt via PTX `rsqrt.approx.f32` — no libdevice needed. */
  rsqrtHw(a: Value): Value {
    return this.inlineAsm("rsqrt.approx.f32 $0, $1;", "=f, f", [a], a.type.shape, a.elem as ScalarElem);
  }

  /** Hardware log2 via PTX `lg2.approx.f32` — no libdevice needed. */
  log2Hw(a: Value): Value {
    return this.inlineAsm("lg2.approx.f32 $0, $1;", "=f, f", [a], a.type.shape, a.elem as ScalarElem);
  }
  sin(a: Value): Value { return this.math1(a, "sin"); }
  cos(a: Value): Value { return this.math1(a, "cos"); }
  tanh(a: Value): Value { return this.math1(a, "tanh"); }
  floor(a: Value): Value { return this.math1(a, "floor"); }
  ceil(a: Value): Value { return this.math1(a, "ceil"); }
  round(a: Value): Value { return this.math1(a, "roundeven"); }

  // ───────────────────────────────────────────────────────────────
  // Reductions (tt.reduce)
  // ───────────────────────────────────────────────────────────────

  private reduce(a: Value, axis: number, kind: "sum" | "max" | "min"): Value {
    if (axis < 0 || axis >= a.rank) throw new Error(`reduce: axis ${axis} out of range for rank ${a.rank}`);
    const outShape = a.type.shape.filter((_, i) => i !== axis);
    const outElem: ScalarElem = a.elem;
    const r = this.fresh("red");
    const e = a.elem;
    const isF = isFloat(e);
    // tt.reduce uses a region combine form:
    //   %r = tt.reduce %a {axis = N : i32} : tensor<...> -> tensor<...> ({
    //     ^bb0(%x : e, %y : e):
    //       %s = <combine> %x, %y : e
    //       tt.reduce.return %s : e
    //   })
    const op = kind === "sum" ? (isF ? "arith.addf" : "arith.addi")
              : kind === "max" ? (isF ? "arith.maximumf" : "arith.maxsi")
              :                  (isF ? "arith.minimumf" : "arith.minsi");
    this.emit(`%${r} = "tt.reduce"(%${a.name.slice(1)}) ({`);
    this.lines.push(`      ^bb0(%rx : ${e}, %ry : ${e}):`);
    this.lines.push(`        %rv = ${op} %rx, %ry : ${e}`);
    this.lines.push(`        "tt.reduce.return"(%rv) : (${e}) -> ()`);
    this.lines.push(`    }) {axis = ${axis} : i32} : (${typeText(a.type)}) -> ${typeText({ shape: outShape, elem: outElem })}`);
    return tensor(`%${r}`, outShape, outElem);
  }

  /** Sum along axis (drops the axis). */
  sum(a: Value, axis: number): Value { return this.reduce(a, axis, "sum"); }
  /** Max along axis. */
  max(a: Value, axis: number): Value { return this.reduce(a, axis, "max"); }
  /** Min along axis. */
  min(a: Value, axis: number): Value { return this.reduce(a, axis, "min"); }

  // ───────────────────────────────────────────────────────────────
  // Layout ops
  // ───────────────────────────────────────────────────────────────

  /** `tt.trans` — transpose last two dims (order = [...,N-1,N-2]). */
  trans(a: Value): Value {
    if (a.rank < 2) throw new Error(`trans: requires rank >= 2`);
    const shape = [...a.type.shape];
    [shape[a.rank - 1], shape[a.rank - 2]] = [shape[a.rank - 2], shape[a.rank - 1]];
    // order = [0, 1, ..., rank-3, rank-1, rank-2]
    const order = Array.from({ length: a.rank }, (_, i) => i);
    [order[a.rank - 1], order[a.rank - 2]] = [order[a.rank - 2], order[a.rank - 1]];
    const r = this.fresh("tr");
    this.emit(`%${r} = tt.trans %${a.name.slice(1)} {order = array<i32: ${order.join(", ")}>} : ${typeText(a.type)} -> ${typeText({ shape, elem: a.elem })}`);
    return tensor(`%${r}`, shape, a.elem);
  }

  /** `tt.reshape`. */
  reshape(a: Value, newShape: number[]): Value {
    const r = this.fresh("rs");
    this.emit(`%${r} = tt.reshape %${a.name.slice(1)} : ${typeText(a.type)} -> ${typeText({ shape: newShape, elem: a.elem })}`);
    return tensor(`%${r}`, newShape, a.elem);
  }

  /** `tt.broadcast` to a new shape. */
  broadcast(a: Value, newShape: number[]): Value {
    const r = this.fresh("bc");
    this.emit(`%${r} = tt.broadcast %${a.name.slice(1)} : ${typeText(a.type)} -> ${typeText({ shape: newShape, elem: a.elem })}`);
    return tensor(`%${r}`, newShape, a.elem);
  }

  /** `tt.expand_dims` — insert a size-1 dim at `axis`. */
  expandDims(a: Value, axis: number): Value {
    const shape = [...a.type.shape];
    shape.splice(axis, 0, 1);
    const r = this.fresh("ex");
    this.emit(`%${r} = tt.expand_dims %${a.name.slice(1)} {axis = ${axis} : i32} : ${typeText(a.type)} -> ${typeText({ shape, elem: a.elem })}`);
    return tensor(`%${r}`, shape, a.elem);
  }

  // ───────────────────────────────────────────────────────────────
  // Matmul
  // ───────────────────────────────────────────────────────────────

  /** `tt.dot` — tensor core matmul. acc defaults to zero tensor of out shape. */
  dot(a: Value, b: Value, acc?: Value): Value {
    if (a.rank !== 2 || b.rank !== 2) throw new Error(`dot: requires 2D operands`);
    const [M, K1] = a.type.shape;
    const [K2, N] = b.type.shape;
    if (K1 !== K2) throw new Error(`dot: K mismatch ${K1} vs ${K2}`);
    const outElem: ScalarElem = (isFloat(a.elem) && (a.elem === "f16" || a.elem === "bf16")) ? "f32" : (isFloat(a.elem) ? a.elem : "i32");
    const outShape = [M, N];
    const accVal = acc ?? this.zeros(outShape, outElem);
    const r = this.fresh("dot");
    this.emit(`%${r} = tt.dot %${a.name.slice(1)}, %${b.name.slice(1)}, %${accVal.name.slice(1)} : ${typeText(a.type)} * ${typeText(b.type)} -> ${typeText({ shape: outShape, elem: outElem })}`);
    return tensor(`%${r}`, outShape, outElem);
  }

  // ───────────────────────────────────────────────────────────────
  // Atomics
  // ───────────────────────────────────────────────────────────────

  private atomic(op: string, tp: Value, val: Value, mask?: Value): Value {
    if (!isPtr(tp.type.elem) && !isTiledPtr(tp.type.elem))
      throw new Error(`atomic: expected tiled pointer or pointer tensor`);
    // tt.atomic_rmw "<op>", "relaxed", "gpu", %ptr, %val : (ptr-ty, val-ty) -> val-ty
    // Float atomics use the f* variant (fadd/max/min); integers use add/max/min.
    const kind = isFloat(val.elem) && (op === "add") ? "fadd" : op;
    const r = this.fresh("at");
    let ptrTy: string, resTy: string, resElem: ScalarElem, resShape: number[];
    if (isTiledPtr(tp.type.elem)) {
      const tile = tp.type.elem.tile;
      ptrTy = elemText(tp.type.elem);
      resTy = typeText({ shape: tile.shape, elem: tile.elem });
      resElem = tile.elem; resShape = tile.shape;
    } else {
      const e = (tp.type.elem as { ptr: ScalarElem }).ptr;
      ptrTy = typeText(tp.type);
      resTy = typeText({ shape: tp.type.shape, elem: e });
      resElem = e; resShape = tp.type.shape;
    }
    const maskPart = mask ? `, ${mask.name}` : "";
    const maskTy = mask ? `, ${typeText(mask.type)}` : "";
    this.emit(`%${r} = tt.atomic_rmw "${kind}", "relaxed", "gpu", ${tp.name}${maskPart}, ${val.name} : (${ptrTy}${maskTy}, ${typeText(val.type)}) -> ${resTy}`);
    return tensor(`%${r}`, resShape, resElem);
  }
  atomicAdd(tp: Value, val: Value, mask?: Value): Value { return this.atomic("add", tp, val, mask); }
  atomicMax(tp: Value, val: Value, mask?: Value): Value { return this.atomic("max", tp, val, mask); }
  atomicMin(tp: Value, val: Value, mask?: Value): Value { return this.atomic("min", tp, val, mask); }
  atomicXchg(tp: Value, val: Value, mask?: Value): Value { return this.atomic("exch", tp, val, mask); }

  // ───────────────────────────────────────────────────────────────
  // Control flow (scf dialect)
  // ───────────────────────────────────────────────────────────────

  /**
   * `scf.if` — executes one of two blocks based on a scalar i1 condition.
   *   b.if_(cond, (then) => { ... }, (els) => { ... });
   * Yields are not yet supported (void if); use for side-effects (stores, atomics).
   */
  if_(cond: Value, thenFn: (b: TTIRBuilder) => void, elseFn?: (b: TTIRBuilder) => void): void {
    if (cond.elem !== "i1") throw new Error(`if_: cond must be i1, got ${cond.elem}`);
    const indent = "    ";
    this.lines.push(`${indent}scf.if ${cond.name} -> () {`);
    const saved = this.lines; this.lines = [];
    thenFn(this);
    for (const l of this.lines) this.saved_push(saved, "      " + l);
    this.lines = saved;
    if (elseFn) {
      this.lines.push(`${indent}} else {`);
      const saved2 = this.lines; this.lines = [];
      elseFn(this);
      for (const l of this.lines) this.saved_push(saved2, "      " + l);
      this.lines = saved2;
    }
    this.lines.push(`${indent}}`);
  }
  private saved_push(arr: string[], v: string) { arr.push(v); }

  /** Cast a value to `index` if it isn't already (for scf.for bounds). */
  private toIndex(v: Value): Value {
    if (v.type.elem === "index") return v;
    return this.indexCast(v, "index");
  }

  /**
   * `scf.for` (void, no iter-args) — counted loop for side effects.
   *   b.for_(start, end, step, (b, iv) => { ... });
   */
  for_(start: Value, end: Value, step: Value, bodyFn: (b: TTIRBuilder, iv: Value) => void): void {
    const indent = "    ";
    const ivName = `%iv${this.uid++}`;
    const s = this.toIndex(start), e = this.toIndex(end), st = this.toIndex(step);
    this.lines.push(`${indent}scf.for ${ivName} = ${s.name} to ${e.name} step ${st.name} : index {`);
    const saved = this.lines; this.lines = [];
    bodyFn(this, new Value(ivName, { shape: [], elem: "index" as any }));
    for (const l of this.lines) this.saved_push(saved, "      " + l);
    this.lines = saved;
    this.lines.push(`${indent}}`);
  }

  /**
   * `scf.for` with iter-args (accumulation loops). Returns the yielded results.
   *   const [acc] = b.forIter(start, end, step, [init], (b, iv, [a]) => {
   *     const next = b.dot(..., a);
   *     return [next];
   *   });
   */
  forIter(
    start: Value, end: Value, step: Value,
    initArgs: Value[],
    bodyFn: (b: TTIRBuilder, iv: Value, iterArgs: Value[]) => Value[],
  ): Value[] {
    const indent = "    ";
    const ivName = `%iv${this.uid++}`;
    const s = this.toIndex(start), e = this.toIndex(end), st = this.toIndex(step);
    const iterNames = initArgs.map((_, i) => `%ia${this.uid++}_${i}`);
    const iterDecls = initArgs.map((a, i) => `${iterNames[i]} = ${a.name}`).join(", ");
    const iterTys = initArgs.map(a => typeText(a.type)).join(", ");
    // Result SSA names (one per iter arg).
    const resNames = initArgs.map((_, i) => `%fr${this.uid++}`);
    this.lines.push(`${indent}${resNames.join(", ")} = scf.for ${ivName} = ${s.name} to ${e.name} step ${st.name} iter_args(${iterDecls}) -> (${iterTys}) {`);
    const iterArgVals = initArgs.map((a, i) => new Value(iterNames[i], a.type));
    const saved = this.lines; this.lines = [];
    const yielded = bodyFn(this, new Value(ivName, { shape: [], elem: "index" as any }), iterArgVals);
    for (const l of this.lines) this.saved_push(saved, "      " + l);
    this.lines = saved;
    this.lines.push(`${indent}  scf.yield ${yielded.map(y => y.name).join(", ")} : ${yielded.map(y => typeText(y.type)).join(", ")}`);
    this.lines.push(`${indent}}`);
    return yielded.map((y, i) => new Value(resNames[i], y.type));
  }

  /**
   * `scf.while` — general loop with iter-args. Returns the loop's result
   * values (the iter-args at loop exit). beforeFn receives the loop-carried
   * args and returns {cond, iterArgs}; bodyFn receives the forwarded iterArgs
   * and returns the next iteration's values. For counted loops prefer forIter().
   *   const [acc] = b.while_([init], (b, [a]) => ({cond, iterArgs: [a]}),
   *                                   (b, [a]) => [next]);
   */
  while_(
    initArgs: Value[],
    beforeFn: (b: TTIRBuilder, args: Value[]) => { cond: Value; iterArgs: Value[] },
    bodyFn: (b: TTIRBuilder, args: Value[]) => Value[],
  ): Value[] {
    const indent = "    ";
    const n = initArgs.length;
    const resNames = initArgs.map((_, i) => `%fw${this.uid++}_${i}`);   // while results
    const bwNames = initArgs.map((_, i) => `%bw${this.uid++}_${i}`);   // before-region block args
    const dwNames = initArgs.map((_, i) => `%dw${this.uid++}_${i}`);   // do-region block args
    const tys = initArgs.map(a => typeText(a.type)).join(", ");
    const initList = initArgs.map((a, i) => `${bwNames[i]} = ${a.name}`).join(", ");
    const header = n ? `${resNames.join(", ")} = scf.while (${initList}) : (${tys}) -> (${tys}) {`
                     : `scf.while : () -> () {`;
    this.lines.push(`${indent}${header}`);
    // before region: cond + scf.condition(cond) <forwarded iterArgs>
    const beforeArgs = initArgs.map((a, i) => new Value(bwNames[i], a.type));
    const saved = this.lines; this.lines = [];
    const before = beforeFn(this, beforeArgs);
    if (before.cond.elem !== "i1") throw new Error(`while_: cond must be i1`);
    for (const l of this.lines) this.saved_push(saved, "      " + l);
    this.lines = saved;
    const fwdVals = before.iterArgs.map(a => a.name).join(", ");
    const fwdTys = before.iterArgs.map(a => typeText(a.type)).join(", ");
    this.lines.push(`${indent}  scf.condition(${before.cond.name})${fwdVals ? " " + fwdVals + " : " + fwdTys : ""}`);
    this.lines.push(`${indent}} do {`);
    // do region: block args = forwarded iterArgs (typed)
    const doArgs = before.iterArgs.map((a, i) => new Value(dwNames[i], a.type));
    const doBlock = doArgs.map(a => `${a.name} : ${typeText(a.type)}`).join(", ");
    this.lines.push(`${indent}  ^bb0(${doBlock}):`);
    const saved2 = this.lines; this.lines = [];
    const nextArgs = bodyFn(this, doArgs);
    for (const l of this.lines) this.saved_push(saved2, "        " + l);
    this.lines = saved2;
    const yVals = nextArgs.map(a => a.name).join(", ");
    const yTys = nextArgs.map(a => typeText(a.type)).join(", ");
    this.lines.push(`${indent}  scf.yield${yVals ? " " + yVals + " : " + yTys : ""}`);
    this.lines.push(`${indent}}`);
    return nextArgs.map((a, i) => new Value(resNames[i], a.type));
  }

  // ───────────────────────────────────────────────────────────────
  // Module assembly
  // ───────────────────────────────────────────────────────────────

  /** Emit the final TTIR module string. */
  build(kernelName = "kernel", numWarps = 4, numStages = 0): string {
    const paramDecls = this.params.map((p, i) =>
      `%arg${i}: ${paramTypeText(p)}`,
    ).join(", ");
    const stagesAttr = numStages > 0 ? `, "ttg.num-stages" = ${numStages} : i32` : "";
    const header =
`module attributes {"ttg.num-warps" = ${numWarps} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32${stagesAttr}} {
  tt.func @${kernelName}(${paramDecls}) {`;
    const footer = `\n    tt.return\n  }\n}\n`;
    return header + "\n" + this.lines.join("\n") + footer;
  }

  /** Compile this builder's TTIR to PTX via the shim. */
  compile(kernelName = "kernel", numWarps = 4): CompileResult {
    return compileTTIR(this.build(kernelName, numWarps), numWarps);
  }
}

// ═══════════════════════════════════════════════════════════════════
// Stub types (editor-only — no runtime effect)
// ═══════════════════════════════════════════════════════════════════
export class i8 {}
export class i16 {}
export class i32 {}
export class i64 {}
export class f16 {}
export class f32 {}
export class f64 {}
export class ptr<T> {}
export const blockIdx = { x: 0, y: 0, z: 0 };
export const threadIdx = { x: 0, y: 0, z: 0 };
export const blockDim = { x: 0, y: 0, z: 0 };
export const gridDim = { x: 0, y: 0, z: 0 };

// Stub builtins for the arrow-function lift. The arrow body is *parsed* by
// kernel_ttir, never executed — these exist only so the source type-checks
// and the editor offers completion. Signatures are intentionally `any`.
export const programId = (axis?: 0 | 1 | 2): any => undefined as any;
export const numPrograms = (axis?: 0 | 1 | 2): any => undefined as any;
export const arange = (start: number, end: number): any => undefined as any;
export const load = (ptr: any, opts?: any): any => undefined as any;
export const store = (ptr: any, val: any, opts?: any): any => undefined as any;
export const dot = (a: any, b: any, acc?: any): any => undefined as any;
export const sum = (a: any, axis: number): any => undefined as any;
export const max = (a: any, axis: number): any => undefined as any;
export const min = (a: any, axis: number): any => undefined as any;
export const exp = (a: any): any => undefined as any;
export const log = (a: any): any => undefined as any;
export const sqrt = (a: any): any => undefined as any;
export const rsqrt = (a: any): any => undefined as any;
export const select = (cond: any, a: any, b: any): any => undefined as any;
export const atomicAdd = (ptr: any, val: any, mask?: any): any => undefined as any;

// ═══════════════════════════════════════════════════════════════════
// kernel_ttir — arrow-function AST → TTIR lift
//
// Authoring surface. Write a plain TS arrow function with typed params;
// kernel_ttir parses it (re-reading the caller's source for types) and lowers
// it to TTIR via TTIRBuilder. The result is a {ttir, compile, source} object.
//
// Supported subset (see AGENTS.md "Feature targets"):
//   - params: ptr<E>, i32, f32, i16, i8, f16, f64, i64
//   - let/const with numeric/string-typed inits
//   - arithmetic (+ - * /), comparisons (< > <= >= == !=), && || !
//   - if/else, for loops, while loops
//   - array indexing A[i] (load) and A[i] = v (store) for ptr params
//   - builtin calls: arange, programId, numPrograms, load, store, dot, sum,
//     max, min, exp, log, sqrt, rsqrt, sin, cos, select, atomicAdd, ...
//   - blockIdx.x/y/z, threadIdx.x/y/z, blockDim.x/y/z, gridDim.x/y/z
// ═══════════════════════════════════════════════════════════════════

/** Builtins callable inside a kernel_ttir arrow function. */
export interface KernelBuiltins {
  arange(start: number, end: number): Value;
  programId(axis?: 0 | 1 | 2): Value;
  numPrograms(axis?: 0 | 1 | 2): Value;
  load(tp: Value, opts?: { boundaryCheck?: number[]; mask?: Value; other?: Value }): Value;
  store(tp: Value, val: Value, opts?: { boundaryCheck?: number[]; mask?: Value }): void;
  dot(a: Value, b: Value, acc?: Value): Value;
  sum(a: Value, axis: number): Value;
  max(a: Value, axis: number): Value;
  min(a: Value, axis: number): Value;
  exp(a: Value): Value; log(a: Value): Value; sqrt(a: Value): Value; rsqrt(a: Value): Value;
  sin(a: Value): Value; cos(a: Value): Value; tanh(a: Value): Value;
  select(cond: Value, a: Value, b: Value): Value;
  atomicAdd(tp: Value, val: Value, mask?: Value): Value;
}

/** Parse a param type string like `ptr<f32>` or `i32` into an Elem. */
function parseParamElem(text: string): ScalarElem | { ptr: ScalarElem } {
  const t = text.trim();
  const m = t.match(/^ptr<(.+)>$/);
  if (m) return { ptr: m[1] as ScalarElem };
  if (["i1","i8","i16","i32","i64","f16","bf16","f32","f64"].includes(t)) return t as ScalarElem;
  throw new Error(`kernel_ttir: unknown param type '${t}'`);
}

/**
 * Lift a TS arrow function to a TTIR kernel.
 *
 *   const add = kernel_ttir((A: ptr<f32>, B: ptr<f32>, C: ptr<f32>, N: i32) => {
 *     const pid = programId(0);
 *     const offs = arange(0, 256).add(pid.mul(256));   // ← fluent Value API
 *     ...
 *   });
 *   const { ptx } = add.compile(8);
 *
 * NOTE: the arrow function is *parsed*, not executed. It references stub
 * builtins (arange, programId, load, store, …) and stub types (ptr, i32, …)
 * imported from this module. The lift walks the AST and calls TTIRBuilder.
 */
export function kernel_ttir<F extends (...a: any[]) => void>(fn: F): {
  source: string; ttir: string; compile: (numWarps?: number) => CompileResult;
} {
  const stripped = fn.toString();

  // Re-read the caller source file to recover type annotations.
  let typedSource: string | null = null;
  try {
    const stack = new Error().stack ?? "";
    for (const line of stack.split("\n")) {
      const m = line.match(/([^\s()]+\.ts):\d+(?::\d+)?\)?/);
      if (m) {
        const file = m[1];
        // Skip the ttir.ts module itself (basename match), but NOT user files
        // like test_kernel_ttir.ts that merely contain the substring.
        const base = file.split("/").pop() ?? file;
        if (base !== "ttir.ts" && !file.includes("typescript")) {
          try { typedSource = readFileSync(file, "utf-8"); } catch {}
          if (typedSource) break;
        }
      }
    }
  } catch {}

  // Parse the stripped arrow function (types removed by Bun).
  let strippedFn: ts.ArrowFunction | undefined;
  const sf = ts.createSourceFile("k.ts", `const __k__ = ${stripped}`, ts.ScriptTarget.Latest, true);
  ts.forEachChild(sf, n => {
    if (ts.isVariableStatement(n)) {
      const init = n.declarationList.declarations[0].initializer;
      if (init && ts.isArrowFunction(init)) strippedFn = init;
    }
  });
  if (!strippedFn) throw new Error("kernel_ttir: must pass an arrow function");

  // Extract typed param types from the caller source.
  const paramTypes: (string | null)[] = strippedFn.parameters.map(() => null);
  if (typedSource) {
    const typedSF = ts.createSourceFile("k2.ts", typedSource, ts.ScriptTarget.Latest, true);
    ts.forEachChild(typedSF, function findKernelCall(node: ts.Node) {
      if (ts.isCallExpression(node) && node.expression.getText(typedSF) === "kernel_ttir") {
        const arg = node.arguments[0];
        if (arg && ts.isArrowFunction(arg)) {
          arg.parameters.forEach((p, i) => {
            if (p.type) paramTypes[i] = p.type.getText(typedSF);
          });
        }
      }
      ts.forEachChild(node, findKernelCall);
    });
  }

  const builder = new TTIRBuilder();
  // Bind params as Values in the env.
  const env = new Map<string, Value>();

  // Resolve captured numeric constants (e.g. `const BLOCK = 256;`) from the
  // caller source file, so kernels can reference outer consts like Triton Python.
  const constMap = new Map<string, number>();
  if (typedSource) {
    const constSF = ts.createSourceFile("k3.ts", typedSource, ts.ScriptTarget.Latest, true);
    const visitConst = (node: ts.Node) => {
      if (ts.isVariableStatement(node)) {
        for (const d of node.declarationList.declarations) {
          if (d.initializer && ts.isNumericLiteral(d.initializer)) {
            constMap.set(d.name.getText(constSF), +d.initializer.text);
          }
        }
      }
      ts.forEachChild(node, visitConst);
    };
    ts.forEachChild(constSF, visitConst);
  }
  strippedFn.parameters.forEach((p, i) => {
    const name = p.name.getText(sf);
    const elemText = paramTypes[i];
    if (!elemText) throw new Error(`kernel_ttir: missing type for param '${name}'`);
    const elem = parseParamElem(elemText);
    const v = builder.param(name, elem);
    env.set(name, v);
  });

  // Evaluate a numeric-literal (or captured const) expression to a JS number.
  function numExpr(x: ts.Expression): number {
    if (ts.isNumericLiteral(x)) return +x.text;
    if (ts.isIdentifier(x) && constMap.has(x.text)) return constMap.get(x.text)!;
    if (ts.isParenthesizedExpression(x)) return numExpr(x.expression);
    if (ts.isBinaryExpression(x)) {
      const l = numExpr(x.left), r = numExpr(x.right);
      switch (x.operatorToken.kind) {
        case ts.SyntaxKind.PlusToken: return l + r;
        case ts.SyntaxKind.MinusToken: return l - r;
        case ts.SyntaxKind.AsteriskToken: return l * r;
        case ts.SyntaxKind.SlashToken: return l / r;
      }
    }
    throw new Error(`kernel_ttir: expected numeric literal, got '${x.getText(sf)}'`);
  }

  // Builtin dispatch table.
  const builtins: Record<string, (args: ts.Expression[], sf: ts.SourceFile) => Value> = {
    arange: (a, sf) => builder.arange(numExpr(a[0]), numExpr(a[1])),
    programId: (a, sf) => builder.programId(numExpr(a[0]) as 0 | 1 | 2),
    numPrograms: (a, sf) => builder.numPrograms(numExpr(a[0]) as 0 | 1 | 2),
    load: (a, sf) => builder.load(emitExpr(a[0], sf), a[1] ? parseLoadOpts(a[1], sf) : {}),
    store: (a, sf) => { builder.store(emitExpr(a[0], sf), emitExpr(a[1], sf), a[2] ? parseStoreOpts(a[2], sf) : {}); return builder.i32(0); },
    dot: (a, sf) => builder.dot(emitExpr(a[0], sf), emitExpr(a[1], sf), a[2] ? emitExpr(a[2], sf) : undefined),
    sum: (a, sf) => builder.sum(emitExpr(a[0], sf), numExpr(a[1])),
    max: (a, sf) => builder.max(emitExpr(a[0], sf), numExpr(a[1])),
    min: (a, sf) => builder.min(emitExpr(a[0], sf), numExpr(a[1])),
    exp: (a, sf) => builder.exp(emitExpr(a[0], sf)),
    log: (a, sf) => builder.log(emitExpr(a[0], sf)),
    sqrt: (a, sf) => builder.sqrt(emitExpr(a[0], sf)),
    rsqrt: (a, sf) => builder.rsqrt(emitExpr(a[0], sf)),
    sin: (a, sf) => builder.sin(emitExpr(a[0], sf)),
    cos: (a, sf) => builder.cos(emitExpr(a[0], sf)),
    tanh: (a, sf) => builder.tanh(emitExpr(a[0], sf)),
    select: (a, sf) => builder.select(emitExpr(a[0], sf), emitExpr(a[1], sf), emitExpr(a[2], sf)),
    atomicAdd: (a, sf) => builder.atomicAdd(emitExpr(a[0], sf), emitExpr(a[1], sf), a[2] ? emitExpr(a[2], sf) : undefined),
  };

  function parseLoadOpts(node: ts.Expression, sf: ts.SourceFile): { boundaryCheck?: number[]; mask?: Value; other?: Value } {
    if (!ts.isObjectLiteralExpression(node)) throw new Error("load opts must be object literal");
    const out: any = {};
    for (const p of node.properties) {
      if (!ts.isPropertyAssignment(p)) continue;
      const key = p.name.getText(sf);
      if (key === "boundaryCheck") {
        out.boundaryCheck = (p.initializer as ts.ArrayLiteralExpression).elements.map(e => numExpr(e));
      } else {
        out[key] = emitExpr(p.initializer, sf);
      }
    }
    return out;
  }
  function parseStoreOpts(node: ts.Expression, sf: ts.SourceFile): { boundaryCheck?: number[]; mask?: Value } {
    return parseLoadOpts(node, sf);
  }

  // Expression emitter: AST node → Value (emitting builder calls).
  function emitExpr(x: ts.Expression, sf: ts.SourceFile): Value {
    if (ts.isNumericLiteral(x)) {
      return x.text.includes(".") ? builder.f32(+x.text) : builder.i32(+x.text);
    }
    if (ts.isStringLiteral(x)) {
      // Used for asm string args; not common in TTIR lift.
      return builder.i32(0);
    }
    if (ts.isIdentifier(x)) {
      const v = env.get(x.text);
      if (v) return v;
      // Resolve captured numeric constant (const NAME = <number> at module scope).
      if (constMap.has(x.text)) {
        const n = constMap.get(x.text)!;
        return Number.isInteger(n) ? builder.i32(n) : builder.f32(n);
      }
      throw new Error(`kernel_ttir: undefined identifier '${x.text}'`);
    }
    if (ts.isParenthesizedExpression(x)) return emitExpr(x.expression, sf);
    if (ts.isPropertyAccessExpression(x)) {
      const obj = x.expression.getText(sf);
      const dim = x.name.text;
      if ((obj === "blockIdx" || obj === "threadIdx" || obj === "blockDim" || obj === "gridDim")) {
        if (obj === "blockDim") {
          // blockDim.x in TTIR land — emit as a constant if known, else numPrograms*?
          // For TTIR kernels blockDim is determined by num_warps; provide 1D x size.
          if (dim === "x") return builder.i32(0); // placeholder: kernels should use arange, not blockDim
        }
        if (obj === "gridDim") {
          return builder.numPrograms(dim === "x" ? 0 : dim === "y" ? 1 : 2);
        }
        return builder.programId(dim === "x" ? 0 : dim === "y" ? 1 : 2);
      }
      throw new Error(`kernel_ttir: unsupported property access ${obj}.${dim}`);
    }
    if (ts.isBinaryExpression(x)) {
      const k = x.operatorToken.kind;
      // Arithmetic
      const arith: Record<number, "add" | "sub" | "mul" | "divi" | "divf"> = {
        [ts.SyntaxKind.PlusToken]: "add",
        [ts.SyntaxKind.MinusToken]: "sub",
        [ts.SyntaxKind.AsteriskToken]: "mul",
        [ts.SyntaxKind.SlashToken]: "divi",
      };
      if (k in arith) {
        const a = emitExpr(x.left, sf); const b = emitExpr(x.right, sf);
        const op = arith[k];
        // Triton-Python-style pointer arithmetic: ptr + offsets → pointer-tensor.
        if (op === "add") {
          if (isPtr(a.type.elem) && a.isScalar && !b.isScalar)
            return builder.addptr(builder.splatPtr(a, b.type.shape[0], (a.type.elem as any).ptr), b);
          if (isPtr(b.type.elem) && b.isScalar && !a.isScalar)
            return builder.addptr(builder.splatPtr(b, a.type.shape[0], (b.type.elem as any).ptr), a);
          return builder.add(a, b);
        }
        if (op === "sub") return builder.sub(a, b);
        if (op === "mul") return builder.mul(a, b);
        if (op === "divi") return isFloat(a.elem) ? builder.divf(a, b) : builder.divi(a, b);
      }
      // Comparison
      const cmpMap: Record<number, "eq" | "ne" | "lt" | "gt" | "le" | "ge"> = {
        [ts.SyntaxKind.LessThanToken]: "lt",
        [ts.SyntaxKind.GreaterThanToken]: "gt",
        [ts.SyntaxKind.LessThanEqualsToken]: "le",
        [ts.SyntaxKind.GreaterThanEqualsToken]: "ge",
        [ts.SyntaxKind.EqualsEqualsToken]: "eq",
        [ts.SyntaxKind.ExclamationEqualsToken]: "ne",
      };
      if (k in cmpMap) {
        const a = emitExpr(x.left, sf); const b = emitExpr(x.right, sf);
        return builder[cmpMap[k]](a, b);
      }
      // Logical
      if (k === ts.SyntaxKind.AmpersandAmpersandToken) {
        const a = emitExpr(x.left, sf); const b = emitExpr(x.right, sf);
        return builder.and(a, b);
      }
      if (k === ts.SyntaxKind.BarBarToken) {
        const a = emitExpr(x.left, sf); const b = emitExpr(x.right, sf);
        return builder.or(a, b);
      }
      // Bitwise
      const bitMap: Record<number, "and" | "or" | "xor" | "shl"> = {
        [ts.SyntaxKind.AmpersandToken]: "and",
        [ts.SyntaxKind.BarToken]: "or",
        [ts.SyntaxKind.CaretToken]: "xor",
        [ts.SyntaxKind.LessThanLessThanToken]: "shl",
      };
      if (k in bitMap) {
        const a = emitExpr(x.left, sf); const b = emitExpr(x.right, sf);
        return builder[bitMap[k]](a, b);
      }
      throw new Error(`kernel_ttir: unsupported binary op ${ts.SyntaxKind[k]}`);
    }
    if (ts.isPrefixUnaryExpression(x)) {
      if (x.operator === ts.SyntaxKind.ExclamationToken) {
        const v = emitExpr(x.operand, sf);
        return builder.select(v, builder.zeros(v.type.shape, v.elem), builder.splat(builder.i32(1), v.type.shape, v.elem));
      }
      if (x.operator === ts.SyntaxKind.MinusToken) {
        return builder.neg(emitExpr(x.operand, sf));
      }
      return emitExpr(x.operand, sf);
    }
    if (ts.isCallExpression(x)) {
      const name = x.expression.getText(sf);
      const fn = builtins[name];
      if (!fn) throw new Error(`kernel_ttir: unknown builtin '${name}'`);
      return fn(x.arguments, sf);
    }
    if (ts.isElementAccessExpression(x)) {
      // A[i] on a ptr param → tt.load via addptr+splat pattern (1D vector load)
      const base = emitExpr(x.expression, sf);
      const idx = emitExpr(x.argumentExpression, sf);
      if (!isPtr(base.type.elem)) throw new Error(`kernel_ttir: indexing non-pointer ${base.name}`);
      // Only handle the common case: idx is a tensor<i32> → pointer-tensor load.
      if (idx.rank >= 1) {
        const n = idx.type.shape[0];
        const elemPtr = (base.type.elem as { ptr: ScalarElem }).ptr;
        const sp = builder.splatPtr(base, n, elemPtr);
        const ap = builder.addptr(sp, idx);
        return builder.load(ap);
      }
      throw new Error(`kernel_ttir: scalar indexing not yet supported (use arange)`);
    }
    if (ts.isConditionalExpression(x)) {
      const cond = emitExpr(x.condition, sf);
      const t = emitExpr(x.whenTrue, sf);
      const f = emitExpr(x.whenFalse, sf);
      return builder.select(cond, t, f);
    }
    throw new Error(`kernel_ttir: unsupported expression ${ts.SyntaxKind[x.kind]}`);
  }

  // Statement emitter.
  function emitStmt(n: ts.Statement, sf: ts.SourceFile): void {
    if (ts.isBlock(n)) { for (const s of n.statements) emitStmt(s, sf); return; }
    if (ts.isVariableStatement(n)) {
      for (const d of n.declarationList.declarations) {
        const name = d.name.getText(sf);
        if (!d.initializer) throw new Error(`kernel_ttir: '${name}' needs initializer`);
        env.set(name, emitExpr(d.initializer, sf));
      }
      return;
    }
    if (ts.isExpressionStatement(n)) { emitExpr(n.expression, sf); return; }
    if (ts.isIfStatement(n)) {
      const cond = emitExpr(n.expression, sf);
      builder.if_(cond, (b) => emitStmtIn(b, n.thenStatement, sf),
        n.elseStatement ? (b) => emitStmtIn(b, n.elseStatement, sf) : undefined);
      return;
    }
    if (ts.isForStatement(n)) {
      // for (let i = start; i < end; i += step) { ... }
      if (!n.initializer || !ts.isVariableDeclarationList(n.initializer))
        throw new Error("kernel_ttir: for needs `let i = ...` initializer");
      const decl = n.initializer.declarations[0];
      const ivName = decl.name.getText(sf);
      const start = emitExpr(decl.initializer!, sf);
      // condition: i < end
      if (!n.condition || !ts.isBinaryExpression(n.condition) || n.condition.operatorToken.kind !== ts.SyntaxKind.LessThanToken)
        throw new Error("kernel_ttir: for condition must be `i < end`");
      const end = emitExpr(n.condition.right, sf);
      // increment: i += step  (default 1)
      let step = builder.i32(1);
      if (n.incrementor && ts.isBinaryExpression(n.incrementor) && n.incrementor.operatorToken.kind === ts.SyntaxKind.PlusEqualsToken) {
        step = emitExpr(n.incrementor.right, sf);
      }
      builder.for_(start, end, step, (b, iv) => {
        env.set(ivName, iv);
        emitStmtIn(b, n.statement, sf);
        env.delete(ivName);
      });
      return;
    }
    if (ts.isWhileStatement(n)) {
      // while (cond) { ... } — lower via scf.while with no iter args.
      builder.while_(
        [],
        (b) => {
          const cond = emitExprIn(b, n.expression, sf);
          return { cond, iterArgs: [] };
        },
        (b) => {
          emitStmtIn(b, n.statement, sf);
          return [];
        },
      );
      return;
    }
    if (ts.isReturnStatement(n)) return;
    throw new Error(`kernel_ttir: unsupported statement ${ts.SyntaxKind[n.kind]}`);
  }
  // Statement emitters that bind to a specific builder (for nested regions).
  function emitStmtIn(b: TTIRBuilder, n: ts.Statement, sf: ts.SourceFile): void {
    // For now, region-local statements share the global env. This is fine for
    // side-effecting stores; true block-scoped SSA would require region-local envs.
    emitStmt(n, sf);
  }
  function emitExprIn(b: TTIRBuilder, x: ts.Expression, sf: ts.SourceFile): Value {
    return emitExpr(x, sf);
  }

  // Walk the body.
  const body = strippedFn.body;
  if (ts.isBlock(body)) { for (const s of body.statements) emitStmt(s, sf); }
  else { emitStmt(body as ts.Statement, sf); }

  const ttir = builder.build("kernel", 4);
  return {
    source: stripped,
    ttir,
    compile: (numWarps = 4) => compileTTIR(ttir, numWarps),
  };
}

export default { TTIRBuilder, kernel_ttir, compileTTIR, Value };
