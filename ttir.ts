// ttir.ts — TTIR codegen from TypeScript DSL (parallel to dsl.ts)
// Generates TTIR MLIR text for compilation via libtriton_shim.so
// Reuses dsl.ts's HM type inference and AST parsing

import ts from "typescript";
import { readFileSync } from "fs";
import { dlopen, ptr as ffiPtr, CString } from "bun:ffi";

// Load the shim
let _shimLib: any = null;
function getShim() {
  if (!_shimLib) {
    _shimLib = dlopen(`${__dirname}/libtriton_shim.so`, {
      triton_compile: { args: ["ptr", "i32"], returns: "ptr" },
      triton_free: { args: ["ptr"], returns: "void" },
    }).symbols;
  }
  return _shimLib;
}

// ── Type Stubs (for editor) ──
export const T = {
  i32: {} as i32,
  f32: {} as f32,
  f16: {} as f16,
  ptr: <T>(t?: T) => ({} as ptr<T>),
};

// ── TTIR Builder ──
export class TTIRBuilder {
  private uid = 0;
  private body: string[] = [];
  private ssaNames = new Map<string, string>();
  private blockSize: number;
  private kernelName = "kernel";

  constructor(blockSize = 256) {
    this.blockSize = blockSize;
  }

  private fresh(name = "v"): string {
    const n = `${name}${this.uid++}`;
    this.ssaNames.set(n, n);
    return n;
  }

  private emit(line: string) {
    this.body.push(`    ${line}`);
  }

  // Create a range tensor [0..N)
  makeRange(N: number): string {
    const r = this.fresh("r");
    this.emit(`%${r} = tt.make_range {end = ${N} : i32, start = 0 : i32} : tensor<${N}xi32>`);
    return r;
  }

  // Broadcast a scalar to a tensor
  splat(scalar: string, scalarType: string, N: number, elemType: string = "f32"): string {
    const s = this.fresh("s");
    this.emit(`%${s} = tt.splat %${scalar} : ${scalarType} -> tensor<${N}x${elemType}>`);
    return s;
  }

  // Splat a pointer
  splatPtr(ptr: string, N: number, elemType: string = "f32"): string {
    const s = this.fresh("sp");
    this.emit(`%${s} = tt.splat %${ptr} : !tt.ptr<${elemType}> -> tensor<${N}x!tt.ptr<${elemType}>>`);
    return s;
  }

  // Add pointers and offsets
  addptr(ptrs: string, offs: string, N: number, elemType: string = "f32"): string {
    const a = this.fresh("a");
    this.emit(`%${a} = tt.addptr %${ptrs}, %${offs} : tensor<${N}x!tt.ptr<${elemType}>>, tensor<${N}xi32>`);
    return a;
  }

  // Load a tile
  load(ptrs: string, N: number, elemType: string = "f32", mask?: string, other?: string): string {
    const l = this.fresh("l");
    const maskPart = mask ? `, %${mask}` : "";
    const otherPart = other ? `, %${other}` : "";
    this.emit(`%${l} = tt.load %${ptrs}${maskPart}${otherPart} : tensor<${N}x!tt.ptr<${elemType}>>`);
    return l;
  }

  // Store a tile
  store(ptrs: string, val: string, N: number, elemType: string = "f32", mask?: string) {
    const maskPart = mask ? `, %${mask}` : "";
    this.emit(`tt.store %${ptrs}${maskPart}, %${val} : tensor<${N}x!tt.ptr<${elemType}>>`);
  }

  // Element-wise add (on tensors)
  add(a: string, b: string, N: number, type: string = "f32"): string {
    const r = this.fresh("r");
    this.emit(`%${r} = arith.addf %${a}, %${b} : tensor<${N}x${type}>`);
    return r;
  }

  // Element-wise mul
  mul(a: string, b: string, N: number, type: string = "f32"): string {
    const r = this.fresh("r");
    this.emit(`%${r} = arith.mulf %${a}, %${b} : tensor<${N}x${type}>`);
    return r;
  }

  // Integer add
  addi(a: string, b: string, N: number): string {
    const r = this.fresh("r");
    this.emit(`%${r} = arith.addi %${a}, %${b} : tensor<${N}xi32>`);
    return r;
  }

  // Integer mul
  muli(a: string, b: string, N: number): string {
    const r = this.fresh("r");
    this.emit(`%${r} = arith.muli %${a}, %${b} : tensor<${N}xi32>`);
    return r;
  }

  // Constant tensor (zero fill)
  constantZero(N: number, type: string = "f32"): string {
    const r = this.fresh("c");
    this.emit(`%${r} = arith.constant dense<0.000000e+00> : tensor<${N}x${type}>`);
    return r;
  }

  // Comparison (signed less than)
  cmplt(a: string, b: string, N: number): string {
    const r = this.fresh("p");
    this.emit(`%${r} = arith.cmpi slt, %${a}, %${b} : tensor<${N}xi32>`);
    return r;
  }

  // Get program ID
  getProgramId(dim: string = "x"): string {
    const r = this.fresh("pid");
    this.emit(`%${r} = tt.get_program_id ${dim} : i32`);
    return r;
  }

  // Integer constant (scalar)
  constI32(val: number): string {
    const r = this.fresh("c");
    this.emit(`%${r} = arith.constant ${val} : i32`);
    return r;
  }

  // Float constant (scalar)
  constF32(val: number): string {
    const r = this.fresh("c");
    this.emit(`%${r} = arith.constant ${val > 0 ? val.toExponential() : "0.000000e+00"} : f32`);
    return r;
  }

  // Generate the complete TTIR module
  build(params: { name?: string; numWarps?: number; ptrTypes?: string[] } = {}): string {
    const { name = this.kernelName, numWarps = 4 } = params;

    // Count unique pointer args for !tt.ptr types
    const ptrArgs: { name: string; type: string }[] = [];
    const scalarArgs: { name: string; type: string }[] = [];

    // Collect SSA args referenced in body that are marked as ptr inputs
    for (const [ssa, _name] of this.ssaNames) {
      if (ssa.startsWith("arg")) {
        // Already tracked as an argument
      }
    }

    const header = `module attributes {"ttg.num-warps" = ${numWarps} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @${name}(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {`;

    const footer = `    tt.return
  }
}
`;

    return header + "\n" + this.body.join("\n") + "\n" + footer;
  }

  // Compile TTIR → PTX via shim
  compile(ttir: string, numWarps = 4): string {
    const shim = getShim();
    const buf = Buffer.from(ttir + "\0", "utf-8");
    const rp = shim.triton_compile(ffiPtr(buf), numWarps);
    if (!rp) throw Error("triton_compile returned null");
    const result = new CString(rp);
    shim.triton_free(rp);
    if (result.startsWith("ERROR:")) throw Error(result);
    return result;
  }
}

// ── kernel_ttir: parse a TS arrow function and emit TTIR ──
export function kernel_ttir<F extends (...a: any[]) => void>(
  fn: F,
  blockSize = 256
): F & { source: string; ttir: string; compile: () => string } {
  // Reuse the same AST parsing from dsl.ts, but use our TTIR builder
  const stripped = fn.toString();
  let typedSource: string | null = null;
  try {
    const stack = new Error().stack;
    for (const line of stack!.split("\n")) {
      const m = line.match(/([^\s()]+\.ts):\d+(?::\d+)?\)?/);
      if (m) {
        const file = m[1];
        if (!file.includes("ttir.ts") && !file.includes("typescript")) {
          try { typedSource = readFileSync(file, "utf-8"); } catch {}
          if (typedSource) break;
        }
      }
    }
  } catch {}

  let strippedFn: ts.ArrowFunction | undefined;
  try {
    const sf = ts.createSourceFile("k.ts", `const __k__ = ${stripped}`, ts.ScriptTarget.Latest, true);
    ts.forEachChild(sf, n => {
      if (ts.isVariableStatement(n) && ts.isArrowFunction(n.declarationList.declarations[0].initializer!))
        strippedFn = n.declarationList.declarations[0].initializer as ts.ArrowFunction;
    });
  } catch {}
  if (!strippedFn) throw Error("Must pass arrow function");

  // For now, we return a placeholder that uses the builder
  // The actual AST → TTIR mapping is complex; we provide the builder API
  // for constructing TTIR kernels programmatically

  const result = Object.assign(
    function(this: any, ...args: any[]) {},
    { source: "", ttir: "", compile: () => "" }
  );
  return result as any;
}

export default { TTIRBuilder, kernel_ttir };
