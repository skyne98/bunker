// dsl.ts — TypeScript GPU kernels: proper types, HM type checking, LLVM → PTX
import ts from "typescript";
import { dlopen, ptr as ffiPtr, CString } from "bun:ffi";
import { readFileSync } from "fs";

// ═══════════════════════════════════════════════════════════════════
// 1. Stub types (for your TS editor — no runtime effect)
// ═══════════════════════════════════════════════════════════════════
export class i32 {}
export class f32 {}
export class i16 {}
export class i8 {}
export class ptr<T> {}
export const blockIdx = { x: 0, y: 0, z: 0 };
export const threadIdx = { x: 0, y: 0, z: 0 };
export const blockDim  = { x: 0, y: 0, z: 0 };

// ═══════════════════════════════════════════════════════════════════
// Struct/array descriptor system
// ═══════════════════════════════════════════════════════════════════
export type FieldType = { kind: "base"; name: string } | { kind: "array"; base: FieldType; count: number };
export interface StructField { name: string; type: FieldType; offset: number; size: number; }
export interface StructDesc { kind: "struct"; name: string; fields: StructField[]; size: number; }

export function array(base: { __type?: FieldType }, count: number): { __type: FieldType } & { __array: true } {
  const bt = (base as any).__type ?? { kind: "base", name: "i8" };
  return { __type: { kind: "array", base: bt, count }, __array: true } as any;
}

const _baseStruct = function(fields: Record<string, any>, name = "anon"): StructDesc & { __size: number } {
  let offset = 0;
  const fds: StructField[] = [];
  for (const [key, val] of Object.entries(fields)) {
    const ft = resolveFieldType(val);
    const size = fieldSize(ft);
    fds.push({ name: key, type: ft, offset, size });
    offset += size;
  }
  return { kind: "struct", name, fields: fds, size: offset, __size: offset } as any;
};

function resolveFieldType(val: any): FieldType {
  if (val && val.__type) return val.__type; // array
  if (val === i32 || val?.name === "i32") return { kind: "base", name: "i32" };
  if (val === i16 || val?.name === "i16") return { kind: "base", name: "i16" };
  if (val === i8 || val?.name === "i8") return { kind: "base", name: "i8" };
  if (val === f32 || val?.name === "f32") return { kind: "base", name: "f32" };
  if (typeof val === "string") return { kind: "base", name: val };
  return { kind: "base", name: "i8" };
}

function fieldSize(t: FieldType): number {
  if (t.kind === "base") {
    if (t.name === "i8") return 1;
    if (t.name === "i16") return 2;
    if (t.name === "i32") return 4;
    if (t.name === "f32") return 4;
    return 4;
  }
  if (t.kind === "array") return t.count * fieldSize(t.base);
  return 4;
}

export function sizeof(s: any): number {
  if (s && s.__size) return s.__size;
  if (s && s.kind === "struct") return s.size;
  return 4;
}

function typeToFieldType(t: FieldType): string {
  if (t.kind === "base") return t.name;
  if (t.kind === "array") return `[${t.count} x ${typeToFieldType(t.base)}]`;
  return "i8";
}

function fieldTypeToHMType(ft: FieldType): Type {
  if (ft.kind === "base") return base(ft.name);
  if (ft.kind === "array") {
    // Array fields return a pointer to the element type (for indexing)
    return mkPtr(fieldTypeToHMType(ft.base));
  }
  return I32;
}

// Global struct registry
const STRUCT_REGISTRY = new Map<string, StructDesc>();
export function registerStruct(name: string, desc: StructDesc) { STRUCT_REGISTRY.set(name, desc); }

// Registering struct() that auto-registers
export function struct(fields: Record<string, any>, name = "anon"): StructDesc & { __size: number } {
  const desc = _baseStruct(fields, name);
  STRUCT_REGISTRY.set(desc.name, desc);
  return desc as any;
}

// ═══════════════════════════════════════════════════════════════════
// 2. HM Type System
// ═══════════════════════════════════════════════════════════════════
let _tv = 0;
type TVar = number;

type Type =
  | { tag: "var"; id: TVar }
  | { tag: "base"; name: string }
  | { tag: "ptr"; elem: Type }
  | { tag: "fn"; args: Type[]; ret: Type }
  | { tag: "struct"; desc: StructDesc };

function tv(): Type { return { tag: "var", id: _tv++ }; }
function base(name: string): Type { return { tag: "base", name }; }
function mkPtr(elem: Type): Type { return { tag: "ptr", elem }; }
const I32 = base("i32"), F32 = base("f32"), I1 = base("i1"), VOID = base("void");

function typeEq(a: Type, b: Type): boolean {
  if (a.tag !== b.tag) return false;
  if (a.tag === "var") return a.id === (b as any).id;
  if (a.tag === "base") return a.name === (b as any).name;
  if (a.tag === "ptr") return typeEq(a.elem, (b as any).elem);
  if (a.tag === "fn") return a.args.length === (b as any).args.length && a.args.every((x, i) => typeEq(x, (b as any).args[i])) && typeEq(a.ret, (b as any).ret);
  if (a.tag === "struct") return a.desc.name === (b as any).desc.name;
  return false;
}

function typeClone(t: Type): Type {
  if (t.tag === "var") return { tag: "var", id: t.id };
  if (t.tag === "base") return { tag: "base", name: t.name };
  if (t.tag === "ptr") return { tag: "ptr", elem: typeClone(t.elem) };
  if (t.tag === "fn") return { tag: "fn", args: t.args.map(typeClone), ret: typeClone(t.ret) };
  if (t.tag === "struct") return { tag: "struct", desc: t.desc };
  return t;
}

function typeShow(t: Type): string {
  if (t.tag === "var") return `'${t.id}`;
  if (t.tag === "base") return t.name;
  if (t.tag === "ptr") return `ptr<${typeShow(t.elem)}>`;
  if (t.tag === "fn") return `(${t.args.map(typeShow).join(", ")}) -> ${typeShow(t.ret)}`;
  if (t.tag === "struct") return `struct ${t.desc.name}`;
  return "?";
}

class UnifyError extends Error {
  constructor(public a: Type, public b: Type) { super(`Type mismatch: ${typeShow(a)} ≠ ${typeShow(b)}`); }
}

class Subst {
  map = new Map<number, Type>();

  get(id: number): Type | undefined { return this.map.get(id); }
  set(id: number, t: Type) { this.map.set(id, t); }

  apply(t: Type): Type {
    if (t.tag === "var") {
      const found = this.get(t.id);
      if (found) return this.apply(found);
      return t;
    }
    if (t.tag === "ptr") return { tag: "ptr", elem: this.apply(t.elem) };
    if (t.tag === "fn") return { tag: "fn", args: t.args.map(a => this.apply(a)), ret: this.apply(t.ret) };
    return t;
  }

  unify(a: Type, b: Type) {
    a = this.apply(a);
    b = this.apply(b);
    if (typeEq(a, b)) return;
    if (a.tag === "var") {
      if (b.tag === "var" && a.id === b.id) return;
      if (this.occurs(a.id, b)) throw new UnifyError(a, b);
      this.set(a.id, b);
      return;
    }
    if (b.tag === "var") {
      if (this.occurs(b.id, a)) throw new UnifyError(a, b);
      this.set(b.id, a);
      return;
    }
    if (a.tag === "ptr" && b.tag === "ptr") { this.unify(a.elem, b.elem); return; }
    if (a.tag === "fn" && b.tag === "fn") {
      a.args.forEach((x, i) => this.unify(x, b.args[i]));
      this.unify(a.ret, b.ret);
      return;
    }
    throw new UnifyError(a, b);
  }

  private occurs(id: number, t: Type): boolean {
    if (t.tag === "var") return t.id === id;
    if (t.tag === "ptr") return this.occurs(id, t.elem);
    if (t.tag === "fn") return t.args.some(a => this.occurs(id, a)) || this.occurs(id, t.ret);
    return false;
  }
}

// Type environment: maps variable name → Type
class Env {
  map = new Map<string, Type>();
  get(n: string): Type { return this.map.get(n)!; }
  set(n: string, t: Type) { this.map.set(n, t); }
  clone(): Env { const e = new Env(); e.map = new Map(this.map); return e; }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Kernel function with source-file type parsing
// ═══════════════════════════════════════════════════════════════════
let _uid = 0, _lid = 0;
function fresh() { return `v${_uid++}`; }
function label() { return `b${_lid++}`; }

export function kernel<F extends (...a: any[]) => void>(fn: F): F & { source: string } {
  _uid = 0; _lid = 0;

  // Get function source (stripped of types by Bun)
  const stripped = fn.toString();

  // Try to get the caller file to read typed source
  let typedSource: string | null = null;
  try {
    const stack = new Error().stack;
    const lines = stack!.split("\n");
    for (const line of lines) {
      const m = line.match(/([^\s()]+\.ts):\d+(?::\d+)?\)?/);
      if (m) {
        const file = m[1];
        if (!file.includes("dsl.ts") && !file.includes("typescript")) {
          try { typedSource = readFileSync(file, "utf-8"); } catch {}
          if (typedSource) break;
        }
      }
    }
  } catch {}

  // Parse with TS compiler: first with types (from file), then without (from fn)
  let typedFn: ts.ArrowFunction | undefined;
  let strippedFn: ts.ArrowFunction | undefined;
  let sourceFile: ts.SourceFile | undefined;

  try {
    sourceFile = ts.createSourceFile("k.ts", `const __k__ = ${stripped}`, ts.ScriptTarget.Latest, true);
    ts.forEachChild(sourceFile, n => {
      if (ts.isVariableStatement(n) && ts.isArrowFunction(n.declarationList.declarations[0].initializer!))
        strippedFn = n.declarationList.declarations[0].initializer as ts.ArrowFunction;
    });
  } catch {}
  if (!strippedFn) throw Error("Must pass arrow function");

  // Parse typed source (from file) to extract parameter types
  const paramTypes: (string | null)[] = strippedFn.parameters.map(() => null);
  if (typedSource) {
    try {
      const typedSF = ts.createSourceFile("k2.ts", typedSource, ts.ScriptTarget.Latest, true);
      ts.forEachChild(typedSF, function findKernelCall(node: ts.Node) {
        if (ts.isCallExpression(node) && node.expression.getText(typedSF) === "kernel") {
          const arg = node.arguments[0];
          if (arg && ts.isArrowFunction(arg)) {
            typedFn = arg;
            arg.parameters.forEach((p, i) => {
              if (p.type) paramTypes[i] = p.type.getText(typedSF);
            });
          }
        }
        ts.forEachChild(node, findKernelCall);
      });
    } catch {}
  }

  const target = strippedFn;

  // ══════ HM Type Checking ══════════════════════════════════════
  const subst = new Subst();
  const env = new Env();
  const exprTypes = new Map<ts.Node, Type>(); // stores inferred type per expression node

  target.parameters.forEach((p, i) => {
    const name = p.name.getText(sourceFile!);
    if (paramTypes[i]) {
      const at = paramTypes[i]!;
      const m = at.match(/^ptr<(.+)>$/);
      const baseM = at.match(/^i32|f32|i16|i8|f64$/);
      if (m) {
        const innerName = m[1];
        const structDesc = STRUCT_REGISTRY.get(innerName);
        if (structDesc) env.set(name, mkPtr({ tag: "struct", desc: structDesc }));
        else env.set(name, mkPtr(base(innerName)));
      } else if (baseM) env.set(name, base(at));
      else env.set(name, tv());
    } else {
      env.set(name, tv());
    }
  });

  // Expression type inference with HM
  function inferExpr(x: ts.Expression): Type {
    const cached = exprTypes.get(x);
    if (cached) return cached;
    const result = inferExprInner(x);
    exprTypes.set(x, result);
    return result;
  }

  function inferExprInner(x: ts.Expression): Type {
    if (ts.isIdentifier(x)) {
      const t = x.text;
      if (t === "blockIdx" || t === "threadIdx" || t === "blockDim") return tv(); // will be constrained
      return env.get(t) ?? tv();
    }
    if (ts.isNumericLiteral(x)) return x.text.includes(".") ? F32 : I32;
    if (ts.isParenthesizedExpression(x)) return inferExpr(x.expression);
    if (ts.isPrefixUnaryExpression(x)) {
      const v = inferExpr(x.operand);
      if (x.operator === ts.SyntaxKind.ExclamationToken) { subst.unify(v, I1); return I1; }
      if (x.operator === ts.SyntaxKind.MinusToken) { subst.unify(v, I32); return I32; }
      return v;
    }
    if (ts.isPropertyAccessExpression(x)) {
      const obj = inferExpr(x.expression);
      const p = x.name.text;
      // blockIdx.x, threadIdx.x → i32
      if (ts.isIdentifier(x.expression) && (x.expression.text === "blockIdx" || x.expression.text === "threadIdx")) {
        return I32;
      }
      if (ts.isIdentifier(x.expression) && x.expression.text === "blockDim") {
        return I32;
      }
      // Struct field access: ptr->field or struct.field
      const objR = subst.apply(obj);
      if (objR.tag === "struct") {
        const field = objR.desc.fields.find(f => f.name === p);
        if (field) return fieldTypeToHMType(field.type);
      }
      if (objR.tag === "ptr" && objR.elem.tag === "struct") {
        const field = objR.elem.desc.fields.find(f => f.name === p);
        if (field) return fieldTypeToHMType(field.type);
      }
      throw Error(`${x.expression.getText(sourceFile!)}.${p}`);
    }
    if (ts.isBinaryExpression(x)) {
      const k = x.operatorToken.kind;
      const lhs = inferExpr(x.left);
      const rhs = inferExpr(x.right);
      // Assignment = 
      if (k === ts.SyntaxKind.EqualsToken) {
        const val = inferExpr(x.right);
        if (ts.isElementAccessExpression(x.left)) {
          const arr = inferExpr(x.left.expression);
          const idx = inferExpr(x.left.argumentExpression);
          const elem = tv();
          subst.unify(arr, mkPtr(elem));
          subst.unify(idx, I32);
          const valA = subst.apply(val);
          const elemA = subst.apply(elem);
          if (valA.tag === "base" && valA.name === "i32" && elemA.tag === "base" && elemA.name === "f32") {
            // i32→f32 promotion on store
          } else {
            subst.unify(val, elem);
          }
          return val;
        }
        if (ts.isIdentifier(x.left)) {
          env.set(x.left.text, val);
          return val;
        }
      }
      // Compound assignment (+=, -=, *=, /=) — promote variable type
      if ([ts.SyntaxKind.PlusEqualsToken, ts.SyntaxKind.MinusEqualsToken,
           ts.SyntaxKind.AsteriskEqualsToken, ts.SyntaxKind.SlashEqualsToken].includes(k)) {
        const rhsType = inferExpr(x.right);
        if (ts.isIdentifier(x.left)) {
          const oldType = env.get(x.left.text);
          // Promote i32 → f32 if RHS is f32
          if (oldType && rhsType && oldType.tag === "base" && oldType.name === "i32" &&
              rhsType.tag === "base" && rhsType.name === "f32") {
            env.set(x.left.text, rhsType);
          }
          return rhsType;
        }
        return I32;
      }
      // Arithmetic — promote to i32 or f32
      if ([ts.SyntaxKind.PlusToken, ts.SyntaxKind.AsteriskToken, ts.SyntaxKind.MinusToken, ts.SyntaxKind.SlashToken].includes(k)) {
        const isF = lhs.tag === "base" && lhs.name === "f32";
        const resultType = isF ? F32 : I32;
        return resultType;
      }
      // Logical ops
      if (k === ts.SyntaxKind.BarBarToken || k === ts.SyntaxKind.AmpersandAmpersandToken) {
        subst.unify(lhs, I1); subst.unify(rhs, I1);
        return I1;
      }
      // Bitwise ops → integer (both sides widen to i32)
      if ([ts.SyntaxKind.AmpersandToken, ts.SyntaxKind.BarToken,
           ts.SyntaxKind.LessThanLessThanToken, ts.SyntaxKind.GreaterThanGreaterThanToken].includes(k)) {
        return I32;
      }
      // Comparison
      if ([ts.SyntaxKind.LessThanToken, ts.SyntaxKind.GreaterThanToken, ts.SyntaxKind.LessThanEqualsToken,
           ts.SyntaxKind.GreaterThanEqualsToken, ts.SyntaxKind.EqualsEqualsToken, ts.SyntaxKind.ExclamationEqualsToken].includes(k)) {
        subst.unify(lhs, rhs);
        return I1;
      }
      throw Error(`Op ${ts.SyntaxKind[k]}`);
    }
    if (ts.isElementAccessExpression(x)) {
      const arr = inferExpr(x.expression);
      const idx = inferExpr(x.argumentExpression);
      subst.unify(idx, I32);
      const arrType = subst.apply(arr);
      if (arrType.tag === "ptr") {
        const elem = arrType.elem;
        if (elem.tag === "struct") return elem;
        return I32;
      }
      const elemVar = tv();
      subst.unify(arr, mkPtr(elemVar));
      return I32;
    }
    if (ts.isCallExpression(x)) {
      const name = x.expression.getText(sourceFile!);
      const args = x.arguments.map(a => inferExpr(a));
      if (name === "sizeof") return I32;
      if (name === "half_to_float") return F32;
      if (name === "shared") {
        const structName = x.arguments[0].getText(sourceFile!);
        const sd = STRUCT_REGISTRY.get(structName);
        if (sd) return mkPtr({ tag: "struct", desc: sd });
        return tv();
      }
      if (name === "memcpy") return I32;
      if (name === "asm") { return I32; }
      if (name.startsWith("llvm.")) return I32;
      const ret = tv();
      args.forEach(a => { /* any type OK for now */ });
      return ret;
    }
    if (ts.isConditionalExpression(x)) {
      const cond = inferExpr(x.condition);
      const t = inferExpr(x.whenTrue);
      const f = inferExpr(x.whenFalse);
      subst.unify(cond, I1);
      subst.unify(t, f);
      return t;
    }
    return tv();
  }

  // Statement type checking
  function checkStmt(n: ts.Statement) {
    if (ts.isBlock(n)) { for (const st of n.statements) checkStmt(st); return; }
    if (ts.isVariableStatement(n)) {
      for (const d of n.declarationList.declarations) {
        const name = d.name.getText(sourceFile!);
        if (d.initializer) {
          const t = inferExpr(d.initializer);
          env.set(name, t);
        }
      }
      return;
    }
    if (ts.isIfStatement(n)) {
      const cond = inferExpr(n.expression);
      subst.unify(cond, I1);
      checkStmt(n.thenStatement);
      if (n.elseStatement) checkStmt(n.elseStatement);
      return;
    }
    if (ts.isForStatement(n)) {
      if (n.initializer) checkStmt(n.initializer);
      if (n.condition) subst.unify(inferExpr(n.condition), I1);
      if (n.incrementor) inferExpr(n.incrementor);
      checkStmt(n.statement);
      return;
    }
    if (ts.isExpressionStatement(n)) { inferExpr(n.expression); return; }
    if (ts.isReturnStatement(n)) { return; }
    if (ts.isPostfixUnaryExpression(n)) { inferExpr(n.operand); return; }
  }

  // Run HM type inference on the body
  const fb = target.body;
  if (ts.isBlock(fb)) { for (const st of fb.statements) checkStmt(st); }
  else { checkStmt(fb as ts.Statement); }

  // Apply the substitution to get concrete types for params
  const finalParamTypes = target.parameters.map(p => {
    const name = p.name.getText(sourceFile!);
    return subst.apply(env.get(name));
  });

  // ══════ LLVM IR Generation ════════════════════════════════════
  const ssaTypes = new Map<string, string>();
  const vars = new Map<string, string>();
  const allocas = new Set<string>();
  const body: string[] = [];
  const globals: string[] = [];
  let _sid = 0;
  const emit = (s: string) => body.push(`  ${s}`);
  const emitGlobal = (s: string) => globals.push(s);
  const ptrAS = new Map<string, number>(); // SSA name → address space

  // Helper to get LLVM type from HM type
  function toLLVMType(t: Type): string {
    t = subst.apply(t);
    if (t.tag === "base" && t.name === "i32") return "i32";
    if (t.tag === "base" && t.name === "i16") return "i16";
    if (t.tag === "base" && t.name === "i8") return "i8";
    if (t.tag === "base" && t.name === "f32") return "float";
    if (t.tag === "base" && t.name === "f64") return "double";
    if (t.tag === "base" && t.name === "i1") return "i1";
    if (t.tag === "ptr") return "ptr";
    if (t.tag === "var") return "i32";
    return "i32";
  }

  // Known device functions: name → { args: string[], ret: string }
  const DEVICE_FNS: Record<string, { args: string[]; ret: string }> = {
    "half_to_float": { args: ["i32"], ret: "float" },
    "__syncthreads": { args: [], ret: "void", llvm: "@llvm.nvvm.barrier0" },
    "mma_m16n8k32": { args: ["i32","i32","i32","i32","i32","i32","i32","i32","i32","i32","i32","i32","i32","i32","i32","i32","i32","i32"], ret: "void", llvm: "@llvm.nvvm.mma.m16n8k32.row.col.s8.s8.s32" },
  };

  // Track addrspace per SSA/global value. Returns the value with addrspace annotation.
  function gep(ptrVal: string, ty: string, idx: string): string {
    const as = ptrAS.has(ptrVal) ? ` addrspace(${ptrAS.get(ptrVal)})` : "";
    const r = fresh(); emit(`%${r} = getelementptr ${ty}, ptr${as} ${ptrVal}, i32 ${idx}`);
    if (ptrAS.has(ptrVal)) ptrAS.set(`%${r}`, ptrAS.get(ptrVal)!);
    return `%${r}`;
  }
  function gepStruct(ptrVal: string, structName: string, idx: string, field: number): string {
    const as = ptrAS.has(ptrVal) ? ` addrspace(${ptrAS.get(ptrVal)})` : "";
    const r = fresh(); emit(`%${r} = getelementptr %${structName}, ptr${as} ${ptrVal}, i32 ${idx}, i32 ${field}`);
    if (ptrAS.has(ptrVal)) ptrAS.set(`%${r}`, ptrAS.get(ptrVal)!);
    return `%${r}`;
  }
  function loadLLVM(ty: string, ptrVal: string): string {
    const as = ptrAS.has(ptrVal) ? ` addrspace(${ptrAS.get(ptrVal)})` : "";
    const r = fresh(); emit(`%${r} = load ${ty}, ptr${as} ${ptrVal}`);
    return `%${r}`;
  }
  function storeLLVM(ty: string, val: string, ptrVal: string): void {
    const as = ptrAS.has(ptrVal) ? ` addrspace(${ptrAS.get(ptrVal)})` : "";
    emit(`store ${ty} ${val}, ptr${as} ${ptrVal}`);
  }

  // EMIT: expression → SSA name (emits instructions)
  function emitExpr(x: ts.Expression): string {
    if (ts.isIdentifier(x)) {
      const t = x.text;
      if (vars.has(t)) {
        const v = vars.get(t)!;
        if (allocas.has(t)) {
          const ty = env.get(t) ? toLLVMType(env.get(t)!) : "i32";
          return loadLLVM(ty, v);
        }
        return v;
      }
      return `%${t}`;
    }
    if (ts.isNumericLiteral(x)) {
      const numTy = exprTypes.get(x);
      if (numTy) { const aty = subst.apply(numTy); if (aty.tag === "base" && aty.name === "f32") { return parseFloat(x.text) + ".0"; } }
      return x.text;
    }
    if (ts.isParenthesizedExpression(x)) return emitExpr(x.expression);
    if (ts.isPropertyAccessExpression(x)) {
      const objText = x.expression.getText(sourceFile!);
      if (objText === "blockIdx" || objText === "threadIdx") {
        const dim = x.name.text;
        if (dim === "x" || dim === "y" || dim === "z") {
          const reg = objText === "blockIdx" ? "ctaid" : "tid";
          const r = fresh(); emit(`%${r} = call i32 @llvm.nvvm.read.ptx.sreg.${reg}.${dim}()`); return `%${r}`;
        }
      }
      if (objText === "blockDim") {
        const dim = x.name.text;
        if (dim === "x") return "256";
        if (dim === "y") return "1";
        if (dim === "z") return "1";
      }
      // Struct field access: ptr->field
      const objVal = emitExpr(x.expression);
      // Determine type from env (for variables) or exprTypes (for expressions)
      let objType: Type | null = null;
      if (ts.isIdentifier(x.expression)) {
        const n = x.expression.text;
        const t = env.get(n);
        if (t) objType = subst.apply(t);
      }
      if (!objType && exprTypes.has(x.expression)) objType = subst.apply(exprTypes.get(x.expression)!);
      let innerType = objType;
      if (innerType && innerType.tag === "ptr") innerType = innerType.elem;
      if (innerType && innerType.tag === "struct") {
        const sd = innerType.desc;
        const fi = sd.fields.findIndex(f => f.name === x.name.text);
        if (fi >= 0) {
          const llvmStructName = `${sd.name}`;
          const llvmTy = typeToFieldType(sd.fields[fi].type);
          const gepR = gepStruct(objVal, llvmStructName, "0", fi);
          if (sd.fields[fi].type.kind === "array") return gepR;
          const ld = loadLLVM(llvmTy, gepR);
          if (llvmTy === "i8") { const ext = fresh(); emit(`%${ext} = sext i8 ${ld} to i32`); return `%${ext}`; }
          if (llvmTy === "i16") { const ext = fresh(); emit(`%${ext} = zext i16 ${ld} to i32`); return `%${ext}`; }
          return ld;
        }
      }
      throw Error(`${objText}.${x.name.text}`);
    }
    if (ts.isBinaryExpression(x)) {
      const k = x.operatorToken.kind;
      if (k === ts.SyntaxKind.EqualsToken) {
        const val = emitExpr(x.right);
        if (ts.isElementAccessExpression(x.left)) {
          const arr = emitExpr(x.left.expression);
          const idx = emitExpr(x.left.argumentExpression);
          const arrType = exprTypes.has(x.left.expression) ? subst.apply(exprTypes.get(x.left.expression)!) : I32;
          const elemLLVM = arrType.tag === "ptr" ? toLLVMType((arrType as any).elem) : "i32";
          const gp = gep(arr, elemLLVM, idx);
          const rhsType = exprTypes.has(x.right) ? subst.apply(exprTypes.get(x.right)!) : null;
          const storeVal = elemLLVM === "float" && rhsType && rhsType.tag === "base" && rhsType.name === "i32"
            ? (() => { const cv = fresh(); emit(`%${cv} = sitofp i32 ${val} to float`); return `%${cv}`; })()
            : val;
          storeLLVM(elemLLVM, storeVal, gp);
          return val;
        }
        if (ts.isIdentifier(x.left)) {
          const name = x.left.text;
          const ptr = vars.get(name);
          const hmTy = env.get(name);
          if (ptr && allocas.has(name)) {
            const ty = hmTy ? toLLVMType(hmTy) : "i32";
            const rhsType = exprTypes.has(x.right) ? subst.apply(exprTypes.get(x.right)!) : null;
            const storeVal = ty === "float" && rhsType && rhsType.tag === "base" && rhsType.name === "i32"
              ? (() => { const cv = fresh(); emit(`%${cv} = sitofp i32 ${val} to float`); return `%${cv}`; })() : val;
            storeLLVM(ty, storeVal, ptr);
          } else if (ptr) {
            vars.set(name, val);
          }
          return val;
        }
        throw Error("Can only assign to variables or array elements");
      }
      const opMap: Record<number, string> = { 
        [ts.SyntaxKind.PlusToken]: "add", [ts.SyntaxKind.AsteriskToken]: "mul", [ts.SyntaxKind.MinusToken]: "sub", [ts.SyntaxKind.SlashToken]: "sdiv",
        [ts.SyntaxKind.PlusEqualsToken]: "add", [ts.SyntaxKind.MinusEqualsToken]: "sub", 
        [ts.SyntaxKind.AsteriskEqualsToken]: "mul", [ts.SyntaxKind.SlashEqualsToken]: "sdiv",
      };
      // Compound assignment: += -= *= /= (handle BEFORE lhs/rhs emit to avoid double emission)
      if ([ts.SyntaxKind.PlusEqualsToken, ts.SyntaxKind.MinusEqualsToken,
           ts.SyntaxKind.AsteriskEqualsToken, ts.SyntaxKind.SlashEqualsToken].includes(k)) {
        if (ts.isIdentifier(x.left)) {
          const name = x.left.text;
          const ptr = vars.get(name) ?? `%${name}`;
          const rhs = emitExpr(x.right);
          const hmType = env.get(name);
          const ty = hmType ? toLLVMType(hmType) : "i32";
          const isF = ty === "float";
          const loaded = allocas.has(name)
            ? loadLLVM(ty, ptr)
            : ptr;
          let lhsv = loaded, rhsv = rhs;
          if (isF) {
            if (loaded.match(/^-?\d+$/)) { const c = fresh(); emit(`%${c} = sitofp i32 ${loaded} to float`); lhsv = `%${c}`; }
            if (rhs.match(/^-?\d+$/)) { const c = fresh(); emit(`%${c} = sitofp i32 ${rhs} to float`); rhsv = `%${c}`; }
          }
          const op = opMap[k]!;
          const r = fresh(); emit(`%${r} = ${isF ? "f" : ""}${op} ${ty} ${lhsv}, ${rhsv}`);
          if (allocas.has(name)) storeLLVM(ty, `%${r}`, ptr);
          else vars.set(name, `%${r}`);
          return `%${r}`;
        }
      }

      const lhs = emitExpr(x.left);
      const rhs = emitExpr(x.right);
      const lt = exprTypes.has(x.left) ? subst.apply(exprTypes.get(x.left)!) : I32;
      const isFloat = toLLVMType(lt) === "float";

      // Logical: || → or, && → select
      if (k === ts.SyntaxKind.BarBarToken) {
        const r = fresh(); emit(`%${r} = or i1 ${lhs}, ${rhs}`); return `%${r}`;
      }
      if (k === ts.SyntaxKind.AmpersandAmpersandToken) {
        const r = fresh(); emit(`%${r} = and i1 ${lhs}, ${rhs}`); return `%${r}`;
      }
      // Bitwise: & | << >>
      const bitMap: Record<number, string> = {
        [ts.SyntaxKind.AmpersandToken]: "and",
        [ts.SyntaxKind.BarToken]: "or",
        [ts.SyntaxKind.LessThanLessThanToken]: "shl",
        [ts.SyntaxKind.GreaterThanGreaterThanToken]: "ashr",
      };
      const bit = bitMap[k];
      if (bit) {
        const r = fresh(); emit(`%${r} = ${bit} i32 ${lhs}, ${rhs}`);
        return `%${r}`;
      }

      if ([ts.SyntaxKind.PlusToken, ts.SyntaxKind.AsteriskToken, ts.SyntaxKind.MinusToken, ts.SyntaxKind.SlashToken].includes(k)) {
        const opMap: Record<number, string> = { [ts.SyntaxKind.PlusToken]: "add", [ts.SyntaxKind.AsteriskToken]: "mul", [ts.SyntaxKind.MinusToken]: "sub", [ts.SyntaxKind.SlashToken]: "sdiv" };
        const op = opMap[k]!;
        const ty = toLLVMType(lt);
        const r = fresh();
        if (isFloat) {
          // Check left operand type
          let lhsv = lhs;
          if (ts.isIdentifier(x.left)) {
            const lt2 = env.get(x.left.text);
            if (lt2 && lt2.tag === "base" && (lt2.name === "i32" || lt2.name === "i8" || lt2.name === "i16")) {
              const c = fresh(); emit(`%${c} = sitofp i32 ${lhs} to float`); lhsv = `%${c}`;
            }
          } else if (lhs.match(/^-?\d+$/)) {
            const c = fresh(); emit(`%${c} = sitofp i32 ${lhs} to float`); lhsv = `%${c}`;
          }
          // Check right operand type
          let rhsv = rhs;
          if (ts.isIdentifier(x.right)) {
            const rt = env.get(x.right.text);
            if (rt && rt.tag === "base" && (rt.name === "i32" || rt.name === "i8" || rt.name === "i16")) {
              const c = fresh(); emit(`%${c} = sitofp i32 ${rhs} to float`); rhsv = `%${c}`;
            }
          } else if (rhs.match(/^-?\d+$/)) {
            const c = fresh(); emit(`%${c} = sitofp i32 ${rhs} to float`); rhsv = `%${c}`;
          }
          emit(`%${r} = f${op} ${ty} ${lhsv}, ${rhsv}`);
        } else {
          emit(`%${r} = ${op} ${ty} ${lhs}, ${rhs}`);
        }
        return `%${r}`;
      }

      const cmpMap: Record<number, string> = {
        [ts.SyntaxKind.LessThanToken]: "slt", [ts.SyntaxKind.GreaterThanToken]: "sgt",
        [ts.SyntaxKind.LessThanEqualsToken]: "sle", [ts.SyntaxKind.GreaterThanEqualsToken]: "sge",
        [ts.SyntaxKind.EqualsEqualsToken]: "eq", [ts.SyntaxKind.ExclamationEqualsToken]: "ne",
      };
      const cmp = cmpMap[k];
      if (cmp) {
        const ty = toLLVMType(exprTypes.has(x.left) ? subst.apply(exprTypes.get(x.left)!) : I32);
        const r = fresh(); emit(`%${r} = ${isFloat ? "fcmp" : "icmp"} ${cmp} ${ty} ${lhs}, ${rhs}`);
        return `%${r}`;
      }
      throw Error(`Op: ${ts.SyntaxKind[k]}`);
    }
    if (ts.isElementAccessExpression(x)) {
      const arr = emitExpr(x.expression);
      const idx = emitExpr(x.argumentExpression);
      const arrType = exprTypes.has(x.expression) ? subst.apply(exprTypes.get(x.expression)!) : I32;
      if (arrType.tag === "ptr" && arrType.elem.tag === "struct") {
        const sd = arrType.elem.desc;
        return gep(arr, `%${sd.name}`, idx);
      }
      const elemLLVM = arrType.tag === "ptr" ? toLLVMType(arrType.elem) : "i32";
      const gp = gep(arr, elemLLVM, idx);
      const ld = loadLLVM(elemLLVM, gp);
      if (elemLLVM === "i8") { const ext = fresh(); emit(`%${ext} = sext i8 ${ld} to i32`); return `%${ext}`; }
      if (elemLLVM === "i16") { const ext = fresh(); emit(`%${ext} = zext i16 ${ld} to i32`); return `%${ext}`; }
      return ld;
    }
    if (ts.isPrefixUnaryExpression(x)) {
      if (x.operator === ts.SyntaxKind.ExclamationToken) { const v = emitExpr(x.operand); const r = fresh(); emit(`%${r} = xor i1 ${v}, true`); return `%${r}`; }
      if (x.operator === ts.SyntaxKind.MinusToken) { const v = emitExpr(x.operand); const r = fresh(); emit(`%${r} = sub i32 0, ${v}`); return `%${r}`; }
      return emitExpr(x.operand);
    }
    if (ts.isPostfixUnaryExpression(x)) {
      const v = emitExpr(x.operand);
      if (x.operator === ts.SyntaxKind.PlusPlusToken) {
        const r = fresh(); emit(`%${r} = add i32 ${v}, 1`);
        if (ts.isIdentifier(x.operand)) {
          const name = x.operand.text;
          if (allocas.has(name)) storeLLVM("i32", `%${r}`, vars.get(name)!);
          vars.set(name, `%${r}`);
        }
        return `%${r}`;
      }
      if (x.operator === ts.SyntaxKind.MinusMinusToken) {
        const r = fresh(); emit(`%${r} = sub i32 ${v}, 1`);
        if (ts.isIdentifier(x.operand)) {
          const name = x.operand.text;
          if (allocas.has(name)) storeLLVM("i32", `%${r}`, vars.get(name)!);
          vars.set(name, `%${r}`);
        }
        return `%${r}`;
      }
      return v;
    }
    if (ts.isCallExpression(x)) {
      const name = x.expression.getText(sourceFile!);
      if (name === "sizeof") {
        const arg = x.arguments[0];
        if (arg && ts.isIdentifier(arg)) {
          const s = STRUCT_REGISTRY.get(arg.text);
          if (s) return `${s.size}`;
        }
        return "4";
      }
      if (name === "asm") {
        let asmStr = ((x.arguments[0] as any).text || emitExpr(x.arguments[0])).replace(/"/g,'');
        if (!asmStr.endsWith(";")) asmStr += ";";
        const constraints = ((x.arguments[1] as any).text || emitExpr(x.arguments[1])).replace(/"/g,'');
        const argExprs = x.arguments.slice(2).map(a => emitExpr(a));
        // Count output constraints (start with =)
        const parts = constraints.split(",").map(s => s.trim());
        const numOutputs = parts.filter(p => p.startsWith("=")).length;
        const hasResult = numOutputs > 0;
        if (!hasResult) {
          emit(`call void asm sideeffect "${asmStr}", "${constraints}"(${argExprs.map(a => `i32 ${a}`).join(", ")})`);
          return "0";
        }
        if (numOutputs === 1) {
          const r = fresh(); emit(`%${r} = call i32 asm "${asmStr}", "${constraints}"(${argExprs.map(a => `i32 ${a}`).join(", ")})`);
          return `%${r}`;
        }
        // Multiple outputs: use struct return type  
        const structTy = "{" + Array(numOutputs).fill("i32").join(",") + "}";
        const r = fresh(); emit(`%${r} = call ${structTy} asm sideeffect "${asmStr}", "${constraints}"(${argExprs.map(a => `i32 ${a}`).join(", ")})`);
        // Extract all outputs to prevent dead-code elimination
        const outs = Array.from({length: numOutputs}, (_, i) => {
          const e = fresh(); emit(`%${e} = extractvalue ${structTy} %${r}, ${i}`); return `%${e}`;
        });
        // Return first output as the "value" of the asm expression
        return outs[0];
      }
      if (name === "memcpy") {
        const dst = emitExpr(x.arguments[0]);
        const src = emitExpr(x.arguments[1]);
        const sz = emitExpr(x.arguments[2]);
        const castDst = ptrAS.has(dst) ? (() => { const r = fresh(); emit(`%${r} = addrspacecast ptr addrspace(${ptrAS.get(dst)}) ${dst} to ptr`); return `%${r}`; })() : dst;
        const castSrc = ptrAS.has(src) ? (() => { const r = fresh(); emit(`%${r} = addrspacecast ptr addrspace(${ptrAS.get(src)}) ${src} to ptr`); return `%${r}`; })() : src;
        emit(`call void @llvm.memcpy.p0.p0.i64(ptr ${castDst}, ptr ${castSrc}, i64 ${sz}, i1 false)`);
        return "0";
      }
      if (name === "shared") {
        const structName = x.arguments[0].getText(sourceFile!);
        const count = parseInt(emitExpr(x.arguments[1]));
        const sd = STRUCT_REGISTRY.get(structName);
        if (!sd) throw Error(`Unknown struct '${structName}' for shared`);
        const totalBytes = sd.size * count;
        const sharedGlobal = `__shared_${_sid++}`;
        emitGlobal(`@${sharedGlobal} = internal addrspace(3) global [${totalBytes} x i8] undef, align 16`);
        ptrAS.set(`@${sharedGlobal}`, 3);
        return `@${sharedGlobal}`;
      }
      const args = x.arguments.map(a => emitExpr(a));
      const fn = DEVICE_FNS[name];
      const fnName = fn?.llvm ?? name;
      const fnCall = fnName.startsWith("@") ? fnName.substring(1) : fnName;
      const argTypes = fn ? fn.args.join(", ") : args.map(() => "i32").join(", ");
      const retType = fn ? fn.ret : "i32";
      if (retType === "void") {
        emit(`call void @${fnCall}(${args.map((a, i) => `${fn ? fn.args[i] : "i32"} ${a}`).join(", ")})`);
        return "0";
      }
      const r = fresh(); emit(`%${r} = call ${retType} @${fnCall}(${args.map((a, i) => `${fn ? fn.args[i] : "i32"} ${a}`).join(", ")})`);
      return `%${r}`;
    }
    throw Error(`Expr: ${ts.SyntaxKind[x.kind]}`);
  }

  function emitStmt(n: ts.Statement) {
    if (ts.isBlock(n)) { for (const st of n.statements) emitStmt(st); return; }
    if (ts.isVariableStatement(n)) {
      for (const d of n.declarationList.declarations) {
        const name = d.name.getText(sourceFile!);
        const hmTy = env.get(name);
        if (hmTy && hmTy.tag !== "base") {
          // Non-base types (struct/ptr) — keep SSA value, no alloca
          if (d.initializer) vars.set(name, emitExpr(d.initializer));
        } else {
          const initVal = d.initializer ? emitExpr(d.initializer) : "0";
          const initTy = hmTy ? toLLVMType(hmTy) : (d.initializer && exprTypes.has(d.initializer) ? toLLVMType(subst.apply(exprTypes.get(d.initializer)!)) : "i32");
          const slot = fresh(); emit(`%${slot} = alloca ${initTy}`);
          allocas.add(name);
          vars.set(name, `%${slot}`);
          const storeV = initTy === "float" && initVal.match(/^\d+$/)
            ? (() => { const cv = fresh(); emit(`%${cv} = sitofp i32 ${initVal} to float`); return `%${cv}`; })() : initVal;
          storeLLVM(initTy, storeV, `%${slot}`);
        }
      }
      return;
    }
    if (ts.isIfStatement(n)) {
      const cond = emitExpr(n.expression);
      const t = label(), el = label(), end = label();
      emit(`br i1 ${cond}, label %${t}, label %${el}`);
      emit(`${t}:`); emitStmt(n.thenStatement);
      if (!endsWithReturn(n.thenStatement)) emit(`br label %${end}`);
      emit(`${el}:`);
      if (n.elseStatement) { emitStmt(n.elseStatement); if (!endsWithReturn(n.elseStatement)) emit(`br label %${end}`); }
      else emit(`br label %${end}`);
      emit(`${end}:`);
      return;
    }
    if (ts.isForStatement(n)) {
      const cond = label(), body = label(), inc = label(), end = label();
      // Handle for loop initializer
      if (n.initializer) {
        if (ts.isVariableDeclarationList(n.initializer)) {
          for (const d of n.initializer.declarations) {
            const name = d.name.getText(sourceFile!);
            const ty = "i32";
            const slot = fresh(); emit(`%${slot} = alloca ${ty}`);
            allocas.add(name);
            vars.set(name, `%${slot}`);
            if (d.initializer) {
              const val = emitExpr(d.initializer);
              storeLLVM(ty, val, `%${slot}`);
            }
          }
        } else {
          emitStmt(n.initializer as ts.Statement);
        }
      }
      emit(`br label %${cond}`); emit(`${cond}:`);
      const c = n.condition ? emitExpr(n.condition) : "true";
      emit(`br i1 ${c}, label %${body}, label %${end}`);
      emit(`${body}:`); emitStmt(n.statement); emit(`br label %${inc}`);
      emit(`${inc}:`); if (n.incrementor) emitExpr(n.incrementor); emit(`br label %${cond}`);
      emit(`${end}:`);
      return;
    }
    if (ts.isExpressionStatement(n)) { emitExpr(n.expression); return; }
    if (ts.isReturnStatement(n)) { emit("ret void"); return; }
    if (ts.isPostfixUnaryExpression(n)) { emitExpr(n.operand); return; }
  }

  function endsWithReturn(n: ts.Statement): boolean {
    if (ts.isBlock(n)) return n.statements.length > 0 && ts.isReturnStatement(n.statements[n.statements.length - 1]);
    return ts.isReturnStatement(n);
  }

  // Walk body to emit LLVM IR
  if (ts.isBlock(fb)) { for (const st of fb.statements) emitStmt(st); }
  else { emitStmt(fb as ts.Statement); }

  // Build LLVM IR — include struct type definitions
  const structDefs: string[] = [];
  for (const [name, desc] of STRUCT_REGISTRY) {
    const members = desc.fields.map(f => typeToFieldType(f.type));
    structDefs.push(`%${name} = type { ${members.join(", ")} }`);
  }

  const sig = target.parameters.map((p, i) => `${toLLVMType(finalParamTypes[i])} %${p.name.getText(sourceFile!)}`).join(", ");
  const source = [
    ...(structDefs.length ? structDefs : []),
    ...(globals.length ? globals : []),
    `define void @kernel(${sig}) {`,
    ...body,
    `  ret void`,
    `}`,
    `declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()`,
    `declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()`,
    `declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()`,
    `declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()`,
    // @llvm.nvvm.barrier0 is an intrinsic, no declare needed
    `declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)`,
    `define float @half_to_float(i32 %h) {`,
    `  %s = and i32 %h, 65535`,
    `  %sign = and i32 %s, 32768`,
    `  %exp = and i32 %s, 31744`,
    `  %exp2 = lshr i32 %exp, 10`,
    `  %mant = and i32 %s, 1023`,
    `  %is_zero = icmp eq i32 %exp2, 0`,
    `  br i1 %is_zero, label %zero, label %normal`,
    `zero:`,
    `  ret float 0.0`,
    `normal:`,
    `  %exp_adj = sub i32 %exp2, 15`,
    `  %exp_bias = add i32 %exp_adj, 127`,
    `  %exp_shifted = shl i32 %exp_bias, 23`,
    `  %mant_shifted = shl i32 %mant, 13`,
    `  %sign_shifted = shl i32 %sign, 16`,
    `  %result_int = or i32 %sign_shifted, %exp_shifted`,
    `  %result_int2 = or i32 %result_int, %mant_shifted`,
    `  %result = bitcast i32 %result_int2 to float`,
    `  ret float %result`,
    `}`,
    `!0 = !{ptr @kernel, !"kernel", i32 1}`,
    `!nvvm.annotations = !{!0}`,
  ].join("\n") + "\n";

  const compiled = Object.assign(
    function(this: any, ...args: any[]) { runGPUKernel(source, args); },
    { source }
  );
  return compiled as any;
}

// ═══════════════════════════════════════════════════════════════════
// 4. GPU Runtime
// ═══════════════════════════════════════════════════════════════════
const LLVM = "/nix/store/6r234y6pkbyyr8pk1wh7nfsmnzdxyswx-llvm-19.1.7-lib/lib/libLLVM-19.so";
const CUDA = "/run/opengl-driver/lib/libcuda.so";

const ls = dlopen(LLVM, {
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
ls.LLVMInitializeNVPTXTargetInfo(); ls.LLVMInitializeNVPTXTarget();
ls.LLVMInitializeNVPTXTargetMC(); ls.LLVMInitializeNVPTXAsmPrinter();

const cs = dlopen(CUDA, {
  cuInit: { args: ["u32"], returns: "i32" },
  cuDeviceGet: { args: ["pointer", "i32"], returns: "i32" },
  cuDeviceGetName: { args: ["pointer", "i32", "i32"], returns: "i32" },
  cuCtxCreate_v2: { args: ["pointer", "u32", "i32"], returns: "i32" },
  cuModuleLoadData: { args: ["pointer", "pointer"], returns: "i32" },
  cuModuleGetFunction: { args: ["pointer", "pointer", "pointer"], returns: "i32" },
  cuMemAlloc_v2: { args: ["pointer", "i64"], returns: "i32" },
  cuMemcpyHtoD_v2: { args: ["i64", "pointer", "i64"], returns: "i32" },
  cuMemcpyDtoH_v2: { args: ["pointer", "i64", "i64"], returns: "i32" },
  cuLaunchKernel: { args: ["pointer", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "pointer", "pointer", "pointer"], returns: "i32" },
  cuMemFree_v2: { args: ["i64"], returns: "i32" },
  cuCtxSynchronize: { args: [], returns: "i32" },
}).symbols;

function llvmToPTX(src: string): string {
  const ctx = ls.LLVMContextCreate();
  const irBuf = Buffer.from(src + "\0");
  const mb = ls.LLVMCreateMemoryBufferWithMemoryRange(ffiPtr(irBuf), BigInt(irBuf.length - 1), ffiPtr(Buffer.from("k.ll\0")), 1);
  const ma = new BigUint64Array(1);
  if (ls.LLVMParseIRInContext(ctx, mb, ffiPtr(ma), ffiPtr(new BigUint64Array(1)))) throw Error("LLVM IR parse failed");
  const mod = Number(ma[0]);
  const tp = Buffer.from("nvptx64-nvidia-cuda\0"); const ta = new BigUint64Array(1);
  ls.LLVMGetTargetFromTriple(ffiPtr(tp), ffiPtr(ta), ffiPtr(new BigUint64Array(1)));
  const tm = ls.LLVMCreateTargetMachine(Number(ta[0]), ffiPtr(tp), ffiPtr(Buffer.from("sm_86\0")), ffiPtr(Buffer.from("\0")), 2, 0, 0);
  const pa = new BigUint64Array(1);
  ls.LLVMTargetMachineEmitToMemoryBuffer(tm, mod, 0, ffiPtr(new BigUint64Array(1)), ffiPtr(pa));
  return new CString(ls.LLVMGetBufferStart(Number(pa[0])), 0, Number(ls.LLVMGetBufferSize(Number(pa[0])))).toString();
}

function runGPUKernel(source: string, args: any[]) {
  cs.cuInit(0);
  const dev = new Int32Array(1); cs.cuDeviceGet(dev, 0);
  const nb = new Uint8Array(256); cs.cuDeviceGetName(nb, 256, dev[0]);
  const cb = Buffer.alloc(8); cs.cuCtxCreate_v2(cb, 0, dev[0]);
  console.log(`GPU: ${new CString(ffiPtr(nb))}`);

  const ptx = llvmToPTX(source);
  const mod = Buffer.alloc(8);
  if (cs.cuModuleLoadData(mod, Buffer.from(ptx)) !== 0) throw Error("PTX load failed");
  const fn = Buffer.alloc(8);
  cs.cuModuleGetFunction(fn, Number(mod.readBigUInt64LE(0)), Buffer.from("kernel\0"));

  const devPtrs: number[] = [];
  const copies: { b: ArrayBufferView; d: number }[] = [];
  const slots: Buffer[] = [];
  for (const a of args) {
    if (a && typeof a === "object" && a.byteLength !== undefined) {
      const db = Buffer.alloc(8); cs.cuMemAlloc_v2(db, BigInt(a.byteLength));
      const dp = Number(db.readBigUInt64LE(0));
      devPtrs.push(dp); cs.cuMemcpyHtoD_v2(dp, a, BigInt(a.byteLength));
      const slot = Buffer.alloc(8); slot.writeBigUInt64LE(BigInt(dp), 0);
      slots.push(slot); copies.push({ b: a, d: dp });
    } else if (typeof a === "number") {
      const slot = Buffer.alloc(8); slot.writeInt32LE(a, 0);
      slots.push(slot);
    }
  }
  const pb = Buffer.concat(slots);
  const kp = Buffer.alloc((slots.length + 1) * 8);
  const pp = ffiPtr(pb);
  for (let i = 0; i < slots.length; i++) kp.writeBigUInt64LE(BigInt(pp + i * 8), i * 8);
  kp.writeBigUInt64LE(0n, slots.length * 8);
  cs.cuLaunchKernel(Number(fn.readBigUInt64LE(0)), 1, 1, 1, 256, 1, 1, 0, null, kp, null);
  cs.cuCtxSynchronize();
  for (const c of copies) cs.cuMemcpyDtoH_v2(c.b, c.d, BigInt(c.b.byteLength));
  for (const p of devPtrs) cs.cuMemFree_v2(p);
}
