// prototypes/graph_ir.ts — Graph IR with automatic fusion
// Prototype: build a computation graph, fuse compatible nodes, generate fused TTIR kernels.

import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch, LoadedKernel } from "../src/ttir";

// ═══════════════════════════════════════════════════════════════════
// Type system
// ═══════════════════════════════════════════════════════════════════

type DType = "f32" | "bf16" | "f16" | "i32" | "i1";

interface TensorType {
  shape: number[];
  dtype: DType;
  strides: number[];
  custom?: { name: string; fields?: { name: string; type: TensorType }[] };
  desc?: string; // human-readable
}

type TensorRole = "data" | "weight" | "state" | "scratch" | "scalar";

interface NodeIO {
  name: string;           // input/output name on this node
  tensorName: string;     // name of the source tensor in the graph
  type: TensorType;
  role: TensorRole;
}

interface GraphNode {
  id: number;
  op: string;
  inputs: NodeIO[];
  outputs: NodeIO[];
  grid: [number, number, number];
  params: Record<string, any>;
  // Set by fusion pass:
  groupId?: number;
}

interface Tensor {
  name: string;
  type: TensorType;
  role: TensorRole;
  producer: GraphNode | null;   // null = external input
  consumers: GraphNode[];
  // Set by fusion pass:
  isFusedAway?: boolean;        // true = SSA value within a group, no global memory
}

// ═══════════════════════════════════════════════════════════════════
// Graph builder
// ═══════════════════════════════════════════════════════════════════

class Graph {
  nodes: GraphNode[] = [];
  tensors = new Map<string, Tensor>();
  private nextId = 0;

  // Declare an external tensor (input, weight, state)
  input(name: string, type: TensorType, role: TensorRole = "data"): Tensor {
    const t: Tensor = { name, type, role, producer: null, consumers: [] };
    this.tensors.set(name, t);
    return t;
  }

  // Add a node to the graph
  node(
    op: string,
    inputs: { tensor: string | Tensor; name: string }[],
    outputs: { name: string; type: TensorType; role?: TensorRole }[],
    grid: [number, number, number],
    params: Record<string, any> = {},
  ): GraphNode {
    const inputIOs: NodeIO[] = inputs.map(i => {
      const t = typeof i.tensor === "string" ? this.tensors.get(i.tensor)! : i.tensor;
      t.consumers.push(this.nodes[this.nodes.length]); // will be set below
      return { name: i.name, tensorName: t.name, type: t.type, role: t.role };
    });

    const node: GraphNode = {
      id: this.nextId++,
      op,
      inputs: inputIOs,
      outputs: outputs.map(o => ({
        name: o.name,
        tensorName: `n${this.nextId}_${o.name}`,
        type: o.type,
        role: o.role ?? "data",
      })),
      grid,
      params,
    };

    // Fix consumer references (push happened before node was assigned)
    for (const inp of inputs) {
      const t = typeof inp.tensor === "string" ? this.tensors.get(inp.tensor)! : inp.tensor;
      t.consumers[t.consumers.length - 1] = node;
    }

    // Create output tensors
    for (const out of outputs) {
      const fullName = `n${node.id}_${out.name}`;
      const t: Tensor = {
        name: fullName,
        type: out.type,
        role: out.role ?? "data",
        producer: node,
        consumers: [],
      };
      this.tensors.set(fullName, t);
    }

    this.nodes.push(node);
    return node;
  }

  // Convenience: get a tensor by name
  t(name: string): Tensor {
    const t = this.tensors.get(name);
    if (!t) throw new Error(`tensor "${name}" not found`);
    return t;
  }

  // Convenience: get a node's output tensor name
  out(node: GraphNode, outputName: string = "out"): string {
    return `n${node.id}_${outputName}`;
  }

  // ═════════════════════════════════════════════════════════════════
  // Fusion pass
  // ═════════════════════════════════════════════════════════════════

  // ═════════════════════════════════════════════════════════════════
  // Fusion pass — vertical (producer-consumer) + horizontal (siblings)
  // ═════════════════════════════════════════════════════════════════

  fuse(): GraphNode[][] {
    const groups: GraphNode[][] = [];
    const nodeGroup = new Map<number, number>();

    // Pass 1: Vertical fusion — producer-consumer chains with same grid
    for (const node of this.nodes) {
      let fused = false;
      for (let gi = groups.length - 1; gi >= 0; gi--) {
        const group = groups[gi];
        const lastNode = group[group.length - 1];
        if (!this.gridsCompatible(lastNode.grid, node.grid)) continue;

        let hasDataEdge = false;
        let allDataEdgesFusable = true;
        for (const inp of node.inputs) {
          const tensor = this.findTensor(inp.name, node);
          if (!tensor?.producer) continue;
          if (group.includes(tensor.producer)) {
            hasDataEdge = true;
            if (tensor.role !== "data") allDataEdgesFusable = false;
            const otherConsumers = tensor.consumers.filter(c => !group.includes(c) && c !== node);
            if (otherConsumers.length > 0) allDataEdgesFusable = false;
          }
        }
        if (hasDataEdge && allDataEdgesFusable) {
          group.push(node);
          nodeGroup.set(node.id, gi);
          fused = true;
          break;
        }
      }
      if (!fused) {
        groups.push([node]);
        nodeGroup.set(node.id, groups.length - 1);
      }
    }

    // Pass 2: Horizontal fusion — sibling GEMMs with same input + same grid
    // Merges a_proj+b_proj, gate_proj+up_proj into single groups
    for (let i = 0; i < groups.length; i++) {
      if (groups[i].length !== 1) continue; // only merge single-node groups
      const nodeA = groups[i][0];
      if (nodeA.op !== "gemm") continue;
      for (let j = i + 1; j < groups.length; j++) {
        if (groups[j].length !== 1) continue;
        const nodeB = groups[j][0];
        if (nodeB.op !== "gemm") continue;
        if (!this.gridsCompatible(nodeA.grid, nodeB.grid)) continue;
        // Check if they share at least one input tensor
        const sharedInput = nodeA.inputs.some(inpA =>
          nodeB.inputs.some(inpB => inpA.tensorName === inpB.tensorName)
        );
        if (!sharedInput) continue;
        // Merge: group j into group i, mark as dual GEMM
        groups[i].push(nodeB);
        nodeGroup.set(nodeB.id, i);
        groups.splice(j, 1);
        // Update all nodeGroup entries after j
        for (const [nid, gid] of nodeGroup) if (gid > j) nodeGroup.set(nid, gid - 1);
        j--; // recheck this slot
        break;
      }
    }

    // Mark fused-away tensors
    for (const [, tensor] of this.tensors) {
      if (!tensor.producer || tensor.role !== "data") continue;
      const producerGroup = nodeGroup.get(tensor.producer.id);
      if (producerGroup === undefined) continue;
      const allSameGroup = tensor.consumers.length > 0 && tensor.consumers.every(c =>
        nodeGroup.get(c.id) === producerGroup
      );
      if (allSameGroup) tensor.isFusedAway = true;
    }

    return groups;
  }

  private findTensor(ioName: string, node: GraphNode): Tensor | null {
    // Find the tensor by the NodeIO's tensorName
    const inp = node.inputs.find(i => i.name === ioName);
    if (!inp) return null;
    return this.tensors.get(inp.tensorName) ?? null;
  }

  private gridsCompatible(g1: [number, number, number], g2: [number, number, number]): boolean {
    // Exact match
    if (g1[0] === g2[0] && g1[1] === g2[1] && g1[2] === g2[2]) return true;
    // Could add more compatibility rules here (e.g., one is a subset)
    return false;
  }

  // ═════════════════════════════════════════════════════════════════
  // Print
  // ═════════════════════════════════════════════════════════════════

  printGraph(): void {
    console.log("=== Computation Graph ===");
    for (const node of this.nodes) {
      const ins = node.inputs.map(i => `${i.name}:${i.type.dtype}[${i.type.shape.join(",")}]${i.role !== "data" ? `/${i.role}` : ""}`).join(", ");
      const outs = node.outputs.map(o => `${o.name}:${o.type.dtype}[${o.type.shape.join(",")}]${o.role !== "data" ? `/${o.role}` : ""}`).join(", ");
      console.log(`  n${node.id} ${node.op.padEnd(20)} grid=[${node.grid.join(",")}]  (${ins}) → (${outs})`);
    }
  }

  printFusion(groups: GraphNode[][]): void {
    console.log("\n=== Fusion Groups ===");
    for (let gi = 0; gi < groups.length; gi++) {
      const group = groups[gi];
      const ops = group.map(n => `n${n.id}:${n.op}`).join(" + ");
      const grid = group[0].grid.join(",");
      const fusedTensors = [...this.tensors.values()]
        .filter(t => t.isFusedAway && t.producer && group.includes(t.producer))
        .map(t => t.name);
      console.log(`  Group ${gi}: [${grid}] ${ops}`);
      if (fusedTensors.length > 0) {
        console.log(`           fused away: ${fusedTensors.join(", ")}`);
      }
    }
    const totalNodes = this.nodes.length;
    const totalGroups = groups.length;
    const fusedAway = [...this.tensors.values()].filter(t => t.isFusedAway).length;
    console.log(`\n  ${totalNodes} nodes → ${totalGroups} kernels (${totalNodes - totalGroups} fusions)`);
    console.log(`  ${fusedAway} intermediate tensors eliminated (SSA values)`);
  }
}

// ═══════════════════════════════════════════════════════════════════
// Prototype: build a single GDN layer as a graph
// ═══════════════════════════════════════════════════════════════════

const H = 1024, INTER = 3584, QKVD = 6144, ZD = 2048, LVH = 16, LKD = 128, LVD = 128;

const g = new Graph();

// External inputs
g.input("x", { shape: [1, H], dtype: "bf16", strides: [H, 1] }, "data");
g.input("input_layernorm_weight", { shape: [H], dtype: "bf16", strides: [H, 1] }, "weight");
g.input("qkv_weight", { shape: [QKVD, H], dtype: "bf16", strides: [H, 1] }, "weight");
g.input("z_weight", { shape: [ZD, H], dtype: "bf16", strides: [H, 1] }, "weight");
g.input("a_weight", { shape: [LVH, H], dtype: "bf16", strides: [H, 1] }, "weight");
g.input("b_weight", { shape: [LVH, H], dtype: "bf16", strides: [H, 1] }, "weight");
g.input("conv_weight", { shape: [QKVD, 4], dtype: "bf16", strides: [4, 1] }, "weight");
g.input("A_log", { shape: [LVH], dtype: "f32", strides: [1] }, "weight");
g.input("dt_bias", { shape: [LVH], dtype: "bf16", strides: [1] }, "weight");
g.input("norm_weight", { shape: [LVD], dtype: "f32", strides: [1] }, "weight");
g.input("out_proj_weight", { shape: [H, ZD], dtype: "bf16", strides: [ZD, 1] }, "weight");
g.input("post_attn_norm_weight", { shape: [H], dtype: "bf16", strides: [H, 1] }, "weight");
g.input("gate_weight", { shape: [INTER, H], dtype: "bf16", strides: [H, 1] }, "weight");
g.input("up_weight", { shape: [INTER, H], dtype: "bf16", strides: [H, 1] }, "weight");
g.input("down_weight", { shape: [H, INTER], dtype: "bf16", strides: [INTER, 1] }, "weight");
g.input("conv_state", { shape: [3, QKVD], dtype: "bf16", strides: [QKVD, 1], custom: { name: "ConvState" } }, "state");
g.input("s_state", { shape: [LVH, LKD, LVD], dtype: "f32", strides: [LKD * LVD, LVD, 1], custom: { name: "GDNState" } }, "state");

// Build computation graph
// 1. RMSNorm — grid [1]
const normed = g.node("rmsnorm",
  [{ tensor: "x", name: "x" }, { tensor: "input_layernorm_weight", name: "w" }],
  [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
  [1, 1, 1], { N: H }
);

// 2. QKV proj GEMM — grid [1, 96] (ceil(QKVD/64))
const qkvF32 = g.node("gemm",
  [{ tensor: g.out(normed), name: "A" }, { tensor: "qkv_weight", name: "B" }],
  [{ name: "out", type: { shape: [1, QKVD], dtype: "f32", strides: [QKVD, 1] } }],
  [1, Math.ceil(QKVD / 64), 1], { M: 1, N: QKVD, K: H }
);

// 3. Cast QKV — grid [1, 96] (SAME as qkv GEMM → can fuse!)
const qkvB = g.node("cast",
  [{ tensor: g.out(qkvF32), name: "x" }],
  [{ name: "out", type: { shape: [1, QKVD], dtype: "bf16", strides: [QKVD, 1] } }],
  [1, Math.ceil(QKVD / 64), 1], { N: QKVD, from: "f32", to: "bf16" }
);

// 4. Z proj GEMM — grid [1, 32] (different grid → can't fuse with qkv)
const zF32 = g.node("gemm",
  [{ tensor: g.out(normed), name: "A" }, { tensor: "z_weight", name: "B" }],
  [{ name: "out", type: { shape: [1, ZD], dtype: "f32", strides: [ZD, 1] } }],
  [1, Math.ceil(ZD / 64), 1], { M: 1, N: ZD, K: H }
);

// 5. Cast Z — grid [1, 32] (same as z GEMM → can fuse!)
const zB = g.node("cast",
  [{ tensor: g.out(zF32), name: "x" }],
  [{ name: "out", type: { shape: [1, ZD], dtype: "bf16", strides: [ZD, 1] } }],
  [1, Math.ceil(ZD / 64), 1], { N: ZD, from: "f32", to: "bf16" }
);

// 6. A proj GEMM — grid [1, 1]
const aProj = g.node("gemm",
  [{ tensor: g.out(normed), name: "A" }, { tensor: "a_weight", name: "B" }],
  [{ name: "out", type: { shape: [1, LVH], dtype: "f32", strides: [LVH, 1] } }],
  [1, 1, 1], { M: 1, N: LVH, K: H }
);

// 7. B proj GEMM — grid [1, 1] (same as a proj, shares input → can fuse as dual GEMM!)
const bProj = g.node("gemm",
  [{ tensor: g.out(normed), name: "A" }, { tensor: "b_weight", name: "B" }],
  [{ name: "out", type: { shape: [1, LVH], dtype: "f32", strides: [LVH, 1] } }],
  [1, 1, 1], { M: 1, N: LVH, K: H }
);

// 8. Conv1d decode — grid [6] (different grid → can't fuse)
const convOut = g.node("conv1d_decode",
  [
    { tensor: g.out(qkvB), name: "qkv" },
    { tensor: "conv_state", name: "state" },
    { tensor: "conv_weight", name: "w" },
  ],
  [
    { name: "out", type: { shape: [1, QKVD], dtype: "bf16", strides: [QKVD, 1] } },
    { name: "state_new", type: { shape: [3, QKVD], dtype: "bf16", strides: [QKVD, 1], custom: { name: "ConvState" } }, role: "state" },
  ],
  [Math.ceil(QKVD / 1024), 1, 1], { kernelSize: 4, numChannels: QKVD }
);

// 9. GDN delta rule — grid [16] (different grid → can't fuse)
const gdnOut = g.node("gdn_delta_rule",
  [
    { tensor: g.out(convOut), name: "conv_out" },
    { tensor: g.out(zB), name: "z" },
    { tensor: "A_log", name: "A_log" },
    { tensor: "dt_bias", name: "dt_bias" },
    { tensor: g.out(aProj), name: "a_proj" },
    { tensor: g.out(bProj), name: "b_proj" },
    { tensor: "norm_weight", name: "norm_w" },
    { tensor: "s_state", name: "s_state" },
  ],
  [
    { name: "out", type: { shape: [1, ZD], dtype: "bf16", strides: [ZD, 1] } },
    { name: "s_state_new", type: { shape: [LVH, LKD, LVD], dtype: "f32", strides: [LKD * LVD, LVD, 1], custom: { name: "GDNState" } }, role: "state" },
  ],
  [LVH, 1, 1], { numHeads: LVH, keyDim: LKD, valDim: LVD }
);

// 10. Out proj GEMM — grid [1, 16] (ceil(H/64))
const outProjF32 = g.node("gemm",
  [{ tensor: g.out(gdnOut), name: "A" }, { tensor: "out_proj_weight", name: "B" }],
  [{ name: "out", type: { shape: [1, H], dtype: "f32", strides: [H, 1] } }],
  [1, Math.ceil(H / 64), 1], { M: 1, N: H, K: ZD }
);

// 11. Cast — grid [1, 16] (same as out proj → can fuse!)
const attnBf = g.node("cast",
  [{ tensor: g.out(outProjF32), name: "x" }],
  [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
  [1, Math.ceil(H / 64), 1], { N: H, from: "f32", to: "bf16" }
);

// 12. Add residual — grid [1, 16] (same as cast → can fuse!)
const afterAttn = g.node("add",
  [{ tensor: "x", name: "a" }, { tensor: g.out(attnBf), name: "b" }],
  [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
  [1, Math.ceil(H / 64), 1], { N: H }
);

// 13. RMSNorm2 — grid [1]
const normed2 = g.node("rmsnorm",
  [{ tensor: g.out(afterAttn), name: "x" }, { tensor: "post_attn_norm_weight", name: "w" }],
  [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
  [1, 1, 1], { N: H }
);

// 14. Gate proj GEMM — grid [1, 56] (ceil(INTER/64))
const gateF32 = g.node("gemm",
  [{ tensor: g.out(normed2), name: "A" }, { tensor: "gate_weight", name: "B" }],
  [{ name: "out", type: { shape: [1, INTER], dtype: "f32", strides: [INTER, 1] } }],
  [1, Math.ceil(INTER / 64), 1], { M: 1, N: INTER, K: H }
);

// 15. Up proj GEMM — grid [1, 56] (same as gate, shares input → can fuse as dual GEMM!)
const upF32 = g.node("gemm",
  [{ tensor: g.out(normed2), name: "A" }, { tensor: "up_weight", name: "B" }],
  [{ name: "out", type: { shape: [1, INTER], dtype: "f32", strides: [INTER, 1] } }],
  [1, Math.ceil(INTER / 64), 1], { M: 1, N: INTER, K: H }
);

// 16. SwiGLU — grid [4] (ceil(INTER/1024))
const act = g.node("swiglu",
  [{ tensor: g.out(gateF32), name: "g" }, { tensor: g.out(upF32), name: "u" }],
  [{ name: "out", type: { shape: [1, INTER], dtype: "bf16", strides: [INTER, 1] } }],
  [Math.ceil(INTER / 1024), 1, 1], { N: INTER }
);

// 17. Down proj GEMM — grid [1, 16]
const downF32 = g.node("gemm",
  [{ tensor: g.out(act), name: "A" }, { tensor: "down_weight", name: "B" }],
  [{ name: "out", type: { shape: [1, H], dtype: "f32", strides: [H, 1] } }],
  [1, Math.ceil(H / 64), 1], { M: 1, N: H, K: INTER }
);

// 18. Cast — grid [1, 16] (same as down proj → can fuse!)
const mlpBf = g.node("cast",
  [{ tensor: g.out(downF32), name: "x" }],
  [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
  [1, Math.ceil(H / 64), 1], { N: H, from: "f32", to: "bf16" }
);

// 19. Add residual — grid [1, 16] (same as cast → can fuse!)
const xNew = g.node("add",
  [{ tensor: g.out(afterAttn), name: "a" }, { tensor: g.out(mlpBf), name: "b" }],
  [{ name: "out", type: { shape: [1, H], dtype: "bf16", strides: [H, 1] } }],
  [1, Math.ceil(H / 64), 1], { N: H }
);

// ═══════════════════════════════════════════════════════════════════
// Run fusion pass and print results
// ═══════════════════════════════════════════════════════════════════

g.printGraph();
const groups = g.fuse();
g.printFusion(groups);
