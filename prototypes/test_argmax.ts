import { TTIRBuilder, compileAndLoad, cuAlloc, cuHtoD, cuDtoH, cuSync, cuLaunch } from "../src/ttir";
const VOCAB = 248320;
function buildArgmax() {
  const b = new TTIRBuilder();
  const L = b.param("L", {ptr: "f32"}), OutVal = b.param("V", {ptr: "f32"}), OutIdx = b.param("I", {ptr: "i32"});
  const CHUNK = 4096;
  const tpL = b.makeTensorPtr(L, [1, VOCAB], [VOCAB, 1], [b.i32(0), b.i32(0)], [1, CHUNK], "f32", [1, 0]);
  const initMax = b.broadcastTo(b.f32(-1000), [1]);
  const initIdx = b.broadcastTo(b.i32(0), [1]);
  const initOff = b.broadcastTo(b.i32(0), [1]);
  const [finalMax, finalIdx] = b.forIter(b.index(0), b.index(VOCAB), b.index(CHUNK), [initMax, initIdx, initOff, tpL], (bb, _, [curMax, curIdx, curOff, tp]) => {
    const chunk = bb.load(tp, {boundaryCheck: [0, 1], padding: 1});
    const localMax = bb.max(chunk, 1); // [1] f32
    // Find index of local max
    const mask = bb.eq(chunk, bb.broadcastTo(localMax, [1, CHUNK])); // [1, CHUNK] i1
    const arange = bb.arange(0, CHUNK); // [CHUNK] i32
    const arangeBc = bb.broadcast(bb.expandDims(arange, 0), [1, CHUNK]); // [1, CHUNK] i32
    const masked = bb.select(mask, arangeBc, bb.broadcastTo(bb.i32(0), [1, CHUNK])); // [1, CHUNK] i32
    const localIdx = bb.sum(masked, 1); // [1] i32
    const globalIdx = bb.add(localIdx, curOff); // [1] i32
    // Update if better
    const isBetter = bb.gt(localMax, curMax); // [1] i1
    const newMax = bb.select(isBetter, localMax, curMax);
    const newIdx = bb.select(isBetter, globalIdx, curIdx);
    return [newMax, newIdx, bb.add(curOff, bb.broadcastTo(bb.i32(CHUNK), [1])), bb.advance(tp, [bb.i32(0), bb.i32(CHUNK)])];
  });
  b.store(b.makeTensorPtr(OutVal, [1, 1], [1, 1], [b.i32(0), b.i32(0)], [1, 1], "f32", [1, 0]), b.broadcastTo(finalMax, [1, 1]), {boundaryCheck: [0, 1]});
  b.store(b.makeTensorPtr(OutIdx, [1, 1], [1, 1], [b.i32(0), b.i32(0)], [1, 1], "i32", [1, 0]), b.broadcastTo(finalIdx, [1, 1]), {boundaryCheck: [0, 1]});
  return b.build("argmax", 4, 3);
}
const k = compileAndLoad(buildArgmax(), "argmax", 4);
console.log("Compiled!");
const data = new Float32Array(VOCAB);
for (let i = 0; i < VOCAB; i++) data[i] = Math.sin(i) * 10;
data[123456] = 999.0;
const d = cuAlloc(BigInt(VOCAB * 4));
cuHtoD(d, data.buffer); cuSync();
const valD = cuAlloc(BigInt(4)), idxD = cuAlloc(BigInt(4));
cuLaunch(k, [1, 1, 1], [128, 1, 1], [d, valD, idxD]);
cuSync();
const hVal = new Float32Array(1), hIdx = new Int32Array(1);
cuDtoH(hVal.buffer, valD, BigInt(4));
cuDtoH(hIdx.buffer, idxD, BigInt(4));
console.log(`Max: ${hVal[0]} at index ${hIdx[0]} (expected 999.0 at 123456)`);
