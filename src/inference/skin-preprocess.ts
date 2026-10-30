// Skin image preprocessing. MUST match training/cpu/skin_common.py (squash_u8 + to_tensor) exactly.
// area-weighted squash resize (no crop; box filter with fractional pixel coverage, so a 640 px photo is averaged,
// not point-sampled) -> round to uint8 -> /255 -> ImageNet mean/std -> NCHW float32, RGB order.
// The caller (camera.ts) has already downscaled the photo so that its longest side is <= 640 px.
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/** For each output cell i of s: first source index and the overlap weights (sum 1) over source pixels of length n. */
function areaWeights(n: number, s: number): { start: number[]; w: Float64Array[] } {
  const scale = n / s;
  const start: number[] = [];
  const w: Float64Array[] = [];
  for (let i = 0; i < s; i++) {
    const a = i * scale;
    const b = (i + 1) * scale;
    const j0 = Math.floor(a);
    const j1 = Math.min(n, Math.ceil(b));
    const row = new Float64Array(Math.max(0, j1 - j0));
    for (let j = j0; j < j1; j++) row[j - j0] = (Math.min(b, j + 1) - Math.max(a, j)) / scale;
    start.push(j0);
    w.push(row);
  }
  return { start, w };
}

export function skinTensor(img: { data: ArrayLike<number>; width: number; height: number }, S = 224): Float32Array {
  const { width: W, height: H, data } = img;
  const wy = areaWeights(H, S);
  const wx = areaWeights(W, S);
  // vertical pass: tmp[i][x][c]
  const tmp = new Float64Array(S * W * 3);
  for (let i = 0; i < S; i++) {
    const row = wy.w[i];
    for (let k = 0; k < row.length; k++) {
      const sy = wy.start[i] + k;
      const wt = row[k];
      for (let x = 0; x < W; x++) {
        const p = (sy * W + x) * 4;
        const o = (i * W + x) * 3;
        tmp[o] += wt * data[p];
        tmp[o + 1] += wt * data[p + 1];
        tmp[o + 2] += wt * data[p + 2];
      }
    }
  }
  const out = new Float32Array(3 * S * S);
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < S; j++) {
      const row = wx.w[j];
      const acc = [0, 0, 0];
      for (let k = 0; k < row.length; k++) {
        const o = (i * W + wx.start[j] + k) * 3;
        acc[0] += row[k] * tmp[o];
        acc[1] += row[k] * tmp[o + 1];
        acc[2] += row[k] * tmp[o + 2];
      }
      for (let c = 0; c < 3; c++) {
        const u8 = Math.min(255, Math.max(0, Math.round(acc[c])));
        out[c * S * S + i * S + j] = (u8 / 255 - MEAN[c]) / STD[c];
      }
    }
  }
  return out;
}
