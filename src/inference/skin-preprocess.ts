// Skin image preprocessing. MUST match training/cpu/skin_common.py (squash_u8 + to_tensor) exactly.
// area-weighted resize to 224x224 (box filter with fractional pixel coverage, so a 448 px square is averaged, not point-sampled)
// -> round to uint8 -> /255 -> ImageNet mean/std -> NCHW float32, RGB order.
// The caller (camera.ts) has already cropped the SQUARE under the guide box (camera) or the centred square (upload) and shrunk it
// to <= 448 px, so the image given here is square; a non-square image would be squashed (skin-v1 behaviour).
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

/** Pure-pixel copy of the square `r` (RGBA in, RGBA out). Used by tests; the app crops with canvas (camera.ts cropSquare). */
export function cropRgba(img: { data: ArrayLike<number>; width: number; height: number }, r: { x: number; y: number; s: number }): { data: Uint8ClampedArray; width: number; height: number } {
  const out = new Uint8ClampedArray(r.s * r.s * 4);
  for (let y = 0; y < r.s; y++) for (let x = 0; x < r.s * 4; x++) out[y * r.s * 4 + x] = img.data[((r.y + y) * img.width + r.x) * 4 + x];
  return { data: out, width: r.s, height: r.s };
}

/** Area-resize to S x S and round: uint8 HWC RGB. This is exactly the image the network sees (before /255 and mean/std). */
export function skinU8(img: { data: ArrayLike<number>; width: number; height: number }, S = 224): Uint8Array {
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
  const out = new Uint8Array(S * S * 3);
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
      for (let c = 0; c < 3; c++) out[(i * S + j) * 3 + c] = Math.min(255, Math.max(0, Math.round(acc[c])));
    }
  }
  return out;
}

/** uint8 HWC -> ImageNet-normalised NCHW float32. */
export function u8ToTensor(u8: Uint8Array, S = 224): Float32Array {
  const out = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) out[c * S * S + i] = (u8[i * 3 + c] / 255 - MEAN[c]) / STD[c];
  return out;
}

export function skinTensor(img: { data: ArrayLike<number>; width: number; height: number }, S = 224): Float32Array {
  return u8ToTensor(skinU8(img, S), S);
}

export interface DarkSpotParams {
  minArea: number;
  relLuma: number;
  maxLuma: number;
}

/**
 * Area (fraction of the box) of the largest very dark blob that does NOT touch the border: an enclosed black spot, not a shadow, vignette or
 * dark background. Luma on a 56x56 grid (4x4 means of the 224 px model input), dark = luma < relLuma * median and < maxLuma, 4-connected.
 * MUST match training/cpu/skin_decide.py dark_spot_area.
 */
export function darkSpotArea(u8: Uint8Array, p: Pick<DarkSpotParams, "relLuma" | "maxLuma"> = { relLuma: 0.45, maxLuma: 70 }): number {
  const G = 56;
  const Y = new Float64Array(G * G);
  for (let gy = 0; gy < G; gy++)
    for (let gx = 0; gx < G; gx++) {
      let sum = 0;
      for (let dy = 0; dy < 4; dy++)
        for (let dx = 0; dx < 4; dx++) {
          const o = ((gy * 4 + dy) * 224 + gx * 4 + dx) * 3;
          sum += 0.299 * u8[o] + 0.587 * u8[o + 1] + 0.114 * u8[o + 2];
        }
      Y[gy * G + gx] = sum / 16;
    }
  const sorted = Float64Array.from(Y).sort();
  const med = (sorted[(G * G) / 2 - 1] + sorted[(G * G) / 2]) / 2;
  const dark = Y.map((v) => (v < p.relLuma * med && v < p.maxLuma ? 1 : 0));
  const seen = new Uint8Array(G * G);
  let best = 0;
  for (let s = 0; s < G * G; s++) {
    if (!dark[s] || seen[s]) continue;
    const stack = [s];
    seen[s] = 1;
    let size = 0;
    let border = false;
    while (stack.length) {
      const c = stack.pop()!;
      size++;
      const y = Math.floor(c / G);
      const x = c % G;
      if (y === 0 || x === 0 || y === G - 1 || x === G - 1) border = true;
      for (const [ny, nx] of [[y - 1, x], [y + 1, x], [y, x - 1], [y, x + 1]]) {
        if (ny < 0 || nx < 0 || ny >= G || nx >= G) continue;
        const n = ny * G + nx;
        if (dark[n] && !seen[n]) {
          seen[n] = 1;
          stack.push(n);
        }
      }
    }
    if (!border && size > best) best = size;
  }
  return best / (G * G);
}
