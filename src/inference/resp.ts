// Respiration rate from a per-frame motion/intensity signal. Pure TS, no DOM.
import { fft, nextPow2 } from "./fft";

export const RESP_BAND_HZ = [0.1, 0.7] as const; // 6-42 breaths/min
/** Below this quality the prediction is "uncertain". */
export const RESP_MIN_QUALITY = 0.3;
const MIN_SECONDS = 8;

export interface RespEstimate {
  breathsPerMin: number;
  /** 0..1, from the in-band SNR of the spectral peak. */
  quality: number;
}

function detrend(x: Float32Array): Float64Array {
  const n = x.length;
  let sx = 0;
  let sy = 0;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sx += i;
    sy += x[i];
    sxy += i * x[i];
    sxx += i * i;
  }
  const den = n * sxx - sx * sx;
  const slope = den === 0 ? 0 : (n * sxy - sx * sy) / den;
  const icpt = (sy - slope * sx) / n;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = x[i] - (icpt + slope * i);
  return out;
}

/**
 * detrend -> Hann window -> zero-padded FFT -> band mask 0.1-0.7 Hz -> peak + parabolic interpolation.
 * Quality = SNR of (power within +-1.5 resolution bins of the peak) vs (rest of the band), mapped to 0..1.
 */
export function estimateBreathingRate(signal: Float32Array, fps: number): RespEstimate {
  const n = signal.length;
  const bad = { breathsPerMin: 0, quality: 0 };
  if (!(fps > 0) || n < fps * MIN_SECONDS) return bad;
  const x = detrend(signal);
  const nFft = nextPow2(Math.max(n * 8, 4096));
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  for (let i = 0; i < n; i++) {
    const v = Number.isFinite(x[i]) ? x[i] : 0;
    re[i] = v * (0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / n));
  }
  fft(re, im);
  const binHz = fps / nFft;
  const lo = Math.max(1, Math.ceil(RESP_BAND_HZ[0] / binHz));
  const hi = Math.min(nFft / 2 - 1, Math.floor(RESP_BAND_HZ[1] / binHz));
  const mag = (k: number) => Math.hypot(re[k], im[k]);
  let peak = lo;
  let total = 0;
  for (let k = lo; k <= hi; k++) {
    total += mag(k) ** 2;
    if (mag(k) > mag(peak)) peak = k;
  }
  if (total <= 1e-18) return bad;
  // Parabolic interpolation on log magnitude.
  let delta = 0;
  if (peak > lo && peak < hi) {
    const a = Math.log(mag(peak - 1) + 1e-30);
    const b = Math.log(mag(peak) + 1e-30);
    const c = Math.log(mag(peak + 1) + 1e-30);
    const d = a - 2 * b + c;
    if (d < 0) delta = Math.max(-0.5, Math.min(0.5, (0.5 * (a - c)) / d));
  }
  const hz = (peak + delta) * binHz;
  const half = Math.ceil(1.5 / (n / fps) / binHz);
  let sig = 0;
  for (let k = Math.max(lo, peak - half); k <= Math.min(hi, peak + half); k++) sig += mag(k) ** 2;
  const noise = Math.max(total - sig, 1e-18 * total);
  const snrDb = 10 * Math.log10(sig / noise);
  const quality = Math.max(0, Math.min(1, (snrDb - 2) / 8));
  return { breathsPerMin: hz * 60, quality };
}

export type RespLabel = "slow" | "normal" | "fast";

export function respLabel(bpm: number): RespLabel {
  return bpm < 12 ? "slow" : bpm > 20 ? "fast" : "normal";
}

/** Adult resting tiers: 12-20 green; 10-11 or 21-24 amber; <10 or >24 red. */
export function respTier(bpm: number): "green" | "amber" | "red" {
  if (bpm < 10 || bpm > 24) return "red";
  if (bpm < 12 || bpm > 20) return "amber";
  return "green";
}

/**
 * Vertical motion proxy inside `box` between two frames: 1-D optical flow along y,
 * v = -sum(Iy * It) / sum(Iy^2), in pixels (positive = content moved down). Subsampled, pure.
 * Integrating or not, the breathing frequency is the same, so the UI may push either to resp.
 */
export function frameMotion(
  prev: ImageData,
  cur: ImageData,
  box: { x: number; y: number; w: number; h: number },
): number {
  if (prev.width !== cur.width || prev.height !== cur.height) return 0;
  const W = cur.width;
  const H = cur.height;
  const x0 = Math.max(0, Math.floor(box.x));
  const y0 = Math.max(1, Math.floor(box.y));
  const x1 = Math.min(W, Math.ceil(box.x + box.w));
  const y1 = Math.min(H - 1, Math.ceil(box.y + box.h));
  if (x1 <= x0 || y1 <= y0) return 0;
  const step = Math.max(1, Math.floor(Math.min(x1 - x0, y1 - y0) / 32));
  const lum = (d: Uint8ClampedArray, x: number, y: number) => {
    const i = (y * W + x) * 4;
    return 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  };
  let num = 0;
  let den = 0;
  for (let y = y0; y < y1; y += step) {
    for (let x = x0; x < x1; x += step) {
      const gy = (lum(cur.data, x, y + 1) - lum(cur.data, x, y - 1) + lum(prev.data, x, y + 1) - lum(prev.data, x, y - 1)) / 4;
      const dt = lum(cur.data, x, y) - lum(prev.data, x, y);
      num += gy * dt;
      den += gy * gy;
    }
  }
  return den < 1e-6 ? 0 : -num / den || 0;
}
