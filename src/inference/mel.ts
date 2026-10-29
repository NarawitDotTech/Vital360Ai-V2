// Log-mel spectrogram in pure TS. Mirrors librosa defaults (center=True reflect padding, periodic Hann,
// Slaney mel scale + Slaney area normalisation, power=2). Exact numeric parity with the Python training
// preprocessing still needs a fixture exported from the training notebook (future work).
import { fft } from "./fft";

export interface MelOptions {
  sampleRate?: number;
  nFft?: number;
  hop?: number;
  nMels?: number;
  fMin?: number;
  fMax?: number;
  eps?: number;
}

export const MEL_DEFAULTS = { sampleRate: 16000, nFft: 1024, hop: 320, nMels: 64, eps: 1e-10 } as const;

export interface LogMel {
  /** Row-major [frames][nMels]. */
  data: Float32Array;
  nFrames: number;
  nMels: number;
}

/** Simple resampler: linear interpolation up, box average (crude anti-alias) down. */
export function resample(x: Float32Array, from: number, to: number): Float32Array {
  if (from === to || x.length === 0) return x;
  const n = Math.max(1, Math.floor((x.length * to) / from));
  const out = new Float32Array(n);
  const r = from / to;
  if (r > 1) {
    const cs = new Float64Array(x.length + 1);
    for (let i = 0; i < x.length; i++) cs[i + 1] = cs[i] + x[i];
    for (let i = 0; i < n; i++) {
      const a = Math.min(x.length - 1, Math.floor(i * r));
      const b = Math.min(x.length, Math.max(a + 1, Math.floor((i + 1) * r)));
      out[i] = (cs[b] - cs[a]) / (b - a);
    }
  } else {
    for (let i = 0; i < n; i++) {
      const t = i * r;
      const a = Math.floor(t);
      const b = Math.min(x.length - 1, a + 1);
      out[i] = x[a] + (x[b] - x[a]) * (t - a);
    }
  }
  return out;
}

const F_SP = 200 / 3;
const MIN_LOG_HZ = 1000;
const MIN_LOG_MEL = MIN_LOG_HZ / F_SP;
const LOG_STEP = Math.log(6.4) / 27;

export function hzToMel(hz: number): number {
  return hz >= MIN_LOG_HZ ? MIN_LOG_MEL + Math.log(hz / MIN_LOG_HZ) / LOG_STEP : hz / F_SP;
}
export function melToHz(mel: number): number {
  return mel >= MIN_LOG_MEL ? MIN_LOG_HZ * Math.exp(LOG_STEP * (mel - MIN_LOG_MEL)) : mel * F_SP;
}

const fbCache = new Map<string, Float32Array[]>();

/** Slaney-normalised triangular filterbank: nMels rows of (nFft/2+1) weights. */
export function melFilterbank(sampleRate: number, nFft: number, nMels: number, fMin = 0, fMax = sampleRate / 2): Float32Array[] {
  const key = `${sampleRate}/${nFft}/${nMels}/${fMin}/${fMax}`;
  const hit = fbCache.get(key);
  if (hit) return hit;
  const nBins = nFft / 2 + 1;
  const lo = hzToMel(fMin);
  const hi = hzToMel(fMax);
  const pts: number[] = [];
  for (let i = 0; i < nMels + 2; i++) pts.push(melToHz(lo + ((hi - lo) * i) / (nMels + 1)));
  const rows: Float32Array[] = [];
  for (let m = 0; m < nMels; m++) {
    const row = new Float32Array(nBins);
    const norm = 2 / (pts[m + 2] - pts[m]);
    for (let k = 0; k < nBins; k++) {
      const f = (k * sampleRate) / nFft;
      const up = (f - pts[m]) / (pts[m + 1] - pts[m]);
      const down = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
      row[k] = Math.max(0, Math.min(up, down)) * norm;
    }
    rows.push(row);
  }
  fbCache.set(key, rows);
  return rows;
}

/** Centre frequency (Hz) of each mel band. */
export function melCenterFreqs(nMels: number, sampleRate: number, fMin = 0, fMax = sampleRate / 2): number[] {
  const lo = hzToMel(fMin);
  const hi = hzToMel(fMax);
  return Array.from({ length: nMels }, (_, m) => melToHz(lo + ((hi - lo) * (m + 1)) / (nMels + 1)));
}

function reflect(i: number, n: number): number {
  if (n === 1) return 0;
  const period = 2 * (n - 1);
  let j = ((i % period) + period) % period;
  if (j >= n) j = period - j;
  return j;
}

/** Log-mel spectrogram: log(eps + |STFT|^2 mapped to mel). `samples` are resampled to 16 kHz first. */
export function logMel(samples: Float32Array, inputRate: number, opts: MelOptions = {}): LogMel {
  const sr = opts.sampleRate ?? MEL_DEFAULTS.sampleRate;
  const nFft = opts.nFft ?? MEL_DEFAULTS.nFft;
  const hop = opts.hop ?? MEL_DEFAULTS.hop;
  const nMels = opts.nMels ?? MEL_DEFAULTS.nMels;
  const eps = opts.eps ?? MEL_DEFAULTS.eps;
  const x = resample(samples, inputRate, sr);
  const nFrames = 1 + Math.floor(x.length / hop);
  const fb = melFilterbank(sr, nFft, nMels, opts.fMin ?? 0, opts.fMax ?? sr / 2);
  const win = new Float64Array(nFft);
  for (let i = 0; i < nFft; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft);
  const nBins = nFft / 2 + 1;
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  const power = new Float64Array(nBins);
  const out = new Float32Array(nFrames * nMels);
  const pad = nFft / 2;
  for (let f = 0; f < nFrames; f++) {
    const start = f * hop - pad;
    for (let i = 0; i < nFft; i++) {
      const v = x.length ? x[reflect(start + i, x.length)] : 0;
      re[i] = Number.isFinite(v) ? v * win[i] : 0;
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k < nBins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < nMels; m++) {
      const row = fb[m];
      let s = 0;
      for (let k = 0; k < nBins; k++) s += row[k] * power[k];
      out[f * nMels + m] = Math.log(eps + s);
    }
  }
  return { data: out, nFrames, nMels };
}
