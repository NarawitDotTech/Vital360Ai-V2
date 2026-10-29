// Pure capture-quality checks (no DOM). Thresholds are conservative heuristics, to be tuned on real phones.

export interface PixelData {
  data: Uint8ClampedArray | Uint8Array | number[];
  width: number;
  height: number;
}

export type ImageIssue = "tooDark" | "tooBright" | "blurry";

export const IMAGE_LIMITS = { dark: 40, bright: 228, blurVariance: 20 } as const;

function luma(d: PixelData["data"], i: number): number {
  return 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
}

/** Mean brightness 0-255 and sharpness (variance of the 4-neighbour Laplacian of luma). */
export function measureImage(img: PixelData): { brightness: number; sharpness: number } {
  const { width: w, height: h, data } = img;
  if (w < 3 || h < 3) return { brightness: 0, sharpness: 0 };
  const g = new Float32Array(w * h);
  let sum = 0;
  for (let i = 0, p = 0; p < g.length; i += 4, p++) {
    g[p] = luma(data, i);
    sum += g[p];
  }
  let lapSum = 0;
  let lapSq = 0;
  let n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const c = y * w + x;
      const l = g[c - 1] + g[c + 1] + g[c - w] + g[c + w] - 4 * g[c];
      lapSum += l;
      lapSq += l * l;
      n++;
    }
  }
  const mean = lapSum / n;
  return { brightness: sum / g.length, sharpness: lapSq / n - mean * mean };
}

/** Returns the first problem found, or null when the picture is good enough to analyse. */
export function checkImage(img: PixelData): ImageIssue | null {
  const { brightness, sharpness } = measureImage(img);
  if (brightness < IMAGE_LIMITS.dark) return "tooDark";
  if (brightness > IMAGE_LIMITS.bright) return "tooBright";
  if (sharpness < IMAGE_LIMITS.blurVariance) return "blurry";
  return null;
}

export type AudioIssue = "tooNoisy" | "tooQuiet";

export const AUDIO_LIMITS = {
  /** ambient RMS above this (dBFS) means a noisy room */
  noisyDb: -35,
  /** whole-recording peak below this (linear) means the mic captured nothing */
  silentPeak: 0.005,
} as const;

export function rmsDb(samples: Float32Array): number {
  if (samples.length === 0) return -Infinity;
  let s = 0;
  for (let i = 0; i < samples.length; i++) s += samples[i] * samples[i];
  const rms = Math.sqrt(s / samples.length);
  return rms > 0 ? 20 * Math.log10(rms) : -Infinity;
}

export function peak(samples: Float32Array): number {
  let m = 0;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > m) m = a;
  }
  return m;
}

/** Pre-record check on ~1.5 s of room noise. */
export function checkAmbient(samples: Float32Array): AudioIssue | null {
  if (peak(samples) < 1e-5) return "tooQuiet"; // muted / blocked microphone
  if (rmsDb(samples) > AUDIO_LIMITS.noisyDb) return "tooNoisy";
  return null;
}

/** Post-record check: did we capture anything at all? */
export function checkRecording(samples: Float32Array): AudioIssue | null {
  return peak(samples) < AUDIO_LIMITS.silentPeak ? "tooQuiet" : null;
}

/** Chest-video signal: needs enough frames and some variation. */
export function checkMotionSignal(signal: Float32Array, fps: number, minSeconds = 8): "noMotion" | null {
  if (!(fps > 0) || signal.length < fps * minSeconds) return "noMotion";
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < signal.length; i++) {
    if (!Number.isFinite(signal[i])) return "noMotion";
    if (signal[i] < min) min = signal[i];
    if (signal[i] > max) max = signal[i];
  }
  return max - min > 1e-9 ? null : "noMotion";
}
