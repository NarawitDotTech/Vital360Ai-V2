// Deterministic mock backend: hash of simple input statistics -> logits. Offline, no randomness.
import type { ModalityInput } from "./types";

function fnv1a(parts: number[]): number {
  let h = 0x811c9dc5;
  for (const p of parts) {
    let v = Math.round(p) | 0;
    for (let i = 0; i < 4; i++) {
      h ^= v & 0xff;
      h = Math.imul(h, 0x01000193);
      v >>>= 8;
    }
  }
  return h >>> 0;
}

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Simple statistics, quantised so tiny float noise does not flip the hash. */
export function inputStats(input: ModalityInput, pixels?: ImageData | null): number[] {
  switch (input.modality) {
    case "skin": {
      const w = "width" in input.image ? input.image.width : 0;
      const h = "height" in input.image ? input.image.height : 0;
      if (!pixels) return [1, w, h];
      const d = pixels.data;
      const stride = Math.max(1, Math.floor(d.length / 4 / 2048)) * 4;
      let n = 0;
      const sum = [0, 0, 0];
      let sq = 0;
      for (let i = 0; i + 2 < d.length; i += stride) {
        sum[0] += d[i];
        sum[1] += d[i + 1];
        sum[2] += d[i + 2];
        const l = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        sq += l * l;
        n++;
      }
      n = Math.max(n, 1);
      const mean = (sum[0] * 0.299 + sum[1] * 0.587 + sum[2] * 0.114) / n;
      const sd = Math.sqrt(Math.max(0, sq / n - mean * mean));
      return [1, w, h, sum[0] / n, sum[1] / n, sum[2] / n, sd].map((v, i) => (i < 3 ? v : v * 2));
    }
    case "audio": {
      const x = input.samples;
      const step = Math.max(1, Math.floor(x.length / 4096));
      let s = 0;
      let sq = 0;
      let zc = 0;
      let n = 0;
      for (let i = 0; i < x.length; i += step) {
        s += x[i];
        sq += x[i] * x[i];
        if (i >= step && x[i] * x[i - step] < 0) zc++;
        n++;
      }
      n = Math.max(n, 1);
      return [2, x.length, input.sampleRate, (s / n) * 1e4, Math.sqrt(sq / n) * 1e4, (zc / n) * 1e3];
    }
    case "resp": {
      const x = input.signal;
      let s = 0;
      for (let i = 0; i < x.length; i++) s += x[i];
      return [3, x.length, input.fps * 10, (s / Math.max(x.length, 1)) * 1e3];
    }
    case "fusion":
      return [4, input.parts.length];
  }
}

/** Logits in about [-2, 2]; same input => same output. */
export function mockLogits(input: ModalityInput, nLabels: number, pixels?: ImageData | null): number[] {
  const rand = mulberry32(fnv1a(inputStats(input, pixels)));
  return Array.from({ length: nLabels }, () => (rand() - 0.5) * 4);
}
