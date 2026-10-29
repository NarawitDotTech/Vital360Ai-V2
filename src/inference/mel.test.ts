import { describe, expect, it } from "vitest";
import { logMel, melCenterFreqs, melFilterbank, resample } from "./mel";

function tone(freq: number, sr: number, secs: number) {
  const x = new Float32Array(Math.round(sr * secs));
  for (let i = 0; i < x.length; i++) x[i] = 0.5 * Math.sin((2 * Math.PI * freq * i) / sr);
  return x;
}
function bandMeans(m: { data: Float32Array; nFrames: number; nMels: number }) {
  const means = new Array<number>(m.nMels).fill(0);
  for (let f = 2; f < m.nFrames - 2; f++) for (let b = 0; b < m.nMels; b++) means[b] += m.data[f * m.nMels + b];
  return means.map((v) => v / (m.nFrames - 4));
}
const argmax = (a: number[]) => a.indexOf(Math.max(...a));

describe("logMel", () => {
  it("has expected shape", () => {
    const m = logMel(new Float32Array(16000), 16000);
    expect(m.nMels).toBe(64);
    expect(m.nFrames).toBe(1 + Math.floor(16000 / 320));
    expect(m.data.length).toBe(m.nFrames * 64);
  });
  it("puts a 1 kHz tone in the band whose centre is nearest 1 kHz", () => {
    const m = logMel(tone(1000, 16000, 1), 16000);
    const b = argmax(bandMeans(m));
    const centers = melCenterFreqs(64, 16000);
    const expected = argmax(centers.map((c) => -Math.abs(c - 1000)));
    expect(Math.abs(b - expected)).toBeLessThanOrEqual(1);
    expect(Math.abs(centers[b] - 1000)).toBeLessThan(150);
  });
  it("tone at 3 kHz lands higher than 500 Hz", () => {
    const lo = argmax(bandMeans(logMel(tone(500, 16000, 1), 16000)));
    const hi = argmax(bandMeans(logMel(tone(3000, 16000, 1), 16000)));
    expect(hi).toBeGreaterThan(lo + 10);
  });
  it("resamples 44.1 kHz input to 16 kHz and keeps the tone band", () => {
    const m = logMel(tone(1000, 44100, 1), 44100);
    expect(m.nFrames).toBeGreaterThan(48);
    const centers = melCenterFreqs(64, 16000);
    expect(Math.abs(centers[argmax(bandMeans(m))] - 1000)).toBeLessThan(150);
  });
  it("silence and empty input give finite values", () => {
    for (const x of [new Float32Array(8000), new Float32Array(0), new Float32Array(10)]) {
      const m = logMel(x, 16000);
      expect(m.data.every(Number.isFinite)).toBe(true);
    }
  });
  it("filterbank rows are non-negative and non-empty", () => {
    const fb = melFilterbank(16000, 1024, 64);
    expect(fb).toHaveLength(64);
    for (const r of fb) expect(r.some((v) => v > 0)).toBe(true);
  });
  it("resample changes length proportionally", () => {
    expect(resample(new Float32Array(48000), 48000, 16000).length).toBe(16000);
    expect(resample(new Float32Array(8000), 8000, 16000).length).toBe(16000);
  });
});
