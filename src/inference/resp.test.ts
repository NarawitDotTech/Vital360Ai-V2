import { describe, expect, it } from "vitest";
import { estimateBreathingRate, frameMotion, respLabel, respTier, RESP_MIN_QUALITY } from "./resp";

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

function sine(bpm: number, fps: number, secs: number, noise: number, seed = 1) {
  const r = rng(seed);
  const x = new Float32Array(Math.round(fps * secs));
  for (let i = 0; i < x.length; i++) x[i] = Math.sin((2 * Math.PI * (bpm / 60) * i) / fps) + 0.02 * i + noise * gauss(r);
  return x;
}

describe("estimateBreathingRate", () => {
  it.each([8, 15, 28])("recovers %i bpm with noise", (bpm) => {
    const e = estimateBreathingRate(sine(bpm, 30, 20, 0.5), 30);
    expect(Math.abs(e.breathsPerMin - bpm)).toBeLessThan(1);
    expect(e.quality).toBeGreaterThan(RESP_MIN_QUALITY);
  });
  it("pure noise has low quality (many seeds)", () => {
    for (let s = 1; s <= 20; s++) {
      const r = rng(s);
      const x = new Float32Array(600).map(() => gauss(r));
      expect(estimateBreathingRate(x, 30).quality).toBeLessThan(RESP_MIN_QUALITY);
    }
  });
  it("too-short or invalid input => quality 0", () => {
    expect(estimateBreathingRate(new Float32Array(30), 30).quality).toBe(0);
    expect(estimateBreathingRate(new Float32Array(600), 0).quality).toBe(0);
    expect(estimateBreathingRate(new Float32Array(600), 30).quality).toBe(0);
  });
  it("labels and tiers", () => {
    expect(respLabel(15)).toBe("normal");
    expect(respLabel(11)).toBe("slow");
    expect(respLabel(22)).toBe("fast");
    expect([12, 20].map(respTier)).toEqual(["green", "green"]);
    expect([10, 11, 21, 24].map(respTier)).toEqual(["amber", "amber", "amber", "amber"]);
    expect([9, 25].map(respTier)).toEqual(["red", "red"]);
  });
});

function img(w: number, h: number, shift: number): ImageData {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = 128 + 100 * Math.sin((y - shift) / 6);
      const i = (y * w + x) * 4;
      data[i] = data[i + 1] = data[i + 2] = v;
      data[i + 3] = 255;
    }
  return { data, width: w, height: h, colorSpace: "srgb" } as ImageData;
}

describe("frameMotion", () => {
  const box = { x: 0, y: 0, w: 64, h: 64 };
  it("is positive for downward motion, negative for upward, ~0 for static", () => {
    expect(frameMotion(img(64, 64, 0), img(64, 64, 1), box)).toBeGreaterThan(0.5);
    expect(frameMotion(img(64, 64, 1), img(64, 64, 0), box)).toBeLessThan(-0.5);
    expect(frameMotion(img(64, 64, 0), img(64, 64, 0), box)).toBe(0);
  });
  it("handles out-of-range boxes and size mismatch", () => {
    expect(frameMotion(img(64, 64, 0), img(32, 32, 0), box)).toBe(0);
    expect(frameMotion(img(64, 64, 0), img(64, 64, 1), { x: 500, y: 500, w: 10, h: 10 })).toBe(0);
  });
});
