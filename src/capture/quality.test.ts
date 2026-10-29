import { describe, expect, it } from "vitest";
import { checkAmbient, checkImage, checkMotionSignal, checkRecording, measureImage, rmsDb } from "./quality";

function img(w: number, h: number, fn: (x: number, y: number) => number) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = fn(x, y);
      const i = (y * w + x) * 4;
      data[i] = data[i + 1] = data[i + 2] = v;
      data[i + 3] = 255;
    }
  return { data, width: w, height: h };
}
// deterministic pseudo-texture
const tex = (x: number, y: number) => 110 + ((x * 37 + y * 91 + x * y * 13) % 60);

describe("image quality", () => {
  it("flags dark, bright and blurry pictures", () => {
    expect(checkImage(img(64, 64, () => 10))).toBe("tooDark");
    expect(checkImage(img(64, 64, () => 250))).toBe("tooBright");
    expect(checkImage(img(64, 64, (x) => 100 + x))).toBe("blurry"); // smooth gradient
  });
  it("accepts a well lit, textured picture", () => {
    expect(checkImage(img(64, 64, tex))).toBeNull();
    expect(measureImage(img(64, 64, tex)).brightness).toBeGreaterThan(100);
  });
  it("handles tiny images", () => {
    expect(measureImage(img(2, 2, () => 100))).toEqual({ brightness: 0, sharpness: 0 });
  });
});

describe("audio quality", () => {
  const tone = (amp: number, n = 16000) => Float32Array.from({ length: n }, (_, i) => amp * Math.sin(i / 5));
  it("detects muted mic, noisy room and ok room", () => {
    expect(checkAmbient(new Float32Array(1000))).toBe("tooQuiet");
    expect(checkAmbient(tone(0.5))).toBe("tooNoisy");
    expect(checkAmbient(tone(0.003))).toBeNull();
    expect(rmsDb(tone(1))).toBeCloseTo(-3, 0);
  });
  it("detects an empty recording", () => {
    expect(checkRecording(tone(0.001))).toBe("tooQuiet");
    expect(checkRecording(tone(0.1))).toBeNull();
  });
});

describe("motion signal", () => {
  it("needs enough frames and some variation", () => {
    expect(checkMotionSignal(new Float32Array(10), 15)).toBe("noMotion");
    expect(checkMotionSignal(new Float32Array(300), 15)).toBe("noMotion");
    expect(checkMotionSignal(Float32Array.from({ length: 300 }, (_, i) => Math.sin(i / 10)), 15)).toBeNull();
    expect(checkMotionSignal(Float32Array.from({ length: 300 }, () => NaN), 15)).toBe("noMotion");
  });
});
