import { describe, expect, it } from "vitest";
import { mockLogits } from "./mock";

const audio = (n: number, f: number) => {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(i * f);
  return { modality: "audio" as const, samples: x, sampleRate: 16000 };
};
const px = (v: number) => ({ data: new Uint8ClampedArray(16 * 16 * 4).fill(v), width: 16, height: 16, colorSpace: "srgb" }) as ImageData;

describe("mockLogits", () => {
  it("is deterministic", () => {
    expect(mockLogits(audio(16000, 0.1), 4)).toEqual(mockLogits(audio(16000, 0.1), 4));
  });
  it("differs for different input and has the right size and range", () => {
    const a = mockLogits(audio(16000, 0.1), 4);
    const b = mockLogits(audio(16000, 0.3), 4);
    expect(a).not.toEqual(b);
    expect(a).toHaveLength(4);
    expect(a.every((v) => Math.abs(v) <= 2)).toBe(true);
  });
  it("is deterministic for images and sensitive to pixels", () => {
    const i1 = { modality: "skin" as const, image: px(100) };
    expect(mockLogits(i1, 3, px(100))).toEqual(mockLogits(i1, 3, px(100)));
    expect(mockLogits(i1, 3, px(100))).not.toEqual(mockLogits(i1, 3, px(200)));
  });
});
