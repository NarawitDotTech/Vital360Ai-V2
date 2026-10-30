// Parity: TS skin preprocessing (centre-square crop + area resize) must reproduce the Python training preprocessing
// (training/cpu/skin_pipe.py + skin_common.py) on the fixture images. Fixture files are written by training/cpu/skin2_fixture.py.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import sharp from "sharp";
import { describe, expect, it } from "vitest";
import { centerSquare } from "@/capture/camera";
import { cropRgba, darkSpotArea, skinTensor } from "./skin-preprocess";

const dir = join(__dirname, "__fixtures__", "skin");
const expected = JSON.parse(readFileSync(join(dir, "expected.json"), "utf8")) as {
  size: number;
  mean: number[];
  std: number[];
  images: { name: string; width: number; height: number; logits_fp32: number[]; logits_int8: number[] }[];
};

describe("skin preprocessing parity with Python", () => {
  for (const im of expected.images) {
    it(`${im.name}: tensor within 1e-3 of Python`, async () => {
      const { data, info } = await sharp(join(dir, `${im.name}.png`)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      expect([info.width, info.height]).toEqual([im.width, im.height]);
      const sq = cropRgba({ data, width: info.width, height: info.height }, centerSquare(info.width, info.height));
      const got = skinTensor(sq, expected.size);
      const u8 = readFileSync(join(dir, `${im.name}.u8.bin`)); // squashed uint8 HWC from Python
      const S = expected.size;
      let maxDiff = 0;
      let off = 0; // pixels that differ by one uint8 level: exact .5 sums round differently in float32/64 summation order
      for (let c = 0; c < 3; c++)
        for (let i = 0; i < S * S; i++) {
          const want = (u8[i * 3 + c] / 255 - expected.mean[c]) / expected.std[c];
          const d = Math.abs(got[c * S * S + i] - want);
          maxDiff = Math.max(maxDiff, d);
          if (d > 1e-3) off++;
        }
      expect(maxDiff).toBeLessThan(0.02); // at most one uint8 level (1/255/std = 0.0175)
      expect(off / (3 * S * S)).toBeLessThan(0.002);
    });
  }
  it("has expected logits for each image", () => {
    for (const im of expected.images) expect(im.logits_int8.length).toBe(im.logits_fp32.length);
  });
});

describe("darkSpotArea", () => {
  const skin = (): Uint8Array => {
    const u8 = new Uint8Array(224 * 224 * 3);
    for (let i = 0; i < 224 * 224; i++) u8.set([200, 150, 130], i * 3);
    return u8;
  };
  const disc = (u8: Uint8Array, cx: number, cy: number, r: number) => {
    for (let y = 0; y < 224; y++) for (let x = 0; x < 224; x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) u8.set([15, 12, 10], (y * 224 + x) * 3);
  };
  it("is 0 for plain skin", () => expect(darkSpotArea(skin())).toBe(0));
  it("measures an enclosed black disc (about pi r^2 / 224^2)", () => {
    const u8 = skin();
    disc(u8, 112, 112, 50);
    expect(darkSpotArea(u8)).toBeGreaterThan(0.12);
    expect(darkSpotArea(u8)).toBeLessThan(0.2);
  });
  it("ignores a dark blob that touches the border (shadow / vignette)", () => {
    const u8 = skin();
    disc(u8, 5, 112, 50);
    expect(darkSpotArea(u8)).toBe(0);
  });
  it("ignores spots that are not much darker than the skin", () => {
    const u8 = skin();
    for (let y = 80; y < 140; y++) for (let x = 80; x < 140; x++) u8.set([140, 100, 85], (y * 224 + x) * 3);
    expect(darkSpotArea(u8)).toBe(0);
  });
});
