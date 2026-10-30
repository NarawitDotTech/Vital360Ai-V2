// Parity: TS skin preprocessing must reproduce the Python training preprocessing (training/cpu/skin_common.py)
// on the fixture images. Fixture files are written by training/cpu/skin_fixture.py.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import sharp from "sharp";
import { describe, expect, it } from "vitest";
import { skinTensor } from "./skin-preprocess";

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
      const got = skinTensor({ data, width: info.width, height: info.height }, expected.size);
      const u8 = readFileSync(join(dir, `${im.name}.u8.bin`)); // squashed uint8 HWC from Python
      const S = expected.size;
      let maxDiff = 0;
      for (let c = 0; c < 3; c++)
        for (let i = 0; i < S * S; i++) {
          const want = (u8[i * 3 + c] / 255 - expected.mean[c]) / expected.std[c];
          maxDiff = Math.max(maxDiff, Math.abs(got[c * S * S + i] - want));
        }
      expect(maxDiff).toBeLessThan(1e-3);
    });
  }
  it("has expected logits for each image", () => {
    for (const im of expected.images) expect(im.logits_int8.length).toBe(im.logits_fp32.length);
  });
});
