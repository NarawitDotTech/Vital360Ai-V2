// Parity: the TS tier decision (tierRule "thresholds", calibrate.ts) must equal the Python one (training/cpu/skin_decide.py) on
// real int8 logits of held-out test images. expected.json is written by training/cpu/skin2_fixture.py.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { calibrateLogits } from "./calibrate";
import { darkSpotArea } from "./skin-preprocess";
import type { Manifest } from "./types";
import manifest from "../../public/models/manifest.json";

const expected = JSON.parse(readFileSync(join(__dirname, "__fixtures__", "skin", "expected.json"), "utf8")) as {
  temperature: number;
  images: { name: string; logits_int8: number[]; dark_spot: number; tier: string; label: string }[];
  decisions: { logits: number[]; dark_spot: number; tier: string; label: string }[];
};
const skin = (manifest as unknown as Manifest).models.find((m) => m.id === "skin")!;

describe("skin manifest entry", () => {
  it("uses the sensitivity-first rule with a healthy-skin class", () => {
    expect(skin.tierRule).toBe("thresholds");
    expect(skin.tierThresholds!.red).toBeLessThanOrEqual(skin.tierThresholds!.amber);
    expect(skin.labels).toContain("healthy-skin");
    expect(skin.tiers["healthy-skin"]).toBe("green");
    expect(skin.labels).toHaveLength(20);
    expect(skin.darkSpot!.minArea).toBeGreaterThan(0);
    expect(skin.temperature).toBeCloseTo(expected.temperature, 3);
  });
});

describe("TS decision == Python decision", () => {
  it("fixture images", () => {
    for (const im of expected.images) {
      const r = calibrateLogits(skin, im.logits_int8, { darkSpot: im.dark_spot });
      expect({ tier: r.tier, label: r.label }, im.name).toEqual({ tier: im.tier, label: im.label });
    }
  });
  it("300 held-out test logits", () => {
    let bad = 0;
    for (const d of expected.decisions) {
      const r = calibrateLogits(skin, d.logits, { darkSpot: d.dark_spot });
      if (r.tier !== d.tier || r.label !== d.label) bad++;
    }
    // logits are rounded to 4 decimals in the fixture: a probability exactly on a threshold may flip, nothing else may
    expect(bad).toBeLessThanOrEqual(1);
  });
});

describe("dark spot feature parity", () => {
  it("darkSpotArea on the fixture model inputs equals Python", () => {
    for (const im of expected.images) {
      const u8 = new Uint8Array(readFileSync(join(__dirname, "__fixtures__", "skin", `${im.name}.u8.bin`)));
      expect(Math.abs(darkSpotArea(u8, skin.darkSpot) - im.dark_spot)).toBeLessThan(1e-4);
    }
  });
});
