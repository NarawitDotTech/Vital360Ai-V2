import { describe, expect, it } from "vitest";
import { fuse, fusionWeights, severity, type FusionWeights } from "./fusion";
import type { Prediction } from "./types";

const w: FusionWeights = {
  classes: ["low", "high"],
  features: ["skin", "audio"],
  missingPrior: { skin: 0.5, audio: 0 },
  bias: [0, 0],
  weights: [
    [0, 0],
    [1, 2],
  ],
};

describe("fuse", () => {
  it("matches hand-computed logistic math", () => {
    const { probs, present } = fuse({ skin: 1, audio: 0.5 }, w);
    // z_high = 1*1 + 2*0.5 = 2; p_high = e^2/(1+e^2)
    expect(present).toBe(2);
    expect(probs[1]).toBeCloseTo(Math.exp(2) / (1 + Math.exp(2)), 10);
    expect(probs[0] + probs[1]).toBeCloseTo(1, 12);
  });
  it("uses the documented prior for a missing modality", () => {
    const a = fuse({ skin: 1 }, w);
    const b = fuse({ skin: 1, audio: 0 }, w);
    expect(a.present).toBe(1);
    expect(a.probs).toEqual(b.probs);
    expect(fuse({}, w).probs[1]).toBeCloseTo(1 / (1 + Math.exp(-0.5)), 10);
  });
  it("placeholder weights are marked untrained and monotone in severity", () => {
    expect(fusionWeights.untrained).toBe(true);
    const lo = fuse({ skin: 0, audio: 0, resp: 0 }).probs;
    const hi = fuse({ skin: 2, audio: 2, resp: 2 }).probs;
    expect(lo[0]).toBeGreaterThan(lo[2]);
    expect(hi[2]).toBeGreaterThan(hi[0]);
  });
  it("severity is P(amber)+2P(red)", () => {
    const p = { probs: { a: 0.5, b: 0.3, c: 0.2 } } as unknown as Prediction;
    const tiers: Record<string, "green" | "amber" | "red"> = { a: "green", b: "amber", c: "red" };
    expect(severity(p, (l) => tiers[l])).toBeCloseTo(0.7, 12);
  });
});
