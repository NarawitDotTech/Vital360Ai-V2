import { describe, expect, it } from "vitest";
import { calibrateLogits, finalize, softmax, tierOf } from "./calibrate";
import type { ModelEntry } from "./types";

const entry: ModelEntry = {
  id: "skin",
  version: "t",
  url: null,
  sha256: null,
  bytes: 0,
  labels: ["a", "b", "c"],
  tiers: { a: "green", b: "amber", c: "red" },
  threshold: 0.6,
  temperature: 2,
};

describe("calibrate", () => {
  it("softmax sums to 1 and is stable for large logits", () => {
    const p = softmax([1000, 1001, 999]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(p[1]).toBeGreaterThan(p[0]);
  });
  it("higher temperature flattens", () => {
    expect(Math.max(...softmax([3, 0, 0], 1))).toBeGreaterThan(Math.max(...softmax([3, 0, 0], 3)));
  });
  it("maps top label to tier", () => {
    const r = calibrateLogits(entry, [0, 0, 20]);
    expect(r.label).toBe("c");
    expect(r.tier).toBe("red");
  });
  it("is uncertain below threshold", () => {
    const r = finalize(entry, [0.4, 0.35, 0.25]);
    expect(r.label).toBe("uncertain");
    expect(r.tier).toBe("uncertain");
    expect(r.probs.a).toBe(0.4);
  });
  it("rejects wrong output size and unknown labels", () => {
    expect(() => finalize(entry, [1])).toThrow();
    expect(() => tierOf(entry, "zzz")).toThrow();
  });
  it("tierRule sum: tier from summed probs, label stays the top class", () => {
    const e: ModelEntry = { ...entry, labels: ["g1", "g2", "r1", "r2"], tiers: { g1: "green", g2: "green", r1: "red", r2: "red" }, tierRule: "sum", threshold: 0.5 };
    // top class is g1 (0.34) but red classes sum to 0.46 < green 0.54: green wins
    const r = finalize(e, [0.34, 0.2, 0.23, 0.23]);
    expect(r.tier).toBe("green");
    expect(r.label).toBe("g1");
    expect(r.tierProbs?.green).toBeCloseTo(0.54, 10);
    // top class is red but red total beats green: red
    expect(finalize(e, [0.1, 0.1, 0.5, 0.3]).tier).toBe("red");
    // no tier reaches threshold
    const u = finalize({ ...e, threshold: 0.6 }, [0.3, 0.2, 0.25, 0.25]);
    expect(u.tier).toBe("uncertain");
    expect(u.label).toBe("uncertain");
    expect(u.tierProbs?.red).toBeCloseTo(0.5, 10);
  });

  describe('tierRule "thresholds" (sensitivity-first)', () => {
    const e: ModelEntry = {
      ...entry,
      labels: ["mel", "bcc", "ak", "nev", "sk", "healthy"],
      tiers: { mel: "red", bcc: "red", ak: "amber", nev: "green", sk: "green", healthy: "green" },
      tierRule: "thresholds",
      tierThresholds: { red: 0.2, amber: 0.4 },
      guard: { topIn: ["nev", "sk"], watch: "mel" },
      threshold: 0.6,
    };
    it("red as soon as P(red) reaches the red threshold, even if green is the top tier", () => {
      const r = finalize(e, [0.12, 0.1, 0.03, 0.5, 0.05, 0.2]); // P(red) = 0.22
      expect(r.tier).toBe("red");
      expect(r.label).toBe("mel"); // best class of the shown tier
      expect(r.tierProbs?.red).toBeCloseTo(0.22, 10);
    });
    it("amber when P(amber)+P(red) reaches the amber threshold but red does not", () => {
      const r = finalize(e, [0.05, 0.05, 0.3, 0.3, 0.1, 0.2]); // red 0.10, amber 0.30 -> 0.40
      expect(r.tier).toBe("amber");
      expect(r.label).toBe("ak");
    });
    it("green only when both thresholds are missed and P(green) >= uncertain threshold", () => {
      const r = finalize(e, [0.02, 0.02, 0.06, 0.2, 0.1, 0.6]);
      expect(r.tier).toBe("green");
      expect(r.label).toBe("healthy");
    });
    it("uncertain when would-be green has P(green) below the threshold", () => {
      const e2 = { ...e, tierThresholds: { red: 0.5, amber: 0.6 }, threshold: 0.75 };
      const r = finalize(e2, [0.1, 0.1, 0.1, 0.3, 0.2, 0.2]);
      expect(r.tier).toBe("uncertain");
      expect(r.label).toBe("uncertain");
    });
    it("guard: top class nevus/SK with melanoma runner-up is shown amber, label stays the top class", () => {
      const r = finalize({ ...e, tierThresholds: { red: 0.5, amber: 0.6 } }, [0.3, 0.02, 0.02, 0.4, 0.05, 0.21]);
      expect(r.tier).toBe("amber");
      expect(r.label).toBe("nev");
    });
    it("guard does not fire when melanoma is only third, or the top class is not a look-alike", () => {
      const hi = { ...e, tierThresholds: { red: 0.5, amber: 0.6 } };
      expect(finalize(hi, [0.1, 0.02, 0.02, 0.4, 0.08, 0.38]).tier).toBe("green"); // melanoma is only third
      expect(finalize(hi, [0.3, 0.02, 0.02, 0.1, 0.06, 0.5]).tier).toBe("green"); // top is healthy
    });
    it("guard minWatchProb: a runner-up melanoma below the floor does not raise amber", () => {
      const hi = { ...e, tierThresholds: { red: 0.5, amber: 0.6 }, guard: { topIn: ["nev", "sk"], watch: "mel", minWatchProb: 0.1 } };
      expect(finalize(hi, [0.2, 0.0, 0.0, 0.7, 0.0, 0.1]).tier).toBe("amber"); // melanoma 0.20 >= 0.10 and runner-up
      expect(finalize(hi, [0.09, 0.0, 0.0, 0.86, 0.0, 0.05]).tier).toBe("green"); // melanoma is runner-up but 0.09 < 0.10
    });
    it("darkSpot: a big enclosed dark spot is never green (raised to amber), red stays red", () => {
      const d = { ...e, tierThresholds: { red: 0.5, amber: 0.6 }, darkSpot: { minArea: 0.03, relLuma: 0.45, maxLuma: 70 } };
      const green = [0.02, 0.02, 0.06, 0.2, 0.1, 0.6];
      expect(finalize(d, green, { darkSpot: 0.02 }).tier).toBe("green");
      expect(finalize(d, green, {}).tier).toBe("green"); // no pixels (mock / worker without canvas)
      const r = finalize(d, green, { darkSpot: 0.1 });
      expect(r.tier).toBe("amber");
      expect(r.label).toBe("healthy"); // label stays the top class
      expect(finalize(d, [0.6, 0.1, 0.0, 0.1, 0.1, 0.1], { darkSpot: 0.1 }).tier).toBe("red");
    });
    it("throws without tierThresholds", () => {
      expect(() => finalize({ ...e, tierThresholds: undefined }, [1, 0, 0, 0, 0, 0])).toThrow();
    });
  });
});
