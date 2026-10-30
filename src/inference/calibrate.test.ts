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
});
