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
});
