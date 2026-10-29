import { describe, expect, it } from "vitest";
import type { Prediction } from "./types";
import { inference } from "@/inference";

describe("inference (main-thread fallback, bundled manifest)", () => {
  it("skin: deterministic, filled metadata", async () => {
    const image = { data: new Uint8ClampedArray(8 * 8 * 4).fill(90), width: 8, height: 8, colorSpace: "srgb" } as ImageData;
    const a = await inference.predict({ modality: "skin", image });
    const b = await inference.predict({ modality: "skin", image });
    expect({ ...a, latencyMs: 0 }).toEqual({ ...b, latencyMs: 0 });
    expect(a.modelVersion).toBe("mock-0.1.0");
    expect(a.latencyMs).toBeGreaterThanOrEqual(0);
    expect(Object.keys(a.probs)).toEqual(["benign", "needs-review", "urgent-referral"]);
  });
  it("audio never goes red", async () => {
    for (let k = 1; k < 30; k++) {
      const s = new Float32Array(16000).map((_, i) => Math.sin(i * k * 0.01));
      const p = await inference.predict({ modality: "audio", samples: s, sampleRate: 16000 });
      expect(p.tier).not.toBe("red");
    }
  });
  it("resp: 15 bpm green, noise uncertain, fast red", async () => {
    const mk = (bpm: number) => new Float32Array(600).map((_, i) => Math.sin((2 * Math.PI * (bpm / 60) * i) / 30));
    const ok = await inference.predict({ modality: "resp", signal: mk(15), fps: 30 });
    expect(ok.tier).toBe("green");
    expect(ok.extras?.breathsPerMin).toBeCloseTo(15, 0);
    const fast = await inference.predict({ modality: "resp", signal: mk(28), fps: 30 });
    expect([fast.label, fast.tier]).toEqual(["fast", "red"]);
    const bad = await inference.predict({ modality: "resp", signal: new Float32Array(600), fps: 30 });
    expect([bad.label, bad.tier]).toEqual(["uncertain", "uncertain"]);
  });
  it("fusion: uncertain with no usable parts; escalates with severe parts", async () => {
    const none = await inference.predict({ modality: "fusion", parts: [] });
    expect(none.tier).toBe("uncertain");
    const part = (modality: "skin" | "audio" | "resp", label: string, tier: "red") =>
      ({ modality, label, tier, probs: { [label]: 1 }, confidence: 1, modelVersion: "x", latencyMs: 0 }) as Prediction;
    const hi = await inference.predict({
      modality: "fusion",
      parts: [part("skin", "urgent-referral", "red"), part("resp", "fast", "red")],
    });
    expect(hi.label).toBe("high");
    expect(hi.tier).toBe("red");
  });
  it("explain is undefined for mock; loadModel resolves", async () => {
    await expect(inference.loadModel("skin")).resolves.toBeUndefined();
    expect(await inference.explain({ modality: "resp", signal: new Float32Array(1), fps: 30 })).toBeUndefined();
  });
});
