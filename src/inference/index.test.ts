import { describe, expect, it } from "vitest";
import { Engine } from "./engine";
import type { Manifest, Prediction } from "./types";
import { inference } from "@/inference";

// The bundled manifest now points skin (and audio) at real ONNX models, which need a browser to run (see e2e/skin-model.spec.ts).
// These unit tests exercise the deterministic mock backend through an injected mock manifest instead.
const mockManifest: Manifest = {
  schemaVersion: 1,
  models: [
    { id: "skin", version: "mock-0.1.0", url: null, sha256: null, bytes: 0, labels: ["benign", "needs-review", "urgent-referral"], tiers: { benign: "green", "needs-review": "amber", "urgent-referral": "red" }, threshold: 0.55, temperature: 1.5 },
    { id: "audio", version: "mock-0.1.0", url: null, sha256: null, bytes: 0, labels: ["normal", "wheeze", "crackle", "cough-abnormal"], tiers: { normal: "green", wheeze: "amber", crackle: "amber", "cough-abnormal": "amber" }, threshold: 0.5, temperature: 1.5 },
  ],
};
const mockEngine = () => new Engine({ manifest: async () => mockManifest });

describe("inference (main-thread fallback, bundled manifest)", () => {
  it("skin: deterministic, filled metadata", async () => {
    const image = { data: new Uint8ClampedArray(8 * 8 * 4).fill(90), width: 8, height: 8, colorSpace: "srgb" } as ImageData;
    const eng = mockEngine();
    const a = await eng.predict({ modality: "skin", image });
    const b = await eng.predict({ modality: "skin", image });
    expect({ ...a, latencyMs: 0 }).toEqual({ ...b, latencyMs: 0 });
    expect(a.modelVersion).toBe("mock-0.1.0");
    expect(a.latencyMs).toBeGreaterThanOrEqual(0);
    expect(Object.keys(a.probs)).toEqual(["benign", "needs-review", "urgent-referral"]);
  });
  it("audio never goes red", async () => {
    const eng = mockEngine();
    for (let k = 1; k < 30; k++) {
      const s = new Float32Array(16000).map((_, i) => Math.sin(i * k * 0.01));
      const p = await eng.predict({ modality: "audio", samples: s, sampleRate: 16000 });
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
    await expect(inference.loadModel("resp")).resolves.toBeUndefined();
    expect(await inference.explain({ modality: "resp", signal: new Float32Array(1), fps: 30 })).toBeUndefined();
  });
});
