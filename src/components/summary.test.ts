import { describe, expect, it } from "vitest";
import { dictionaries } from "@/i18n";
import type { Prediction } from "@/inference/types";
import { summarize } from "./summary";

const en = dictionaries.en;
const p = (o: Partial<Prediction>): Prediction => ({
  modality: "skin",
  probs: { nevus: 0.7 },
  label: "nevus",
  tier: "green",
  confidence: 0.7,
  modelVersion: "skin-v1",
  latencyMs: 1,
  ...o,
});

describe("summarize", () => {
  it("one check: green, lists the two missing", () => {
    const s = summarize([p({})], null, en);
    expect(s.headline).toBe("Nothing of concern found");
    expect(s.missing).toEqual(["audio", "resp"]);
    expect(s.lines[0]).toMatch(/Closest match/);
  });
  it("all three: nothing missing", () => {
    const s = summarize(
      [
        p({}),
        p({ modality: "audio", label: "normal", probs: { normal: 0.9 } }),
        p({ modality: "resp" }),
      ],
      null,
      en,
    );
    expect(s.missing).toEqual([]);
    expect(s.missingText).toBeNull();
    expect(s.speech).toContain(en.summary.allDone);
  });
  it("mixed tiers: worst wins (red > amber > uncertain > green)", () => {
    expect(
      summarize(
        [p({ tier: "amber" }), p({ modality: "resp", tier: "red" })],
        null,
        en,
      ).tier,
    ).toBe("red");
    expect(
      summarize(
        [p({ tier: "green" }), p({ modality: "resp", tier: "uncertain" })],
        null,
        en,
      ).tier,
    ).toBe("uncertain");
    expect(
      summarize(
        [p({ tier: "uncertain" }), p({ modality: "resp", tier: "amber" })],
        null,
        en,
      ).headline,
    ).toBe("Worth having a clinician look");
  });
  it("uncertain: cannot tell yet, no closest match", () => {
    const s = summarize(
      [p({ tier: "uncertain", label: "uncertain" })],
      null,
      en,
    );
    expect(s.headline).toBe("Cannot tell yet");
    expect(s.lines[0]).not.toMatch(/Closest/);
  });
  it("mock results are demo only and do not drive the headline", () => {
    const only = summarize(
      [p({ modelVersion: "mock-0.1", tier: "red" })],
      null,
      en,
    );
    expect(only.headline).toBe("Cannot tell yet");
    expect(only.demoOnly).toBe(true);
    expect(only.lines[0]).toMatch(/demo only/);
    const mixed = summarize(
      [
        p({ modelVersion: "mock-0.1", tier: "red" }),
        p({ modality: "resp", tier: "green" }),
      ],
      null,
      en,
    );
    expect(mixed.headline).toBe("Nothing of concern found");
    expect(
      summarize(
        [p({ tier: "green" })],
        p({ modality: "fusion", tier: "amber" }),
        en,
      ).tier,
    ).toBe("amber");
  });
  it("breaths per minute and Thai", () => {
    const r = p({ modality: "resp", extras: { breathsPerMin: 17.4 } });
    expect(summarize([r], null, en).lines[0]).toContain(
      "About 17 breaths per minute",
    );
    expect(summarize([r], null, dictionaries.th).lines[0]).toContain("17");
    expect(
      summarize([p({ tier: "red" })], null, dictionaries.th).headline,
    ).toBe("ควรพบแพทย์โดยเร็ว");
  });
});
