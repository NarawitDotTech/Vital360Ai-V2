import { describe, expect, it } from "vitest";
import { alternatives, canNameMatch, className } from "./names";
import { dictionaries } from "@/i18n";

const r = dictionaries.en.results;
const pred = (over: object) => ({
  modality: "skin" as const,
  modelVersion: "skin-v1.0.0",
  label: "nevus",
  probs: { nevus: 0.6, melanoma: 0.25, eczema: 0.1, zzz: 0.05 },
  ...over,
});

describe("class names", () => {
  it("names skin classes and prefixes audio ones", () => {
    expect(className(r, "skin", "nevus")).toMatch(/Mole/);
    expect(className(r, "audio", "wheeze")).toMatch(/Wheeze/);
  });
  it("never names mock or uncertain results", () => {
    expect(canNameMatch(pred({}), r)).toBe(true);
    expect(canNameMatch(pred({ modelVersion: "mock-0.1.0" }), r)).toBe(false);
    expect(canNameMatch(pred({ label: "uncertain" }), r)).toBe(false);
    expect(canNameMatch(pred({ modality: "resp" }), r)).toBe(false);
  });
  it("lists the next two named alternatives", () => {
    expect(alternatives(pred({}) as never, r).map((a) => a.key)).toEqual([
      "melanoma",
      "eczema",
    ]);
  });
  it("every skin class in the manifest has a plain name in both languages", async () => {
    const m = (await import("../../public/models/manifest.json")).default;
    const skin = m.models.find((x) => x.id === "skin")!;
    for (const l of skin.labels)
      for (const d of [dictionaries.en, dictionaries.th])
        expect(d.results.names[l], l).toBeTruthy();
  });
});
