import { describe, expect, it } from "vitest";
import type { Prediction } from "@/inference/types";
import { clearSessionResults, doneModalities, getSessionResults, sessionParts, setSessionResult } from "./results";

const p = (modality: Prediction["modality"], extra: Partial<Prediction> = {}): Prediction => ({
  modality,
  probs: { a: 1 },
  label: "a",
  tier: "green",
  confidence: 1,
  modelVersion: "t",
  latencyMs: 1,
  ...extra,
});

describe("session results", () => {
  it("replaces per modality, strips heatmaps, survives bad JSON", () => {
    clearSessionResults();
    expect(getSessionResults()).toEqual([]);
    setSessionResult(p("skin"));
    setSessionResult(p("skin", { tier: "red" }));
    setSessionResult(p("resp", { extras: { breathsPerMin: 16, heatmap: {} as ImageData } }));
    const r = getSessionResults();
    expect(r).toHaveLength(2);
    expect(r.find((x) => x.modality === "skin")?.tier).toBe("red");
    expect(r.find((x) => x.modality === "resp")?.extras).toEqual({ breathsPerMin: 16 });
    expect([...doneModalities(r)].sort()).toEqual(["resp", "skin"]);
    setSessionResult(p("fusion"));
    expect(sessionParts(getSessionResults())).toHaveLength(2);
    sessionStorage.setItem("v360.session", "{oops");
    expect(getSessionResults()).toEqual([]);
  });
});
