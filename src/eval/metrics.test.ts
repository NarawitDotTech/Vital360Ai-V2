import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { formatBytes, metricsToCsv, parseMetrics } from "./metrics";

const sample = () => JSON.parse(readFileSync("eval/metrics/sample.json", "utf8"));

describe("eval metrics", () => {
  it("parses the sample file and marks it as sample", () => {
    const f = parseMetrics(sample(), "sample.json");
    expect(f.isSample).toBe(true);
    expect(Object.keys(f.models).sort()).toEqual(["audio", "fusion", "resp", "skin"]);
    expect(f.models.fusion!.accuracy).toBeGreaterThan(f.models.skin!.accuracy);
  });
  it("rejects malformed files", () => {
    const bad = sample();
    bad.models.skin.confusionMatrix = [[1]];
    expect(() => parseMetrics(bad, "x.json")).toThrow(/confusionMatrix/);
    expect(() => parseMetrics({ schemaVersion: 2 }, "x.json")).toThrow(/schemaVersion/);
  });
  it("exports CSV with sample marker, model rows and class rows, and escapes commas", () => {
    const j = sample();
    j.models.skin.testSetDescription = 'a, "b"';
    const csv = metricsToCsv(parseMetrics(j, "s.json"));
    const lines = csv.trim().split("\n");
    expect(lines[0]).toMatch(/SAMPLE DATA/);
    expect(lines[1]).toMatch(/^kind,model,class/);
    expect(lines.filter((l) => l.startsWith("model,"))).toHaveLength(4);
    expect(lines.filter((l) => l.startsWith("class,"))).toHaveLength(12);
    expect(csv).toContain('"a, ""b"""');
  });
  it("formats bytes", () => {
    expect(formatBytes(15000)).toBe("15.0 KB");
    expect(formatBytes(9400000)).toBe("9.4 MB");
    expect(formatBytes(12)).toBe("12 B");
  });
});
