import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseMetrics, type MetricsFile } from "./metrics";

/** Build-time only (Node fs). Reads every eval/metrics/*.json. */
export function loadMetricFiles(dir = join(process.cwd(), "eval", "metrics")): MetricsFile[] {
  let names: string[] = [];
  try {
    names = readdirSync(dir).filter((n) => n.endsWith(".json")).sort();
  } catch {
    return [];
  }
  return names.map((n) => parseMetrics(JSON.parse(readFileSync(join(dir, n), "utf8")), n));
}
