// Evaluation metrics: types, validation and CSV export (pure, no fs). Schema: eval/metrics/README.md
export const MODEL_KEYS = ["skin", "audio", "resp", "fusion"] as const;
export type ModelKey = (typeof MODEL_KEYS)[number];

export interface ClassStat {
  sensitivity: number;
  specificity: number;
}

export interface ModelMetrics {
  label: string;
  classes: string[];
  accuracy: number;
  sensitivity: number;
  specificity: number;
  f1: number;
  perClass: Record<string, ClassStat>;
  confusionMatrix: number[][];
  latencyMs: number;
  sizeBytes: number;
  testSetDescription: string;
}

export interface MetricsFile {
  file: string;
  schemaVersion: number;
  isSample: boolean;
  generatedAt: string;
  description: string;
  models: Partial<Record<ModelKey, ModelMetrics>>;
}

const num = (v: unknown) => typeof v === "number" && Number.isFinite(v);

export function parseMetrics(json: unknown, file: string): MetricsFile {
  const fail = (m: string): never => {
    throw new Error(`eval/metrics/${file}: ${m}`);
  };
  if (!json || typeof json !== "object") return fail("not an object");
  const o = json as Record<string, unknown>;
  if (o.schemaVersion !== 1) fail("schemaVersion must be 1");
  if (!o.models || typeof o.models !== "object") fail("models missing");
  const models: MetricsFile["models"] = {};
  for (const key of MODEL_KEYS) {
    const m = (o.models as Record<string, unknown>)[key] as Record<string, unknown> | undefined;
    if (!m) continue;
    for (const f of ["accuracy", "sensitivity", "specificity", "f1", "latencyMs", "sizeBytes"]) {
      if (!num(m[f])) fail(`${key}.${f} must be a number`);
    }
    const classes = m.classes;
    const cm = m.confusionMatrix;
    if (!Array.isArray(classes) || !classes.every((c) => typeof c === "string")) fail(`${key}.classes invalid`);
    if (!Array.isArray(cm) || cm.length !== (classes as string[]).length || !cm.every((r) => Array.isArray(r) && r.length === cm.length && r.every(num)))
      fail(`${key}.confusionMatrix must be ${(classes as string[]).length}x${(classes as string[]).length}`);
    const perClass = m.perClass as Record<string, ClassStat> | undefined;
    if (!perClass || !(classes as string[]).every((c) => perClass[c] && num(perClass[c].sensitivity) && num(perClass[c].specificity)))
      fail(`${key}.perClass must cover every class`);
    models[key] = {
      label: typeof m.label === "string" ? m.label : key,
      classes: classes as string[],
      accuracy: m.accuracy as number,
      sensitivity: m.sensitivity as number,
      specificity: m.specificity as number,
      f1: m.f1 as number,
      perClass: perClass as Record<string, ClassStat>,
      confusionMatrix: cm as number[][],
      latencyMs: m.latencyMs as number,
      sizeBytes: m.sizeBytes as number,
      testSetDescription: typeof m.testSetDescription === "string" ? m.testSetDescription : "",
    };
  }
  return {
    file,
    schemaVersion: 1,
    isSample: o.isSample === true,
    generatedAt: typeof o.generatedAt === "string" ? o.generatedAt : "",
    description: typeof o.description === "string" ? o.description : "",
    models,
  };
}

function cell(v: string | number): string {
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** One CSV: a summary row per model, then one row per model+class. Prefixed with a SAMPLE marker when needed. */
export function metricsToCsv(f: MetricsFile): string {
  const rows: (string | number)[][] = [];
  if (f.isSample) rows.push(["# SAMPLE DATA, not measured"]);
  rows.push(["kind", "model", "class", "accuracy", "sensitivity", "specificity", "f1", "latencyMs", "sizeBytes", "testSet"]);
  for (const key of MODEL_KEYS) {
    const m = f.models[key];
    if (!m) continue;
    rows.push(["model", key, "", m.accuracy, m.sensitivity, m.specificity, m.f1, m.latencyMs, m.sizeBytes, m.testSetDescription]);
    for (const c of m.classes) rows.push(["class", key, c, "", m.perClass[c].sensitivity, m.perClass[c].specificity, "", "", "", ""]);
  }
  return rows.map((r) => r.map(cell).join(",")).join("\n") + "\n";
}

export function formatBytes(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}
