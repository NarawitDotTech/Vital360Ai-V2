import type { Modality, Prediction, Tier } from "@/inference/types";
import { format } from "@/i18n";
import type { Dict } from "@/i18n";
import { canNameMatch, className } from "./names";

export type SummaryT = Dict;
export type CheckId = Exclude<Modality, "fusion">;
export const ALL_CHECKS: CheckId[] = ["skin", "audio", "resp"];

const RANK: Record<Tier, number> = { green: 0, uncertain: 1, amber: 2, red: 3 };
const isMock = (p: Prediction) => p.modelVersion.startsWith("mock");

export interface Summary {
  tier: Tier;
  headline: string;
  lines: string[];
  next: string;
  missing: CheckId[];
  missingText: string | null;
  /** True when every part is a demo (mock) result. */
  demoOnly: boolean;
  speech: string;
}

/** Deterministic, on-device summary of all checks in the session. Demo (mock) results never drive the headline. */
export function summarize(
  parts: Prediction[],
  fusion: Prediction | null,
  t: SummaryT,
): Summary {
  const real = parts.filter((p) => !isMock(p));
  const tiered = fusion && !isMock(fusion) ? [...real, fusion] : real;
  const tier = tiered.reduce<Tier>(
    (w, p) => (RANK[p.tier] > RANK[w] ? p.tier : w),
    "green",
  );
  const demoOnly = parts.length > 0 && real.length === 0;
  const finalTier: Tier = tiered.length === 0 ? "uncertain" : tier;

  const lines = parts.map((p) => {
    const name = t.results.modality[p.modality];
    if (isMock(p)) return format(t.summary.demoLine, { name });
    let s = format(t.summary.line, { name, word: t.tiers[p.tier].word });
    if (canNameMatch(p, t.results)) {
      s +=
        " " +
        format(t.results.similarTo, {
          name: className(t.results, p.modality, p.label) ?? "",
          pct: Math.round((p.probs[p.label] ?? 0) * 100),
        }) +
        ".";
    }
    const bpm = p.extras?.breathsPerMin;
    if (bpm !== undefined)
      s += " " + format(t.results.breaths, { n: Math.round(bpm) }) + ".";
    return s;
  });

  const done = new Set(parts.map((p) => p.modality));
  const missing = ALL_CHECKS.filter((m) => !done.has(m));
  const missingText = missing.length
    ? format(t.summary.missing, {
        list: missing.map((m) => t.results.modality[m]).join(", "),
      })
    : null;
  const next = demoOnly ? t.summary.demoOnly : t.tiers[finalTier].next;
  const headline = t.summary.headline[finalTier];
  const speech = [
    t.summary.title,
    headline,
    ...lines,
    next,
    missingText ?? t.summary.allDone,
  ].join(". ");
  return {
    tier: finalTier,
    headline,
    lines,
    next,
    missing,
    missingText,
    demoOnly,
    speech,
  };
}
