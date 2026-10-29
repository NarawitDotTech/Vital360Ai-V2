// Calibration: temperature-scaled softmax, max-prob threshold -> "uncertain", label -> tier mapping.
import type { ModelEntry, Prediction, Tier } from "./types";

export function softmax(logits: ArrayLike<number>, temperature = 1): number[] {
  const t = temperature > 0 ? temperature : 1;
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i] / t);
  const exps: number[] = [];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] / t - max);
    exps.push(e);
    sum += e;
  }
  return exps.map((e) => e / sum);
}

export function tierOf(entry: ModelEntry, label: string): Exclude<Tier, "uncertain"> {
  const t = entry.tiers[label];
  if (!t) throw new Error(`No tier for label "${label}" in model ${entry.id}`);
  return t;
}

export interface Calibrated {
  probs: Record<string, number>;
  label: string;
  tier: Tier;
  confidence: number;
}

/** Turn class probabilities (already calibrated) into label/tier, applying the uncertainty threshold. */
export function finalize(entry: ModelEntry, probs: ArrayLike<number>): Calibrated {
  if (probs.length !== entry.labels.length) {
    throw new Error(`Model ${entry.id}: got ${probs.length} outputs, manifest has ${entry.labels.length} labels`);
  }
  const rec: Record<string, number> = {};
  let best = 0;
  for (let i = 0; i < probs.length; i++) {
    rec[entry.labels[i]] = probs[i];
    if (probs[i] > probs[best]) best = i;
  }
  const confidence = probs[best];
  if (confidence < entry.threshold) {
    return { probs: rec, label: "uncertain", tier: "uncertain", confidence };
  }
  const label = entry.labels[best];
  return { probs: rec, label, tier: tierOf(entry, label), confidence };
}

/** Raw logits -> temperature softmax -> finalize. */
export function calibrateLogits(entry: ModelEntry, logits: ArrayLike<number>): Calibrated {
  return finalize(entry, softmax(logits, entry.temperature));
}

export type PredictionCore = Pick<Prediction, "probs" | "label" | "tier" | "confidence">;
