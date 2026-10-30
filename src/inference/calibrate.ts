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
  tierProbs?: Prediction["tierProbs"];
}

/** Optional image-derived features (skin): area of the largest enclosed dark spot, see skin-preprocess.ts darkSpotArea. */
export interface Extra {
  darkSpot?: number;
}

/** Turn class probabilities (already calibrated) into label/tier, applying the uncertainty threshold. */
export function finalize(entry: ModelEntry, probs: ArrayLike<number>, extra: Extra = {}): Calibrated {
  if (probs.length !== entry.labels.length) {
    throw new Error(`Model ${entry.id}: got ${probs.length} outputs, manifest has ${entry.labels.length} labels`);
  }
  const rec: Record<string, number> = {};
  let best = 0;
  for (let i = 0; i < probs.length; i++) {
    rec[entry.labels[i]] = probs[i];
    if (probs[i] > probs[best]) best = i;
  }
  if (entry.tierRule === "sum") return finalizeBySum(entry, probs, rec, best);
  if (entry.tierRule === "thresholds") return finalizeByThresholds(entry, probs, rec, best, extra);
  const confidence = probs[best];
  if (confidence < entry.threshold) {
    return { probs: rec, label: "uncertain", tier: "uncertain", confidence };
  }
  const label = entry.labels[best];
  return { probs: rec, label, tier: tierOf(entry, label), confidence };
}

const TIER_ORDER = ["green", "amber", "red"] as const;

/** tierRule "sum": tier probabilities are the sums of class probabilities; shown tier = argmax, uncertain below threshold. */
function finalizeBySum(entry: ModelEntry, probs: ArrayLike<number>, rec: Record<string, number>, best: number): Calibrated {
  const tp = tierSums(entry, probs);
  let tier: (typeof TIER_ORDER)[number] = "green";
  for (const t of TIER_ORDER) if (tp[t] > tp[tier]) tier = t;
  const confidence = tp[tier];
  if (confidence < entry.threshold) return { probs: rec, label: "uncertain", tier: "uncertain", confidence, tierProbs: tp };
  return { probs: rec, label: entry.labels[best], tier, confidence, tierProbs: tp };
}

/** Summed class probability per tier. */
function tierSums(entry: ModelEntry, probs: ArrayLike<number>): { green: number; amber: number; red: number } {
  const tp = { green: 0, amber: 0, red: 0 };
  for (let i = 0; i < probs.length; i++) tp[tierOf(entry, entry.labels[i])] += probs[i];
  return tp;
}

/** Index of the best class whose tier is `tier`, or -1. */
function bestInTier(entry: ModelEntry, probs: ArrayLike<number>, tier: string): number {
  let b = -1;
  for (let i = 0; i < probs.length; i++) if (entry.tiers[entry.labels[i]] === tier && (b < 0 || probs[i] > probs[b])) b = i;
  return b;
}

/** True when the top class is in guard.topIn and guard.watch is the runner-up with P >= minWatchProb (melanoma hiding behind a "harmless" look-alike). */
function guardHit(entry: ModelEntry, probs: ArrayLike<number>, best: number): boolean {
  const g = entry.guard;
  if (!g || !g.topIn.includes(entry.labels[best])) return false;
  let second = -1;
  for (let i = 0; i < probs.length; i++) if (i !== best && (second < 0 || probs[i] > probs[second])) second = i;
  return second >= 0 && entry.labels[second] === g.watch && probs[second] >= (g.minWatchProb ?? 0);
}

/**
 * tierRule "thresholds" (sensitivity-first): red if P(red) >= t.red; else amber if P(amber)+P(red) >= t.amber; else amber when the
 * look-alike guard fires or an enclosed dark spot is large (entry.darkSpot); else "uncertain" if P(green) < entry.threshold; else green. The label is the top class of the shown tier
 * (so a red result names the most likely red class); the guard keeps the overall top class.
 */
function finalizeByThresholds(entry: ModelEntry, probs: ArrayLike<number>, rec: Record<string, number>, best: number, extra: Extra): Calibrated {
  const t = entry.tierThresholds;
  if (!t) throw new Error(`Model ${entry.id}: tierRule "thresholds" needs tierThresholds`);
  const tp = tierSums(entry, probs);
  const pick = (tier: "green" | "amber" | "red", confidence: number, forceTop = false): Calibrated => {
    const j = bestInTier(entry, probs, tier);
    const i = forceTop || j < 0 || tierOf(entry, entry.labels[best]) === tier ? best : j;
    return { probs: rec, label: entry.labels[i], tier, confidence, tierProbs: tp };
  };
  if (tp.red >= t.red) return pick("red", tp.red);
  if (tp.amber + tp.red >= t.amber) return pick("amber", tp.amber + tp.red);
  const darkHit = entry.darkSpot !== undefined && extra.darkSpot !== undefined && extra.darkSpot >= entry.darkSpot.minArea;
  if (darkHit || guardHit(entry, probs, best)) return pick("amber", Math.max(tp.amber + tp.red, probs[best]), true);
  if (tp.green < entry.threshold) return { probs: rec, label: "uncertain", tier: "uncertain", confidence: tp.green, tierProbs: tp };
  return pick("green", tp.green);
}

/** Raw logits -> temperature softmax -> finalize. */
export function calibrateLogits(entry: ModelEntry, logits: ArrayLike<number>, extra: Extra = {}): Calibrated {
  return finalize(entry, softmax(logits, entry.temperature), extra);
}

export type PredictionCore = Pick<Prediction, "probs" | "label" | "tier" | "confidence">;
