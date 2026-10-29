// Late fusion: multinomial logistic regression over per-modality expected severity.
import weights from "./fusion-weights.json";
import type { Modality, Prediction } from "./types";

export interface FusionWeights {
  untrained?: boolean;
  classes: string[];
  features: string[];
  missingPrior: Record<string, number>;
  bias: number[];
  weights: number[][];
}

export const fusionWeights = weights as FusionWeights;

const SEVERITY = { green: 0, amber: 1, red: 2 } as const;

/** Expected severity 0..2 of a modality prediction: P(amber)+2P(red), via the label -> tier map in `tierOfLabel`. */
export function severity(p: Prediction, tierOfLabel: (label: string) => "green" | "amber" | "red" | undefined): number {
  let s = 0;
  for (const [label, prob] of Object.entries(p.probs)) {
    const t = tierOfLabel(label);
    if (t) s += prob * SEVERITY[t];
  }
  return s;
}

/** Plain softmax over class logits. */
function softmaxPlain(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/**
 * features: modality -> severity, or undefined when missing/uncertain (replaced by the documented prior).
 * Returns class probabilities in `fusionWeights.classes` order, plus how many modalities were really present.
 */
export function fuse(
  features: Partial<Record<Modality, number | undefined>>,
  w: FusionWeights = fusionWeights,
): { probs: number[]; present: number } {
  let present = 0;
  const x = w.features.map((f) => {
    const v = features[f as Modality];
    if (v === undefined || !Number.isFinite(v)) return w.missingPrior[f] ?? 0;
    present++;
    return v;
  });
  const z = w.classes.map((_, c) => w.bias[c] + x.reduce((a, xi, i) => a + xi * w.weights[c][i], 0));
  return { probs: softmaxPlain(z), present };
}
