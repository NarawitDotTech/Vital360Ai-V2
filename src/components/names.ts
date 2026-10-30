import type { Prediction } from "@/inference/types";

type Results = { names: Record<string, string>; labels: Record<string, string> };

/** Plain-language name of a model class: skin classes use their own key, audio classes are prefixed ("audio-wheeze"). */
export function className(r: Results, modality: Prediction["modality"], key: string): string | undefined {
  return r.names[modality === "audio" ? `audio-${key}` : key];
}

/** True when the prediction comes from a real fine-grained model that can name the closest match. */
export function canNameMatch(p: Pick<Prediction, "modality" | "modelVersion" | "label">, r: Results): boolean {
  return (p.modality === "skin" || p.modality === "audio") && !p.modelVersion.startsWith("mock") && p.label !== "uncertain" && !!className(r, p.modality, p.label);
}

/** The next `n` most likely classes after the top one (those with a plain name). */
export function alternatives(p: Prediction, r: Results, n = 2): { key: string; name: string; pct: number }[] {
  return Object.entries(p.probs)
    .sort((a, b) => b[1] - a[1])
    .filter(([k]) => k !== p.label || p.label === "uncertain")
    .flatMap(([k, v]) => {
      const name = className(r, p.modality, k);
      return name ? [{ key: k, name, pct: Math.round(v * 100) }] : [];
    })
    .slice(0, n);
}
