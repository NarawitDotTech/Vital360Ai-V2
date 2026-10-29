"use client";

import type { Prediction } from "@/inference/types";
import { format } from "@/i18n";
import { useI18n } from "./I18nProvider";
import { TierIcon } from "./TierBadge";

export function resultSpeech(p: Prediction, t: ReturnType<typeof useI18n>["t"]): string {
  const tier = t.tiers[p.tier];
  const parts = [`${t.results.modality[p.modality]}. ${tier.word}.`, tier.sentence, `${t.results.nextStep}: ${tier.next}`];
  const bpm = p.extras?.breathsPerMin;
  if (bpm !== undefined) parts.splice(2, 0, format(t.results.breaths, { n: Math.round(bpm) }) + ".");
  return parts.join(" ");
}

/** Traffic-light result: tier (icon + word + color), one sentence, one next step, details collapsed. */
export function ResultCard({ p, headingLevel = 2 }: { p: Prediction; headingLevel?: 2 | 3 }) {
  const { t } = useI18n();
  const tier = t.tiers[p.tier];
  const bpm = p.extras?.breathsPerMin;
  const H = headingLevel === 2 ? "h2" : "h3";
  const probs = Object.entries(p.probs).sort((a, b) => b[1] - a[1]);
  return (
    <section className={`tier tier-${p.tier}`} aria-labelledby={`tier-${p.modality}`} data-testid={`result-${p.modality}`} data-tier={p.tier}>
      <H id={`tier-${p.modality}`} className="font-semibold">
        {t.results.modality[p.modality]}
      </H>
      <div className="mt-2 flex items-center gap-4">
        <TierIcon tier={p.tier} />
        <p className="tier-word">{tier.word}</p>
      </div>
      <p className="mt-3 text-xl font-semibold">{tier.sentence}</p>
      {bpm !== undefined && (
        <div className="mt-3">
          <p className="text-xl font-bold">{format(t.results.breaths, { n: Math.round(bpm) })}</p>
          <p className="text-base">{t.results.breathsNote}</p>
        </div>
      )}
      <p className="mt-3 rounded-lg bg-white/70 p-3 text-lg">
        <strong>{t.results.nextStep}: </strong>
        {tier.next}
      </p>
      <details className="mt-3">
        <summary>{t.results.details}</summary>
        <div className="mt-2 space-y-2 text-base">
          <p>{format(t.results.confidence, { pct: Math.round(p.confidence * 100) })}</p>
          <p className="font-semibold">{t.results.chances}</p>
          <ul className="space-y-1">
            {probs.map(([k, v]) => (
              <li key={k} className="flex items-center gap-2">
                <span className="w-1/2 break-words">{k}</span>
                <span className="font-semibold tabular-nums">{Math.round(v * 100)}%</span>
              </li>
            ))}
          </ul>
          <p className="text-sm">{format(t.results.modelVersion, { v: p.modelVersion })}</p>
        </div>
      </details>
    </section>
  );
}
