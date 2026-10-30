"use client";

import { ArrowRight } from "lucide-react";
import dynamic from "next/dynamic";
import { Card, CardContent } from "@/components/ui/card";
import type { Prediction } from "@/inference/types";
import { format } from "@/i18n";
import { useI18n } from "./I18nProvider";
import { isMockModel, MockNotice } from "./MockNotice";
import { TierBadge } from "./TierBadge";
import { canNameMatch, className } from "./names";

// The collapsed details (accordion + bars) load after the page is interactive, which keeps the initial JS small.
const ResultDetails = dynamic(() => import("./ResultDetails").then((m) => m.ResultDetails), { ssr: false });

export function resultSpeech(p: Prediction, t: ReturnType<typeof useI18n>["t"]): string {
  const tier = t.tiers[p.tier];
  const parts: string[] = [];
  if (isMockModel(p.modelVersion)) parts.push(`${t.results.mockTitle}. ${t.results.mockText}`);
  parts.push(`${t.results.modality[p.modality]}. ${tier.word}.`, tier.sentence);
  const bpm = p.extras?.breathsPerMin;
  if (bpm !== undefined) parts.push(format(t.results.breaths, { n: Math.round(bpm) }) + ".");
  parts.push(`${t.results.nextStep}: ${tier.next}`, `${t.disclaimer.title}. ${t.disclaimer.text}`, `${t.urgent.title}. ${t.urgent.intro}`);
  return parts.join(" ");
}

/** Summary card: tier badge (icon + word), one sentence, one next step; details collapsed. */
export function ResultCard({ p, headingLevel = 2 }: { p: Prediction; headingLevel?: 2 | 3 }) {
  const { t } = useI18n();
  const tier = t.tiers[p.tier];
  const bpm = p.extras?.breathsPerMin;
  const matchName = canNameMatch(p, t.results) ? className(t.results, p.modality, p.label) : undefined;
  const H = headingLevel === 2 ? "h2" : "h3";
  return (
    <Card aria-labelledby={`tier-${p.modality}`} data-testid={`result-${p.modality}`} data-tier={p.tier} role="region">
      <CardContent className="gap-4">
        <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
          <H id={`tier-${p.modality}`} className="text-lg font-semibold">
            {t.results.modality[p.modality]}
          </H>
          <TierBadge tier={p.tier} size="lg" />
        </div>
        {isMockModel(p.modelVersion) && <MockNotice />}
        <p className="text-lg leading-relaxed font-medium">{tier.sentence}</p>
        {matchName && (
          <div data-testid="similar-match">
            <p className="text-lg font-semibold">{format(t.results.similarTo, { name: matchName, pct: Math.round((p.probs[p.label] ?? 0) * 100) })}</p>
            <p className="text-sm text-muted-foreground">{t.results.similarNote}</p>
          </div>
        )}
        {bpm !== undefined && (
          <div>
            <p className="text-lg font-semibold">{format(t.results.breaths, { n: Math.round(bpm) })}</p>
            <p className="text-sm text-muted-foreground">{t.results.breathsNote}</p>
          </div>
        )}
        <div className="flex gap-3 rounded-lg bg-muted/70 p-3.5">
          <ArrowRight aria-hidden="true" className="mt-1 size-5 shrink-0 text-primary" />
          <p>
            <strong className="font-semibold">{t.results.nextStep}: </strong>
            {tier.next}
          </p>
        </div>
        <ResultDetails p={p} />
      </CardContent>
    </Card>
  );
}
