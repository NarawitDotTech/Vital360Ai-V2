"use client";

import { Sparkles } from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { format } from "@/i18n";
import type { Prediction } from "@/inference/types";
import { useI18n } from "./I18nProvider";
import { SpeakButton } from "./SpeakButton";
import { TierIcon } from "./TierBadge";
import { summarize } from "./summary";

const TINT = {
  green: "border-tier-green-border bg-tier-green-bg text-tier-green-fg",
  amber: "border-tier-amber-border bg-tier-amber-bg text-tier-amber-fg",
  red: "border-tier-red-border bg-tier-red-bg text-tier-red-fg",
  uncertain:
    "border-tier-uncertain-border bg-tier-uncertain-bg text-tier-uncertain-fg",
} as const;

export function SummaryCard({
  parts,
  fusion,
}: {
  parts: Prediction[];
  fusion: Prediction | null;
}) {
  const { locale, t } = useI18n();
  const s = useMemo(() => summarize(parts, fusion, t), [parts, fusion, t]);
  return (
    <Card
      className="v-enter border-primary/40"
      role="region"
      aria-labelledby="ai-summary-title"
      data-testid="ai-summary"
      data-tier={s.tier}
    >
      <CardContent className="gap-3">
        <h2
          id="ai-summary-title"
          className="flex items-center gap-2 text-xl font-semibold"
        >
          <Sparkles aria-hidden="true" className="size-6 text-primary" />
          {t.summary.title}
        </h2>
        <p
          className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-lg font-semibold ${TINT[s.tier]}`}
          data-testid="ai-summary-headline"
        >
          <TierIcon tier={s.tier} className="size-6 shrink-0" />
          {s.headline}
        </p>
        <ul className="flex flex-col gap-2 leading-relaxed">
          {s.lines.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
        <p className="font-medium">{s.next}</p>
        {s.missingText ? <p>{s.missingText}</p> : <p>{t.summary.allDone}</p>}
        {s.missing.length > 0 && (
          <div className="flex flex-col gap-2">
            {s.missing.map((m) => (
              <Button key={m} asChild variant="outline" size="lg">
                <Link href={`/${locale}/capture/${m}/`}>
                  {format(t.summary.doCheck, { name: t.results.modality[m] })}
                </Link>
              </Button>
            ))}
          </div>
        )}
        <p className="text-sm text-muted-foreground">{t.summary.note}</p>
        <SpeakButton text={s.speech} label={t.summary.speak} />
      </CardContent>
    </Card>
  );
}
