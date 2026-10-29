"use client";

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import type { Prediction } from "@/inference/types";
import { format } from "@/i18n";
import { useI18n } from "./I18nProvider";

/** Collapsed "details for the curious": confidence, rough shares per label, model version. */
export function ResultDetails({ p }: { p: Prediction }) {
  const { t } = useI18n();
  const probs = Object.entries(p.probs).sort((a, b) => b[1] - a[1]);
  return (
    <Accordion type="single" collapsible className="border-t">
      <AccordionItem value="details">
        <AccordionTrigger>{t.results.details}</AccordionTrigger>
        <AccordionContent className="space-y-3">
          <p>{format(t.results.confidence, { pct: Math.round(p.confidence * 100) })}</p>
          <p className="font-semibold">{t.results.chances}</p>
          <ul className="space-y-2.5">
            {probs.map(([k, v]) => (
              <li key={k}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="break-words">{t.results.labels[k] ?? k}</span>
                  <span className="font-semibold tabular-nums">{Math.round(v * 100)}%</span>
                </div>
                <div aria-hidden="true" className="mt-1 h-2 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full bg-primary" style={{ width: `${Math.round(v * 100)}%` }} />
                </div>
              </li>
            ))}
          </ul>
          <p className="text-sm text-muted-foreground">{format(t.results.modelVersion, { v: p.modelVersion })}</p>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
