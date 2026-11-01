"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import type { Prediction } from "@/inference/types";
import { format } from "@/i18n";
import { useI18n } from "./I18nProvider";
import { alternatives, canNameMatch, className } from "./names";

/** Collapsed "details for the curious": confidence, rough shares per label, model version. */
export function ResultDetails({ p }: { p: Prediction }) {
  const { t } = useI18n();
  const probs = Object.entries(p.probs).sort((a, b) => b[1] - a[1]);
  const alts = canNameMatch(p, t.results) ? alternatives(p, t.results) : [];
  const named = !p.modelVersion.startsWith("mock");
  const shown = named && p.modality === "skin" ? probs.slice(0, 6) : probs;
  return (
    <Accordion type="single" collapsible className="border-t">
      <AccordionItem value="details">
        <AccordionTrigger>{t.results.details}</AccordionTrigger>
        <AccordionContent className="space-y-3">
          <p>
            {format(t.results.confidence, {
              pct: Math.round(p.confidence * 100),
            })}
          </p>
          {alts.length > 0 && (
            <div data-testid="alternatives">
              <p className="font-semibold">{t.results.alternatives}</p>
              <ul className="list-disc ps-5">
                {alts.map((a) => (
                  <li key={a.key}>
                    {a.name}{" "}
                    <span className="font-semibold tabular-nums">{a.pct}%</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="font-semibold">{t.results.chances}</p>
          <ul className="space-y-2.5">
            {shown.map(([k, v]) => (
              <li key={k}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="break-words">
                    {(named && className(t.results, p.modality, k)) ||
                      t.results.labels[k] ||
                      k}
                  </span>
                  <span className="font-semibold tabular-nums">
                    {Math.round(v * 100)}%
                  </span>
                </div>
                <div
                  aria-hidden="true"
                  className="mt-1 h-2 overflow-hidden rounded-full bg-muted"
                >
                  <div
                    className="v-bar h-full rounded-full bg-primary"
                    style={{ width: `${Math.round(v * 100)}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
          <p className="text-sm text-muted-foreground">
            {format(t.results.modelVersion, { v: p.modelVersion })}
          </p>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
