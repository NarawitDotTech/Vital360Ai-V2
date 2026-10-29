"use client";

import { Check, History, Loader2, RotateCcw, Save } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import type { Prediction } from "@/inference/types";
import { sessionParts } from "@/session/results";
import { useSessionResults } from "@/session/useSession";
import { addHistory } from "@/storage/db";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Disclaimer } from "./Disclaimer";
import { useI18n } from "./I18nProvider";
import { PageTitle } from "./PageTitle";
import { ResultCard, resultSpeech } from "./ResultCard";
import { SpeakButton } from "./SpeakButton";
import { UrgentGuide } from "./UrgentGuide";

export function ResultsView() {
  const { locale, t } = useI18n();
  const results = useSessionResults();
  const parts = useMemo(() => (results === null ? null : sessionParts(results)), [results]);
  // fusion result tagged with the modalities it was computed from, so stale results are never shown
  const key = parts ? parts.map((p) => p.modality).join(",") : "";
  const [fusionState, setFusionState] = useState<{ key: string; value: Prediction | null } | null>(null);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");

  useEffect(() => {
    if (!parts || parts.length < 2) return;
    let cancelled = false;
    import("@/inference")
      .then(({ inference }) => inference.predict({ modality: "fusion", parts }))
      .then((f) => !cancelled && setFusionState({ key, value: f }))
      .catch(() => !cancelled && setFusionState({ key, value: null }));
    return () => {
      cancelled = true;
    };
  }, [parts, key]);

  const fusing = !!parts && parts.length >= 2 && fusionState?.key !== key;
  const fusion = fusionState?.key === key ? fusionState.value : null;

  if (parts === null) return <p role="status">{t.common.loading}</p>;

  if (parts.length === 0) {
    return (
      <div className="flex flex-col gap-5">
        <PageTitle>{t.results.title}</PageTitle>
        <p>{t.results.empty}</p>
        <Button asChild size="lg" className="w-full">
          <Link href={`/${locale}/capture/`}>{t.results.startCheck}</Link>
        </Button>
      </div>
    );
  }

  const primary = fusion ?? parts[parts.length - 1];
  const shown = fusion ? [fusion, ...parts] : parts;

  async function save() {
    setSaveState("saving");
    try {
      for (const p of shown) {
        await addHistory({
          locale,
          modality: p.modality,
          tier: p.tier,
          label: p.label,
          confidence: p.confidence,
          probs: p.probs,
          modelVersion: p.modelVersion,
          ...(p.extras?.breathsPerMin !== undefined ? { breathsPerMin: p.extras.breathsPerMin } : {}),
        });
      }
      setSaveState("saved");
    } catch {
      setSaveState("error");
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <PageTitle>{t.results.title}</PageTitle>

      {fusing && (
        <Alert role="status">
          <Loader2 aria-hidden="true" className="animate-spin" />
          <AlertDescription>{t.results.combinedPending}</AlertDescription>
        </Alert>
      )}
      {fusion && <p className="text-lg font-semibold">{t.results.combinedTitle}</p>}
      <ResultCard p={primary} />
      <SpeakButton text={resultSpeech(primary, t)} />

      {fusion && (
        <section className="flex flex-col gap-3" aria-labelledby="each-title">
          <h2 id="each-title" className="text-xl font-semibold">
            {t.results.eachTitle}
          </h2>
          {parts.map((p) => (
            <ResultCard key={p.modality} p={p} headingLevel={3} />
          ))}
        </section>
      )}

      <UrgentGuide />
      <Disclaimer />

      <div className="flex flex-col gap-3">
        <Button type="button" size="lg" className="w-full" onClick={save} disabled={saveState === "saving" || saveState === "saved"}>
          {saveState === "saved" ? <Check aria-hidden="true" /> : <Save aria-hidden="true" />}
          {saveState === "saved" ? t.results.saved : t.results.save}
        </Button>
        {saveState === "error" && (
          <Alert variant="destructive" role="alert">
            <AlertDescription>{t.results.saveFailed}</AlertDescription>
          </Alert>
        )}
        {saveState === "saved" && (
          <p role="status" className="sr-only">
            {t.results.saved}
          </p>
        )}
        <div className="grid gap-3 sm:grid-cols-2">
          <Button asChild variant="outline" size="lg">
            <Link href={`/${locale}/capture/`}>
              <RotateCcw aria-hidden="true" />
              {t.results.again}
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg">
            <Link href={`/${locale}/history/`}>
              <History aria-hidden="true" />
              {t.results.viewHistory}
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
