"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import type { Prediction } from "@/inference/types";
import { sessionParts } from "@/session/results";
import { useSessionResults } from "@/session/useSession";
import { addHistory } from "@/storage/db";
import { Disclaimer } from "./Disclaimer";
import { useI18n } from "./I18nProvider";
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
      <div className="stack">
        <h1>{t.results.title}</h1>
        <p>{t.results.empty}</p>
        <Link href={`/${locale}/capture/`} className="btn btn-primary btn-big">
          {t.results.startCheck}
        </Link>
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
    <div className="stack">
      <h1>{t.results.title}</h1>

      {fusing && <p role="status" className="notice">{t.results.combinedPending}</p>}
      {fusion && <p className="text-lg font-semibold">{t.results.combinedTitle}</p>}
      <ResultCard p={primary} />
      <SpeakButton text={resultSpeech(primary, t)} />

      {fusion && (
        <section className="stack-sm" aria-labelledby="each-title">
          <h2 id="each-title">{t.results.eachTitle}</h2>
          {parts.map((p) => (
            <ResultCard key={p.modality} p={p} headingLevel={3} />
          ))}
        </section>
      )}

      <UrgentGuide />
      <Disclaimer />

      <div className="stack-sm">
        <button type="button" className="btn btn-primary" onClick={save} disabled={saveState === "saving" || saveState === "saved"}>
          {saveState === "saved" ? `✓ ${t.results.saved}` : t.results.save}
        </button>
        {saveState === "error" && (
          <p role="alert" className="notice notice-danger">
            {t.results.saveFailed}
          </p>
        )}
        {saveState === "saved" && (
          <p role="status" className="sr-only">
            {t.results.saved}
          </p>
        )}
        <Link href={`/${locale}/capture/`} className="btn btn-secondary">
          {t.results.again}
        </Link>
        <Link href={`/${locale}/history/`} className="btn btn-secondary">
          {t.results.viewHistory}
        </Link>
      </div>
    </div>
  );
}
