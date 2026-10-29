"use client";

import Link from "next/link";
import { useMemo } from "react";
import { doneModalities } from "@/session/results";
import { useSessionResults } from "@/session/useSession";
import type { Modality } from "@/inference/types";
import { useI18n } from "./I18nProvider";
import { SpeakButton } from "./SpeakButton";

export function CaptureHub() {
  const { locale, t } = useI18n();
  const results = useSessionResults();
  const done = useMemo<Set<Modality>>(() => doneModalities(results ?? []), [results]);

  const items = [
    { id: "skin", label: t.capture.skin, hint: t.capture.skinHint, icon: "📷" },
    { id: "audio", label: t.capture.audio, hint: t.capture.audioHint, icon: "🎤" },
    { id: "resp", label: t.capture.resp, hint: t.capture.respHint, icon: "🎥" },
  ] as const;

  return (
    <div className="stack">
      <h1>{t.capture.title}</h1>
      <p>{t.capture.intro}</p>
      <ul className="stack-sm">
        {items.map((it) => (
          <li key={it.id}>
            <Link href={`/${locale}/capture/${it.id}/`} className="btn btn-primary btn-big !justify-start !text-left">
              <span aria-hidden="true">{it.icon}</span>
              <span className="flex flex-col">
                <span>{it.label}</span>
                <span className="text-base font-normal">{it.hint}</span>
              </span>
              {done.has(it.id) && <span className="ml-auto rounded-lg bg-white px-2 py-1 text-base text-brand-dark">✓ {t.capture.done}</span>}
            </Link>
          </li>
        ))}
      </ul>
      {done.size > 0 ? (
        <Link href={`/${locale}/results/`} className="btn btn-secondary btn-big">
          {t.capture.viewResults}
        </Link>
      ) : (
        <p className="text-muted">{t.capture.noneYet}</p>
      )}
      <SpeakButton text={`${t.capture.title}. ${t.capture.intro}`} label={t.capture.speak} />
    </div>
  );
}
