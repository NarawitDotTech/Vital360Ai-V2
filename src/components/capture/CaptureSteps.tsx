"use client";

import { SpeakButton } from "@/components/SpeakButton";
import { useI18n } from "@/components/I18nProvider";
import Link from "next/link";

/** Shared header for the capture screens: title, 3 numbered steps (an actual sequence), spoken prompt, back link. */
export function CaptureSteps({ title, steps, spoken }: { title: string; steps: string[]; spoken: string }) {
  const { locale, t } = useI18n();
  return (
    <>
      <Link href={`/${locale}/capture/`} className="inline-flex min-h-12 items-center">
        ← {t.common.back}
      </Link>
      <h1>{title}</h1>
      <ol className="card card-tinted list-decimal space-y-2 pl-9 text-lg">
        {steps.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ol>
      <SpeakButton text={spoken} label={t.capture.speak} />
    </>
  );
}
