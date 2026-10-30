"use client";

import { useI18n } from "@/components/I18nProvider";
import { PageTitle } from "@/components/PageTitle";
import { SpeakButton } from "@/components/SpeakButton";
import { Card, CardContent } from "@/components/ui/card";

/** Shared header for the capture screens: title, 3 numbered steps (an actual sequence), spoken prompt. */
export function CaptureSteps({ title, steps, spoken }: { title: string; steps: string[]; spoken: string }) {
  const { t } = useI18n();
  return (
    <>
      <PageTitle>{title}</PageTitle>
      <Card size="sm">
        <CardContent>
          <ol role="list" className="flex flex-col gap-3">
            {steps.map((s, i) => (
              <li key={s} className="flex items-start gap-3">
                <span
                  aria-hidden="true"
                  className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-semibold text-primary-foreground"
                >
                  {i + 1}
                </span>
                <span>{s}</span>
              </li>
            ))}
          </ol>
        </CardContent>
      </Card>
      <SpeakButton text={spoken} label={t.capture.speak} />
    </>
  );
}
