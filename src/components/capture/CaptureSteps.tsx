"use client";

import { ChevronLeft } from "lucide-react";
import Link from "next/link";
import { useI18n } from "@/components/I18nProvider";
import { PageTitle } from "@/components/PageTitle";
import { SpeakButton } from "@/components/SpeakButton";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

/** Shared header for the capture screens: back link, title, 3 numbered steps (an actual sequence), spoken prompt. */
export function CaptureSteps({ title, steps, spoken }: { title: string; steps: string[]; spoken: string }) {
  const { locale, t } = useI18n();
  return (
    <>
      <Button asChild variant="ghost" size="sm" className="-ml-3 self-start">
        <Link href={`/${locale}/capture/`}>
          <ChevronLeft aria-hidden="true" />
          {t.common.back}
        </Link>
      </Button>
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
