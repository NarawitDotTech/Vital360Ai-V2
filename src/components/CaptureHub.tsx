"use client";

import { Camera, ChevronRight, CircleCheck, ClipboardList, Mic, Video } from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { doneModalities } from "@/session/results";
import { useSessionResults } from "@/session/useSession";
import type { Modality } from "@/inference/types";
import { useI18n } from "./I18nProvider";
import { PageTitle } from "./PageTitle";
import { SpeakButton } from "./SpeakButton";

export function CaptureHub() {
  const { locale, t } = useI18n();
  const results = useSessionResults();
  const done = useMemo<Set<Modality>>(() => doneModalities(results ?? []), [results]);

  const items = [
    { id: "skin", label: t.capture.skin, hint: t.capture.skinHint, Icon: Camera },
    { id: "audio", label: t.capture.audio, hint: t.capture.audioHint, Icon: Mic },
    { id: "resp", label: t.capture.resp, hint: t.capture.respHint, Icon: Video },
  ] as const;

  return (
    <div className="flex flex-col gap-5">
      <PageTitle>{t.capture.title}</PageTitle>
      <p>{t.capture.intro}</p>
      <ul className="flex flex-col gap-3">
        {items.map(({ id, label, hint, Icon }) => (
          <li key={id}>
            <Link
              href={`/${locale}/capture/${id}/`}
              className="flex min-h-20 items-center gap-4 rounded-xl border bg-card p-4 text-foreground no-underline shadow-sm transition-colors hover:bg-accent"
            >
              <span aria-hidden="true" className="flex size-11 shrink-0 items-center justify-center rounded-lg bg-accent text-primary">
                <Icon className="size-6" />
              </span>
              <span className="flex min-w-0 flex-1 flex-col">
                <span className="font-semibold">{label}</span>
                <span className="text-sm text-muted-foreground">{hint}</span>
              </span>
              {done.has(id) && (
                <Badge variant="outline" className="border-tier-green-border bg-tier-green-bg text-tier-green-fg">
                  <CircleCheck aria-hidden="true" />
                  {t.capture.done}
                </Badge>
              )}
              <ChevronRight aria-hidden="true" className="size-5 shrink-0 text-muted-foreground" />
            </Link>
          </li>
        ))}
      </ul>
      {done.size > 0 ? (
        <Button asChild size="lg" className="w-full">
          <Link href={`/${locale}/results/`}>
            <ClipboardList aria-hidden="true" />
            {t.capture.viewResults}
          </Link>
        </Button>
      ) : (
        <p className="text-muted-foreground">{t.capture.noneYet}</p>
      )}
      <SpeakButton text={`${t.capture.title}. ${t.capture.intro}`} label={t.capture.speak} />
    </div>
  );
}
