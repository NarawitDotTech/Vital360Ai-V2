"use client";

import { Loader2, Mic, Square } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useI18n } from "@/components/I18nProvider";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { format } from "@/i18n";
import { checkAmbient, checkRecording } from "@/capture/quality";
import { setSessionResult } from "@/session/results";
import { CaptureSteps } from "./CaptureSteps";
import { ScanBar, Waveform } from "./Motion";

const AMBIENT_SEC = 1.5;
const RECORD_SEC = 10;

export function AudioCapture() {
  const { locale, t } = useI18n();
  const router = useRouter();
  const abortRef = useRef<AbortController | null>(null);
  const primaryRef = useRef<HTMLButtonElement>(null);
  const prevActive = useRef(false);
  const [phase, setPhase] = useState<"idle" | "ambient" | "recording" | "analysing">("idle");
  const [left, setLeft] = useState(RECORD_SEC);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  async function start() {
    setMessage(null);
    setLeft(RECORD_SEC);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        setMessage(t.quality.notAvailable);
        return;
      }
      const { recordMono } = await import("@/capture/audio");
      let ambientIssue: "tooNoisy" | "tooQuiet" | null = null;
      setPhase("ambient");
      const res = await recordMono({
        ambientSec: AMBIENT_SEC,
        recordSec: RECORD_SEC,
        signal: ac.signal,
        onPhase: setPhase,
        onTick: setLeft,
        onAmbient: (a) => {
          ambientIssue = checkAmbient(a);
          return ambientIssue === null;
        },
      });
      if (!res.ok) {
        setMessage(t.quality[ambientIssue ?? "tooNoisy"]);
        setPhase("idle");
        return;
      }
      const issue = checkRecording(res.samples);
      if (issue) {
        setMessage(t.quality[issue]);
        setPhase("idle");
        return;
      }
      setPhase("analysing");
      const { inference } = await import("@/inference");
      const p = await inference.predict({ modality: "audio", samples: res.samples, sampleRate: res.sampleRate });
      setSessionResult(p);
      router.push(`/${locale}/results/`);
    } catch (e) {
      if ((e as Error)?.name === "AbortError") return;
      const denied = (e as Error)?.name === "NotAllowedError" || (e as Error)?.name === "NotFoundError";
      setMessage(denied ? t.audio.micOff : t.common.problem);
      setPhase("idle");
    }
  }

  function stop() {
    abortRef.current?.abort();
    setPhase("idle");
  }

  const active = phase === "ambient" || phase === "recording";
  // the primary button is swapped (start <-> stop): keep keyboard focus on the new one
  useEffect(() => {
    if (prevActive.current !== active) primaryRef.current?.focus();
    prevActive.current = active;
  }, [active]);

  return (
    <div className="flex flex-col gap-5">
      <CaptureSteps title={t.audio.title} steps={[t.audio.step1, t.audio.step2, t.audio.step3]} spoken={t.audio.spoken} />

      {message && (
        <Alert variant="warning" role="alert">
          <AlertDescription className="font-medium">{message}</AlertDescription>
        </Alert>
      )}

      {phase === "ambient" && (
        <Alert role="status">
          <Loader2 aria-hidden="true" className="animate-spin" />
          <AlertDescription>{t.audio.checking}</AlertDescription>
        </Alert>
      )}
      {phase === "recording" && (
        <Card size="sm">
          <CardContent className="gap-3">
            {/* announced at milestones only (start, 5 s left), not every second */}
            <p role="status" className="sr-only">
              {left <= 5 ? format(t.audio.remaining, { n: 5 }) : t.audio.recording}
            </p>
            <p className="flex items-center gap-2 text-lg font-semibold">
              <Mic aria-hidden="true" className="size-5 text-primary" />
              {t.audio.recording}
            </p>
            <Waveform />
            <Progress value={((RECORD_SEC - left) / RECORD_SEC) * 100} aria-label={t.audio.recording} />
            <p className="text-muted-foreground" aria-hidden="true">
              {format(t.audio.remaining, { n: left })}
            </p>
          </CardContent>
        </Card>
      )}
      {phase === "analysing" && (
        <Alert role="status">
          <Loader2 aria-hidden="true" className="animate-spin" />
          <AlertDescription>
            {t.common.working}
            <span className="mt-2 block"><ScanBar /></span>
          </AlertDescription>
        </Alert>
      )}

      {active ? (
        <Button ref={primaryRef} type="button" variant="outline" size="lg" className="w-full" onClick={stop}>
          <Square aria-hidden="true" />
          {t.audio.stop}
        </Button>
      ) : (
        <Button ref={primaryRef} type="button" size="lg" className="w-full" onClick={start} disabled={phase === "analysing"}>
          <Mic aria-hidden="true" />
          {t.audio.start}
        </Button>
      )}
    </div>
  );
}
