"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useI18n } from "@/components/I18nProvider";
import { format } from "@/i18n";
import { checkAmbient, checkRecording } from "@/capture/quality";
import { setSessionResult } from "@/session/results";
import { CaptureSteps } from "./CaptureSteps";

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
    <div className="stack">
      <CaptureSteps title={t.audio.title} steps={[t.audio.step1, t.audio.step2, t.audio.step3]} spoken={t.audio.spoken} />

      {message && (
        <p role="alert" className="notice notice-warn text-lg font-semibold">
          {message}
        </p>
      )}

      {phase === "ambient" && <p role="status" className="notice">{t.audio.checking}</p>}
      {phase === "recording" && (
        <div className="card card-tinted stack-sm">
          {/* announced at milestones only (start, 5 s left), not every second */}
          <p role="status" className="sr-only">
            {left <= 5 ? format(t.audio.remaining, { n: 5 }) : t.audio.recording}
          </p>
          <p className="text-xl font-bold">
            <span aria-hidden="true">🎤 </span>
            {t.audio.recording}
          </p>
          <progress max={RECORD_SEC} value={RECORD_SEC - left} aria-label={t.audio.recording} />
          <p className="text-lg" aria-hidden="true">{format(t.audio.remaining, { n: left })}</p>
        </div>
      )}
      {phase === "analysing" && <p role="status">{t.common.working}</p>}

      {active ? (
        <button ref={primaryRef} type="button" className="btn btn-secondary" onClick={stop}>
          {t.audio.stop}
        </button>
      ) : (
        <button ref={primaryRef} type="button" className="btn btn-primary btn-big" onClick={start} disabled={phase === "analysing"}>
          <span aria-hidden="true">🎤</span> {t.audio.start}
        </button>
      )}
    </div>
  );
}
