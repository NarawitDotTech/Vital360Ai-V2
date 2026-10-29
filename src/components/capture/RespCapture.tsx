"use client";

import { Loader2, Video, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "@/components/I18nProvider";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { format } from "@/i18n";
import { openCamera, stopStream } from "@/capture/camera";
import { checkMotionSignal } from "@/capture/quality";
import { CHEST_BOX } from "@/capture/video";
import { setSessionResult } from "@/session/results";
import { CaptureSteps } from "./CaptureSteps";

const SECONDS = 20;

export function RespCapture() {
  const { locale, t } = useI18n();
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(false);
  const openingRef = useRef(false);
  const primaryRef = useRef<HTMLButtonElement>(null);
  const prevCameraOn = useRef(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [aspect, setAspect] = useState(4 / 3);
  const [phase, setPhase] = useState<"idle" | "measuring" | "analysing">("idle");
  const [left, setLeft] = useState(SECONDS);
  const [message, setMessage] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    abortRef.current?.abort();
    stopStream(streamRef.current);
    streamRef.current = null;
    setCameraOn(false);
    setPhase("idle");
  }, []);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stopCamera();
    };
  }, [stopCamera]);

  // the primary button is swapped (open <-> start): keep keyboard focus on the new one
  useEffect(() => {
    if (prevCameraOn.current !== cameraOn) primaryRef.current?.focus();
    prevCameraOn.current = cameraOn;
  }, [cameraOn]);

  async function startCamera() {
    if (streamRef.current || openingRef.current) return;
    openingRef.current = true;
    setMessage(null);
    try {
      const stream = await openCamera("environment");
      if (!mountedRef.current || streamRef.current) {
        stopStream(stream); // unmounted during the permission prompt, or a second stream already exists
        return;
      }
      streamRef.current = stream;
      setCameraOn(true);
      requestAnimationFrame(() => {
        const v = videoRef.current;
        if (!v) return;
        v.srcObject = stream;
        v.onloadedmetadata = () => v.videoWidth && setAspect(v.videoWidth / v.videoHeight);
        void v.play().catch(() => {});
      });
    } catch {
      if (mountedRef.current) setMessage(t.resp.cameraOff);
    } finally {
      openingRef.current = false;
    }
  }

  async function measure() {
    const v = videoRef.current;
    if (!v || !v.videoWidth) {
      setMessage(t.common.problem);
      return;
    }
    setMessage(null);
    setLeft(SECONDS);
    setPhase("measuring");
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const [{ collectMotion }, { inference, frameMotion }] = await Promise.all([
        import("@/capture/video"),
        import("@/inference"),
      ]);
      const { signal, fps } = await collectMotion({ video: v, seconds: SECONDS, frameMotion, onTick: setLeft, signal: ac.signal });
      const issue = checkMotionSignal(signal, fps);
      if (issue) {
        setMessage(t.quality[issue]);
        setPhase("idle");
        return;
      }
      setPhase("analysing");
      const p = await inference.predict({ modality: "resp", signal, fps });
      setSessionResult(p);
      stopCamera();
      router.push(`/${locale}/results/`);
    } catch (e) {
      if ((e as Error)?.name === "AbortError") return;
      setMessage(t.common.problem);
      setPhase("idle");
    }
  }

  const box = { left: `${CHEST_BOX.x * 100}%`, top: `${CHEST_BOX.y * 100}%`, width: `${CHEST_BOX.w * 100}%`, height: `${CHEST_BOX.h * 100}%` };

  return (
    <div className="flex flex-col gap-5">
      <CaptureSteps title={t.resp.title} steps={[t.resp.step1, t.resp.step2, t.resp.step3]} spoken={t.resp.spoken} />

      {cameraOn && (
        <div className="video-frame" style={{ aspectRatio: String(aspect), maxHeight: "65dvh" }}>
          <video ref={videoRef} playsInline muted aria-label={t.resp.guide} />
          <div className="guide-box" style={box} aria-hidden="true" />
          <p className="guide-label">{t.resp.guide}</p>
        </div>
      )}

      {message && (
        <Alert variant="warning" role="alert">
          <AlertDescription className="font-medium">{message}</AlertDescription>
        </Alert>
      )}

      {phase === "measuring" && (
        <Card size="sm">
          <CardContent className="gap-3">
            {/* announced at milestones only (start, 10 s, 5 s), not every second */}
            <p role="status" className="sr-only">
              {left <= 5 ? format(t.resp.remaining, { n: 5 }) : left <= 10 ? format(t.resp.remaining, { n: 10 }) : t.resp.measuring}
            </p>
            <p className="flex items-center gap-2 text-lg font-semibold">
              <Video aria-hidden="true" className="size-5 text-primary" />
              {t.resp.measuring}
            </p>
            <Progress value={((SECONDS - left) / SECONDS) * 100} aria-label={t.resp.measuring} />
            <p className="text-muted-foreground" aria-hidden="true">
              {format(t.resp.remaining, { n: left })}
            </p>
          </CardContent>
        </Card>
      )}
      {phase === "analysing" && (
        <Alert role="status">
          <Loader2 aria-hidden="true" className="animate-spin" />
          <AlertDescription>{t.common.working}</AlertDescription>
        </Alert>
      )}

      {cameraOn ? (
        <div className="flex flex-col gap-3">
          <Button ref={primaryRef} type="button" size="lg" className="w-full" onClick={measure} disabled={phase !== "idle"}>
            <Video aria-hidden="true" />
            {t.resp.start}
          </Button>
          <Button type="button" variant="outline" size="lg" onClick={stopCamera}>
            <X aria-hidden="true" />
            {t.resp.stopCamera}
          </Button>
        </div>
      ) : (
        <Button ref={primaryRef} type="button" size="lg" className="w-full" onClick={startCamera}>
          <Video aria-hidden="true" />
          {t.resp.openCamera}
        </Button>
      )}
    </div>
  );
}
