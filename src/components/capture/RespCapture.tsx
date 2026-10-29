"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "@/components/I18nProvider";
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
    <div className="stack">
      <CaptureSteps title={t.resp.title} steps={[t.resp.step1, t.resp.step2, t.resp.step3]} spoken={t.resp.spoken} />

      {cameraOn && (
        <div className="video-frame" style={{ aspectRatio: String(aspect), maxHeight: "65dvh" }}>
          <video ref={videoRef} playsInline muted aria-label={t.resp.guide} />
          <div className="guide-box" style={box} aria-hidden="true" />
          <p className="guide-label">{t.resp.guide}</p>
        </div>
      )}

      {message && (
        <p role="alert" className="notice notice-warn text-lg font-semibold">
          {message}
        </p>
      )}

      {phase === "measuring" && (
        <div className="card card-tinted stack-sm">
          {/* announced at milestones only (start, 10 s, 5 s), not every second */}
          <p role="status" className="sr-only">
            {left <= 5 ? format(t.resp.remaining, { n: 5 }) : left <= 10 ? format(t.resp.remaining, { n: 10 }) : t.resp.measuring}
          </p>
          <p className="text-xl font-bold">{t.resp.measuring}</p>
          <progress max={SECONDS} value={SECONDS - left} aria-label={t.resp.measuring} />
          <p className="text-lg" aria-hidden="true">{format(t.resp.remaining, { n: left })}</p>
        </div>
      )}
      {phase === "analysing" && <p role="status">{t.common.working}</p>}

      {cameraOn ? (
        <>
          <button ref={primaryRef} type="button" className="btn btn-primary btn-big" onClick={measure} disabled={phase !== "idle"}>
            <span aria-hidden="true">🎥</span> {t.resp.start}
          </button>
          <button type="button" className="btn btn-secondary" onClick={stopCamera}>
            {t.resp.stopCamera}
          </button>
        </>
      ) : (
        <button ref={primaryRef} type="button" className="btn btn-primary btn-big" onClick={startCamera}>
          <span aria-hidden="true">🎥</span> {t.resp.openCamera}
        </button>
      )}
    </div>
  );
}
