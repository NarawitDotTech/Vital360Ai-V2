"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "@/components/I18nProvider";
import { fileToImageData, openCamera, stopStream, toImageData } from "@/capture/camera";
import { checkImage } from "@/capture/quality";
import { setSessionResult } from "@/session/results";
import { CaptureSteps } from "./CaptureSteps";

export function SkinCapture() {
  const { locale, t } = useI18n();
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    stopStream(streamRef.current);
    streamRef.current = null;
    setCameraOn(false);
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  async function startCamera() {
    setMessage(null);
    try {
      const stream = await openCamera("environment");
      streamRef.current = stream;
      setCameraOn(true);
      // the <video> element is mounted by the state change above
      requestAnimationFrame(() => {
        const v = videoRef.current;
        if (v) {
          v.srcObject = stream;
          void v.play().catch(() => {});
        }
      });
    } catch {
      setMessage(t.skin.cameraOff);
    }
  }

  async function analyse(img: ImageData) {
    const issue = checkImage(img);
    if (issue) {
      setMessage(t.quality[issue]);
      setBusy(false);
      return;
    }
    try {
      const { inference } = await import("@/inference");
      const p = await inference.predict({ modality: "skin", image: img });
      setSessionResult(p);
      stopCamera();
      router.push(`/${locale}/results/`);
    } catch {
      setMessage(t.common.problem);
      setBusy(false);
    }
  }

  async function takePhoto() {
    const v = videoRef.current;
    if (!v || !v.videoWidth) {
      setMessage(t.common.problem);
      return;
    }
    setBusy(true);
    setMessage(null);
    await analyse(toImageData(v, v.videoWidth, v.videoHeight));
  }

  async function onFile(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setMessage(null);
    let img: ImageData;
    try {
      img = await fileToImageData(file);
    } catch {
      setMessage(t.skin.badFile);
      setBusy(false);
      return;
    }
    await analyse(img);
  }

  return (
    <div className="stack">
      <CaptureSteps title={t.skin.title} steps={[t.skin.step1, t.skin.step2, t.skin.step3]} spoken={t.skin.spoken} />

      {cameraOn && (
        <div className="video-frame" style={{ aspectRatio: "4 / 3" }}>
          <video ref={videoRef} playsInline muted aria-label={t.skin.guide} />
          <div className="guide-box" style={{ left: "20%", top: "15%", width: "60%", height: "70%" }} aria-hidden="true" />
          <p className="guide-label">{t.skin.guide}</p>
        </div>
      )}

      {message && (
        <p role="alert" className="notice notice-warn text-lg font-semibold">
          {message}
        </p>
      )}
      {busy && <p role="status">{t.skin.analyzing}</p>}

      <div className="stack-sm">
        {cameraOn ? (
          <>
            <button type="button" className="btn btn-primary btn-big" onClick={takePhoto} disabled={busy}>
              <span aria-hidden="true">📷</span> {t.skin.takePhoto}
            </button>
            <button type="button" className="btn btn-secondary" onClick={stopCamera} disabled={busy}>
              {t.skin.stopCamera}
            </button>
          </>
        ) : (
          <button type="button" className="btn btn-primary btn-big" onClick={startCamera} disabled={busy}>
            <span aria-hidden="true">📷</span> {t.skin.openCamera}
          </button>
        )}
        <button type="button" className="btn btn-secondary" onClick={() => fileRef.current?.click()} disabled={busy}>
          {t.skin.chooseFile}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          capture="environment"
          hidden
          aria-label={t.skin.fileLabel}
          data-testid="skin-file"
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            void onFile(f);
          }}
        />
      </div>
    </div>
  );
}
