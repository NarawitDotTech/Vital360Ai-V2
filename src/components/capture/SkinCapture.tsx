"use client";

import { Camera, ImageUp, Loader2, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "@/components/I18nProvider";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
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
  const mountedRef = useRef(false);
  const openingRef = useRef(false);
  const primaryRef = useRef<HTMLButtonElement>(null);
  const prevCameraOn = useRef(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    stopStream(streamRef.current);
    streamRef.current = null;
    setCameraOn(false);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stopCamera();
    };
  }, [stopCamera]);

  // the primary button is swapped (open <-> take photo): keep keyboard focus on the new one
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
      // the <video> element is mounted by the state change above
      requestAnimationFrame(() => {
        const v = videoRef.current;
        if (v) {
          v.srcObject = stream;
          void v.play().catch(() => {});
        }
      });
    } catch {
      if (mountedRef.current) setMessage(t.skin.cameraOff);
    } finally {
      openingRef.current = false;
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
    <div className="flex flex-col gap-5">
      <CaptureSteps title={t.skin.title} steps={[t.skin.step1, t.skin.step2, t.skin.step3]} spoken={t.skin.spoken} />

      {cameraOn && (
        <div className="video-frame" style={{ aspectRatio: "4 / 3" }}>
          <video ref={videoRef} playsInline muted aria-label={t.skin.guide} />
          <div className="guide-box" style={{ left: "20%", top: "15%", width: "60%", height: "70%" }} aria-hidden="true" />
          <p className="guide-label">{t.skin.guide}</p>
        </div>
      )}

      {message && (
        <Alert variant="warning" role="alert">
          <AlertDescription className="font-medium">{message}</AlertDescription>
        </Alert>
      )}
      {busy && (
        <Alert role="status">
          <Loader2 aria-hidden="true" className="animate-spin" />
          <AlertDescription>{t.skin.analyzing}</AlertDescription>
        </Alert>
      )}

      <div className="flex flex-col gap-3">
        {cameraOn ? (
          <Button ref={primaryRef} type="button" size="lg" className="w-full" onClick={takePhoto} disabled={busy}>
            <Camera aria-hidden="true" />
            {t.skin.takePhoto}
          </Button>
        ) : (
          <Button ref={primaryRef} type="button" size="lg" className="w-full" onClick={startCamera} disabled={busy}>
            <Camera aria-hidden="true" />
            {t.skin.openCamera}
          </Button>
        )}
        <div className="grid gap-3 sm:grid-cols-2">
          {cameraOn && (
            <Button type="button" variant="outline" size="lg" onClick={stopCamera} disabled={busy}>
              <X aria-hidden="true" />
              {t.skin.stopCamera}
            </Button>
          )}
          <Button type="button" variant="outline" size="lg" onClick={() => fileRef.current?.click()} disabled={busy}>
            <ImageUp aria-hidden="true" />
            {t.skin.chooseFile}
          </Button>
        </div>
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
