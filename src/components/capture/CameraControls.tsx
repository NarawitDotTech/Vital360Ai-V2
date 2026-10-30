"use client";

import { SwitchCamera, Zap, ZapOff } from "lucide-react";
import { useEffect, useState } from "react";
import { useI18n } from "@/components/I18nProvider";
import { countVideoInputs, getTorchSupport, setTorch } from "@/capture/camera";

const BTN =
  "inline-flex size-12 items-center justify-center rounded-full bg-black/60 text-white outline-none " +
  "hover:bg-black/75 focus-visible:ring-[3px] focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-black " +
  "disabled:opacity-40";

/** Flip + flash buttons overlaid bottom-right of a .video-frame. */
export function CameraControls({
  stream,
  facing,
  onFlip,
  flipDisabled = false,
}: {
  stream: MediaStream | null;
  facing: "environment" | "user";
  onFlip: () => void;
  flipDisabled?: boolean;
}) {
  const { t } = useI18n();
  const [multi, setMulti] = useState(false);
  // torch state belongs to one stream; a new stream (switch/stop) starts fresh and off
  const [torch, setTorchState] = useState<{ stream: MediaStream | null; on: boolean; failed: boolean }>({
    stream: null,
    on: false,
    failed: false,
  });
  const mine = torch.stream === stream;
  const torchOn = mine && torch.on;
  const torchOk = facing === "environment" && !(mine && torch.failed) && getTorchSupport(stream);

  useEffect(() => {
    let alive = true;
    void countVideoInputs().then((n) => alive && setMulti(n > 1));
    return () => {
      alive = false;
    };
  }, [stream]);

  async function toggleTorch() {
    const next = !torchOn;
    const ok = await setTorch(stream, next);
    setTorchState({ stream, on: ok && next, failed: !ok });
  }

  if (!multi && !torchOk) return null;
  return (
    <div className="absolute bottom-3 right-3 z-10 flex gap-2">
      {torchOk && (
        <button type="button" className={BTN} aria-label={t.camera.torch} aria-pressed={torchOn} onClick={toggleTorch}>
          {torchOn ? <Zap aria-hidden="true" className="size-6" /> : <ZapOff aria-hidden="true" className="size-6" />}
        </button>
      )}
      {multi && (
        <button type="button" className={BTN} aria-label={t.camera.switch} onClick={onFlip} disabled={flipDisabled}>
          <SwitchCamera aria-hidden="true" className="size-6" />
        </button>
      )}
    </div>
  );
}
