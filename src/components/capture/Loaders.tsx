"use client";

import dynamic from "next/dynamic";

// Capture code (camera, audio, canvas) is only downloaded on the capture screens.
const Loading = () => <p role="status">…</p>;

export const SkinCaptureLoader = dynamic(
  () => import("./SkinCapture").then((m) => m.SkinCapture),
  { ssr: false, loading: Loading },
);
export const AudioCaptureLoader = dynamic(
  () => import("./AudioCapture").then((m) => m.AudioCapture),
  { ssr: false, loading: Loading },
);
export const RespCaptureLoader = dynamic(
  () => import("./RespCapture").then((m) => m.RespCapture),
  { ssr: false, loading: Loading },
);
