// Browser camera / canvas helpers (client only).

export async function openCamera(facing: "environment" | "user" = "environment"): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("no getUserMedia");
  return navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 960 } },
    audio: false,
  });
}

export function stopStream(stream: MediaStream | null | undefined): void {
  if (!stream) return;
  const stop = () => stream.getTracks().forEach((t) => t.stop());
  // Some Android phones keep the flash lit after track.stop(): switch it off explicitly first.
  if (getTorchSupport(stream)) void setTorch(stream, false).finally(stop);
  else stop();
}

export type Facing = "environment" | "user";

type TorchCaps = MediaTrackCapabilities & { torch?: boolean };

/** True when the stream's video track reports torch (flash) support. */
export function getTorchSupport(stream: MediaStream | null | undefined): boolean {
  const track = stream?.getVideoTracks()[0];
  if (!track || typeof track.getCapabilities !== "function") return false;
  return (track.getCapabilities() as TorchCaps).torch === true;
}

/** Turn the torch on/off. Resolves false (never throws) when the device refuses. */
export async function setTorch(stream: MediaStream | null | undefined, on: boolean): Promise<boolean> {
  const track = stream?.getVideoTracks()[0];
  if (!track) return false;
  try {
    await track.applyConstraints({ advanced: [{ torch: on } as MediaTrackConstraintSet] });
    return true;
  } catch {
    return false;
  }
}

/** Number of cameras; only meaningful after camera permission is granted. */
export async function countVideoInputs(): Promise<number> {
  try {
    const list = await navigator.mediaDevices.enumerateDevices();
    return list.filter((d) => d.kind === "videoinput").length;
  } catch {
    return 0;
  }
}

/** Draw a source to a canvas no larger than maxSide and return its pixels. Photos never leave memory. */
export function toImageData(
  src: CanvasImageSource,
  srcW: number,
  srcH: number,
  maxSide = 640,
): ImageData {
  const scale = Math.min(1, maxSide / Math.max(srcW, srcH));
  const w = Math.max(1, Math.round(srcW * scale));
  const h = Math.max(1, Math.round(srcH * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("no 2d context");
  ctx.drawImage(src, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

export async function fileToImageData(file: File, maxSide = 640): Promise<ImageData> {
  const bmp = await createImageBitmap(file);
  try {
    return toImageData(bmp, bmp.width, bmp.height, maxSide);
  } finally {
    bmp.close();
  }
}
