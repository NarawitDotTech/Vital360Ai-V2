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

/** Longest side of the square canvas handed to the model preprocessing (skin-preprocess.ts then area-resizes it to 224). */
export const SQUARE_MAX = 448;
/** On-screen guide box: a centred square whose side is this fraction of the visible frame height (see SkinCapture + globals.css). */
export const GUIDE_HEIGHT_FRAC = 0.7;
/** The camera preview container is 4:3 and the <video> uses object-fit: cover. */
export const FRAME_ASPECT = 4 / 3;

export interface SquareRect {
  x: number;
  y: number;
  s: number;
}

/** Centred square covering the whole short side (file uploads). Same rule as training (training/cpu/skin_pipe.py center_square). */
export function centerSquare(w: number, h: number): SquareRect {
  const s = Math.max(1, Math.min(w, h));
  return { x: Math.floor((w - s) / 2), y: Math.floor((h - s) / 2), s };
}

/**
 * Square of video pixels that sits under the drawn guide box. The preview is object-fit: cover in a frameAspect container, so only
 * part of the video is visible when the stream is not exactly frameAspect; the guide is centred in the visible part.
 */
export function guideRect(videoW: number, videoH: number, frameAspect = FRAME_ASPECT, heightFrac = GUIDE_HEIGHT_FRAC): SquareRect {
  const visibleH = videoW / videoH > frameAspect ? videoH : videoW / frameAspect;
  const s = Math.max(1, Math.min(Math.round(visibleH * heightFrac), videoW, videoH));
  return { x: Math.floor((videoW - s) / 2), y: Math.floor((videoH - s) / 2), s };
}

/**
 * Copy the square `r` of a source into an ImageData of side min(r.s, maxSide). Large crops are shrunk in steps of at most 2x
 * (a single big canvas downscale aliases in some browsers), so the result is close to a box average like the Python training pipeline.
 * Transparent pixels (PNG) are composited on white.
 */
export function cropSquare(src: CanvasImageSource, r: SquareRect, maxSide = SQUARE_MAX): ImageData {
  const target = Math.max(1, Math.min(r.s, maxSide));
  let cur: CanvasImageSource = src;
  let cx = r.x;
  let cy = r.y;
  let cs = r.s;
  const draw = (side: number): HTMLCanvasElement => {
    const canvas = document.createElement("canvas");
    canvas.width = side;
    canvas.height = side;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("no 2d context");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, side, side);
    ctx.drawImage(cur, cx, cy, cs, cs, 0, 0, side, side);
    return canvas;
  };
  while (cs > 2 * target) {
    const half = Math.ceil(cs / 2);
    cur = draw(half);
    cx = 0;
    cy = 0;
    cs = half;
  }
  const out = draw(target);
  return out.getContext("2d", { willReadFrequently: true })!.getImageData(0, 0, target, target);
}

/** Camera frame -> the part under the guide box. Photos never leave memory. */
export function videoToGuideImage(v: HTMLVideoElement): ImageData {
  return cropSquare(v, guideRect(v.videoWidth, v.videoHeight));
}

/** Uploaded photo -> centred square crop (EXIF orientation applied by createImageBitmap). */
export async function fileToImageData(file: File): Promise<ImageData> {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  try {
    return cropSquare(bmp, centerSquare(bmp.width, bmp.height));
  } finally {
    bmp.close();
  }
}
