// Chest-video motion signal collection. Client only.

/** Normalised (0-1) chest box shown as the framing guide and used as the ROI. */
export const CHEST_BOX = { x: 0.2, y: 0.35, w: 0.6, h: 0.4 } as const;

export type MotionFn = (
  prev: ImageData,
  cur: ImageData,
  box: { x: number; y: number; w: number; h: number },
) => number;

export interface CollectOptions {
  video: HTMLVideoElement;
  seconds: number;
  targetFps?: number;
  frameMotion: MotionFn;
  onTick: (secondsLeft: number) => void;
  signal: AbortSignal;
}

/** Per-frame ROI motion signal plus the measured fps. Frames are downscaled to ~160 px and discarded. */
export function collectMotion(o: CollectOptions): Promise<{ signal: Float32Array; fps: number }> {
  const { video, seconds, frameMotion } = o;
  const interval = 1000 / (o.targetFps ?? 15);
  const W = 160;
  const H = Math.max(1, Math.round((W * (video.videoHeight || 3)) / (video.videoWidth || 4)));
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const box = { x: CHEST_BOX.x * W, y: CHEST_BOX.y * H, w: CHEST_BOX.w * W, h: CHEST_BOX.h * H };

  return new Promise((resolve, reject) => {
    if (!ctx) return reject(new Error("no 2d context"));
    const values: number[] = [];
    let prev: ImageData | null = null;
    let start = 0;
    let last = -Infinity;
    let lastLeft = -1;
    let raf = 0;

    const onAbort = () => {
      cancelAnimationFrame(raf);
      reject(new DOMException("aborted", "AbortError"));
    };
    if (o.signal.aborted) return onAbort();
    o.signal.addEventListener("abort", onAbort, { once: true });

    const step = (now: number) => {
      if (!start) start = now;
      const elapsed = (now - start) / 1000;
      if (now - last >= interval) {
        last = now;
        ctx.drawImage(video, 0, 0, W, H);
        const cur = ctx.getImageData(0, 0, W, H);
        if (prev) values.push(frameMotion(prev, cur, box));
        prev = cur;
      }
      const left = Math.max(0, Math.ceil(seconds - elapsed));
      if (left !== lastLeft) {
        lastLeft = left;
        o.onTick(left);
      }
      if (elapsed >= seconds) {
        o.signal.removeEventListener("abort", onAbort);
        resolve({ signal: Float32Array.from(values), fps: values.length / elapsed });
        return;
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  });
}
