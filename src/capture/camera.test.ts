import { describe, expect, it, vi } from "vitest";
import { centerSquare, getTorchSupport, guideRect, setTorch } from "./camera";

function streamWith(track: Partial<MediaStreamTrack> | null): MediaStream {
  return { getVideoTracks: () => (track ? [track] : []) } as unknown as MediaStream;
}

describe("getTorchSupport", () => {
  it("detects torch capability", () => {
    expect(getTorchSupport(streamWith({ getCapabilities: () => ({ torch: true }) as MediaTrackCapabilities }))).toBe(true);
  });
  it("is false without torch, track, or getCapabilities", () => {
    expect(getTorchSupport(streamWith({ getCapabilities: () => ({}) }))).toBe(false);
    expect(getTorchSupport(streamWith({}))).toBe(false);
    expect(getTorchSupport(streamWith(null))).toBe(false);
    expect(getTorchSupport(null)).toBe(false);
  });
});

describe("setTorch", () => {
  it("applies the advanced torch constraint", async () => {
    const applyConstraints = vi.fn().mockResolvedValue(undefined);
    expect(await setTorch(streamWith({ applyConstraints }), true)).toBe(true);
    expect(applyConstraints).toHaveBeenCalledWith({ advanced: [{ torch: true }] });
  });
  it("returns false when the device rejects", async () => {
    const applyConstraints = vi.fn().mockRejectedValue(new Error("nope"));
    expect(await setTorch(streamWith({ applyConstraints }), true)).toBe(false);
    expect(await setTorch(null, true)).toBe(false);
  });
});

describe("centerSquare (file upload)", () => {
  it("takes the centred square of the short side", () => {
    expect(centerSquare(4000, 3000)).toEqual({ x: 500, y: 0, s: 3000 });
    expect(centerSquare(3000, 4000)).toEqual({ x: 0, y: 500, s: 3000 });
    expect(centerSquare(100, 100)).toEqual({ x: 0, y: 0, s: 100 });
  });
});

describe("guideRect (camera frame)", () => {
  it("4:3 stream: the guide box is a centred square, 70% of the frame height", () => {
    expect(guideRect(1280, 960)).toEqual({ x: 304, y: 144, s: 672 });
  });
  it("wider stream: the visible part is cropped by object-fit cover, guide stays 70% of the visible height", () => {
    expect(guideRect(1920, 1080)).toEqual({ x: 582, y: 162, s: 756 });
  });
  it("taller stream (portrait): visible height = width / (4/3)", () => {
    const r = guideRect(720, 1280);
    expect(r.s).toBe(Math.round((720 / (4 / 3)) * 0.7));
    expect(r.x + r.s / 2).toBe(360);
    expect(r.y + r.s / 2).toBe(640);
  });
  it("never leaves the frame", () => {
    for (const [w, h] of [[640, 480], [100, 2000], [2000, 100], [1, 1]] as const) {
      const r = guideRect(w, h);
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeGreaterThanOrEqual(0);
      expect(r.x + r.s).toBeLessThanOrEqual(w);
      expect(r.y + r.s).toBeLessThanOrEqual(h);
    }
  });
});
