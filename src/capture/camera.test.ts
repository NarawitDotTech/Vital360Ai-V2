import { describe, expect, it, vi } from "vitest";
import { getTorchSupport, setTorch } from "./camera";

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
