// Web Audio recorder -> mono Float32Array. Client only.
// ScriptProcessorNode is deprecated but works in every browser and needs no extra worklet file
// (which would complicate CSP and the offline cache). Revisit if browsers drop it.

export interface RecordOptions {
  ambientSec: number;
  recordSec: number;
  /** Return false to stop after the room-noise check. */
  onAmbient: (ambient: Float32Array) => boolean;
  onPhase: (phase: "ambient" | "recording") => void;
  onTick: (secondsLeft: number) => void;
  signal: AbortSignal;
}

export type RecordResult =
  | { ok: true; samples: Float32Array; sampleRate: number }
  | { ok: false; reason: "ambient" };

function concat(chunks: Float32Array[], total: number): Float32Array {
  const out = new Float32Array(total);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

export async function recordMono(opts: RecordOptions): Promise<RecordResult> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    video: false,
  });
  const AC: typeof AudioContext = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AC();
  const sampleRate = ctx.sampleRate;
  const source = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const mute = ctx.createGain();
  mute.gain.value = 0; // the processor must be connected to run, but we do not want to hear ourselves
  source.connect(proc);
  proc.connect(mute);
  mute.connect(ctx.destination);

  const ambientTarget = Math.round(opts.ambientSec * sampleRate);
  const recordTarget = Math.round(opts.recordSec * sampleRate);
  const chunks: Float32Array[] = [];
  let total = 0;
  let phase: "ambient" | "recording" = "ambient";
  opts.onPhase("ambient");

  const cleanup = () => {
    proc.onaudioprocess = null;
    try {
      source.disconnect();
      proc.disconnect();
      mute.disconnect();
    } catch {
      /* already disconnected */
    }
    stream.getTracks().forEach((t) => t.stop());
    void ctx.close().catch(() => {});
  };

  return new Promise<RecordResult>((resolve, reject) => {
    const onAbort = () => {
      cleanup();
      reject(new DOMException("aborted", "AbortError"));
    };
    if (opts.signal.aborted) return onAbort();
    opts.signal.addEventListener("abort", onAbort, { once: true });

    let ambient: Float32Array | null = null;
    let recorded = 0;
    const record: Float32Array[] = [];
    let lastLeft = -1;

    proc.onaudioprocess = (ev) => {
      const input = ev.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input); // the buffer is reused by the browser
      if (phase === "ambient") {
        chunks.push(copy);
        total += copy.length;
        if (total >= ambientTarget) {
          ambient = concat(chunks, total);
          if (!opts.onAmbient(ambient)) {
            opts.signal.removeEventListener("abort", onAbort);
            cleanup();
            resolve({ ok: false, reason: "ambient" });
            return;
          }
          phase = "recording";
          opts.onPhase("recording");
        }
        return;
      }
      record.push(copy);
      recorded += copy.length;
      const left = Math.max(0, Math.ceil((recordTarget - recorded) / sampleRate));
      if (left !== lastLeft) {
        lastLeft = left;
        opts.onTick(left);
      }
      if (recorded >= recordTarget) {
        opts.signal.removeEventListener("abort", onAbort);
        cleanup();
        resolve({ ok: true, samples: concat(record, recorded).subarray(0, recordTarget), sampleRate });
      }
    };
  });
}
