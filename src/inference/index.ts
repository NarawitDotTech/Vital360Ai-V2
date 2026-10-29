// Public inference API. Runs in one Web Worker; falls back to the main thread when Worker is unavailable.
import { Engine } from "./engine";
import type { WorkerRequest, WorkerResponse } from "./protocol";
import type { InferenceApi, Modality, ModalityInput, Prediction } from "./types";

export * from "./types";
export { frameMotion, estimateBreathingRate, respLabel, respTier } from "./resp";
export { logMel } from "./mel";

type Req = WorkerRequest extends infer R ? (R extends unknown ? Omit<R, "id"> : never) : never;
interface Pending {
  req: WorkerRequest;
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
}

let mainEngine: Engine | null = null;
let worker: Worker | null | undefined; // undefined = not tried, null = unavailable
let seq = 0;
const pending = new Map<number, Pending>();

function runOnMain(req: WorkerRequest): Promise<unknown> {
  mainEngine ??= new Engine();
  return req.type === "load" ? mainEngine.loadModel(req.modality)
    : req.type === "predict" ? mainEngine.predict(req.input)
    : mainEngine.explain(req.input);
}

function getWorker(): Worker | null {
  if (worker !== undefined) return worker;
  worker = null;
  if (typeof Worker === "undefined") return null;
  try {
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const p = pending.get(ev.data.id);
      if (!p) return;
      pending.delete(ev.data.id);
      if (ev.data.ok) p.resolve(ev.data.result);
      else p.reject(new Error(ev.data.error));
    };
    w.onerror = () => {
      // Worker failed to start or crashed: re-run everything pending on the main thread from now on.
      worker = null;
      w.terminate();
      const stuck = [...pending.values()];
      pending.clear();
      for (const p of stuck) runOnMain(p.req).then(p.resolve, p.reject);
    };
    worker = w;
  } catch {
    worker = null;
  }
  return worker;
}

function call<T>(req: Req): Promise<T> {
  const full = { ...req, id: ++seq } as WorkerRequest;
  const w = getWorker();
  if (!w) return runOnMain(full) as Promise<T>;
  return new Promise<T>((resolve, reject) => {
    pending.set(full.id, { req: full, resolve: resolve as (v: unknown) => void, reject });
    w.postMessage(full);
  });
}

export const inference: InferenceApi = {
  loadModel: (modality: Modality) => call<void>({ type: "load", modality }),
  predict: (input: ModalityInput) => call<Prediction>({ type: "predict", input }),
  explain: (input: ModalityInput) => call<Prediction["extras"]>({ type: "explain", input }),
};
