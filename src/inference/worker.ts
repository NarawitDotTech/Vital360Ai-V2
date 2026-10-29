// Worker entry: wires messages to Engine. All logic lives in engine.ts.
import { Engine } from "./engine";
import type { WorkerRequest, WorkerResponse } from "./protocol";

const engine = new Engine();
const ctx = self as unknown as { postMessage(m: WorkerResponse): void; onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null };

ctx.onmessage = async (ev) => {
  const req = ev.data;
  try {
    const result =
      req.type === "load" ? await engine.loadModel(req.modality)
      : req.type === "predict" ? await engine.predict(req.input)
      : await engine.explain(req.input);
    ctx.postMessage({ id: req.id, ok: true, result });
  } catch (err) {
    ctx.postMessage({ id: req.id, ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
