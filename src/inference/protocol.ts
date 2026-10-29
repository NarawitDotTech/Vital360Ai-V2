import type { Modality, ModalityInput } from "./types";

export type WorkerRequest =
  | { id: number; type: "load"; modality: Modality }
  | { id: number; type: "predict"; input: ModalityInput }
  | { id: number; type: "explain"; input: ModalityInput };

export type WorkerResponse = { id: number; ok: true; result: unknown } | { id: number; ok: false; error: string };
