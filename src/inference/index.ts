// Stub so UI work can start; the inference agent replaces the implementation and keeps these exports.
import type { InferenceApi, Modality, ModalityInput, Prediction } from "./types";

export * from "./types";

export const inference: InferenceApi = {
  async loadModel(_m: Modality) {},
  async predict(input: ModalityInput): Promise<Prediction> {
    return {
      modality: input.modality,
      probs: { normal: 1 },
      label: "normal",
      tier: "green",
      confidence: 1,
      modelVersion: "stub",
      latencyMs: 0,
    };
  },
  async explain() {
    return undefined;
  },
};
