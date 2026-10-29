// Shared inference contract. UI code imports only from "@/inference"; swapping mock -> ONNX is a manifest change.

export type Modality = "skin" | "audio" | "resp" | "fusion";

/** Traffic-light tier shown to the user. Never a diagnosis. */
export type Tier = "green" | "amber" | "red" | "uncertain";

export interface Prediction {
  modality: Modality;
  /** Calibrated class probabilities, sum ~1. */
  probs: Record<string, number>;
  /** Top label, or "uncertain" when max prob < threshold or capture quality failed. */
  label: string | "uncertain";
  tier: Tier;
  confidence: number;
  extras?: { breathsPerMin?: number; heatmap?: ImageData };
  modelVersion: string;
  latencyMs: number;
}

/** Inputs per modality. Media stays on the device. */
export type ModalityInput =
  | { modality: "skin"; image: ImageBitmap | ImageData }
  | { modality: "audio"; samples: Float32Array; sampleRate: number }
  | { modality: "resp"; /** per-frame mean ROI motion/intensity signal */ signal: Float32Array; fps: number }
  | { modality: "fusion"; parts: Prediction[] };

/** Entry in public/models/manifest.json. */
export interface ModelEntry {
  id: Modality;
  version: string;
  /** Same-origin URL, e.g. /models/skin-v1.onnx. null = mock backend. */
  url: string | null;
  sha256: string | null;
  bytes: number;
  labels: string[];
  /** label -> tier mapping */
  tiers: Record<string, Exclude<Tier, "uncertain">>;
  /** max prob below this => "uncertain" */
  threshold: number;
  temperature: number;
  /** Optional free-text rationale (thresholds, tier choices). Ignored at runtime. */
  notes?: string;
  /** Optional ONNX I/O hints; defaults are used when absent. */
  onnx?: { input?: string; output?: string; imageSize?: number; frames?: number };
}

/** Shape of public/models/manifest.json. */
export interface Manifest {
  schemaVersion: 1;
  models: ModelEntry[];
}

export interface InferenceApi {
  loadModel(m: Modality): Promise<void>;
  predict(input: ModalityInput): Promise<Prediction>;
  explain(input: ModalityInput): Promise<Prediction["extras"]>;
}
