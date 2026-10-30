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
  /** Only for models with tierRule "sum" / "thresholds": class probabilities summed per tier (green+amber+red ~ 1). */
  tierProbs?: Partial<Record<"green" | "amber" | "red", number>>;
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
  /**
   * How the shown tier is chosen. "label" (default): tier of the top label, uncertain if max class prob < threshold.
   * "sum": tier = argmax over tiers of the summed class probabilities, uncertain if that tier prob < threshold;
   * label stays the top class (or "uncertain"). Used by skin-v1.
   * "thresholds" (sensitivity-first, skin-v1.1+): red if P(red) >= tierThresholds.red, else amber if P(amber)+P(red) >= tierThresholds.amber,
   * else (after the optional `guard`) green, or "uncertain" when P(green) < threshold. Needs tierThresholds.
   */
  tierRule?: "label" | "sum" | "thresholds";
  /** tierRule "thresholds" only: cut-offs on the summed tier probabilities, tuned on the validation split (never on test). */
  tierThresholds?: { red: number; amber: number };
  /**
   * tierRule "thresholds" only: when the top class is in `topIn` (e.g. nevus, seborrheic keratosis) and `watch` (melanoma) is the
   * second most likely class, a green result is raised to amber. `minWatchProb` (default 0 = literal rule) additionally requires
   * P(watch) >= that value; tuned on validation to cut false alarms on look-alikes without losing the guard's gain on cancers.
   */
  guard?: { topIn: string[]; watch: string; minWatchProb?: number };
  /**
   * tierRule "thresholds" only, skin only: an enclosed very dark spot (see skin-preprocess.ts darkSpotArea) covering at least `minArea` of the
   * photographed box is never shown green/uncertain: it is raised to amber (a pigmented lesion that big is a reason to see a clinician, and the
   * network never saw such lesions as "no concern"). Reds stay red.
   */
  darkSpot?: { minArea: number; relLuma: number; maxLuma: number };
  /** Optional free-text rationale (thresholds, tier choices). Ignored at runtime. */
  notes?: string;
  /** Optional ONNX I/O hints; defaults are used when absent. */
  onnx?: { input?: string; output?: string; imageSize?: number; frames?: number; normalization?: string };
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
