// Engine: the whole predict/explain logic, no worker plumbing. Runs inside the worker or on the main thread.
import { calibrateLogits, finalize, type Calibrated, type Extra } from "./calibrate";
import { fuse, severity } from "./fusion";
import { mockLogits } from "./mock";
import { estimateBreathingRate, respLabel, respTier, RESP_MIN_QUALITY } from "./resp";
import type { LogitsRunner } from "./ort";
import { darkSpotArea, skinU8 } from "./skin-preprocess";
import type { Manifest, Modality, ModalityInput, ModelEntry, Prediction } from "./types";
import bundledManifest from "../../public/models/manifest.json";

export const MANIFEST_URL = "/models/manifest.json";

export async function fetchManifest(): Promise<Manifest> {
  try {
    const r = await fetch(MANIFEST_URL, { cache: "no-cache" });
    if (r.ok) return (await r.json()) as Manifest;
  } catch {
    /* offline or no fetch: use the copy bundled at build time */
  }
  return bundledManifest as unknown as Manifest;
}

/** ImageBitmap/ImageData -> ImageData when the context allows it (needs a canvas), else null. */
export function toImageData(img: ImageBitmap | ImageData): ImageData | null {
  if ("data" in img) return img;
  try {
    const canvas: OffscreenCanvas | HTMLCanvasElement | null =
      typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(img.width, img.height)
        : typeof document !== "undefined"
          ? Object.assign(document.createElement("canvas"), { width: img.width, height: img.height })
          : null;
    const ctx = canvas?.getContext("2d") as CanvasRenderingContext2D | null;
    if (!ctx) return null;
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, img.width, img.height);
  } catch {
    return null;
  }
}

export interface EngineDeps {
  manifest?: () => Promise<Manifest>;
  ort?: (entry: ModelEntry) => Promise<LogitsRunner>;
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export class Engine {
  private manifestP?: Promise<Manifest>;
  private runners = new Map<Modality, Promise<LogitsRunner | null>>();

  constructor(private deps: EngineDeps = {}) {}

  private async entry(m: Modality): Promise<ModelEntry> {
    this.manifestP ??= (this.deps.manifest ?? fetchManifest)();
    const e = (await this.manifestP).models.find((x) => x.id === m);
    if (!e) throw new Error(`No manifest entry for ${m}`);
    return e;
  }

  private runner(e: ModelEntry): Promise<LogitsRunner | null> {
    let p = this.runners.get(e.id);
    if (!p) {
      p = e.url && (e.id === "skin" || e.id === "audio")
        ? (this.deps.ort ? this.deps.ort(e) : import("./ort").then((o) => o.createOrtRunner(e)))
        : Promise.resolve(null);
      p.catch(() => this.runners.delete(e.id)); // allow retry after a failed download
      this.runners.set(e.id, p);
    }
    return p;
  }

  async loadModel(m: Modality): Promise<void> {
    await this.runner(await this.entry(m));
  }

  async predict(input: ModalityInput): Promise<Prediction> {
    const t0 = now();
    const e = await this.entry(input.modality);
    const core = await this.core(e, input);
    return {
      modality: input.modality,
      ...core.cal,
      ...(core.extras ? { extras: core.extras } : {}),
      modelVersion: e.version,
      latencyMs: Math.round((now() - t0) * 100) / 100,
    };
  }

  /** Grad-CAM style explanations are future work: undefined for the mock and for ONNX for now. */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async explain(_input?: ModalityInput): Promise<Prediction["extras"]> {
    return undefined;
  }

  private async core(e: ModelEntry, input: ModalityInput): Promise<{ cal: Calibrated; extras?: Prediction["extras"] }> {
    switch (input.modality) {
      case "skin":
      case "audio": {
        const pixels = input.modality === "skin" ? toImageData(input.image) : null;
        const runner = await this.runner(e);
        const logits = runner ? await runner.run(input, pixels) : mockLogits(input, e.labels.length, pixels);
        const extra: Extra = {};
        if (pixels && e.darkSpot) extra.darkSpot = darkSpotArea(skinU8(pixels, e.onnx?.imageSize ?? 224), e.darkSpot);
        return { cal: calibrateLogits(e, logits, extra) };
      }
      case "resp": {
        const { breathsPerMin, quality } = estimateBreathingRate(input.signal, input.fps);
        const label = respLabel(breathsPerMin);
        const top = 0.5 + 0.5 * quality;
        const probs = e.labels.map((l) => (l === label ? top : (1 - top) / (e.labels.length - 1)));
        const cal = finalize(e, probs);
        if (quality < RESP_MIN_QUALITY || breathsPerMin <= 0) {
          return { cal: { ...cal, label: "uncertain", tier: "uncertain", confidence: quality }, extras: { breathsPerMin } };
        }
        return { cal: { ...cal, tier: respTier(breathsPerMin) }, extras: { breathsPerMin } };
      }
      case "fusion": {
        const feats: Partial<Record<Modality, number>> = {};
        const manifest = await (this.manifestP ??= (this.deps.manifest ?? fetchManifest)());
        for (const p of input.parts) {
          if (p.tier === "uncertain" || p.modality === "fusion") continue;
          const pe = manifest.models.find((x) => x.id === p.modality);
          if (!pe) continue;
          feats[p.modality] = severity(p, (l) => pe.tiers[l]);
          // resp tier is rate-dependent: use the realised tier instead of the label default
          if (p.modality === "resp") feats.resp = { green: 0, amber: 1, red: 2 }[p.tier];
        }
        const { probs, present } = fuse(feats);
        const cal = finalize(e, probs);
        if (present === 0) return { cal: { ...cal, label: "uncertain", tier: "uncertain" } };
        return { cal };
      }
    }
  }
}
