// Results of the current screening session, passed between pages via sessionStorage
// (static export: no server state). Holds Prediction summaries only, never media.
import type { Modality, Prediction } from "@/inference/types";
import { KEY_PREFIX } from "@/storage/db";

const KEY = `${KEY_PREFIX}session`;

/** Drop non-serialisable / heavy parts (heatmap ImageData). */
export function toSerializable(p: Prediction): Prediction {
  const { extras, ...rest } = p;
  const slim = extras && extras.breathsPerMin !== undefined ? { breathsPerMin: extras.breathsPerMin } : undefined;
  return slim ? { ...rest, extras: slim } : rest;
}

function isPrediction(v: unknown): v is Prediction {
  if (!v || typeof v !== "object") return false;
  const p = v as Record<string, unknown>;
  return (
    typeof p.modality === "string" &&
    typeof p.tier === "string" &&
    typeof p.label === "string" &&
    typeof p.confidence === "number" &&
    !!p.probs &&
    typeof p.probs === "object"
  );
}

export function getSessionResults(storage: Storage | undefined = safeSession()): Prediction[] {
  if (!storage) return [];
  try {
    const raw = storage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isPrediction) : [];
  } catch {
    return [];
  }
}

/** Store (or replace) the latest result for a modality. */
export function setSessionResult(p: Prediction, storage: Storage | undefined = safeSession()): void {
  if (!storage) return;
  const rest = getSessionResults(storage).filter((x) => x.modality !== p.modality);
  try {
    storage.setItem(KEY, JSON.stringify([...rest, toSerializable(p)]));
  } catch {
    /* storage full or blocked: results page will show the empty state */
  }
}

export function clearSessionResults(storage: Storage | undefined = safeSession()): void {
  try {
    storage?.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

/** The non-fusion parts. */
export function sessionParts(results: Prediction[]): Prediction[] {
  return results.filter((r) => r.modality !== "fusion");
}

export function doneModalities(results: Prediction[]): Set<Modality> {
  return new Set(results.map((r) => r.modality));
}

function safeSession(): Storage | undefined {
  try {
    return typeof sessionStorage === "undefined" ? undefined : sessionStorage;
  } catch {
    return undefined;
  }
}
