// Pure consent gate logic (no browser APIs) so it is easy to test.
export const CONSENT_VERSION = "1.0";

export interface ConsentRecord {
  version: string;
  /** ISO 8601 timestamp */
  acceptedAt: string;
  locale: string;
}

export type GateDecision =
  | { allowed: true }
  | { allowed: false; reason: "missing" | "outdated" | "invalid" };

/** Capture and results pages need a consent record whose version matches the current text. */
export function consentGate(record: unknown, currentVersion: string = CONSENT_VERSION): GateDecision {
  if (record === null || record === undefined) return { allowed: false, reason: "missing" };
  if (!isConsentRecord(record)) return { allowed: false, reason: "invalid" };
  if (record.version !== currentVersion) return { allowed: false, reason: "outdated" };
  return { allowed: true };
}

export function isConsentRecord(v: unknown): v is ConsentRecord {
  if (!v || typeof v !== "object") return false;
  const r = v as Record<string, unknown>;
  return (
    typeof r.version === "string" &&
    r.version !== "" &&
    typeof r.acceptedAt === "string" &&
    !Number.isNaN(Date.parse(r.acceptedAt)) &&
    typeof r.locale === "string"
  );
}

export function makeConsentRecord(locale: string, now: Date = new Date()): ConsentRecord {
  return { version: CONSENT_VERSION, acceptedAt: now.toISOString(), locale };
}
