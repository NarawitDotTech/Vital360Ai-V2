import { beforeEach, describe, expect, it, vi } from "vitest";
import { consentGate, CONSENT_VERSION, makeConsentRecord } from "./consent-gate";
import {
  CACHE_PREFIX,
  DB_NAME,
  addHistory,
  clearHistory,
  deleteAllData,
  deleteHistory,
  getConsent,
  hasValidConsent,
  listHistory,
  saveConsent,
} from "./db";

const base = {
  locale: "th",
  modality: "skin" as const,
  tier: "amber" as const,
  label: "suspicious",
  confidence: 0.7,
  probs: { suspicious: 0.7, benign: 0.3 },
  modelVersion: "mock-1",
};

describe("consentGate (pure)", () => {
  it("blocks when there is no record", () => {
    expect(consentGate(null)).toEqual({ allowed: false, reason: "missing" });
    expect(consentGate(undefined)).toEqual({ allowed: false, reason: "missing" });
  });
  it("blocks malformed records", () => {
    expect(consentGate({})).toEqual({ allowed: false, reason: "invalid" });
    expect(consentGate({ version: CONSENT_VERSION, acceptedAt: "nope", locale: "th" })).toEqual({
      allowed: false,
      reason: "invalid",
    });
  });
  it("blocks an outdated version and allows the current one", () => {
    const rec = makeConsentRecord("th", new Date("2026-01-02T03:04:05Z"));
    expect(rec).toEqual({ version: CONSENT_VERSION, acceptedAt: "2026-01-02T03:04:05.000Z", locale: "th" });
    expect(consentGate(rec)).toEqual({ allowed: true });
    expect(consentGate({ ...rec, version: "0.1" })).toEqual({ allowed: false, reason: "outdated" });
    expect(consentGate(rec, "2.0")).toEqual({ allowed: false, reason: "outdated" });
  });
});

describe("storage", () => {
  beforeEach(async () => {
    await deleteAllData();
  });

  it("starts without consent and stores {version, acceptedAt, locale}", async () => {
    expect(await getConsent()).toBeNull();
    expect(await hasValidConsent()).toBe(false);
    const rec = makeConsentRecord("en");
    await saveConsent(rec);
    expect(await getConsent()).toEqual(rec);
    expect(await hasValidConsent()).toBe(true);
  });

  it("adds, lists (newest first) and deletes history entries", async () => {
    const a = await addHistory({ ...base, createdAt: "2026-01-01T00:00:00.000Z" });
    const b = await addHistory({ ...base, modality: "audio", createdAt: "2026-02-01T00:00:00.000Z" });
    const list = await listHistory();
    expect(list.map((e) => e.id)).toEqual([b.id, a.id]);
    await deleteHistory(b.id);
    expect((await listHistory()).map((e) => e.id)).toEqual([a.id]);
    await clearHistory();
    expect(await listHistory()).toEqual([]);
  });

  it("stores no media fields", async () => {
    const e = await addHistory(base);
    expect(Object.keys(e).sort()).toEqual(
      ["confidence", "createdAt", "id", "label", "locale", "modality", "modelVersion", "probs", "tier"].sort(),
    );
  });

  it("deleteAllData wipes IndexedDB, app caches and app storage keys but not other caches", async () => {
    await saveConsent(makeConsentRecord("th"));
    await addHistory(base);
    sessionStorage.setItem("v360.session", "[]");
    localStorage.setItem("v360.pref", "1");
    localStorage.setItem("other.key", "keep");

    const store = new Map<string, true>([
      [`${CACHE_PREFIX}shell-v1`, true],
      [`${CACHE_PREFIX}models-v1`, true],
      ["someone-elses-cache", true],
    ]);
    (globalThis as unknown as { caches: unknown }).caches = {
      keys: async () => [...store.keys()],
      delete: async (n: string) => store.delete(n),
    };

    await deleteAllData();

    // check the database is gone before reading (reading would re-create it)
    const dbs = await indexedDB.databases();
    expect(dbs.map((d) => d.name)).not.toContain(DB_NAME);
    expect(await getConsent()).toBeNull();
    expect(await listHistory()).toEqual([]);
    expect([...store.keys()]).toEqual(["someone-elses-cache"]);
    expect(sessionStorage.getItem("v360.session")).toBeNull();
    expect(localStorage.getItem("v360.pref")).toBeNull();
    expect(localStorage.getItem("other.key")).toBe("keep");
    delete (globalThis as { caches?: unknown }).caches;
  });

  it("deleteAllData rejects (does not report success) while the database delete stays blocked", async () => {
    vi.useFakeTimers();
    const spy = vi.spyOn(indexedDB, "deleteDatabase").mockImplementation(() => {
      const req = {} as IDBOpenDBRequest;
      setTimeout(() => (req.onblocked as (() => void) | null)?.(), 0);
      return req;
    });
    try {
      const p = deleteAllData();
      const assertion = expect(p).rejects.toThrow(/another tab/);
      await vi.advanceTimersByTimeAsync(10_000);
      await assertion;
    } finally {
      spy.mockRestore();
      vi.useRealTimers();
    }
  });

  it("deleteAllData unregisters service workers", async () => {
    const unregister = vi.fn(async () => true);
    Object.defineProperty(navigator, "serviceWorker", {
      configurable: true,
      value: { getRegistrations: async () => [{ unregister }, { unregister }] },
    });
    await deleteAllData();
    expect(unregister).toHaveBeenCalledTimes(2);
    delete (navigator as unknown as { serviceWorker?: unknown }).serviceWorker;
  });
});
