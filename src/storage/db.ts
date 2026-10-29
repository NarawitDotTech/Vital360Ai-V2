// Tiny IndexedDB wrapper. Stores consent and history summaries only. Never media.
import { consentGate, isConsentRecord, type ConsentRecord } from "./consent-gate";
import type { Modality, Tier } from "@/inference/types";

export const DB_NAME = "vitals360";
const DB_VERSION = 1;
const CONSENT_STORE = "consent";
const HISTORY_STORE = "history";
const CONSENT_KEY = "current";

/** Cache Storage names created by public/sw.js start with this prefix. */
export const CACHE_PREFIX = "vitals360-";
/** sessionStorage / localStorage keys used by the app start with this prefix. */
export const KEY_PREFIX = "v360.";

export interface HistoryEntry {
  id: string;
  /** ISO 8601 */
  createdAt: string;
  locale: string;
  modality: Modality;
  tier: Tier;
  label: string;
  confidence: number;
  breathsPerMin?: number;
  probs: Record<string, number>;
  modelVersion: string;
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("indexedDB unavailable"));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(CONSENT_STORE)) db.createObjectStore(CONSENT_STORE);
      if (!db.objectStoreNames.contains(HISTORY_STORE)) db.createObjectStore(HISTORY_STORE, { keyPath: "id" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("indexedDB open failed"));
  });
}

/** Run one transaction on a short-lived connection (so deleteDatabase is never blocked by us). */
async function tx<T>(
  store: string,
  mode: IDBTransactionMode,
  run: (s: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((resolve, reject) => {
      const t = db.transaction(store, mode);
      const req = run(t.objectStore(store));
      let result: T;
      req.onsuccess = () => {
        result = req.result;
      };
      t.oncomplete = () => resolve(result);
      t.onerror = () => reject(t.error ?? req.error);
      t.onabort = () => reject(t.error ?? new Error("transaction aborted"));
    });
  } finally {
    db.close();
  }
}

// ---- consent ----------------------------------------------------------------

export async function saveConsent(record: ConsentRecord): Promise<void> {
  await tx(CONSENT_STORE, "readwrite", (s) => s.put(record, CONSENT_KEY));
}

export async function getConsent(): Promise<ConsentRecord | null> {
  const v = await tx<unknown>(CONSENT_STORE, "readonly", (s) => s.get(CONSENT_KEY));
  return isConsentRecord(v) ? v : null;
}

/** True when the stored consent is valid for the current consent text. Any storage error counts as "no". */
export async function hasValidConsent(): Promise<boolean> {
  try {
    return consentGate(await getConsent()).allowed;
  } catch {
    return false;
  }
}

// ---- history ----------------------------------------------------------------

export function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export async function addHistory(entry: Omit<HistoryEntry, "id" | "createdAt"> & Partial<Pick<HistoryEntry, "id" | "createdAt">>): Promise<HistoryEntry> {
  const full: HistoryEntry = { id: newId(), createdAt: new Date().toISOString(), ...entry };
  await tx(HISTORY_STORE, "readwrite", (s) => s.put(full));
  return full;
}

/** Newest first. */
export async function listHistory(): Promise<HistoryEntry[]> {
  const all = await tx<HistoryEntry[]>(HISTORY_STORE, "readonly", (s) => s.getAll());
  return all.sort((a, b) => (a.createdAt < b.createdAt ? 1 : a.createdAt > b.createdAt ? -1 : 0));
}

export async function deleteHistory(id: string): Promise<void> {
  await tx(HISTORY_STORE, "readwrite", (s) => s.delete(id));
}

export async function clearHistory(): Promise<void> {
  await tx(HISTORY_STORE, "readwrite", (s) => s.clear());
}

// ---- delete everything ------------------------------------------------------

const DELETE_TIMEOUT_MS = 8000;

/** Resolves only when the database is really gone. If another tab keeps it open past the timeout, rejects. */
function deleteDatabase(name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") return resolve();
    const req = indexedDB.deleteDatabase(name);
    let timer: ReturnType<typeof setTimeout> | undefined;
    req.onsuccess = () => {
      clearTimeout(timer);
      resolve();
    };
    req.onerror = () => {
      clearTimeout(timer);
      reject(req.error ?? new Error("deleteDatabase failed"));
    };
    // Another tab holds a connection open: it should close on versionchange. Keep waiting for success,
    // but never report success while blocked, and do not hang forever.
    req.onblocked = () => {
      clearTimeout(timer);
      timer = setTimeout(
        () => reject(new Error("Could not delete: the app is open in another tab. Close other tabs and try again.")),
        DELETE_TIMEOUT_MS,
      );
    };
  });
}

function clearPrefixed(storage: Storage | undefined) {
  if (!storage) return;
  const keys: string[] = [];
  for (let i = 0; i < storage.length; i++) {
    const k = storage.key(i);
    if (k && k.startsWith(KEY_PREFIX)) keys.push(k);
  }
  keys.forEach((k) => storage.removeItem(k));
}

/**
 * Wipe everything the app stored on this device: the IndexedDB database, Cache Storage entries
 * created by the app (names starting with CACHE_PREFIX), the service worker registration, and the app's session/local storage keys.
 * Account data on a server (if ever used) is deleted via the backend, not here.
 */
export async function deleteAllData(): Promise<void> {
  const errors: unknown[] = [];
  try {
    // first, so a live worker cannot re-fill the caches we are about to delete
    if (typeof navigator !== "undefined" && navigator.serviceWorker?.getRegistrations) {
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map((r) => r.unregister()));
    }
  } catch (e) {
    errors.push(e);
  }
  try {
    await deleteDatabase(DB_NAME);
  } catch (e) {
    errors.push(e);
  }
  try {
    if (typeof caches !== "undefined") {
      const names = await caches.keys();
      await Promise.all(names.filter((n) => n.startsWith(CACHE_PREFIX)).map((n) => caches.delete(n)));
    }
  } catch (e) {
    errors.push(e);
  }
  try {
    clearPrefixed(typeof sessionStorage === "undefined" ? undefined : sessionStorage);
    clearPrefixed(typeof localStorage === "undefined" ? undefined : localStorage);
  } catch (e) {
    errors.push(e);
  }
  if (errors.length) throw errors[0];
}
