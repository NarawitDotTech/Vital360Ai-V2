"use client";

import { useMemo, useSyncExternalStore } from "react";
import type { Prediction } from "@/inference/types";
import { parseSession, readRawSession } from "./results";

const noopSubscribe = () => () => {};

/** Session results; `null` during server render / hydration, an array afterwards. */
export function useSessionResults(): Prediction[] | null {
  const raw = useSyncExternalStore(noopSubscribe, readRawSession, () => null);
  return useMemo(() => (raw === null ? null : parseSession(raw)), [raw]);
}
