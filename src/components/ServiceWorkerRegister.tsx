"use client";

import { useEffect } from "react";

/** Registers /sw.js in production builds only (it would cache stale dev bundles otherwise). */
export function ServiceWorkerRegister() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "production" || !("serviceWorker" in navigator)) return;
    navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => {
      /* offline support is optional; the app still works online */
    });
  }, []);
  return null;
}
