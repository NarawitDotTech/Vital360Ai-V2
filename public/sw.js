/* Vitals 360 AI service worker (hand-written, no dependencies).
 *
 * - Install: precache the app shell = every page of the static export plus the JS/CSS they reference
 *   (found by scanning the fetched HTML, then one level of the JS for lazy chunks). No build step needed.
 * - /models/*  : cache-first in its own cache, so model files survive app updates.
 * - /_next/static/* and icons: cache-first (hashed, immutable).
 * - pages and everything else: network-first, falling back to the cache (offline).
 * Bump VERSION when this file's logic changes. Old vitals360-* caches are deleted on activate.
 * deleteAllData() in src/storage/db.ts deletes every cache whose name starts with "vitals360-".
 */
const VERSION = "v1";
const PREFIX = "vitals360-";
const SHELL = PREFIX + "shell-" + VERSION;
const MODELS = PREFIX + "models-" + VERSION;
const KEEP = [SHELL, MODELS];

const LOCALES = ["th", "en"];
const PAGES = ["", "consent/", "capture/", "capture/skin/", "capture/audio/", "capture/resp/", "results/", "history/", "delete/", "dashboard/"];
const STATIC = ["/", "/manifest.webmanifest", "/icon.svg", "/icon-192.png", "/icon-512.png", "/apple-touch-icon.png", "/favicon-32.png"];

const ASSET_RE = /\/_next\/static\/[^"'\\\s)<>]+\.(?:js|css|woff2?)/g;
const CHUNK_RE = /static\/(?:chunks|media)\/[^"'\\\s)<>]+\.(?:js|css|woff2?)/g;

async function put(cache, url) {
  try {
    const res = await fetch(url, { cache: "reload" });
    if (res.ok) {
      await cache.put(url, res.clone());
      return res;
    }
  } catch {
    /* ignore: runtime caching will pick it up later */
  }
  return null;
}

async function precache() {
  const cache = await caches.open(SHELL);
  const assets = new Set();
  const pageUrls = [];
  for (const l of LOCALES) for (const p of PAGES) pageUrls.push("/" + l + "/" + p);
  await Promise.all(
    pageUrls.concat(STATIC).map(async (u) => {
      const res = await put(cache, u);
      if (res && (res.headers.get("content-type") || "").includes("text/html")) {
        const html = await res.text();
        for (const m of html.match(ASSET_RE) || []) assets.add(m);
      }
    }),
  );
  // one level deeper: lazy chunks referenced from the JS (dynamic imports, workers)
  const second = new Set();
  await Promise.all(
    [...assets].filter((a) => a.endsWith(".js")).map(async (a) => {
      await put(cache, a);
      const res = await cache.match(a);
      if (!res) return;
      for (const m of (await res.text()).match(CHUNK_RE) || []) second.add("/_next/" + m);
    }),
  );
  await Promise.all([...assets].filter((a) => !a.endsWith(".js")).concat([...second].filter((a) => !assets.has(a))).map((a) => put(cache, a)));
}

self.addEventListener("install", (event) => {
  event.waitUntil(precache().then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(names.filter((n) => n.startsWith(PREFIX) && !KEEP.includes(n)).map((n) => caches.delete(n)));
      await self.clients.claim();
    })(),
  );
});

async function cacheFirst(req, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok && res.status === 200) cache.put(req, res.clone());
  return res;
}

async function networkFirst(req) {
  const cache = await caches.open(SHELL);
  try {
    const res = await fetch(req);
    if (res.ok) cache.put(req, res.clone());
    return res;
  } catch (err) {
    const hit = (await cache.match(req)) || (await cache.match(req, { ignoreSearch: true }));
    if (hit) return hit;
    if (req.mode === "navigate") {
      const fallback = await cache.match("/th/");
      if (fallback) return fallback;
    }
    throw err;
  }
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (req.headers.has("range")) return; // let the browser handle partial requests (media)

  if (url.pathname.startsWith("/models/")) {
    event.respondWith(cacheFirst(req, MODELS));
  } else if (url.pathname.startsWith("/_next/static/") || /\.(?:png|svg|ico|webmanifest)$/.test(url.pathname)) {
    event.respondWith(cacheFirst(req, SHELL));
  } else if (url.pathname === "/sw.js") {
    return;
  } else {
    event.respondWith(networkFirst(req));
  }
});
