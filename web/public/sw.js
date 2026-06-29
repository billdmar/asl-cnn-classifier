/*
 * Hand-written service worker for the ASL classifier (offline support).
 *
 * Served verbatim from public/ — NOT processed by Next, so it is plain browser
 * JS with no imports and no precache manifest. Avoiding a precache manifest is
 * deliberate: Next emits content-hashed _next/static filenames, so we cache at
 * runtime instead of pinning a build-time file list.
 *
 * The version comes from the `?v=` query param the registrar appends (a static
 * file can't read build env). Old caches are pruned on activate.
 */

const VERSION = new URL(self.location.href).searchParams.get("v") || "dev";
const CACHE = `asl-cache-${VERSION}`;

/*
 * Mirrors lib/sw-cache.ts (the source of truth). The SW can't import TS, so
 * these path literals are duplicated by hand and parity-tested in
 * lib/__tests__/sw-cache.test.ts to prevent silent drift. Keep in sync.
 */
function shouldCache(url) {
  if (url.origin !== self.location.origin) return false; // cross-origin ORT CDN
  if (url.pathname.startsWith("/model/")) return false; // IndexedDB owns it
  if (url.pathname.startsWith("/mediapipe/")) return false; // cache-first path
  if (url.pathname.startsWith("/metrics/")) return false; // stays live/fresh
  return true;
}

async function putInCache(req, res) {
  // Never store partials, errors, or opaque cross-origin responses.
  if (!res || res.status !== 200 || res.type === "opaque") return;
  const cache = await caches.open(CACHE);
  await cache.put(req, res).catch(() => {});
}

// Cache-first: serve from cache, else fetch + store (MediaPipe ~27MB immutable).
async function cacheFirst(req) {
  const cached = await caches.match(req);
  if (cached) return cached;
  const res = await fetch(req);
  await putInCache(req, res.clone());
  return res;
}

// Network-first for HTML navigations; fall back to cache when offline.
async function networkFirst(req) {
  try {
    const res = await fetch(req);
    await putInCache(req, res.clone());
    return res;
  } catch (err) {
    const cached = (await caches.match(req)) || (await caches.match("/index.html"));
    return cached || Response.error();
  }
}

// Stale-while-revalidate: serve cache immediately, refresh in the background.
async function staleWhileRevalidate(req) {
  const cached = await caches.match(req);
  const network = fetch(req)
    .then((res) => {
      putInCache(req, res.clone());
      return res;
    })
    .catch(() => undefined);
  return cached || (await network) || Response.error();
}

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((k) => k.startsWith("asl-cache-") && k !== CACHE)
          .map((k) => caches.delete(k)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  const sameOrigin = url.origin === self.location.origin;

  // MediaPipe: dedicated cache-first (biggest offline win).
  if (sameOrigin && url.pathname.startsWith("/mediapipe/")) {
    event.respondWith(cacheFirst(req));
    return;
  }

  // Anything we must not cache (cross-origin ORT CDN, /model/, /metrics/):
  // bare return -> no respondWith -> request proceeds to the network untouched.
  if (!shouldCache(url)) return;

  // HTML navigations: network-first with offline fallback.
  if (req.mode === "navigate") {
    event.respondWith(networkFirst(req));
    return;
  }

  // Everything else same-origin (_next/static + small assets): SWR.
  event.respondWith(staleWhileRevalidate(req));
});
