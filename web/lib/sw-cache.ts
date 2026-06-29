/**
 * Single source of truth for the service worker's caching policy.
 *
 * These pure helpers are unit-tested here AND mirrored verbatim (by hand) in
 * `public/sw.js`, which cannot import TypeScript because it is served to the
 * browser as a plain static file. A parity test (`sw-cache.test.ts`) asserts
 * that `public/sw.js` still contains the same path literals so the two copies
 * can never silently drift apart.
 */

/** Path prefixes the SW must NEVER cache, and why. */
const NEVER_CACHE_PREFIXES = [
  // The ONNX model is owned by IndexedDB (lib/model-cache.ts). The SW touching
  // it would break the cache layer's invariants and the "no-refetch" guarantee.
  "/model/",
  // MediaPipe assets are handled separately by a dedicated cache-first path in
  // the SW; shouldCache excludes them so the generic logic doesn't double-own.
  "/mediapipe/",
  // The metrics dashboard JSON must stay live/fresh, never served stale.
  "/metrics/",
] as const;

/** Cache name for a given build version (the SW reads version from `?v=`). */
export function cacheName(version: string): string {
  return `asl-cache-${version}`;
}

/**
 * Whether a request URL is eligible for runtime caching by the service worker.
 *
 * Returns false for:
 *  - cross-origin requests (e.g. the ORT CDN) — `url.origin !== selfOrigin`,
 *  - the IndexedDB-owned model path (`/model/`),
 *  - MediaPipe assets (`/mediapipe/`, cached via a dedicated path),
 *  - live metrics JSON (`/metrics/`).
 * Otherwise true (HTML navigations + same-origin `_next/static` and small assets).
 */
export function shouldCache(url: URL, selfOrigin: string): boolean {
  if (url.origin !== selfOrigin) return false;
  for (const prefix of NEVER_CACHE_PREFIXES) {
    if (url.pathname.startsWith(prefix)) return false;
  }
  return true;
}
