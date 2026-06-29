# Note: in-browser model caching & offline resilience

**Date:** 2026-06-29
**Status:** Shipped (web product round). Not an accuracy change — a load-UX change.

## Problem

The deployed web app runs the real MobileNetV2 ONNX model 100% client-side. The
model file (`web/public/model/model.onnx`, ~9 MB) was re-downloaded on **every**
visit, so each page load paid the full transfer before the webcam/upload could
classify.

## What we did

`lib/inference.ts::getSession()` already streamed the model bytes itself (to
report download progress) before handing them to `ort.InferenceSession.create`.
We backed that path with an IndexedDB cache (`lib/model-cache.ts`):

1. On load, look up the model bytes in IndexedDB keyed by `${url}@${buildSha}`.
2. **Hit** → create the session from cached bytes, zero network.
3. **Miss** → stream + create as before, then best-effort `putCachedModel` and
   `evictOtherVersions` (reclaims entries from prior deploys).

The cache applies to **both** entry points — the webcam warm-up (which passes an
`onProgress` callback) and the upload path (which doesn't). Caching was lifted
out of the progress-callback gate so clicking an example also populates it.

### Design choices (and what we deliberately did NOT cache)

- **IndexedDB, not a service worker.** Under Next `output: "export"` there's no
  emitted SW; hand-authoring one would intercept *all* fetches (including the
  e2e/Lighthouse runs) for marginal gain. IndexedDB stores the raw `ArrayBuffer`
  we already hold — narrower, test-friendly, no new moving parts.
- **MediaPipe `.task` (7.5 MB) is left to the HTTP cache.** It already ships with
  `immutable, max-age=31536000` in `vercel.json`, so repeat visits hit disk
  cache; we don't hold its bytes, so IDB interception there would be extra
  surface for little benefit.
- **The onnxruntime-web WASM runtime is CDN-resolved** (we never set
  `ort.env.wasm.wasmPaths`), so it is not something we cache — the browser's
  HTTP cache handles it.

### Versioning / busting

The cache key includes `NEXT_PUBLIC_BUILD_SHA` (baked at build time, see
`next.config.mjs`). A redeploy with a retrained model gets a new SHA, so the old
bytes are missed and then reclaimed — no stale model is ever served.

### Safety

Caching is strictly **best-effort**: every `model-cache` export guards
`typeof indexedDB === "undefined"` (SSR-safe) and swallows quota/IDB errors, so a
cache failure silently falls back to a normal fetch and never breaks inference.

## Resilience UX

`lib/use-classifier.ts` arms a 12 s warm-up timeout. If the model is still
loading when it fires, a `slow` flag surfaces a "still loading… (first visit
downloads ~9 MB; it's cached after)" note + a Retry button in the webcam panel —
without flipping to an error state while the download is genuinely in flight.

## Verification

- Unit: `lib/__tests__/model-cache.test.ts` (round-trip, version eviction, SSR
  no-op via an in-memory IndexedDB mock — no new dependency).
- E2E: `tests-e2e/inference.spec.ts` classifies once, reloads, and asserts the
  model is **not refetched** on the second visit (zero `/model/model.onnx`
  network requests).
