"use client";

/**
 * Registers the offline service worker (public/sw.js) on window "load".
 *
 * Renders nothing — it exists only for its registration side effect. The
 * integrator mounts it in layout.tsx; it is intentionally not mounted here.
 *
 * The guard is on HOSTNAME, not NODE_ENV: Playwright/Lighthouse e2e run against
 * a real production build, so gating on env would still register the SW under
 * test. Skipping localhost / 127.0.0.1 keeps those runs service-worker-free
 * (no SW caching to confound the inference "no-refetch" assertions).
 */

import { useEffect } from "react";

export function ServiceWorkerRegister(): null {
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;

    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") return;

    const register = (): void => {
      const sha = process.env.NEXT_PUBLIC_BUILD_SHA ?? "dev";
      navigator.serviceWorker
        .register(`/sw.js?v=${encodeURIComponent(sha)}`)
        .catch(() => {});
    };

    window.addEventListener("load", register);
    return () => window.removeEventListener("load", register);
  }, []);

  return null;
}
