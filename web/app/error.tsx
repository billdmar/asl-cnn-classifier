"use client";

import Link from "next/link";
import { useEffect } from "react";

/**
 * Route-level error boundary. A chunk-load failure or a runtime throw in any
 * client component (e.g. a chart) would otherwise white-screen the page; this
 * renders a graceful, on-brand fallback with a retry instead.
 */
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Surface it in the console for debugging; no external error service wired.
    console.error("Page error boundary caught:", error);
  }, [error]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 px-6 text-center">
      <h1 className="text-2xl font-semibold text-fg">Something went wrong</h1>
      <p className="max-w-md text-sm text-fg-muted">
        A part of the page failed to load — often a transient network or
        browser-compatibility issue with the in-browser model. Your camera and
        images never left your device.
      </p>
      <div className="flex gap-3">
        <button
          onClick={reset}
          className="rounded-md bg-accent-gradient px-4 py-2 text-sm font-medium text-bg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          Try again
        </button>
        <Link
          href="/"
          className="rounded-md border border-border px-4 py-2 text-sm font-medium text-fg hover:border-accent/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          Back to home
        </Link>
      </div>
    </main>
  );
}
