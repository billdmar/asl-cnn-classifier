"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";

import type { Prediction } from "@/lib/inference";
import type { SharedResult } from "@/lib/share-link";
import { Card, CardContent } from "@/components/ui/card";
import { ResultBars } from "@/components/upload/result-bars";

/**
 * Render a shared classification result decoded from a permalink.
 *
 * Self-contained and theme-consistent (bg-bg / text-fg / accent gradient). The
 * top-k pairs are mapped back to {@link Prediction}[] so the existing
 * {@link ResultBars} visual can be reused (index is synthetic — only used as a
 * stable React key).
 */
export function ResultCard({ result }: { result: SharedResult }) {
  const ranked: Prediction[] = result.topk.map(([label, prob], index) => ({
    label,
    index,
    prob,
  }));

  const when = formatTimestamp(result.t);

  return (
    <Card>
      <CardContent className="flex flex-col gap-5 p-6 sm:p-8">
        <div className="flex flex-col items-center gap-1 text-center">
          <p className="text-xs uppercase tracking-wide text-fg-subtle">
            Shared prediction
          </p>
          <p className="bg-accent-gradient bg-clip-text font-mono text-7xl font-bold leading-none text-transparent">
            {result.letter}
          </p>
        </div>

        {ranked.length > 0 && <ResultBars ranked={ranked} count={5} />}

        <div className="flex flex-col items-center gap-3 border-t border-border-subtle pt-4 text-center">
          {when && (
            <p className="text-xs text-fg-subtle">Classified {when}</p>
          )}
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 rounded-lg text-sm font-medium text-fg underline decoration-accent decoration-2 underline-offset-4 hover:decoration-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
          >
            Try it yourself
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}

/** Format a share timestamp as a human-readable local string, or "" if junk. */
function formatTimestamp(t: number): string {
  if (!Number.isFinite(t) || t <= 0) return "";
  try {
    return new Date(t).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return "";
  }
}
