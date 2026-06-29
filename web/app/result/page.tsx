"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

import { Footer } from "@/components/footer";
import { SiteHeader } from "@/components/site-header";
import { ResultCard } from "@/components/result-card";
import { decodeResult, type SharedResult } from "@/lib/share-link";

/** Hydration state: undefined = not read yet (server skeleton), null = invalid. */
type HashState = SharedResult | null | undefined;

export default function ResultPage() {
  // Server-prerenders with `undefined` (a neutral skeleton). The hash is read
  // ONLY in the effect below — never at module scope — so static export and
  // hydration stay safe (window/location don't exist during prerender).
  const [decoded, setDecoded] = useState<HashState>(undefined);

  useEffect(() => {
    const hash = window.location.hash; // e.g. "#r=<payload>"
    const raw = hash.startsWith("#r=") ? hash.slice(3) : "";
    setDecoded(raw ? decodeResult(raw) : null);
  }, []);

  return (
    <>
      <SiteHeader />
      <main className="mx-auto flex min-h-[60vh] max-w-xl flex-col px-6 py-16 sm:py-20">
        <h1 className="mb-8 text-center text-3xl font-bold leading-tight text-fg sm:text-4xl">
          Shared ASL result
        </h1>

        {decoded === undefined && <Skeleton />}
        {decoded === null && <EmptyState />}
        {decoded && <ResultCard result={decoded} />}
      </main>
      <Footer />
    </>
  );
}

/** Neutral, non-flashing placeholder shown during prerender/hydration. */
function Skeleton() {
  return (
    <div
      aria-hidden="true"
      className="flex flex-col gap-5 rounded-xl border border-border bg-bg-card p-8"
    >
      <div className="mx-auto h-20 w-20 rounded-lg bg-bg-subtle" />
      <div className="h-3 w-full rounded-full bg-bg-subtle" />
      <div className="h-3 w-4/5 rounded-full bg-bg-subtle" />
      <div className="h-3 w-3/5 rounded-full bg-bg-subtle" />
    </div>
  );
}

/** Friendly fallback when the hash is missing or fails strict validation. */
function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-4 rounded-xl border border-border bg-bg-card p-8 text-center">
      <h2 className="text-lg font-semibold text-fg">No result to show</h2>
      <p className="max-w-sm text-sm text-fg-muted">
        This link doesn&apos;t contain a valid shared prediction. Try the live demo
        and share your own result.
      </p>
      <Link
        href="/"
        className="inline-flex items-center gap-1.5 rounded-lg text-sm font-medium text-fg underline decoration-accent decoration-2 underline-offset-4 hover:decoration-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
      >
        Go to the live demo
        <ArrowRight className="h-4 w-4" aria-hidden="true" />
      </Link>
    </div>
  );
}
