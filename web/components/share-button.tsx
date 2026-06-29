"use client";

import { useCallback, useState } from "react";
import { Check, Share2 } from "lucide-react";

import { encodeResult } from "@/lib/share-link";
import type { InferenceResult } from "@/lib/inference";
import { cn } from "@/lib/utils";

/**
 * Build the absolute permalink for a result, embedding the encoded payload in
 * the URL hash. Reads `Date.now()` (the impure boundary) and `location.origin`,
 * so it must run client-side. Exported standalone so the keyboard "S" shortcut
 * can reuse the exact same URL the button produces.
 */
export function buildShareUrl(result: InferenceResult): string {
  return `${window.location.origin}/result#r=${encodeResult(result, Date.now())}`;
}

/**
 * Share a URL: prefer the native share sheet where available, otherwise copy to
 * the clipboard. Resolves true if it copied to the clipboard (so callers can
 * show a transient "Copied!" affordance), false if it was shared/dismissed.
 */
export async function shareResult(url: string): Promise<boolean> {
  if (typeof navigator !== "undefined" && typeof navigator.share === "function") {
    try {
      await navigator.share({ url });
      return false;
    } catch {
      // User dismissed the sheet, or share failed — fall through to clipboard.
    }
  }
  if (typeof navigator !== "undefined" && navigator.clipboard) {
    await navigator.clipboard.writeText(url);
  }
  return true;
}

interface ShareButtonProps {
  result: InferenceResult;
  className?: string;
}

/**
 * A small, accessible "Share result" button. Builds the permalink, shares or
 * copies it, and shows a 2-second "Copied!" confirmation when it falls back to
 * the clipboard.
 */
export function ShareButton({ result, className }: ShareButtonProps) {
  const [copied, setCopied] = useState(false);

  const onClick = useCallback(async () => {
    try {
      const url = buildShareUrl(result);
      const didCopy = await shareResult(url);
      if (didCopy) {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 2000);
      }
    } catch {
      // Sharing/copying is best-effort; never crash the result view.
    }
  }, [result]);

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={copied ? "Link copied to clipboard" : "Share this result"}
      className={cn(
        "inline-flex items-center gap-2 rounded-lg border border-border bg-bg-card px-3 py-2 text-sm font-medium text-fg transition-colors hover:border-accent/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg",
        className,
      )}
    >
      {copied ? (
        <>
          <Check className="h-4 w-4 text-accent" aria-hidden="true" />
          Copied!
        </>
      ) : (
        <>
          <Share2 className="h-4 w-4 text-fg-subtle" aria-hidden="true" />
          Share result
        </>
      )}
    </button>
  );
}
