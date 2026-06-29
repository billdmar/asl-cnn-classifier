"use client";

/**
 * Reusable top-K confidence bars.
 *
 * Renders a vertical list of animated probability bars (width animated via
 * framer-motion, honoring prefers-reduced-motion). Shared by the live-webcam
 * panel and the upload feature, so it takes only plain props and owns no model
 * or camera logic.
 */

import { motion, useReducedMotion } from "framer-motion";

import { formatPct } from "@/lib/confidence";
import { transition } from "@/lib/motion";
import type { Prediction } from "@/lib/inference";
import { cn } from "@/lib/utils";

export interface ConfidenceBarsProps {
  /** Predictions sorted by probability, descending (e.g. `result.ranked`). */
  ranked: readonly Prediction[];
  /** When true, the top-1 is below the confidence threshold (de-emphasized). */
  unsure?: boolean;
  /** How many bars to show (defaults to 5). */
  topK?: number;
}

/** Animated top-K probability bars. */
export function ConfidenceBars({
  ranked,
  unsure = false,
  topK = 5,
}: ConfidenceBarsProps) {
  const reduceMotion = useReducedMotion();
  const rows = ranked.slice(0, topK);

  return (
    <ul className="flex flex-col gap-2" aria-label="Top predictions">
      {rows.map((pred, i) => {
        const isTop = i === 0;
        const pct = Math.max(0, Math.min(1, pred.prob));
        return (
          <li key={pred.index} className="flex items-center gap-3">
            <span
              className={cn(
                "w-6 shrink-0 text-center font-mono text-sm font-semibold",
                isTop ? "text-fg" : "text-fg-subtle",
              )}
            >
              {pred.label}
            </span>
            <div className="relative h-3 flex-1 overflow-hidden rounded-full bg-bg">
              <motion.div
                className={cn(
                  "h-full rounded-full",
                  isTop && !unsure
                    ? "bg-accent-gradient"
                    : isTop && unsure
                      ? "bg-amber-400/80"
                      : "bg-border",
                )}
                // Top bar also gets a brief opacity fade-in for emphasis.
                initial={reduceMotion ? false : { width: 0, opacity: isTop ? 0 : 1 }}
                animate={{ width: `${pct * 100}%`, opacity: 1 }}
                transition={reduceMotion ? { duration: 0 } : transition}
              />
            </div>
            <span
              className={cn(
                "w-14 shrink-0 text-right font-mono text-xs tabular-nums",
                isTop ? "text-fg-muted" : "text-fg-subtle",
              )}
            >
              {formatPct(pred.prob)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
