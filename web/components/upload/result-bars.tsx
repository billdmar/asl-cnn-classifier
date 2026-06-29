"use client";

import { motion, useReducedMotion } from "framer-motion";

import type { Prediction } from "@/lib/inference";
import { formatPct } from "@/lib/confidence";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { cn } from "@/lib/utils";

interface ResultBarsProps {
  /** Ranked predictions (already sorted descending by probability). */
  ranked: Prediction[];
  /** How many top predictions to render. */
  count?: number;
  /** When true, render the top bar in the amber "unsure" state. */
  unsure?: boolean;
}

/**
 * Inline, self-contained top-N confidence bars for the upload result.
 *
 * Deliberately local to components/upload to avoid a cross-directory dependency
 * on the webcam agent's confidence-bars component. Visually consistent: animated
 * width, accent gradient, percent labels, and an amber state for low confidence.
 */
export function ResultBars({ ranked, count = 5, unsure = false }: ResultBarsProps) {
  const reduceMotion = useReducedMotion();
  const top = ranked.slice(0, count);

  return (
    <motion.ul
      className="flex flex-col gap-2"
      aria-label="Top predictions"
      variants={reduceMotion ? undefined : staggerContainer(0.05)}
      initial={reduceMotion ? false : "hidden"}
      animate={reduceMotion ? false : "visible"}
    >
      {top.map((prediction, rank) => {
        const isTop = rank === 0;
        const amber = isTop && unsure;
        const widthPct = `${Math.max(prediction.prob * 100, 1.5)}%`;
        return (
          <motion.li
            key={prediction.index}
            className="flex items-center gap-3"
            variants={reduceMotion ? undefined : staggerItem}
          >
            <span className="w-5 shrink-0 text-center font-mono text-sm font-semibold text-fg">
              {prediction.label}
            </span>
            <div className="relative h-3 flex-1 overflow-hidden rounded-full bg-bg-subtle">
              <motion.div
                className={cn(
                  "absolute inset-y-0 left-0 rounded-full",
                  amber ? "bg-amber-500" : isTop ? "bg-accent-gradient" : "bg-accent/40",
                )}
                initial={{ width: reduceMotion ? widthPct : 0 }}
                animate={{ width: widthPct }}
                transition={
                  reduceMotion ? { duration: 0 } : { duration: 0.5, ease: "easeOut" }
                }
              />
            </div>
            <span
              className={cn(
                "w-14 shrink-0 text-right font-mono text-xs tabular-nums",
                amber ? "text-amber-400" : "text-fg-muted",
              )}
            >
              {formatPct(prediction.prob)}
            </span>
          </motion.li>
        );
      })}
    </motion.ul>
  );
}
