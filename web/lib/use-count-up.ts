"use client";

/**
 * Animated number counter for headline stats.
 *
 * Eases from 0 → `to` over `durationMs` using requestAnimationFrame. Two
 * correctness guarantees that protect the gates:
 * - The INITIAL render returns `to` (the real value), so the static export /
 *   Lighthouse / no-JS view shows the true number, never a `0`.
 * - Under reduced motion (or `start === false`) it stays at `to` — no animation.
 *
 * The animation only runs once `start` flips true (tie it to in-view), counting
 * up from 0 to the final value exactly once.
 */

import { useEffect, useRef, useState } from "react";

export interface UseCountUpOptions {
  /** Final value to count to. */
  to: number;
  /** Animation duration in ms (default 800). */
  durationMs?: number;
  /** Begin the count-up (e.g. when the card scrolls into view). Default true. */
  start?: boolean;
  /** Skip the animation and jump straight to `to`. */
  reduceMotion?: boolean;
}

/** Cubic ease-out, matching the rest of the motion vocabulary. */
function easeOut(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

/** Returns the current animated value (starts and ends at `to`). */
export function useCountUp({
  to,
  durationMs = 800,
  start = true,
  reduceMotion = false,
}: UseCountUpOptions): number {
  // Initialize at the final value so SSR/first paint shows the real number.
  const [value, setValue] = useState(to);
  const rafRef = useRef<number | null>(null);
  const hasRun = useRef(false);

  useEffect(() => {
    if (reduceMotion || !start || hasRun.current) {
      setValue(to);
      return;
    }
    hasRun.current = true;
    const startTime = performance.now();
    setValue(0);

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const t = Math.min(1, elapsed / durationMs);
      setValue(to * easeOut(t));
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        setValue(to);
      }
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [to, durationMs, start, reduceMotion]);

  return value;
}
