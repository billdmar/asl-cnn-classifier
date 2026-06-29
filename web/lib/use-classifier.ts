"use client";

/**
 * Shared classifier warm-up + classify helper.
 *
 * Both the live-webcam and the (upcoming) upload feature want the same thing:
 * preload the ONNX session and the MediaPipe hand landmarker once, then run
 * classification on demand. This hook centralizes that warm-up so the heavy
 * WASM downloads happen a single time and are shared across features.
 *
 * SSR-safe: all browser API access is deferred into an effect / async handlers,
 * never executed at module load or during render.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { classifyImage, getSession, type InferenceResult } from "@/lib/inference";
import { getHandLandmarker } from "@/lib/handcrop";

/** Lifecycle of the shared warm-up. */
export type WarmupStatus = "idle" | "warming" | "ready" | "error";

/**
 * How long warm-up may run before we flag it as `slow` (still loading, not an
 * error). First visits download ~9 MB; on a slow link the user benefits from a
 * "still working" reassurance + a retry escape hatch.
 */
export const SLOW_WARMUP_MS = 12_000;

/**
 * Pure decision: should the `slow` flag be raised when the timer fires?
 * Only when warm-up is still in flight (`"warming"`). Extracted so the timing
 * logic is unit-testable without a React render environment.
 */
export function isSlowWarmup(statusWhenTimerFired: WarmupStatus): boolean {
  return statusWhenTimerFired === "warming";
}

export interface UseClassifier {
  /** Current warm-up lifecycle state. */
  status: WarmupStatus;
  /** Human-readable error message when {@link status} is `"error"`. */
  error: string | null;
  /** Model-download progress in [0, 1] during `"warming"` (0 until bytes arrive). */
  progress: number;
  /**
   * True when warm-up has been running longer than {@link SLOW_WARMUP_MS} and is
   * still `"warming"`. A loading hint, NOT an error — status stays `"warming"`.
   */
  slow: boolean;
  /** Kick off (or re-try) preloading the model + landmarker. Idempotent. */
  warmUp: () => void;
  /** Run end-to-end classification on a canvas-drawable source. */
  classify: (source: CanvasImageSource) => Promise<InferenceResult>;
}

/**
 * Preload the ONNX session and hand landmarker, exposing a `classify` helper.
 *
 * @param autoWarm - When true, start warm-up on mount (defaults to false so the
 *   caller can defer heavy downloads until the user opts in).
 */
export function useClassifier(autoWarm = false): UseClassifier {
  const [status, setStatus] = useState<WarmupStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [slow, setSlow] = useState(false);
  // Guard against double warm-up (React 18/19 strict-mode double effects).
  const startedRef = useRef(false);
  // Mirror status into a ref so the slow-timer can read the live value.
  const statusRef = useRef<WarmupStatus>("idle");
  // The pending slow-warmup timer, so we can clear it on settle/unmount/retry.
  const slowTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearSlowTimer = useCallback(() => {
    if (slowTimerRef.current !== null) {
      clearTimeout(slowTimerRef.current);
      slowTimerRef.current = null;
    }
  }, []);

  const setWarmStatus = useCallback((next: WarmupStatus) => {
    statusRef.current = next;
    setStatus(next);
  }, []);

  const warmUp = useCallback(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    setWarmStatus("warming");
    setError(null);
    setProgress(0);
    setSlow(false);
    clearSlowTimer();
    slowTimerRef.current = setTimeout(() => {
      // Still loading (not errored) ⇒ surface the "slow" hint, keep "warming".
      if (isSlowWarmup(statusRef.current)) setSlow(true);
    }, SLOW_WARMUP_MS);
    void Promise.all([getSession(undefined, setProgress), getHandLandmarker()])
      .then(() => {
        clearSlowTimer();
        setSlow(false);
        setWarmStatus("ready");
      })
      .catch((err: unknown) => {
        clearSlowTimer();
        setSlow(false);
        // Allow a re-entrant retry: reset the guard so warmUp() runs again.
        startedRef.current = false;
        setWarmStatus("error");
        setError(err instanceof Error ? err.message : "Failed to load the model.");
      });
  }, [clearSlowTimer, setWarmStatus]);

  const classify = useCallback(
    (source: CanvasImageSource): Promise<InferenceResult> => classifyImage(source),
    [],
  );

  useEffect(() => {
    if (autoWarm) warmUp();
  }, [autoWarm, warmUp]);

  // Clear any pending slow-timer on unmount so it can't fire after teardown.
  useEffect(() => clearSlowTimer, [clearSlowTimer]);

  return { status, error, progress, slow, warmUp, classify };
}
