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

export interface UseClassifier {
  /** Current warm-up lifecycle state. */
  status: WarmupStatus;
  /** Human-readable error message when {@link status} is `"error"`. */
  error: string | null;
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
  // Guard against double warm-up (React 18/19 strict-mode double effects).
  const startedRef = useRef(false);

  const warmUp = useCallback(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    setStatus("warming");
    setError(null);
    void Promise.all([getSession(), getHandLandmarker()])
      .then(() => {
        setStatus("ready");
      })
      .catch((err: unknown) => {
        startedRef.current = false;
        setStatus("error");
        setError(err instanceof Error ? err.message : "Failed to load the model.");
      });
  }, []);

  const classify = useCallback(
    (source: CanvasImageSource): Promise<InferenceResult> => classifyImage(source),
    [],
  );

  useEffect(() => {
    if (autoWarm) warmUp();
  }, [autoWarm, warmUp]);

  return { status, error, warmUp, classify };
}
