/**
 * Pure helpers for temporal smoothing of webcam inference.
 *
 * The raw per-frame classifier output flickers between visually-similar letters
 * on a live camera. We stabilize the *displayed* prediction by averaging the
 * last few probability vectors and re-deriving the ranking from that average.
 *
 * These functions are deliberately pure (no DOM, no React, no model access) so
 * they are unit-testable and SSR-safe.
 */

import type { InferenceResult, Prediction } from "@/lib/inference";

/**
 * Number of recent probability vectors to average for the smoothed prediction.
 * Five frames at the ~8 fps classify cadence is roughly a half-second window —
 * long enough to kill single-frame flicker, short enough to stay responsive.
 */
export const SMOOTHING_WINDOW = 5;

/**
 * Element-wise average of a list of equal-length probability vectors.
 *
 * @param buffers - One or more probability vectors (all the same length).
 * @returns A new Float32Array holding the per-class mean. Empty input yields an
 *   empty array; mismatched lengths fall back to the first vector's length and
 *   only average over indices present in each buffer.
 */
export function averageProbs(buffers: readonly Float32Array[]): Float32Array {
  const first = buffers[0];
  if (!first) return new Float32Array(0);

  const length = first.length;
  const sum = new Float32Array(length);
  for (const buf of buffers) {
    for (let i = 0; i < length; i++) {
      // noUncheckedIndexedAccess: indices < length, but a shorter buffer would
      // yield undefined — coalesce to 0 so a malformed buffer can't poison NaN.
      sum[i]! += buf[i] ?? 0;
    }
  }

  const count = buffers.length;
  for (let i = 0; i < length; i++) {
    sum[i]! /= count;
  }
  return sum;
}

/**
 * Build a ranked {@link InferenceResult} from a probability vector and labels.
 *
 * @param probs - Probability vector in class-index order.
 * @param labels - Class labels; `labels[i]` names class index `i`.
 * @returns Ranked predictions (descending), the probs vector, and the top-1.
 *   For an empty vector the top-1 is a synthetic `{ label: "", index: -1,
 *   prob: 0 }` so callers can rely on `top` being present.
 */
export function rankFromProbs(
  probs: Float32Array,
  labels: readonly string[],
): InferenceResult {
  const ranked: Prediction[] = Array.from(probs, (prob, index) => ({
    label: labels[index] ?? String(index),
    index,
    prob,
  })).sort((a, b) => b.prob - a.prob);

  const top: Prediction = ranked[0] ?? { label: "", index: -1, prob: 0 };
  return { ranked, probs, top };
}
