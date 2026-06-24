/**
 * Confidence interpretation and the honest "unsure" state.
 *
 * The model was trained on uniform images and is over-confident on cluttered
 * real-world webcam frames. We render low-confidence predictions as an explicit
 * "unsure" prompt rather than a confident wrong letter.
 *
 * IMPORTANT (honesty): temperature scaling is now WIRED end-to-end (see
 * `lib/inference.ts::applyTemperature`), but it is set to T=1.0 (identity, no
 * behavior change) pending a fit on deployment-like data. A fit on the clean
 * benchmark would sharpen (the model is under-confident there) and is
 * deliberately not shipped. Consequently the threshold below remains a
 * HEURISTIC, not yet a calibrated probability — it is labeled as such in the UI.
 * When a real-world temperature + reliability fit lands, revisit
 * {@link UNSURE_THRESHOLD} as the calibrated operating point and update the copy.
 */

import type { InferenceResult, Prediction } from "./inference";

/**
 * Top-1 probability below which we show "unsure". Chosen heuristically as a
 * conservative operating point; NOT derived from a reliability diagram yet.
 */
export const UNSURE_THRESHOLD = 0.6;

/** How a prediction should be presented to the user. */
export interface ConfidenceVerdict {
  /** True when top-1 confidence is below {@link UNSURE_THRESHOLD}. */
  unsure: boolean;
  /** The top prediction regardless of confidence. */
  top: Prediction;
  /** Short guidance shown in the unsure state. */
  hint: string;
}

const UNSURE_HINT =
  "Unsure — try a plainer background, better lighting, and center your hand in the box.";

/**
 * Decide whether a result should render as confident or "unsure".
 *
 * @param result - An inference result from the classifier.
 * @param threshold - Override for {@link UNSURE_THRESHOLD}.
 * @returns The presentation verdict.
 */
export function interpret(
  result: InferenceResult,
  threshold: number = UNSURE_THRESHOLD,
): ConfidenceVerdict {
  const unsure = result.top.prob < threshold;
  return {
    unsure,
    top: result.top,
    hint: unsure ? UNSURE_HINT : "",
  };
}

/**
 * Format a probability in [0,1] as a percentage string, e.g. `0.973` → "97.3%".
 */
export function formatPct(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`;
}
