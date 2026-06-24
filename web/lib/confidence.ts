/**
 * Confidence interpretation and the honest "unsure" state.
 *
 * The model was trained on uniform images and is over-confident on cluttered
 * real-world webcam frames. Until a measured temperature-scaling calibration
 * ships (the ML calibration workstream), we render low-confidence predictions
 * as an explicit "unsure" prompt rather than a confident wrong letter.
 *
 * IMPORTANT (honesty): the threshold below is a HEURISTIC, not a calibrated
 * probability. It is labeled as such in the UI. When real reliability data is
 * produced, replace {@link UNSURE_THRESHOLD} with the calibrated operating point
 * and update the copy.
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
