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
  /** True when top-1 confidence is below its threshold (per-class or global). */
  unsure: boolean;
  /** The top prediction regardless of confidence. */
  top: Prediction;
  /** Short guidance shown in the unsure state. */
  hint: string;
}

/**
 * Optional per-class acceptance thresholds (from `fit_thresholds.py`, shipped in
 * `calibration.json`). When the top class has an entry, that threshold overrides
 * the global one — used to curb over-predicted "sink" classes (e.g. S, Q) by
 * demanding higher confidence before we present them.
 */
export type ClassThresholds = Record<string, number>;

const UNSURE_HINT =
  "Unsure — try a plainer background, better lighting, and center your hand in the box.";

/**
 * Decide whether a result should render as confident or "unsure".
 *
 * The effective threshold for the top class is, in priority order: its entry in
 * `classThresholds` (if present), else `threshold` (the global operating point).
 * An optional `margin` additionally requires the top1−top2 probability gap to
 * clear it — a small margin means two classes are nearly tied (the T-vs-S call),
 * which we surface as "unsure" rather than guessing.
 *
 * @param result - An inference result from the classifier.
 * @param threshold - Global override for {@link UNSURE_THRESHOLD}.
 * @param classThresholds - Optional per-class acceptance thresholds.
 * @param margin - Optional minimum top1−top2 gap.
 * @returns The presentation verdict.
 */
export function interpret(
  result: InferenceResult,
  threshold: number = UNSURE_THRESHOLD,
  classThresholds?: ClassThresholds,
  margin?: number,
): ConfidenceVerdict {
  const perClass = classThresholds?.[result.top.label];
  const effective = perClass ?? threshold;
  let unsure = result.top.prob < effective;

  if (!unsure && margin !== undefined && result.ranked.length > 1) {
    const gap = result.top.prob - result.ranked[1]!.prob;
    if (gap < margin) unsure = true;
  }

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
