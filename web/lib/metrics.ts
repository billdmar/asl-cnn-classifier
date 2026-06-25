/**
 * Typed loaders + pure helpers for the model-evaluation metrics.
 *
 * HONESTY: every number surfaced in the dashboard originates from the committed
 * JSON under `web/public/metrics/`. This module only fetches and reshapes that
 * data — it never hardcodes a measured value. The metrics were computed on the
 * held-out test set described by {@link Metrics.num_test_samples}.
 *
 * NOTE on calibration: `public/metrics/calibration.json` is a REAL measurement —
 * ECE and the per-bin reliability curve were computed on the held-out real test
 * split with the trained MobileNetV2 checkpoint. {@link fetchCalibration} loads
 * it and {@link toReliabilityRows} reshapes the populated bins for charting.
 */

/** Per-class classification scores (precision/recall/F1 + sample count). */
export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

/** A single (true → predicted) confusion entry with its occurrence count. */
export interface ConfusedPair {
  true: string;
  pred: string;
  count: number;
}

/** Shape of `public/metrics/metrics.json`. */
export interface Metrics {
  overall_accuracy: number;
  macro_f1: number;
  macro_precision: number;
  macro_recall: number;
  per_class: Record<string, ClassMetrics>;
  most_confused_pairs: ConfusedPair[];
  num_test_samples: number;
  checkpoint: string;
  note: string;
}

/** A single training epoch from `public/metrics/training_history.json`. */
export interface TrainingEpoch {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
  lr: number;
}

/** Full training history (one entry per epoch, in epoch order). */
export type TrainingHistory = TrainingEpoch[];

/** A per-class row keyed by its letter, for chart consumption. */
export interface ClassRow extends ClassMetrics {
  letter: string;
}

/** Per-bin reliability arrays from `public/metrics/calibration.json`. */
export interface CalibrationBins {
  bin_lowers: number[];
  bin_uppers: number[];
  bin_acc: number[];
  bin_conf: number[];
  bin_count: number[];
}

/** Shape of `public/metrics/calibration.json`. */
export interface CalibrationData {
  ece: number;
  n_bins: number;
  num_test_samples: number;
  mean_confidence: number;
  accuracy: number;
  bins: CalibrationBins;
  checkpoint: string;
  data_dir: string;
  note: string;
}

/**
 * Shape of `public/metrics/realworld_eval.json` — the HONEST cross-dataset
 * number. The model is evaluated on a DIFFERENT dataset than it trained on
 * (different signers/backgrounds), so this is far lower than the same-dataset
 * `Metrics.overall_accuracy` and must never be conflated with it. `*_ay` fields
 * are the 24-letter A–Y headline (excluding the dynamic motion signs J and Z,
 * which a single static frame cannot represent — the mainstream convention).
 */
export interface RealworldEval {
  source: string;
  num_samples: number;
  hand_crop_used: boolean;
  num_no_hand_fallback: number;
  accuracy: number;
  macro_f1: number;
  macro_precision: number;
  macro_recall: number;
  accuracy_ay: number;
  macro_f1_ay: number;
  num_samples_ay: number;
  per_class: Record<string, ClassMetrics>;
  most_confused_pairs: ConfusedPair[];
  /** Class labels for the confusion matrix axes (row/col order). */
  confusion_labels: string[];
  /** Dense confusion matrix; `confusion_matrix[i][j]` = true i predicted j. */
  confusion_matrix: number[][];
  checkpoint: string;
  note: string;
}

/** A confusion-matrix cell normalized to a row fraction, ready for a heatmap. */
export interface ConfusionCell {
  trueLabel: string;
  predLabel: string;
  count: number;
  /** count / row total (recall-normalized); 0 when the row is empty. */
  fraction: number;
  isDiagonal: boolean;
}

/**
 * Row-normalize a confusion matrix into flat cells for heatmap rendering.
 * Each row is divided by its support so color encodes recall, not raw count
 * (classes have unequal support). Pure.
 */
export function confusionCells(
  matrix: number[][],
  labels: string[],
): ConfusionCell[] {
  const cells: ConfusionCell[] = [];
  for (let i = 0; i < matrix.length; i++) {
    const row = matrix[i] ?? [];
    const rowTotal = row.reduce((a, b) => a + b, 0);
    for (let j = 0; j < row.length; j++) {
      const count = row[j] ?? 0;
      cells.push({
        trueLabel: labels[i] ?? String(i),
        predLabel: labels[j] ?? String(j),
        count,
        fraction: rowTotal > 0 ? count / rowTotal : 0,
        isDiagonal: i === j,
      });
    }
  }
  return cells;
}

/** A single populated reliability-diagram bin, ready for charting. */
export interface ReliabilityRow {
  /** Bin midpoint ((lower + upper) / 2) — the x position. */
  midpoint: number;
  lower: number;
  upper: number;
  /** Empirical accuracy of samples in this bin (the reliability y value). */
  acc: number;
  /** Mean predicted confidence of samples in this bin. */
  conf: number;
  /** Number of test samples that fell into this bin. */
  count: number;
}

/**
 * Fetch the held-out test-set metrics.
 *
 * @returns The parsed {@link Metrics} object.
 * @throws If the request fails or returns a non-OK status.
 */
export async function fetchMetrics(): Promise<Metrics> {
  const res = await fetch("/metrics/metrics.json");
  if (!res.ok) {
    throw new Error(`Failed to load metrics.json: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as Metrics;
}

/**
 * Fetch the per-epoch training history.
 *
 * @returns The parsed {@link TrainingHistory} array.
 * @throws If the request fails or returns a non-OK status.
 */
export async function fetchTrainingHistory(): Promise<TrainingHistory> {
  const res = await fetch("/metrics/training_history.json");
  if (!res.ok) {
    throw new Error(
      `Failed to load training_history.json: ${res.status} ${res.statusText}`,
    );
  }
  return (await res.json()) as TrainingHistory;
}

/**
 * Fetch the held-out test-set calibration measurement (ECE + reliability bins).
 *
 * @returns The parsed {@link CalibrationData} object.
 * @throws If the request fails or returns a non-OK status.
 */
export async function fetchCalibration(): Promise<CalibrationData> {
  const res = await fetch("/metrics/calibration.json");
  if (!res.ok) {
    throw new Error(`Failed to load calibration.json: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as CalibrationData;
}

/**
 * Fetch an honest cross-dataset evaluation. Defaults to the primary gate
 * (`realworld_eval.json`); pass another name (e.g. `"realworld_eval_gate2"`) to
 * load a secondary gate. The caller may tolerate a missing secondary gate via
 * `.catch(() => null)`.
 *
 * @param name - Base filename (without extension) under `/metrics/`.
 * @returns The parsed {@link RealworldEval} object.
 * @throws If the request fails or returns a non-OK status.
 */
export async function fetchRealworldEval(
  name = "realworld_eval",
): Promise<RealworldEval> {
  const res = await fetch(`/metrics/${name}.json`);
  if (!res.ok) {
    throw new Error(`Failed to load ${name}.json: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as RealworldEval;
}

/**
 * Pair each reliability bin's midpoint with its empirical accuracy and
 * confidence, dropping empty bins (`bin_count === 0`) so the diagram only plots
 * regions backed by real samples. Bins are returned in ascending-confidence
 * order (the JSON's natural order).
 *
 * @param bins - The `bins` object from {@link CalibrationData}.
 * @returns One {@link ReliabilityRow} per populated bin.
 */
export function toReliabilityRows(bins: CalibrationBins): ReliabilityRow[] {
  const rows: ReliabilityRow[] = [];
  for (let i = 0; i < bins.bin_count.length; i += 1) {
    const count = bins.bin_count[i];
    const lower = bins.bin_lowers[i];
    const upper = bins.bin_uppers[i];
    const acc = bins.bin_acc[i];
    const conf = bins.bin_conf[i];
    if (
      count === undefined ||
      lower === undefined ||
      upper === undefined ||
      acc === undefined ||
      conf === undefined ||
      count <= 0
    ) {
      continue;
    }
    rows.push({ midpoint: (lower + upper) / 2, lower, upper, acc, conf, count });
  }
  return rows;
}

/**
 * Flatten `per_class` into rows. Defaults to alphabetical (A–Z) order; pass
 * `sortBy: "f1"` to sort ascending by F1 (worst classes first).
 *
 * @param perClass - The `per_class` map from {@link Metrics}.
 * @param sortBy - `"letter"` (default, A–Z) or `"f1"` (ascending F1).
 * @returns A new array of {@link ClassRow}.
 */
export function toClassRows(
  perClass: Record<string, ClassMetrics>,
  sortBy: "letter" | "f1" = "letter",
): ClassRow[] {
  const rows: ClassRow[] = Object.entries(perClass).map(([letter, m]) => ({
    letter,
    ...m,
  }));
  rows.sort((a, b) =>
    sortBy === "f1"
      ? a.f1 - b.f1 || a.letter.localeCompare(b.letter)
      : a.letter.localeCompare(b.letter),
  );
  return rows;
}

/**
 * Return the top-N confused pairs, sorted by descending count (ties broken by
 * true-label then pred-label for stable ordering).
 *
 * @param pairs - `most_confused_pairs` from {@link Metrics}.
 * @param n - Maximum number of pairs to return (default 10).
 * @returns A new, sorted array (never mutates the input).
 */
export function topConfusedPairs(pairs: ConfusedPair[], n = 10): ConfusedPair[] {
  return [...pairs]
    .sort(
      (a, b) =>
        b.count - a.count || a.true.localeCompare(b.true) || a.pred.localeCompare(b.pred),
    )
    .slice(0, n);
}

/**
 * The epoch with the highest validation accuracy.
 *
 * @param history - The training history.
 * @returns The best epoch, or `undefined` if the history is empty.
 */
export function bestValEpoch(history: TrainingHistory): TrainingEpoch | undefined {
  return history.reduce<TrainingEpoch | undefined>((best, e) => {
    if (best === undefined || e.val_acc > best.val_acc) return e;
    return best;
  }, undefined);
}
