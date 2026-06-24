/**
 * Typed loaders + pure helpers for the model-evaluation metrics.
 *
 * HONESTY: every number surfaced in the dashboard originates from the committed
 * JSON under `web/public/metrics/`. This module only fetches and reshapes that
 * data — it never hardcodes a measured value. The metrics were computed on the
 * held-out test set described by {@link Metrics.num_test_samples}.
 *
 * NOTE on calibration: `public/metrics/calibration.json` is NOT yet a real
 * measurement (it was produced on a synthetic fixture), so this module
 * deliberately exposes NO calibration/ECE/reliability loader. The dashboard
 * renders a clearly-labeled "coming with the calibration workstream" placeholder
 * instead.
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
