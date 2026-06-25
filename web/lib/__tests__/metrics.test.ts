/**
 * Unit tests for the pure metrics helpers, asserted against the REAL committed
 * JSON (`public/metrics/*.json`). These tests double as a guard that the helper
 * outputs stay consistent with the data the dashboard renders.
 */

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";

import {
  toClassRows,
  topConfusedPairs,
  bestValEpoch,
  toReliabilityRows,
  confusionCells,
  type CalibrationData,
  type Metrics,
  type RealworldEval,
  type TrainingHistory,
} from "../metrics";

const METRICS_DIR = path.resolve(__dirname, "../../public/metrics");

const metrics = JSON.parse(
  readFileSync(path.join(METRICS_DIR, "metrics.json"), "utf-8"),
) as Metrics;

const history = JSON.parse(
  readFileSync(path.join(METRICS_DIR, "training_history.json"), "utf-8"),
) as TrainingHistory;

const calibration = JSON.parse(
  readFileSync(path.join(METRICS_DIR, "calibration.json"), "utf-8"),
) as CalibrationData;

const realworld = JSON.parse(
  readFileSync(path.join(METRICS_DIR, "realworld_eval.json"), "utf-8"),
) as RealworldEval;

describe("toClassRows", () => {
  it("produces one row per class (26 letters)", () => {
    expect(toClassRows(metrics.per_class)).toHaveLength(26);
  });

  it("defaults to A–Z order and carries through real values", () => {
    const rows = toClassRows(metrics.per_class);
    expect(rows[0]?.letter).toBe("A");
    expect(rows[25]?.letter).toBe("Z");
    // A is a near-perfect class in the real data.
    expect(rows[0]?.f1).toBeCloseTo(0.9838187702265372, 10);
    expect(rows[0]?.support).toBe(155);
  });

  it("sorts ascending by F1 (worst class first)", () => {
    const rows = toClassRows(metrics.per_class, "f1");
    const f1s = rows.map((r) => r.f1);
    const sorted = [...f1s].sort((a, b) => a - b);
    expect(f1s).toEqual(sorted);
    // N has the lowest F1 in the real data.
    expect(rows[0]?.letter).toBe("N");
    expect(rows[0]?.f1).toBeCloseTo(0.9016393442622951, 10);
  });
});

describe("topConfusedPairs", () => {
  it("ranks by descending count and is capped at N", () => {
    const top = topConfusedPairs(metrics.most_confused_pairs, 3);
    expect(top).toHaveLength(3);
    expect(top[0]).toEqual({ true: "M", pred: "N", count: 9 });
    expect(top[1]).toEqual({ true: "T", pred: "N", count: 6 });
    expect(top[0]!.count).toBeGreaterThanOrEqual(top[2]!.count);
  });

  it("does not mutate the input array", () => {
    const original = [...metrics.most_confused_pairs];
    topConfusedPairs(metrics.most_confused_pairs, 2);
    expect(metrics.most_confused_pairs).toEqual(original);
  });
});

describe("toReliabilityRows", () => {
  it("drops empty bins and keeps only populated ones from the real data", () => {
    const rows = toReliabilityRows(calibration.bins);
    // 8 of 10 bins are populated for the deployed model.
    expect(rows).toHaveLength(8);
    expect(rows.every((r) => r.count > 0)).toBe(true);
  });

  it("pairs bin midpoints with the real per-bin accuracy and confidence", () => {
    const rows = toReliabilityRows(calibration.bins);
    // The most-populated bin is the last one: [0.9, 1.0].
    const last = rows[rows.length - 1];
    expect(last?.midpoint).toBeCloseTo(0.95, 10);
    expect(last?.lower).toBe(0.9);
    expect(last?.upper).toBe(1.0);
    expect(last?.acc).toBeCloseTo(0.9970446989287034, 10);
    expect(last?.conf).toBeCloseTo(0.9863262718230662, 10);
    expect(last?.count).toBe(2707);
    // The first populated bin is [0.2, 0.3] with 1 sample.
    expect(rows[0]?.midpoint).toBeCloseTo(0.25, 10);
    expect(rows[0]?.count).toBe(1);
  });

  it("returns rows in ascending-confidence (JSON) order", () => {
    const rows = toReliabilityRows(calibration.bins);
    const mids = rows.map((r) => r.midpoint);
    expect(mids).toEqual([...mids].sort((a, b) => a - b));
  });
});

describe("bestValEpoch", () => {
  it("finds the peak validation-accuracy epoch from the real history", () => {
    const best = bestValEpoch(history);
    expect(best?.epoch).toBe(11);
    expect(best?.val_acc).toBeCloseTo(0.9725465446513095, 10);
  });

  it("returns undefined for empty history", () => {
    expect(bestValEpoch([])).toBeUndefined();
  });
});

describe("realworld_eval (honest cross-dataset number)", () => {
  it("exposes the A-Y headline and full 26-class fields", () => {
    expect(typeof realworld.accuracy).toBe("number");
    expect(typeof realworld.accuracy_ay).toBe("number");
    expect(typeof realworld.macro_f1_ay).toBe("number");
    expect(realworld.num_samples).toBeGreaterThan(0);
  });

  it("is far below the same-dataset benchmark (the whole point)", () => {
    // The honest cross-dataset number must be much lower than same-dataset.
    expect(realworld.accuracy).toBeLessThan(metrics.overall_accuracy - 0.2);
  });

  it("A-Y headline excludes the weak J/Z so it is >= the 26-class number", () => {
    expect(realworld.accuracy_ay).toBeGreaterThanOrEqual(realworld.accuracy);
  });

  it("carries an honest cross-dataset note", () => {
    expect(realworld.note.toLowerCase()).toContain("cross-dataset");
  });

  it("ships a square 26×26 confusion matrix aligned to its labels", () => {
    expect(realworld.confusion_labels).toHaveLength(26);
    expect(realworld.confusion_matrix).toHaveLength(26);
    for (const row of realworld.confusion_matrix) {
      expect(row).toHaveLength(26);
    }
  });

  it("confusion row sums equal per-class support (recall denominator)", () => {
    realworld.confusion_labels.forEach((label, i) => {
      const rowSum = realworld.confusion_matrix[i]!.reduce((a, b) => a + b, 0);
      expect(rowSum).toBe(realworld.per_class[label]!.support);
    });
  });
});

describe("confusionCells", () => {
  it("row-normalizes counts into recall fractions and flags the diagonal", () => {
    const cells = confusionCells(
      [
        [8, 2],
        [0, 5],
      ],
      ["A", "B"],
    );
    expect(cells).toHaveLength(4);
    const aa = cells.find((c) => c.trueLabel === "A" && c.predLabel === "A")!;
    expect(aa.isDiagonal).toBe(true);
    expect(aa.fraction).toBeCloseTo(0.8, 10); // 8 / (8+2)
    const ab = cells.find((c) => c.trueLabel === "A" && c.predLabel === "B")!;
    expect(ab.isDiagonal).toBe(false);
    expect(ab.fraction).toBeCloseTo(0.2, 10);
  });

  it("handles an empty row without dividing by zero", () => {
    const cells = confusionCells([[0, 0]], ["A", "B"]);
    expect(cells.every((c) => c.fraction === 0)).toBe(true);
  });
});
