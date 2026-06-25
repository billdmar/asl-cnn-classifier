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
  type CalibrationData,
  type Metrics,
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

describe("toClassRows", () => {
  it("produces one row per class (26 letters)", () => {
    expect(toClassRows(metrics.per_class)).toHaveLength(26);
  });

  it("defaults to A–Z order and carries through real values", () => {
    const rows = toClassRows(metrics.per_class);
    expect(rows[0]?.letter).toBe("A");
    expect(rows[25]?.letter).toBe("Z");
    // A is a near-perfect class in the real data.
    expect(rows[0]?.f1).toBeCloseTo(0.9965156794425087, 10);
    expect(rows[0]?.support).toBe(144);
  });

  it("sorts ascending by F1 (worst class first)", () => {
    const rows = toClassRows(metrics.per_class, "f1");
    const f1s = rows.map((r) => r.f1);
    const sorted = [...f1s].sort((a, b) => a - b);
    expect(f1s).toEqual(sorted);
    // N has the lowest F1 in the real data.
    expect(rows[0]?.letter).toBe("N");
    expect(rows[0]?.f1).toBeCloseTo(0.9178743961352657, 10);
  });
});

describe("topConfusedPairs", () => {
  it("ranks by descending count and is capped at N", () => {
    const top = topConfusedPairs(metrics.most_confused_pairs, 3);
    expect(top).toHaveLength(3);
    expect(top[0]).toEqual({ true: "N", pred: "M", count: 8 });
    expect(top[1]).toEqual({ true: "X", pred: "I", count: 4 });
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
    // Only the [0.0, 0.1] bin is empty; 9 of 10 remain populated.
    expect(rows).toHaveLength(9);
    expect(rows.every((r) => r.count > 0)).toBe(true);
  });

  it("pairs bin midpoints with the real per-bin accuracy and confidence", () => {
    const rows = toReliabilityRows(calibration.bins);
    // The most-populated bin is the last one: [0.9, 1.0].
    const last = rows[rows.length - 1];
    expect(last?.midpoint).toBeCloseTo(0.95, 10);
    expect(last?.lower).toBe(0.9);
    expect(last?.upper).toBe(1.0);
    expect(last?.acc).toBeCloseTo(0.9987725040916531, 10);
    expect(last?.conf).toBeCloseTo(0.9856753365844564, 10);
    expect(last?.count).toBe(2444);
    // The first populated bin is [0.1, 0.2] with 1 sample.
    expect(rows[0]?.midpoint).toBeCloseTo(0.15, 10);
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
    expect(best?.epoch).toBe(12);
    expect(best?.val_acc).toBeCloseTo(0.9613393165343459, 10);
  });

  it("returns undefined for empty history", () => {
    expect(bestValEpoch([])).toBeUndefined();
  });
});
