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

describe("toClassRows", () => {
  it("produces one row per class (26 letters)", () => {
    expect(toClassRows(metrics.per_class)).toHaveLength(26);
  });

  it("defaults to A–Z order and carries through real values", () => {
    const rows = toClassRows(metrics.per_class);
    expect(rows[0]?.letter).toBe("A");
    expect(rows[25]?.letter).toBe("Z");
    // A is a perfect class in the real data.
    expect(rows[0]?.f1).toBe(1.0);
    expect(rows[0]?.support).toBe(63);
  });

  it("sorts ascending by F1 (worst class first)", () => {
    const rows = toClassRows(metrics.per_class, "f1");
    const f1s = rows.map((r) => r.f1);
    const sorted = [...f1s].sort((a, b) => a - b);
    expect(f1s).toEqual(sorted);
    // N has the lowest F1 in the real data (0.92561…).
    expect(rows[0]?.letter).toBe("N");
    expect(rows[0]?.f1).toBeCloseTo(0.9256198347107438, 10);
  });
});

describe("topConfusedPairs", () => {
  it("ranks by descending count and is capped at N", () => {
    const top = topConfusedPairs(metrics.most_confused_pairs, 3);
    expect(top).toHaveLength(3);
    expect(top[0]).toEqual({ true: "P", pred: "Q", count: 4 });
    expect(top[1]).toEqual({ true: "V", pred: "W", count: 4 });
    expect(top[0]!.count).toBeGreaterThanOrEqual(top[2]!.count);
  });

  it("does not mutate the input array", () => {
    const original = [...metrics.most_confused_pairs];
    topConfusedPairs(metrics.most_confused_pairs, 2);
    expect(metrics.most_confused_pairs).toEqual(original);
  });
});

describe("bestValEpoch", () => {
  it("finds the peak validation-accuracy epoch from the real history", () => {
    const best = bestValEpoch(history);
    expect(best?.epoch).toBe(12);
    expect(best?.val_acc).toBeCloseTo(0.9779276517473943, 10);
  });

  it("returns undefined for empty history", () => {
    expect(bestValEpoch([])).toBeUndefined();
  });
});
