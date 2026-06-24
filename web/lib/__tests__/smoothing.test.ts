/**
 * Unit tests for the pure webcam temporal-smoothing helpers.
 */

import { describe, it, expect } from "vitest";

import { averageProbs, rankFromProbs } from "../../components/webcam/smoothing";

describe("averageProbs", () => {
  it("returns an empty vector for empty input", () => {
    expect(averageProbs([])).toEqual(new Float32Array(0));
  });

  it("returns the same values for a single buffer", () => {
    const probs = new Float32Array([0.1, 0.7, 0.2]);
    const avg = averageProbs([probs]);
    expect(avg[0]).toBeCloseTo(0.1, 6);
    expect(avg[1]).toBeCloseTo(0.7, 6);
    expect(avg[2]).toBeCloseTo(0.2, 6);
  });

  it("averages element-wise across buffers", () => {
    const a = new Float32Array([0.0, 1.0, 0.0]);
    const b = new Float32Array([1.0, 0.0, 0.0]);
    const avg = averageProbs([a, b]);
    expect(avg[0]).toBeCloseTo(0.5, 6);
    expect(avg[1]).toBeCloseTo(0.5, 6);
    expect(avg[2]).toBeCloseTo(0.0, 6);
  });

  it("does not mutate its inputs", () => {
    const a = new Float32Array([0.2, 0.8]);
    const b = new Float32Array([0.6, 0.4]);
    averageProbs([a, b]);
    expect(a[0]).toBeCloseTo(0.2, 6);
    expect(a[1]).toBeCloseTo(0.8, 6);
    expect(b[0]).toBeCloseTo(0.6, 6);
    expect(b[1]).toBeCloseTo(0.4, 6);
  });

  it("smooths out a single-frame flicker", () => {
    // Four frames favor class 0; one outlier favors class 1. The average
    // should still rank class 0 first.
    const steady = new Float32Array([0.8, 0.2]);
    const flicker = new Float32Array([0.3, 0.7]);
    const avg = averageProbs([steady, steady, steady, steady, flicker]);
    expect(avg[0]! > avg[1]!).toBe(true);
  });
});

describe("rankFromProbs", () => {
  const labels = ["A", "B", "C"];

  it("ranks predictions descending and exposes top-1", () => {
    const result = rankFromProbs(new Float32Array([0.1, 0.7, 0.2]), labels);
    expect(result.ranked.map((p) => p.label)).toEqual(["B", "C", "A"]);
    expect(result.top.label).toBe("B");
    expect(result.top.index).toBe(1);
    expect(result.top.prob).toBeCloseTo(0.7, 6);
  });

  it("falls back to the index string when a label is missing", () => {
    const result = rankFromProbs(new Float32Array([0.9, 0.1]), ["A"]);
    expect(result.ranked[0]?.label).toBe("A");
    expect(result.ranked[1]?.label).toBe("1");
  });

  it("returns a synthetic top for an empty vector", () => {
    const result = rankFromProbs(new Float32Array(0), labels);
    expect(result.ranked).toHaveLength(0);
    expect(result.top).toEqual({ label: "", index: -1, prob: 0 });
  });
});
