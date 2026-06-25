import { describe, it, expect } from "vitest";

import { interpret, UNSURE_THRESHOLD } from "../confidence";
import type { InferenceResult } from "../inference";

function makeResult(
  top: { label: string; prob: number },
  second?: { label: string; prob: number },
): InferenceResult {
  const ranked = second ? [top, second] : [top];
  return {
    ranked: ranked.map((p) => ({ label: p.label, index: 0, prob: p.prob })),
    probs: new Float32Array(ranked.map((p) => p.prob)),
    top: { label: top.label, index: 0, prob: top.prob },
  };
}

describe("interpret — global threshold (unchanged default)", () => {
  it("flags unsure below the global threshold", () => {
    const v = interpret(makeResult({ label: "A", prob: 0.5 }));
    expect(v.unsure).toBe(true);
  });

  it("accepts at or above the global threshold", () => {
    const v = interpret(makeResult({ label: "A", prob: UNSURE_THRESHOLD + 0.01 }));
    expect(v.unsure).toBe(false);
  });
});

describe("interpret — per-class thresholds", () => {
  it("a sink class below its per-class floor is unsure even above global", () => {
    // S at 0.65 > global 0.6, but its per-class floor is 0.8 → unsure.
    const v = interpret(
      makeResult({ label: "S", prob: 0.65 }),
      UNSURE_THRESHOLD,
      { S: 0.8 },
    );
    expect(v.unsure).toBe(true);
  });

  it("a class without an entry uses the global threshold", () => {
    const v = interpret(
      makeResult({ label: "W", prob: 0.65 }),
      UNSURE_THRESHOLD,
      { S: 0.8 },
    );
    expect(v.unsure).toBe(false);
  });
});

describe("interpret — margin", () => {
  it("flags unsure when top1-top2 gap is below the margin", () => {
    const v = interpret(
      makeResult({ label: "T", prob: 0.52 }, { label: "S", prob: 0.48 }),
      0.4, // low global threshold so prob alone passes
      undefined,
      0.1, // require a 0.1 gap; actual gap is 0.04
    );
    expect(v.unsure).toBe(true);
  });

  it("accepts when the margin is satisfied", () => {
    const v = interpret(
      makeResult({ label: "T", prob: 0.8 }, { label: "S", prob: 0.1 }),
      0.4,
      undefined,
      0.1,
    );
    expect(v.unsure).toBe(false);
  });
});
