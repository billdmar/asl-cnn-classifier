/**
 * Temperature-scaling unit tests for the inference path.
 *
 * Temperature scaling is WIRED end-to-end but shipped INERT at T=1.0 (identity).
 * These tests prove (a) T=1.0 leaves probabilities exactly equal to the plain
 * softmax the parity gate checks, and (b) T>1 softens the distribution (lower
 * top-1 probability) — so the path is live and ready for a real-world fit, with
 * no behavior change today.
 */

import { describe, it, expect, afterEach, vi } from "vitest";
import { softmax } from "../preprocess";
import {
  applyTemperature,
  getTemperature,
  resetTemperature,
  DEFAULT_TEMPERATURE,
} from "../inference";

const LOGITS = new Float32Array([2.0, 1.0, 0.1, -1.5, 3.0]);

describe("applyTemperature", () => {
  it("T=1.0 is the identity: equals plain softmax (parity stays exact)", () => {
    const plain = softmax(LOGITS);
    const calibrated = applyTemperature(LOGITS, 1.0);
    expect(calibrated.length).toBe(plain.length);
    for (let i = 0; i < plain.length; i++) {
      expect(calibrated[i]).toBe(plain[i]); // exact equality, not just close
    }
  });

  it("T>1 softens the distribution: lower top-1 probability", () => {
    const base = applyTemperature(LOGITS, 1.0);
    const softened = applyTemperature(LOGITS, 2.0);
    const max = (a: Float32Array): number => a.reduce((m, v) => Math.max(m, v), 0);
    expect(max(softened)).toBeLessThan(max(base));
    // Still a valid distribution.
    const sum = softened.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
    // Argmax (the predicted class) is unchanged by positive scaling.
    const argmax = (a: Float32Array): number => {
      let k = 0;
      for (let i = 1; i < a.length; i++) if (a[i]! > a[k]!) k = i;
      return k;
    };
    expect(argmax(softened)).toBe(argmax(base));
  });
});

describe("getTemperature", () => {
  afterEach(() => {
    resetTemperature();
    vi.restoreAllMocks();
  });

  it("reads temperature from /model/calibration.json", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, json: async () => ({ temperature: 1.0 }) })),
    );
    await expect(getTemperature()).resolves.toBe(1.0);
  });

  it("falls back to the identity default when fetch fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("offline");
      }),
    );
    await expect(getTemperature()).resolves.toBe(DEFAULT_TEMPERATURE);
  });

  it("rejects non-positive temperatures, degrading to the identity", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, json: async () => ({ temperature: 0 }) })),
    );
    await expect(getTemperature()).resolves.toBe(DEFAULT_TEMPERATURE);
  });

  it("caches the first result (one fetch only)", async () => {
    const spy = vi.fn(async () => ({
      ok: true,
      json: async () => ({ temperature: 1.0 }),
    }));
    vi.stubGlobal("fetch", spy);
    await getTemperature();
    await getTemperature();
    expect(spy).toHaveBeenCalledTimes(1);
  });
});
