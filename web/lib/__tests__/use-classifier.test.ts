/**
 * Tests for the slow-warmup timeout decision.
 *
 * The full `useClassifier` hook depends on `getSession` (onnxruntime-web / WASM)
 * and `getHandLandmarker` (MediaPipe), both of which are heavy to mock under
 * jsdom and offer little signal. The non-trivial new logic is the *decision* of
 * when to raise the `slow` flag, which we extracted into the pure
 * {@link isSlowWarmup} helper and test directly here. The hook's wiring (timer
 * set on warmUp, cleared on settle/unmount, `slow` reset on retry) is
 * straightforward glue around this predicate.
 */

import { describe, expect, it } from "vitest";

import { isSlowWarmup, SLOW_WARMUP_MS } from "@/lib/use-classifier";

describe("isSlowWarmup", () => {
  it("raises slow only while still warming", () => {
    expect(isSlowWarmup("warming")).toBe(true);
  });

  it("does not raise slow once warm-up has settled", () => {
    expect(isSlowWarmup("ready")).toBe(false);
    expect(isSlowWarmup("error")).toBe(false);
    expect(isSlowWarmup("idle")).toBe(false);
  });

  it("uses a sane, positive timeout threshold", () => {
    expect(SLOW_WARMUP_MS).toBeGreaterThan(0);
  });
});
