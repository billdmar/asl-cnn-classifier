/**
 * Upload-path hand-detection integration logic.
 *
 * The crop geometry (`cropBoxFromLandmarks`) is exercised elsewhere; here we
 * cover the upload entry point `detectHandInImage` — specifically that it maps a
 * landmarker result to the right {@link HandDetection} shape and falls back to
 * `{ found: false }` when no hand is present, without needing the real WASM
 * model. We stub `HandLandmarker.createFromOptions` so the IMAGE-mode landmarker
 * is a controllable fake.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { HandLandmarker } from "@mediapipe/tasks-vision";

import { detectHandInImage, resetImageHandLandmarker } from "../handcrop";

/** Build a fake landmarker whose `.detect()` returns the given landmarks. */
function fakeLandmarker(landmarks: Array<Array<{ x: number; y: number }>>) {
  return {
    detect: vi.fn(() => ({ landmarks })),
  } as unknown as HandLandmarker;
}

afterEach(() => {
  resetImageHandLandmarker();
  vi.restoreAllMocks();
});

describe("detectHandInImage", () => {
  it("falls back to { found: false } when the landmarker finds no hand", async () => {
    vi.spyOn(HandLandmarker, "createFromOptions").mockResolvedValue(
      fakeLandmarker([]),
    );
    // FilesetResolver isn't reached for the fake, but the IMAGE landmarker is
    // created via createFromOptions which we've stubbed above.
    vi.spyOn(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- spying on a sibling export
      (await import("@mediapipe/tasks-vision")).FilesetResolver,
      "forVisionTasks",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- fileset shape is opaque to us
    ).mockResolvedValue({} as any);

    const fakeImg = {} as HTMLImageElement;
    const detection = await detectHandInImage(fakeImg);

    expect(detection.found).toBe(false);
    expect(detection.box).toBeUndefined();
  });

  it("returns a square crop box when the landmarker finds a hand", async () => {
    // A hand spanning x:[0.4,0.6], y:[0.4,0.6] — centered, well inside the image.
    const hand = [
      { x: 0.4, y: 0.4 },
      { x: 0.6, y: 0.6 },
    ];
    vi.spyOn(HandLandmarker, "createFromOptions").mockResolvedValue(
      fakeLandmarker([hand]),
    );
    vi.spyOn(
      (await import("@mediapipe/tasks-vision")).FilesetResolver,
      "forVisionTasks",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- fileset shape is opaque to us
    ).mockResolvedValue({} as any);

    const detection = await detectHandInImage({} as HTMLImageElement);

    expect(detection.found).toBe(true);
    expect(detection.box).toBeDefined();
    // Box must be square (the contract the crop canvas relies on).
    expect(detection.box?.width).toBeCloseTo(detection.box?.height ?? NaN);
  });
});
