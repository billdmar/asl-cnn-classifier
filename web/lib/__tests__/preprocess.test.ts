import { describe, it, expect } from "vitest";
import {
  IMAGE_SIZE,
  IMAGENET_MEAN,
  IMAGENET_STD,
  rgbaToNchwTensor,
  softmax,
} from "../preprocess";

describe("constants are locked to the Python pipeline", () => {
  it("IMAGE_SIZE is 128", () => {
    expect(IMAGE_SIZE).toBe(128);
  });

  it("IMAGENET_MEAN matches src.dataset.IMAGENET_MEAN", () => {
    expect([...IMAGENET_MEAN]).toEqual([0.485, 0.456, 0.406]);
  });

  it("IMAGENET_STD matches src.dataset.IMAGENET_STD", () => {
    expect([...IMAGENET_STD]).toEqual([0.229, 0.224, 0.225]);
  });
});

describe("rgbaToNchwTensor", () => {
  it("produces CHW plane ordering with exact normalize arithmetic", () => {
    // A 2x2 image (4 pixels), RGBA stride 4. Distinct, easy-to-track values.
    // pixel0=(0,0,0) pixel1=(255,255,255) pixel2=(255,0,0) pixel3=(0,128,255)
    // prettier-ignore
    const rgba = new Uint8ClampedArray([
      0, 0, 0, 255,
      255, 255, 255, 255,
      255, 0, 0, 255,
      0, 128, 255, 255,
    ]);
    const size = 2;
    const pixels = size * size; // 4

    const out = rgbaToNchwTensor(rgba, size);

    // Length is 3 channels * pixels.
    expect(out.length).toBe(3 * pixels);

    const [mR, mG, mB] = IMAGENET_MEAN;
    const [sR, sG, sB] = IMAGENET_STD;
    const norm = (v: number, m: number, s: number): number => (v / 255 - m) / s;

    // R plane occupies [0, pixels), G plane [pixels, 2*pixels), B plane [2*pixels, 3*pixels).
    // Expected R channel for the 4 pixels: 0, 255, 255, 0
    const expectedR = [0, 255, 255, 0];
    const expectedG = [0, 255, 0, 128];
    const expectedB = [0, 255, 0, 255];

    for (let i = 0; i < pixels; i++) {
      expect(out[i]).toBeCloseTo(norm(expectedR[i]!, mR, sR), 6);
      expect(out[pixels + i]).toBeCloseTo(norm(expectedG[i]!, mG, sG), 6);
      expect(out[2 * pixels + i]).toBeCloseTo(norm(expectedB[i]!, mB, sB), 6);
    }
  });

  it("throws when given too few bytes", () => {
    // size=2 needs 16 bytes; give only 12.
    const tooShort = new Uint8ClampedArray(12);
    expect(() => rgbaToNchwTensor(tooShort, 2)).toThrow(/Expected at least/);
  });
});

describe("softmax", () => {
  it("sums to ~1", () => {
    const out = softmax([1, 2, 3, 4]);
    const sum = out.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
  });

  it("is monotonic w.r.t. logits", () => {
    const out = softmax([0, 1, 2]);
    expect(out[0]!).toBeLessThan(out[1]!);
    expect(out[1]!).toBeLessThan(out[2]!);
  });

  it("is numerically stable for large logits (no NaN/Inf)", () => {
    const out = softmax([1000, 1001, 999]);
    for (const v of out) {
      expect(Number.isFinite(v)).toBe(true);
    }
    const sum = out.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
    // The largest logit still gets the largest probability.
    expect(out[1]!).toBeGreaterThan(out[0]!);
    expect(out[1]!).toBeGreaterThan(out[2]!);
  });
});
