/**
 * Canonical browser-side preprocessing — the JS half of the project's
 * single-source-of-truth preprocessing contract.
 *
 * This MUST reproduce `src/dataset.py::get_eval_transforms` exactly:
 *
 *   1. Resize to IMAGE_SIZE x IMAGE_SIZE (bilinear).
 *   2. ToTensor: HWC uint8 [0,255] -> CHW float [0,1] (divide by 255).
 *   3. Normalize: (x - mean) / std, per channel, with ImageNet statistics.
 *
 * Output layout is NCHW float32 (batch dim added by the caller) to match the
 * ONNX model's `input` tensor (['batch', 3, 128, 128]).
 *
 * The cross-language parity test (`lib/preprocess.parity.test.ts`) feeds the
 * SAME images this code sees through the committed golden fixtures and asserts
 * the predicted class + probabilities match the Python/ONNX path. Any drift
 * here silently destroys live accuracy, so treat these constants as locked to
 * their Python counterparts.
 */

/** Network input side length. Mirrors `src.dataset.IMAGE_SIZE`. */
export const IMAGE_SIZE = 128;

/** ImageNet channel means. Mirrors `src.dataset.IMAGENET_MEAN`. */
export const IMAGENET_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];

/** ImageNet channel std devs. Mirrors `src.dataset.IMAGENET_STD`. */
export const IMAGENET_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

/**
 * Draw an image source into a fixed-size offscreen canvas and return its RGBA
 * pixel bytes. Resizing happens via the 2D context's bilinear scaling, which
 * matches torchvision's default bilinear `Resize` closely enough for the parity
 * tolerance (atol 1e-3 on probabilities).
 *
 * @param source - Any canvas-drawable image source (img, video, canvas, bitmap).
 * @param size - Target side length (defaults to {@link IMAGE_SIZE}).
 * @returns The `size x size` RGBA pixel data.
 */
export function drawToImageData(
  source: CanvasImageSource,
  size: number = IMAGE_SIZE,
): ImageData {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("Could not acquire a 2D canvas context for preprocessing.");
  }
  ctx.drawImage(source, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size);
}

/**
 * Convert resized RGBA pixel bytes into a normalized NCHW float32 tensor.
 *
 * Pure and DOM-free so it can be unit-tested directly: pass raw RGBA bytes and
 * assert the produced tensor. This is the exact arithmetic the parity gate
 * checks against the Python `ToTensor` + `Normalize` pipeline.
 *
 * @param rgba - RGBA bytes, length `4 * size * size`, row-major.
 * @param size - The image side length the bytes represent.
 * @returns Float32Array of length `3 * size * size` in CHW order, batch-less.
 */
export function rgbaToNchwTensor(
  rgba: Uint8ClampedArray | Uint8Array,
  size: number = IMAGE_SIZE,
): Float32Array {
  const pixels = size * size;
  if (rgba.length < pixels * 4) {
    throw new Error(
      `Expected at least ${pixels * 4} RGBA bytes for a ${size}x${size} image, got ${rgba.length}.`,
    );
  }
  const out = new Float32Array(3 * pixels);
  const [meanR, meanG, meanB] = IMAGENET_MEAN;
  const [stdR, stdG, stdB] = IMAGENET_STD;
  // CHW: channel-major planes. plane offsets: R=0, G=pixels, B=2*pixels.
  for (let i = 0; i < pixels; i++) {
    const r = rgba[i * 4]! / 255;
    const g = rgba[i * 4 + 1]! / 255;
    const b = rgba[i * 4 + 2]! / 255;
    out[i] = (r - meanR) / stdR;
    out[pixels + i] = (g - meanG) / stdG;
    out[2 * pixels + i] = (b - meanB) / stdB;
  }
  return out;
}

/**
 * Full browser preprocessing: image source -> normalized NCHW float32 tensor.
 *
 * @param source - Any canvas-drawable image source.
 * @param size - Target side length (defaults to {@link IMAGE_SIZE}).
 * @returns Float32Array of length `3 * size * size` in CHW order, batch-less.
 */
export function preprocess(
  source: CanvasImageSource,
  size: number = IMAGE_SIZE,
): Float32Array {
  const imageData = drawToImageData(source, size);
  return rgbaToNchwTensor(imageData.data, size);
}

/** Numerically stable softmax over a 1-D logit vector. */
export function softmax(logits: Float32Array | number[]): Float32Array {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i]! - max);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < exps.length; i++) exps[i]! /= sum;
  return exps;
}
