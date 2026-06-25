/**
 * Shared test helpers for the preprocessing / parity suites.
 *
 * These are intentionally tiny and local: PNG decoding via sharp (a transitive
 * dependency available under Node) and the RGB -> RGBA repack that
 * `rgbaToNchwTensor` expects (it reads a stride-4 buffer).
 */

import path from "node:path";
import sharp from "sharp";

/** Absolute path to the committed golden-fixtures directory. */
export const GOLDEN_DIR = path.resolve(__dirname, "..", "..", "test-fixtures", "golden");

/** One fixture entry as recorded in `manifest.json`. */
export interface Fixture {
  file: string;
  true_class: string;
  pred_class: string;
  pred_index: number;
  probs: number[];
  tensor_mean: number;
  tensor_std: number;
  tensor_shape: number[];
}

/** The golden `manifest.json` shape (only the fields the tests use). */
export interface Manifest {
  image_size: number;
  imagenet_mean: number[];
  imagenet_std: number[];
  class_names: string[];
  onnx_input_name: string;
  onnx_model: string;
  fixtures: Fixture[];
}

/**
 * Expand a tightly-packed RGB byte buffer (length `w*h*3`) into the RGBA layout
 * (`w*h*4`, alpha = 255) that {@link rgbaToNchwTensor} consumes. Alpha is
 * irrelevant to the tensor math (only R/G/B are read) but the stride must be 4.
 */
export function rgbToRgba(rgb: Uint8Array | Buffer): Uint8ClampedArray {
  const pixels = rgb.length / 3;
  const rgba = new Uint8ClampedArray(pixels * 4);
  for (let i = 0; i < pixels; i++) {
    rgba[i * 4] = rgb[i * 3]!;
    rgba[i * 4 + 1] = rgb[i * 3 + 1]!;
    rgba[i * 4 + 2] = rgb[i * 3 + 2]!;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

/** Decode an already-128x128 PNG to a raw 3-channel RGB buffer. */
export async function decodeResizedRgb(filePath: string): Promise<Buffer> {
  return sharp(filePath).removeAlpha().raw().toBuffer();
}

/**
 * Decode an arbitrary-size PNG, resize to `size`x`size`, return raw RGB bytes.
 * `kernel` selects the resampling filter (see the e2e parity test for why the
 * choice is documented but tolerance-bounded).
 */
export async function decodeAndResizeRgb(
  filePath: string,
  size: number,
  kernel: keyof sharp.KernelEnum,
): Promise<Buffer> {
  return sharp(filePath)
    .resize(size, size, { kernel, fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer();
}
