/**
 * END-TO-END parity test (includes the resampling step).
 *
 * Decodes the ORIGINAL `<CLASS>.png`, resizes it to 128x128 with sharp, then
 * runs the full JS normalize + CHW-layout + ONNX path and compares to the
 * Python/ONNX golden values.
 *
 * Kernel choice: sharp's `cubic` (bicubic) resampling. The browser canvas and
 * sharp resamplers are NOT bit-identical to PIL's bilinear Resize that produced
 * the golden fixtures, so per-probability values drift. That is expected and is
 * NOT a bug: the strict test (`parity.strict.test.ts`) feeds the pre-resized
 * pixels and proves the normalize/layout/ONNX math is exact (~5e-7). Here we:
 *   - assert the predicted class index matches EXACTLY (class-level parity must
 *     hold across resamplers), and
 *   - assert probabilities only within a deliberately LOOSER atol (3e-2),
 *     reflecting resize-kernel differences alone.
 */

import { describe, it, expect, beforeAll } from "vitest";
import path from "node:path";
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import type { KernelEnum } from "sharp";
import { rgbaToNchwTensor, softmax, IMAGE_SIZE } from "../preprocess";
import { GOLDEN_DIR, rgbToRgba, decodeAndResizeRgb, type Manifest } from "./_helpers";

const manifest = JSON.parse(
  readFileSync(path.join(GOLDEN_DIR, "manifest.json"), "utf-8"),
) as Manifest;

// Documented resampling filter (see file header).
const RESIZE_KERNEL: keyof KernelEnum = "cubic";
// Loose tolerance: accounts for resize-kernel divergence, not math drift.
const PROB_ATOL = 3e-2;

describe("end-to-end parity (original images, resampled in JS)", () => {
  let session: ort.InferenceSession;

  beforeAll(async () => {
    session = await ort.InferenceSession.create("public/model/model.onnx");
  }, 30000);

  for (const fx of manifest.fixtures) {
    it(`${fx.file} -> ${fx.pred_class} (idx ${fx.pred_index})`, async () => {
      const origPath = path.join(GOLDEN_DIR, fx.file);
      const rgb = await decodeAndResizeRgb(origPath, IMAGE_SIZE, RESIZE_KERNEL);
      expect(rgb.length).toBe(IMAGE_SIZE * IMAGE_SIZE * 3);

      const rgba = rgbToRgba(rgb);
      const tensorData = rgbaToNchwTensor(rgba, IMAGE_SIZE);

      const input = new ort.Tensor("float32", tensorData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
      const output = await session.run({ [manifest.onnx_input_name]: input });
      const logits = output[session.outputNames[0]!]!.data as Float32Array;
      const probs = softmax(logits);

      // Class-level parity MUST hold regardless of resampler.
      let argmax = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i]! > probs[argmax]!) argmax = i;
      }
      expect(argmax).toBe(fx.pred_index);

      // Probabilities within a LOOSER tolerance (resize-kernel divergence).
      expect(probs.length).toBe(fx.probs.length);
      for (let i = 0; i < probs.length; i++) {
        expect(Math.abs(probs[i]! - fx.probs[i]!)).toBeLessThanOrEqual(PROB_ATOL);
      }
    }, 30000);
  }
});
