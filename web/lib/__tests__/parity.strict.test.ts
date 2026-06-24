/**
 * STRICT cross-language parity gate.
 *
 * Feeds the EXACT 128x128 pixels Python's bilinear Resize produced
 * (`<CLASS>_resized.png`) through the JS normalize + CHW-layout path and the
 * real ONNX model, then asserts the predicted class index and full probability
 * vector match the Python/ONNX golden values.
 *
 * Because the resize kernel is taken out of the equation (the fixtures are
 * pre-resized), this isolates the normalize + tensor-layout + ONNX-runtime math
 * from the resampling step. Measured agreement is ~5e-7; we assert atol 1e-3 to
 * stay robust while still proving the math is exact.
 */

import { describe, it, expect, beforeAll } from "vitest";
import path from "node:path";
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { rgbaToNchwTensor, softmax, IMAGE_SIZE } from "../preprocess";
import { GOLDEN_DIR, rgbToRgba, decodeResizedRgb, type Manifest } from "./_helpers";

const manifest = JSON.parse(
  readFileSync(path.join(GOLDEN_DIR, "manifest.json"), "utf-8"),
) as Manifest;

const PROB_ATOL = 1e-3;

describe("strict parity (pre-resized images): normalize + layout + ONNX", () => {
  let session: ort.InferenceSession;

  beforeAll(async () => {
    // vitest cwd is web/, so the public path resolves correctly.
    session = await ort.InferenceSession.create("public/model/model.onnx");
  }, 30000);

  for (const fx of manifest.fixtures) {
    it(`${fx.file} -> ${fx.pred_class} (idx ${fx.pred_index})`, async () => {
      const resizedPath = path.join(
        GOLDEN_DIR,
        fx.file.replace(/\.png$/, "_resized.png"),
      );
      const rgb = await decodeResizedRgb(resizedPath);
      expect(rgb.length).toBe(IMAGE_SIZE * IMAGE_SIZE * 3);

      const rgba = rgbToRgba(rgb);
      const tensorData = rgbaToNchwTensor(rgba, IMAGE_SIZE);

      const input = new ort.Tensor("float32", tensorData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
      const output = await session.run({ [manifest.onnx_input_name]: input });
      const logits = output[session.outputNames[0]!]!.data as Float32Array;
      const probs = softmax(logits);

      // Argmax must match the golden prediction exactly.
      let argmax = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i]! > probs[argmax]!) argmax = i;
      }
      expect(argmax).toBe(fx.pred_index);

      // Every probability within atol (real agreement ~5e-7).
      expect(probs.length).toBe(fx.probs.length);
      for (let i = 0; i < probs.length; i++) {
        expect(Math.abs(probs[i]! - fx.probs[i]!)).toBeLessThanOrEqual(PROB_ATOL);
      }
    }, 30000);
  }
});
