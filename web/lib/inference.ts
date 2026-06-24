/**
 * In-browser ONNX inference engine for the ASL classifier.
 *
 * Loads the real exported MobileNetV2 model (`/model/model.onnx`, 26 classes)
 * once and runs it entirely client-side via onnxruntime-web (WASM, with WebGPU
 * attempted first where available). Webcam/image frames never leave the browser.
 *
 * Preprocessing is delegated to `lib/preprocess.ts` so there is exactly one
 * copy of the resize/normalize math, and the cross-language parity test guards
 * it against the Python pipeline.
 */

import * as ort from "onnxruntime-web";
import { CLASS_NAMES } from "./labels";
import { IMAGE_SIZE, preprocess, softmax } from "./preprocess";

/** A single class prediction. */
export interface Prediction {
  label: string;
  index: number;
  prob: number;
}

/** The result of one inference: ranked predictions + raw probability vector. */
export interface InferenceResult {
  /** Predictions sorted by probability, descending. */
  ranked: Prediction[];
  /** Full probability vector in class-index order (sums to ~1). */
  probs: Float32Array;
  /** The top-1 prediction (convenience). */
  top: Prediction;
}

const DEFAULT_MODEL_URL = "/model/model.onnx";

let sessionPromise: Promise<ort.InferenceSession> | null = null;

/**
 * Configure the onnxruntime-web WASM asset location. The package ships its
 * `.wasm`/`.mjs` runtime files; in a Next static export they are served from
 * the package CDN path by default. We point at the bundled copy under
 * `/_next` is not reliable for static export, so we use the public CDN the
 * package documents. Callers may override before {@link getSession}.
 */
export function configureRuntime(): void {
  // Multi-threaded WASM needs cross-origin isolation that a static host may not
  // provide; single-thread is the safe, universally-working default.
  ort.env.wasm.numThreads = 1;
  // Let the package resolve its own wasm assets from its versioned CDN path.
  // (Overridable by tests that run under Node.)
}

/**
 * Lazily create (once) and return the shared ONNX inference session.
 *
 * @param modelUrl - URL/path to the `.onnx` file (defaults to `/model/model.onnx`).
 * @returns A promise resolving to the cached {@link ort.InferenceSession}.
 */
export function getSession(
  modelUrl: string = DEFAULT_MODEL_URL,
): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    configureRuntime();
    // Prefer WebGPU when present, fall back to WASM. Both run client-side.
    const executionProviders: ort.InferenceSession.SessionOptions["executionProviders"] =
      typeof navigator !== "undefined" && "gpu" in navigator
        ? ["webgpu", "wasm"]
        : ["wasm"];
    sessionPromise = ort.InferenceSession.create(modelUrl, { executionProviders });
  }
  return sessionPromise;
}

/** Reset the cached session (used by tests). */
export function resetSession(): void {
  sessionPromise = null;
}

/**
 * Run inference on an already-preprocessed NCHW float32 tensor.
 *
 * @param session - A ready ONNX session.
 * @param tensorData - Float32Array of length `3 * size * size` (no batch dim).
 * @param size - The spatial side length the tensor encodes.
 * @returns Ranked predictions and the full probability vector.
 */
export async function runTensor(
  session: ort.InferenceSession,
  tensorData: Float32Array,
  size: number = IMAGE_SIZE,
): Promise<InferenceResult> {
  const input = new ort.Tensor("float32", tensorData, [1, 3, size, size]);
  const feeds: Record<string, ort.Tensor> = { [session.inputNames[0]!]: input };
  const output = await session.run(feeds);
  const logits = output[session.outputNames[0]!]!.data as Float32Array;
  const probs = softmax(logits);

  const ranked: Prediction[] = Array.from(probs, (prob, index) => ({
    label: CLASS_NAMES[index] ?? String(index),
    index,
    prob,
  })).sort((a, b) => b.prob - a.prob);

  return { ranked, probs, top: ranked[0]! };
}

/**
 * End-to-end classify an image source: preprocess in-browser, then infer.
 *
 * @param source - Any canvas-drawable image source (img, video, canvas).
 * @param modelUrl - Optional model URL override.
 * @returns Ranked predictions and the full probability vector.
 */
export async function classifyImage(
  source: CanvasImageSource,
  modelUrl?: string,
): Promise<InferenceResult> {
  const session = await getSession(modelUrl);
  const tensor = preprocess(source, IMAGE_SIZE);
  return runTensor(session, tensor, IMAGE_SIZE);
}
