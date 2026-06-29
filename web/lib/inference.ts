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
import {
  evictOtherVersions,
  getCachedModel,
  putCachedModel,
} from "./model-cache";
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
const CALIBRATION_URL = "/model/calibration.json";

/** Identity temperature — applying it before softmax is a no-op (current ship). */
export const DEFAULT_TEMPERATURE = 1.0;

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let temperaturePromise: Promise<number> | null = null;

/**
 * Lazily fetch (once, cached) the calibration temperature from
 * {@link CALIBRATION_URL}. SSR/static-export safe: when `fetch` is unavailable
 * (server/build) or the file is missing/unparseable, it degrades gracefully to
 * {@link DEFAULT_TEMPERATURE} (1.0 = identity, no behavior change).
 *
 * @returns A promise resolving to a positive temperature `T`.
 */
export function getTemperature(): Promise<number> {
  if (!temperaturePromise) {
    temperaturePromise =
      typeof fetch === "undefined"
        ? Promise.resolve(DEFAULT_TEMPERATURE)
        : fetch(CALIBRATION_URL)
            .then((res) => (res.ok ? res.json() : null))
            .then((data: unknown) => {
              const t =
                data !== null &&
                typeof data === "object" &&
                "temperature" in data &&
                typeof (data as { temperature: unknown }).temperature === "number"
                  ? (data as { temperature: number }).temperature
                  : DEFAULT_TEMPERATURE;
              // Guard against bad/zero/negative values that would break softmax.
              return Number.isFinite(t) && t > 0 ? t : DEFAULT_TEMPERATURE;
            })
            .catch(() => DEFAULT_TEMPERATURE);
  }
  return temperaturePromise;
}

/** Reset the cached temperature (used by tests). */
export function resetTemperature(): void {
  temperaturePromise = null;
}

/**
 * Divide logits by the calibration temperature `T` before softmax (temperature
 * scaling, Guo et al. 2017). With `T = 1.0` this is the identity, so it leaves
 * the probability vector unchanged; `T > 1` softens the distribution. Pure.
 *
 * @param logits - Raw pre-softmax model outputs.
 * @param temperature - Positive scalar `T`.
 * @returns The probability vector `softmax(logits / T)`.
 */
export function applyTemperature(
  logits: Float32Array,
  temperature: number,
): Float32Array {
  if (temperature === 1.0) return softmax(logits);
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i]! / temperature;
  return softmax(scaled);
}

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
  onProgress?: (fraction: number) => void,
): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    configureRuntime();
    // Prefer WebGPU when present, fall back to WASM. Both run client-side.
    const executionProviders: ort.InferenceSession.SessionOptions["executionProviders"] =
      typeof navigator !== "undefined" && "gpu" in navigator
        ? ["webgpu", "wasm"]
        : ["wasm"];
    sessionPromise = (async () => {
      // We stream the ~9 MB model ourselves (rather than letting
      // ort.InferenceSession.create fetch the URL, which has no progress hook
      // and no cache control) so we can BOTH report download progress AND back
      // it with an IndexedDB cache — returning visitors skip the download
      // entirely. This applies whether or not a progress callback is supplied
      // (the upload path has no callback but still benefits from the cache).
      // Cache version = build SHA, so a new deploy misses the old entry.
      const version = process.env.NEXT_PUBLIC_BUILD_SHA || "dev";
      const cached = await getCachedModel(modelUrl, version);
      if (cached) {
        onProgress?.(1);
        return ort.InferenceSession.create(cached, { executionProviders });
      }
      const bytes = await fetchWithProgress(modelUrl, onProgress);
      if (bytes) {
        // Best-effort: persist + reclaim old versions, but never let a cache
        // failure delay or break the session creation.
        void putCachedModel(modelUrl, version, bytes).then(() =>
          evictOtherVersions(modelUrl, version),
        );
        return ort.InferenceSession.create(bytes, { executionProviders });
      }
      // Streaming unavailable (no Content-Length / ReadableStream) — let ORT
      // fetch the URL directly as a last resort.
      return ort.InferenceSession.create(modelUrl, { executionProviders });
    })();
  }
  return sessionPromise;
}

/**
 * Fetch a binary asset while reporting download progress (0–1). Returns the
 * bytes, or null if the stream/Content-Length isn't usable (caller then falls
 * back to a plain URL load). Reports 1.0 on completion regardless.
 */
async function fetchWithProgress(
  url: string,
  onProgress?: (fraction: number) => void,
): Promise<Uint8Array | null> {
  try {
    const res = await fetch(url);
    if (!res.ok || !res.body) return null;
    const total = Number(res.headers.get("Content-Length") ?? 0);
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        received += value.length;
        if (total > 0) onProgress?.(Math.min(1, received / total));
      }
    }
    onProgress?.(1);
    const out = new Uint8Array(received);
    let offset = 0;
    for (const c of chunks) {
      out.set(c, offset);
      offset += c.length;
    }
    return out;
  } catch {
    return null;
  }
}

/** Reset the cached session and temperature (used by tests). */
export function resetSession(): void {
  sessionPromise = null;
  temperaturePromise = null;
}

/**
 * Run inference on an already-preprocessed NCHW float32 tensor.
 *
 * @param session - A ready ONNX session.
 * @param tensorData - Float32Array of length `3 * size * size` (no batch dim).
 * @param size - The spatial side length the tensor encodes.
 * @param temperature - Optional override for the calibration temperature `T`.
 *   When omitted, the cached value from {@link getTemperature} is used (1.0 by
 *   default, an identity no-op). Tests pass this to assert `T != 1` behavior.
 * @returns Ranked predictions and the full probability vector.
 */
export async function runTensor(
  session: ort.InferenceSession,
  tensorData: Float32Array,
  size: number = IMAGE_SIZE,
  temperature?: number,
): Promise<InferenceResult> {
  const input = new ort.Tensor("float32", tensorData, [1, 3, size, size]);
  const feeds: Record<string, ort.Tensor> = { [session.inputNames[0]!]: input };
  const output = await session.run(feeds);
  const logits = output[session.outputNames[0]!]!.data as Float32Array;
  // Temperature scaling before softmax. T=1.0 (the shipped default) is identity.
  const t = temperature ?? (await getTemperature());
  const probs = applyTemperature(logits, t);

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
