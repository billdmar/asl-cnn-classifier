/**
 * Inference pipeline step capture — freezes a frame and runs the full
 * classification pipeline, caching every intermediate result so the explainer
 * UI can visualize each stage independently.
 *
 * Pure/composable: the caller owns the video/image element and the MediaPipe
 * landmarker mode selection; this module only needs the frozen frame and the
 * landmark result.
 */

import * as ort from "onnxruntime-web";
import { IMAGE_SIZE, rgbaToNchwTensor, drawToImageData } from "@/lib/preprocess";
import {
  cropBoxFromLandmarks,
  cropToCanvas,
  type CropBox,
  type HandDetection,
} from "@/lib/handcrop";
import {
  getSession,
  applyTemperature,
  type InferenceResult,
  type Prediction,
} from "@/lib/inference";
import { CLASS_NAMES } from "@/lib/labels";
import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

// Re-export types the caller will need alongside this module.
export type { CropBox, HandDetection, InferenceResult };

/** A frozen video/image frame stored as a canvas for reprocessing. */
export interface FrozenFrame {
  source: HTMLCanvasElement;
  width: number;
  height: number;
}

/** All intermediate results from one run of the inference pipeline. */
export interface PipelineSnapshot {
  frame: FrozenFrame;
  handDetection: HandDetection;
  cropCanvas: HTMLCanvasElement | null;
  imageData: ImageData;
  tensor: Float32Array;
  logits: Float32Array;
  probs: Float32Array;
  result: InferenceResult;
}

/**
 * Capture the current video element's frame into a retained canvas.
 *
 * The canvas is an independent copy — subsequent video frames do not mutate it.
 *
 * @param video - A playing HTMLVideoElement with a loaded source.
 * @returns A {@link FrozenFrame} with the captured pixel data.
 */
export function captureVideoFrame(video: HTMLVideoElement): FrozenFrame {
  const width = video.videoWidth;
  const height = video.videoHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not acquire a 2D context for frame capture.");
  ctx.drawImage(video, 0, 0, width, height);
  return { source: canvas, width, height };
}

/**
 * Capture an image element into a retained canvas.
 *
 * @param img - A decoded HTMLImageElement.
 * @returns A {@link FrozenFrame} with the captured pixel data.
 */
export function captureImageFrame(img: HTMLImageElement): FrozenFrame {
  const width = img.naturalWidth;
  const height = img.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not acquire a 2D context for image capture.");
  ctx.drawImage(img, 0, 0, width, height);
  return { source: canvas, width, height };
}

/**
 * Run the full pipeline on a frozen frame and return all intermediate data.
 *
 * The hand detection result is passed in so the UI can use VIDEO-mode or
 * IMAGE-mode landmarker as appropriate — the caller manages which one to use.
 *
 * The raw logits (pre-softmax) are captured separately from the final
 * probabilities so the explainer can re-apply temperature scaling interactively.
 *
 * @param frame - A previously captured {@link FrozenFrame}.
 * @param handResult - MediaPipe landmark result (only `.landmarks` is read).
 * @returns A {@link PipelineSnapshot} with every intermediate artifact.
 */
export async function runFullPipeline(
  frame: FrozenFrame,
  handResult: Pick<HandLandmarkerResult, "landmarks">,
): Promise<PipelineSnapshot> {
  // Step 1: Derive the crop box from hand landmarks.
  const handDetection = cropBoxFromLandmarks(handResult);

  // Step 2: Crop to a square canvas around the detected hand, or use the
  // full frame when no hand is found.
  let cropCanvas: HTMLCanvasElement | null = null;
  let preprocessInput: CanvasImageSource;

  if (handDetection.found && handDetection.box) {
    cropCanvas = cropToCanvas(
      frame.source,
      frame.width,
      frame.height,
      handDetection.box,
      IMAGE_SIZE,
    );
    preprocessInput = cropCanvas;
  } else {
    preprocessInput = frame.source;
  }

  // Step 3: Resize/draw into IMAGE_SIZE x IMAGE_SIZE and extract RGBA pixels.
  const imageData = drawToImageData(preprocessInput, IMAGE_SIZE);

  // Step 4: Normalize RGBA → CHW float32 tensor.
  const tensor = rgbaToNchwTensor(imageData.data, IMAGE_SIZE);

  // Step 5: Run the ONNX session directly to capture raw logits before softmax.
  const session = await getSession();
  const input = new ort.Tensor("float32", tensor, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  const feeds: Record<string, ort.Tensor> = { [session.inputNames[0]!]: input };
  const output = await session.run(feeds);
  const logits = new Float32Array(
    output[session.outputNames[0]!]!.data as Float32Array,
  );

  // Step 6: Apply plain softmax (T=1.0) to get uncalibrated probabilities.
  const probs = applyTemperature(logits, 1.0);

  // Step 7: Build the ranked InferenceResult from the probability vector.
  const ranked: Prediction[] = Array.from(probs, (prob, index) => ({
    label: CLASS_NAMES[index] ?? String(index),
    index,
    prob,
  })).sort((a, b) => b.prob - a.prob);

  const result: InferenceResult = { ranked, probs, top: ranked[0]! };

  return {
    frame,
    handDetection,
    cropCanvas,
    imageData,
    tensor,
    logits,
    probs,
    result,
  };
}
