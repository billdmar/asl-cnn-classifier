import * as ort from "onnxruntime-web";
import { IMAGE_SIZE, rgbaToNchwTensor, drawToImageData } from "@/lib/preprocess";
import { cropBoxFromLandmarks, cropToCanvas, type CropBox, type HandDetection } from "@/lib/handcrop";
import { getSession, applyTemperature, type InferenceResult, type Prediction } from "@/lib/inference";
import { CLASS_NAMES } from "@/lib/labels";
import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

export type { CropBox, HandDetection, InferenceResult };

export interface FrozenFrame {
  source: HTMLCanvasElement;
  width: number;
  height: number;
}

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

export async function runFullPipeline(
  frame: FrozenFrame,
  handResult: Pick<HandLandmarkerResult, "landmarks">,
): Promise<PipelineSnapshot> {
  const handDetection = cropBoxFromLandmarks(handResult);
  let cropCanvas: HTMLCanvasElement | null = null;
  let preprocessInput: CanvasImageSource;
  if (handDetection.found && handDetection.box) {
    cropCanvas = cropToCanvas(frame.source, frame.width, frame.height, handDetection.box, IMAGE_SIZE);
    preprocessInput = cropCanvas;
  } else {
    preprocessInput = frame.source;
  }
  const imageData = drawToImageData(preprocessInput, IMAGE_SIZE);
  const tensor = rgbaToNchwTensor(imageData.data, IMAGE_SIZE);
  const session = await getSession();
  const input = new ort.Tensor("float32", tensor, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  const feeds: Record<string, ort.Tensor> = { [session.inputNames[0]!]: input };
  const output = await session.run(feeds);
  const logits = new Float32Array(output[session.outputNames[0]!]!.data as Float32Array);
  const probs = applyTemperature(logits, 1.0);
  const ranked: Prediction[] = Array.from(probs, (prob, index) => ({
    label: CLASS_NAMES[index] ?? String(index),
    index,
    prob,
  })).sort((a, b) => b.prob - a.prob);
  const result: InferenceResult = { ranked, probs, top: ranked[0]! };
  return { frame, handDetection, cropCanvas, imageData, tensor, logits, probs, result };
}
