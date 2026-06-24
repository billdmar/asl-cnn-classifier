/**
 * MediaPipe hand detection + crop — the real-world robustness keystone.
 *
 * Before classifying, we detect the hand and crop to a normalized square box
 * around it. This removes most of the background dependence that makes the
 * model (trained on uniform images) struggle on a cluttered webcam.
 *
 * This is the browser half of a cropping step that must also exist in the
 * Python pipeline for training/eval parity (see the ML robustness workstream).
 * The crop geometry here (square box, margin fraction) is the contract both
 * sides share.
 *
 * Runs 100% client-side via MediaPipe Tasks Vision (WASM). The model and WASM
 * assets are served from the site's own /mediapipe/ path so there is no runtime
 * CDN dependency and frames never leave the browser.
 */

import {
  HandLandmarker,
  FilesetResolver,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

/** A normalized crop box in [0,1] image coordinates. */
export interface CropBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Result of a hand-detection pass. */
export interface HandDetection {
  /** True if at least one hand was found. */
  found: boolean;
  /** The square crop box (normalized), present only when {@link found}. */
  box?: CropBox;
}

/**
 * Fraction of the hand's bounding-box size added as margin on each side, so the
 * crop includes the whole hand plus context (matches the training crop policy).
 */
export const CROP_MARGIN = 0.35;

const WASM_PATH = "/mediapipe/wasm";
const MODEL_PATH = "/mediapipe/hand_landmarker.task";

let landmarkerPromise: Promise<HandLandmarker> | null = null;

/**
 * Lazily create (once) the shared {@link HandLandmarker}, configured for video.
 *
 * @returns A promise resolving to the cached landmarker.
 */
export function getHandLandmarker(): Promise<HandLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
      return HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_PATH },
        runningMode: "VIDEO",
        numHands: 1,
      });
    })();
  }
  return landmarkerPromise;
}

/** Reset the cached landmarker (used by tests). */
export function resetHandLandmarker(): void {
  landmarkerPromise = null;
}

let imageLandmarkerPromise: Promise<HandLandmarker> | null = null;

/**
 * Lazily create (once) a SEPARATE {@link HandLandmarker} configured for stills.
 *
 * The shared {@link getHandLandmarker} instance runs in `"VIDEO"` mode and can
 * only be driven via `detectForVideo`; calling `.detect()` on it throws. Uploads
 * are single still images, so they need their own `"IMAGE"`-mode landmarker.
 * Both share the same WASM fileset and model asset.
 *
 * @returns A promise resolving to the cached image-mode landmarker.
 */
export function getImageHandLandmarker(): Promise<HandLandmarker> {
  if (!imageLandmarkerPromise) {
    imageLandmarkerPromise = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
      return HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_PATH },
        runningMode: "IMAGE",
        numHands: 1,
      });
    })();
  }
  return imageLandmarkerPromise;
}

/** Reset the cached image-mode landmarker (used by tests). */
export function resetImageHandLandmarker(): void {
  imageLandmarkerPromise = null;
}

/**
 * Detect a hand in a single still image and return its normalized crop box.
 *
 * Uses the `"IMAGE"`-mode landmarker (so `.detect()` is valid). Returns the same
 * {@link HandDetection} shape as {@link cropBoxFromLandmarks}: `found:false` when
 * no hand is present, so callers can fall back to whole-image classification.
 *
 * @param image - A decoded image (or other still source) to detect a hand in.
 * @returns A {@link HandDetection} with the square crop box when a hand is found.
 */
export async function detectHandInImage(
  image: ImageBitmapSource | HTMLImageElement | HTMLCanvasElement,
): Promise<HandDetection> {
  const landmarker = await getImageHandLandmarker();
  const result = landmarker.detect(image as Parameters<HandLandmarker["detect"]>[0]);
  return cropBoxFromLandmarks(result);
}

/**
 * Compute a square, margined crop box from MediaPipe landmark results.
 *
 * Pure and dependency-free so it can be unit-tested without the WASM runtime:
 * pass a {@link HandLandmarkerResult}-shaped object and assert the box. The box
 * is the landmarks' bounding square expanded by {@link CROP_MARGIN}, clamped to
 * the unit image.
 *
 * @param result - The landmarker output (uses `result.landmarks[0]`).
 * @param margin - Margin fraction (defaults to {@link CROP_MARGIN}).
 * @returns A {@link HandDetection}; `found:false` when no hand is present.
 */
export function cropBoxFromLandmarks(
  result: Pick<HandLandmarkerResult, "landmarks">,
  margin: number = CROP_MARGIN,
): HandDetection {
  const hand = result.landmarks?.[0];
  if (!hand || hand.length === 0) {
    return { found: false };
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const point of hand) {
    if (point.x < minX) minX = point.x;
    if (point.y < minY) minY = point.y;
    if (point.x > maxX) maxX = point.x;
    if (point.y > maxY) maxY = point.y;
  }

  const w = maxX - minX;
  const h = maxY - minY;
  // Square side = larger dimension, expanded by margin on both sides.
  const side = Math.max(w, h) * (1 + 2 * margin);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;

  let x = cx - side / 2;
  let y = cy - side / 2;
  let boxSide = side;
  // Clamp to the unit image while keeping the box square.
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x + boxSide > 1) boxSide = Math.min(boxSide, 1 - x);
  if (y + boxSide > 1) boxSide = Math.min(boxSide, 1 - y);

  return { found: true, box: { x, y, width: boxSide, height: boxSide } };
}

/**
 * Draw the region of a source defined by a normalized {@link CropBox} into a
 * fresh canvas of `outSize x outSize` pixels (square), ready for preprocessing.
 *
 * @param source - The video/image/canvas to crop from.
 * @param sourceWidth - Source pixel width.
 * @param sourceHeight - Source pixel height.
 * @param box - Normalized crop box.
 * @param outSize - Output canvas side length.
 * @returns The cropped square canvas.
 */
export function cropToCanvas(
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  box: CropBox,
  outSize: number,
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = outSize;
  canvas.height = outSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not acquire a 2D context for cropping.");
  ctx.drawImage(
    source,
    box.x * sourceWidth,
    box.y * sourceHeight,
    box.width * sourceWidth,
    box.height * sourceHeight,
    0,
    0,
    outSize,
    outSize,
  );
  return canvas;
}
