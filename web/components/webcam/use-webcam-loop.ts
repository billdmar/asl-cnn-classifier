"use client";

/**
 * Custom hook owning the rAF/inference loop and all associated mutable state.
 *
 * Extracted from webcam-panel.tsx so the render component is pure JSX. The hook
 * manages:
 *   - Camera lifecycle (idle/requesting/active/denied/no-camera/error)
 *   - MediaPipe hand detection in a rAF loop
 *   - ONNX inference via useClassifier()
 *   - Temporal smoothing of predictions (probsBuffer)
 *   - FPS tracking (rolling average)
 *   - ROI overlay drawing on a canvas
 *   - Hand-lost timeout (clears stale predictions)
 *   - Fingerspelling word-builder (hold-to-lock)
 *   - Eval-set capture (last crop ref + download)
 *
 * SSR-safe by construction: every browser API (navigator, document, rAF,
 * MediaStream) is touched only inside effects / event handlers, never at module
 * scope or during render.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  averageProbs,
  pushConfidencePoint,
  rankFromProbs,
  SMOOTHING_WINDOW,
  type ConfidencePoint,
} from "@/components/webcam/smoothing";
import {
  advanceHold,
  appendLetter,
  INITIAL_HOLD,
  type HoldState,
} from "@/components/webcam/word-builder";
import { downloadCanvasAsPng } from "@/components/webcam/capture";
import { interpret, type ConfidenceVerdict } from "@/lib/confidence";
import { cropBoxFromLandmarks, cropToCanvas, type CropBox } from "@/lib/handcrop";
import { CLASS_NAMES } from "@/lib/labels";
import { IMAGE_SIZE } from "@/lib/preprocess";
import type { InferenceResult } from "@/lib/inference";
import { useClassifier, type WarmupStatus } from "@/lib/use-classifier";

/** Camera lifecycle states, each with distinct UI. */
export type CameraState = "idle" | "requesting" | "active" | "denied" | "no-camera" | "error";

/** Minimum gap between classification runs (ms) — keeps the loop responsive. */
const CLASSIFY_INTERVAL_MS = 120;

/** Frames of top-1 confidence kept for the live sparkline (~5s at 8fps). */
const CONFIDENCE_HISTORY_CAP = 40;
/** Rolling-average window for the FPS readout. */
const FPS_SMOOTHING = 0.9;
/**
 * After the hand has been gone this long (ms), drop the smoothing buffer so a
 * stale letter doesn't linger once the user lowers their hand.
 */
const HAND_LOST_RESET_MS = 600;

/** Everything the webcam panel needs from the loop to render. */
export interface UseWebcamLoopReturn {
  /** Refs the panel must attach to the <video> and overlay <canvas>. */
  videoRef: React.RefObject<HTMLVideoElement | null>;
  overlayRef: React.RefObject<HTMLCanvasElement | null>;

  /** Camera lifecycle state. */
  cameraState: CameraState;
  errorMsg: string | null;

  /** Start the camera + inference loop (user-initiated). */
  startCamera: () => Promise<void>;
  /** Stop the camera + inference loop and reset state. */
  stopCamera: () => void;

  /** Current smoothed inference result (null when no hand / camera off). */
  result: InferenceResult | null;
  /** Confidence verdict (unsure / confident) derived from result. */
  verdict: ConfidenceVerdict | null;
  /** Whether a hand is currently detected in the frame. */
  handFound: boolean;
  /** Rolling FPS readout. */
  fps: number;
  /** Capped confidence history for the live sparkline. */
  confHistory: ConfidencePoint[];

  /** Fingerspelling word built so far. */
  word: string;
  setWord: React.Dispatch<React.SetStateAction<string>>;
  /** 0–1 progress of the current hold toward locking a letter. */
  holdProgress: number;

  /** Selected letter for eval-set capture. */
  captureLetter: string;
  setCaptureLetter: React.Dispatch<React.SetStateAction<string>>;
  /** Number of frames captured this session. */
  captureCount: number;
  /** Download the current hand crop as a labeled PNG. */
  captureFrame: () => void;

  /** Classifier warm-up status (idle/warming/ready/error). */
  warmStatus: WarmupStatus;
  /** Human-readable warm-up error. */
  warmError: string | null;
  /** Model download progress [0, 1]. */
  warmProgress: number;
  /** True when warm-up is taking longer than expected. */
  warmSlow: boolean;
  /** Kick off (or retry) model warm-up. */
  warmUp: () => void;
}

export function useWebcamLoop(): UseWebcamLoopReturn {
  const {
    status: warmStatus,
    error: warmError,
    progress: warmProgress,
    slow: warmSlow,
    warmUp,
    classify,
  } = useClassifier();

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  // Mutable loop state kept in refs so the rAF callback stays stable.
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastClassifyRef = useRef(0);
  const lastFrameRef = useRef(0);
  const inflightRef = useRef(false);
  const runningRef = useRef(false);
  /** Last N raw probability vectors, averaged into the displayed result. */
  const probsBufferRef = useRef<Float32Array[]>([]);
  /** Timestamp of the most recent frame in which a hand was detected. */
  const lastHandSeenRef = useRef(0);
  /** The most recent cropped 128x128 hand canvas — what "Capture" saves. */
  const lastCropRef = useRef<HTMLCanvasElement | null>(null);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [verdict, setVerdict] = useState<ConfidenceVerdict | null>(null);
  const [handFound, setHandFound] = useState(false);
  const [fps, setFps] = useState(0);
  /** Capped history of the smoothed top-1 confidence for the live sparkline. */
  const [confHistory, setConfHistory] = useState<ConfidencePoint[]>([]);
  const confFrameRef = useRef(0);
  /** Fingerspelling word-builder: the running word, hold state, and hold progress. */
  const [word, setWord] = useState("");
  const [holdProgress, setHoldProgress] = useState(0);
  const holdRef = useRef<HoldState>(INITIAL_HOLD);
  const [captureLetter, setCaptureLetter] = useState("A");
  const [captureCount, setCaptureCount] = useState(0);

  /** Draw (or clear) the ROI box on the overlay canvas, sized to the video. */
  const drawOverlay = useCallback((box: CropBox | null) => {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas) return;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (w === 0 || h === 0) return;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    if (!box) return;
    ctx.strokeStyle = "#7c5cff";
    ctx.lineWidth = Math.max(2, Math.round(w / 200));
    ctx.setLineDash([Math.round(w / 40), Math.round(w / 60)]);
    ctx.strokeRect(box.x * w, box.y * h, box.width * w, box.height * h);
  }, []);

  /** The per-frame loop: detect hand, maybe classify, update FPS. */
  const tick = useCallback(async () => {
    if (!runningRef.current) return;
    const video = videoRef.current;

    // FPS rolling average (every animation frame).
    const now = performance.now();
    if (lastFrameRef.current !== 0) {
      const dt = now - lastFrameRef.current;
      if (dt > 0) {
        const inst = 1000 / dt;
        setFps((prev) =>
          prev === 0 ? inst : prev * FPS_SMOOTHING + inst * (1 - FPS_SMOOTHING),
        );
      }
    }
    lastFrameRef.current = now;

    if (video && video.readyState >= 2 && video.videoWidth > 0) {
      try {
        const { getHandLandmarker } = await import("@/lib/handcrop");
        const landmarker = await getHandLandmarker();
        const detection = landmarker.detectForVideo(video, now);
        const crop = cropBoxFromLandmarks(detection);
        setHandFound(crop.found);
        drawOverlay(crop.box ?? null);

        if (crop.found) {
          lastHandSeenRef.current = now;
        } else if (
          lastHandSeenRef.current !== 0 &&
          now - lastHandSeenRef.current > HAND_LOST_RESET_MS &&
          probsBufferRef.current.length > 0
        ) {
          // Hand gone long enough: clear the smoothing buffer + crop so a stale
          // letter doesn't linger.
          probsBufferRef.current = [];
          lastCropRef.current = null;
          setResult(null);
          setVerdict(null);
          setConfHistory([]);
          // Reset the in-progress hold (the built word is intentionally kept).
          holdRef.current = INITIAL_HOLD;
          setHoldProgress(0);
        }

        // Throttle the expensive classify step; skip if one is in flight.
        if (
          crop.found &&
          crop.box &&
          !inflightRef.current &&
          now - lastClassifyRef.current >= CLASSIFY_INTERVAL_MS
        ) {
          lastClassifyRef.current = now;
          inflightRef.current = true;
          const cropCanvas = cropToCanvas(
            video,
            video.videoWidth,
            video.videoHeight,
            crop.box,
            IMAGE_SIZE,
          );
          lastCropRef.current = cropCanvas;
          classify(cropCanvas)
            .then((res) => {
              if (!runningRef.current) return;
              // Push the raw probs into the rolling buffer (drop oldest), then
              // render the element-wise-averaged, re-ranked result so the
              // displayed letter stays stable across frames.
              const buffer = probsBufferRef.current;
              buffer.push(res.probs);
              if (buffer.length > SMOOTHING_WINDOW) buffer.shift();
              const smoothed = rankFromProbs(averageProbs(buffer), CLASS_NAMES);
              setResult(smoothed);
              setVerdict(interpret(smoothed));
              const frame = (confFrameRef.current += 1);
              setConfHistory((h) =>
                pushConfidencePoint(
                  h,
                  { frame, prob: smoothed.top.prob, label: smoothed.top.label },
                  CONFIDENCE_HISTORY_CAP,
                ),
              );
              // Word-builder: advance the hold on the smoothed top letter; a
              // confident letter held HOLD_MS locks one letter into the word.
              const v = interpret(smoothed);
              const hold = advanceHold(
                holdRef.current,
                v.top.label,
                !v.unsure,
                now,
              );
              holdRef.current = hold.state;
              setHoldProgress(hold.progress);
              if (hold.locked) setWord((w) => appendLetter(w, hold.locked!));
            })
            .catch(() => {
              /* transient inference error — keep the loop alive */
            })
            .finally(() => {
              inflightRef.current = false;
            });
        }
      } catch {
        /* detection hiccup (e.g. landmarker still loading) — try next frame */
      }
    }

    rafRef.current = requestAnimationFrame(() => void tick());
  }, [classify, drawOverlay]);

  /** Fully tear down the camera + loop. Safe to call repeatedly. */
  const stopCamera = useCallback(() => {
    runningRef.current = false;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const stream = streamRef.current;
    if (stream) {
      for (const track of stream.getTracks()) track.stop();
      streamRef.current = null;
    }
    const video = videoRef.current;
    if (video) video.srcObject = null;
    const canvas = overlayRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
    lastFrameRef.current = 0;
    lastClassifyRef.current = 0;
    inflightRef.current = false;
    probsBufferRef.current = [];
    lastHandSeenRef.current = 0;
    lastCropRef.current = null;
    setHandFound(false);
    setFps(0);
    setResult(null);
    setVerdict(null);
    setConfHistory([]);
    holdRef.current = INITIAL_HOLD;
    setHoldProgress(0);
    setCameraState("idle");
  }, []);

  /** Request the camera on explicit user action and start the loop. */
  const startCamera = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setCameraState("no-camera");
      setErrorMsg("This browser does not support camera access.");
      return;
    }
    setErrorMsg(null);
    setCameraState("requesting");
    warmUp();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) {
        for (const track of stream.getTracks()) track.stop();
        streamRef.current = null;
        return;
      }
      video.srcObject = stream;
      await video.play().catch(() => {
        /* play() can reject if interrupted; the loop tolerates it */
      });
      setCameraState("active");
      runningRef.current = true;
      lastFrameRef.current = 0;
      probsBufferRef.current = [];
      lastHandSeenRef.current = 0;
      lastCropRef.current = null;
      rafRef.current = requestAnimationFrame(() => void tick());
    } catch (err: unknown) {
      streamRef.current = null;
      const name = err instanceof DOMException ? err.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        setCameraState("denied");
        setErrorMsg("Camera permission was denied. Allow access and try again.");
      } else if (name === "NotFoundError" || name === "OverconstrainedError") {
        setCameraState("no-camera");
        setErrorMsg("No camera was found on this device.");
      } else {
        setCameraState("error");
        setErrorMsg(err instanceof Error ? err.message : "Could not start the camera.");
      }
    }
  }, [tick, warmUp]);

  /** Save the current hand crop as a labeled PNG download (eval-set builder). */
  const captureFrame = useCallback(() => {
    const crop = lastCropRef.current;
    if (!crop) return;
    downloadCanvasAsPng(crop, captureLetter);
    setCaptureCount((n) => n + 1);
  }, [captureLetter]);

  // Clean teardown on unmount.
  useEffect(() => stopCamera, [stopCamera]);

  return {
    videoRef,
    overlayRef,
    cameraState,
    errorMsg,
    startCamera,
    stopCamera,
    result,
    verdict,
    handFound,
    fps,
    confHistory,
    word,
    setWord,
    holdProgress,
    captureLetter,
    setCaptureLetter,
    captureCount,
    captureFrame,
    warmStatus,
    warmError,
    warmProgress,
    warmSlow,
    warmUp,
  };
}
