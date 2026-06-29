"use client";

/**
 * Live webcam ASL inference — the showcase centerpiece.
 *
 * Pipeline per frame (rAF loop):
 *   getUserMedia video -> MediaPipe detectForVideo -> hand crop box ->
 *   cropToCanvas(128) -> classifyImage -> calibrated "unsure" verdict -> UI.
 *
 * Everything runs client-side; frames never leave the browser. The camera is
 * only requested on an explicit user click (privacy + autoplay policy), and all
 * permission / device states are handled without crashing.
 *
 * SSR-safe by construction: every browser API (navigator, document, rAF,
 * MediaStream) is touched only inside effects / event handlers, never at module
 * scope or during render. The component renders a static shell on the server.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Camera, CameraOff, ShieldCheck, VideoOff } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ConfidenceBars } from "@/components/webcam/confidence-bars";
import { ConfidenceTimeseries } from "@/components/webcam/confidence-timeseries";
import { CAPTURE_LETTERS, downloadCanvasAsPng } from "@/components/webcam/capture";
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
  backspace,
  INITIAL_HOLD,
  type HoldState,
} from "@/components/webcam/word-builder";
import { interpret, type ConfidenceVerdict } from "@/lib/confidence";
import { scaleIn, tapScale } from "@/lib/motion";
import { cropBoxFromLandmarks, cropToCanvas, type CropBox } from "@/lib/handcrop";
import { CLASS_NAMES } from "@/lib/labels";
import { IMAGE_SIZE } from "@/lib/preprocess";
import type { InferenceResult } from "@/lib/inference";
import { useClassifier } from "@/lib/use-classifier";
import { cn } from "@/lib/utils";

/** Camera lifecycle states, each with distinct UI. */
type CameraState = "idle" | "requesting" | "active" | "denied" | "no-camera" | "error";

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

export function WebcamPanel() {
  const reduceMotion = useReducedMotion();
  const {
    status: warmStatus,
    error: warmError,
    progress: warmProgress,
    warmUp,
    classify,
  } = useClassifier();

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

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

  const isActive = cameraState === "active";
  const showUnsure = verdict?.unsure ?? false;
  const bigLetter = verdict && !showUnsure ? verdict.top.label : null;

  return (
    <Card className="overflow-hidden">
      <CardContent className="flex flex-col gap-6 p-6 lg:flex-row">
        {/* Video + overlay */}
        <div className="flex flex-1 flex-col gap-3">
          <div className="relative aspect-[4/3] w-full overflow-hidden rounded-lg border border-border bg-bg">
            {/* eslint-disable-next-line jsx-a11y/media-has-caption -- live camera preview, no audio track */}
            <video
              ref={videoRef}
              className="h-full w-full -scale-x-100 object-cover"
              playsInline
              muted
              aria-label="Live camera preview"
            />
            <canvas
              ref={overlayRef}
              className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100"
              aria-hidden="true"
            />

            {/* Non-active state overlays */}
            {!isActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-bg/80 p-6 text-center">
                <CameraStateContent state={cameraState} message={errorMsg} />
              </div>
            )}

            {/* "Show your hand" prompt while active but no hand detected */}
            {isActive && !handFound && (
              <div className="absolute inset-x-0 bottom-0 flex justify-center p-3">
                <Badge variant="default" className="bg-bg/80">
                  <VideoOff className="h-3.5 w-3.5" aria-hidden="true" />
                  Show your hand in the box
                </Badge>
              </div>
            )}

            {/* FPS readout */}
            {isActive && (
              <div className="absolute right-2 top-2">
                <Badge variant="default" className="bg-bg/80 font-mono">
                  {fps.toFixed(0)} fps
                </Badge>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            {isActive ? (
              // motion.button (not <Button>) so the camera toggle gets a subtle
              // tap press; keeps native button semantics + shared button styles.
              <motion.button
                type="button"
                whileTap={reduceMotion ? undefined : tapScale}
                onClick={stopCamera}
                className={buttonVariants({ variant: "outline" })}
              >
                <CameraOff className="h-4 w-4" aria-hidden="true" />
                Stop camera
              </motion.button>
            ) : (
              <motion.button
                type="button"
                whileTap={reduceMotion ? undefined : tapScale}
                onClick={() => void startCamera()}
                disabled={cameraState === "requesting"}
                className={buttonVariants()}
              >
                <Camera className="h-4 w-4" aria-hidden="true" />
                {cameraState === "requesting" ? "Requesting…" : "Start camera"}
              </motion.button>
            )}
            {warmStatus === "warming" && (
              <div className="flex flex-col gap-1" role="status" aria-live="polite">
                <span className="text-sm text-fg-subtle">
                  Loading model…{" "}
                  {warmProgress > 0 ? `${Math.round(warmProgress * 100)}%` : ""}
                </span>
                <div className="h-1 w-40 overflow-hidden rounded-full bg-border">
                  <div
                    className="h-full bg-accent-gradient transition-[width] duration-150"
                    style={{ width: `${Math.round(warmProgress * 100)}%` }}
                    aria-hidden="true"
                  />
                </div>
              </div>
            )}
            {warmStatus === "error" && (
              <span className="text-sm text-amber-400" role="status">
                {warmError ?? "Model failed to load."}
              </span>
            )}
          </div>
        </div>

        {/* Prediction panel */}
        <div className="flex w-full flex-col gap-4 lg:w-80">
          <div
            className={cn(
              "flex min-h-[7rem] flex-col items-center justify-center rounded-lg border p-4 text-center",
              // Subtle accent glow fades in for a confident letter (box-shadow
              // only — no reflow/CLS). Neutralized under reduced motion.
              !reduceMotion && "transition-shadow duration-300",
              showUnsure
                ? "border-amber-400/40 bg-amber-400/5"
                : bigLetter
                  ? "border-accent/40 bg-bg shadow-[0_0_24px_-4px_theme(colors.accent/35%)]"
                  : "border-border bg-bg",
            )}
            aria-live="polite"
          >
            {showUnsure && verdict ? (
              <>
                <span className="text-sm font-semibold text-amber-400">Unsure</span>
                <p className="mt-1 text-pretty text-sm text-fg-muted">{verdict.hint}</p>
              </>
            ) : bigLetter ? (
              <motion.span
                key={bigLetter}
                variants={scaleIn}
                initial={reduceMotion ? false : "hidden"}
                animate="visible"
                className="bg-accent-gradient bg-clip-text font-mono text-6xl font-bold text-transparent"
              >
                {bigLetter}
              </motion.span>
            ) : (
              <span className="text-sm text-fg-subtle">
                {isActive ? "Reading…" : "Start the camera to begin"}
              </span>
            )}
          </div>

          {result && <ConfidenceBars ranked={result.ranked} unsure={showUnsure} />}

          {confHistory.length > 1 && (
            <ConfidenceTimeseries points={confHistory} />
          )}

          {/* Fingerspelling word-builder: hold a letter to spell words. */}
          <div className="rounded-lg border border-border bg-bg p-3">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-medium text-fg-muted">
                Word builder — hold a letter to add it
              </p>
              <span className="text-[11px] text-fg-subtle">
                {word.length} letter{word.length === 1 ? "" : "s"}
              </span>
            </div>
            <div
              className="mt-2 min-h-[2.5rem] rounded-md bg-bg-subtle px-3 py-2 font-mono text-2xl tracking-widest text-fg"
              aria-live="polite"
              aria-label={word ? `Word so far: ${word.split("").join(" ")}` : "No letters yet"}
            >
              {word || <span className="text-base text-fg-subtle">…</span>}
            </div>
            {/* Hold-progress bar (reduced-motion safe: it's a width, not an animation). */}
            <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-border">
              <div
                className="h-full bg-accent-gradient transition-[width] duration-100"
                style={{ width: `${Math.round(holdProgress * 100)}%` }}
                aria-hidden="true"
              />
            </div>
            <div className="mt-2 flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setWord((w) => backspace(w))}
                disabled={!word}
              >
                Backspace
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setWord("")}
                disabled={!word}
              >
                Clear
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void navigator.clipboard?.writeText(word)}
                disabled={!word}
              >
                Copy
              </Button>
            </div>
            <p className="mt-2 text-[11px] text-fg-subtle">
              Hold a confident letter steady for ~1.5s to lock it. Works best for the
              static letters — J and Z are motion signs and won&apos;t hold reliably.
            </p>
          </div>

          {/* Guidance */}
          <div className="rounded-lg border border-border bg-bg p-3 text-xs text-fg-subtle">
            <p className="font-medium text-fg-muted">For best results</p>
            <ul className="mt-1 list-inside list-disc space-y-0.5">
              <li>Use a plain background</li>
              <li>Make sure your hand is well lit</li>
              <li>Keep your hand centered in the box</li>
            </ul>
          </div>

          {/* Eval-set builder — unobtrusive, collapsed by default. */}
          {isActive && (
            <details className="rounded-lg border border-border bg-bg text-xs">
              <summary className="cursor-pointer rounded-lg px-3 py-2 font-medium text-fg-muted outline-none focus-visible:ring-2 focus-visible:ring-accent">
                Build a test set
              </summary>
              <div className="flex flex-col gap-2 px-3 pb-3">
                <div className="flex items-end gap-2">
                  <label className="flex flex-col gap-1 text-fg-subtle">
                    <span>Label</span>
                    <select
                      value={captureLetter}
                      onChange={(e) => setCaptureLetter(e.target.value)}
                      className="rounded-md border border-border bg-bg-card px-2 py-1 font-mono text-sm text-fg outline-none focus-visible:ring-2 focus-visible:ring-accent"
                      aria-label="Letter to label the captured frame"
                    >
                      {CAPTURE_LETTERS.map((letter) => (
                        <option key={letter} value={letter}>
                          {letter}
                        </option>
                      ))}
                    </select>
                  </label>
                  <Button
                    variant="outline"
                    onClick={captureFrame}
                    disabled={!handFound}
                  >
                    Capture
                  </Button>
                </div>
                <p className="text-fg-subtle">
                  Captured frames download locally — use them to build a
                  real-world test set.
                  {captureCount > 0 && (
                    <span className="text-fg-muted"> ({captureCount} saved)</span>
                  )}
                </p>
              </div>
            </details>
          )}

          <p className="flex items-center gap-1.5 text-xs text-fg-subtle">
            <ShieldCheck className="h-3.5 w-3.5 text-accent" aria-hidden="true" />
            Frames never leave your browser.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

/** Centered overlay content for each non-active camera state. */
function CameraStateContent({
  state,
  message,
}: {
  state: CameraState;
  message: string | null;
}) {
  switch (state) {
    case "requesting":
      return (
        <p className="text-sm text-fg-muted" role="status">
          Requesting camera access…
        </p>
      );
    case "denied":
      return (
        <>
          <CameraOff className="h-8 w-8 text-amber-400" aria-hidden="true" />
          <p className="text-sm text-fg-muted">
            {message ?? "Camera permission denied."}
          </p>
        </>
      );
    case "no-camera":
      return (
        <>
          <VideoOff className="h-8 w-8 text-fg-subtle" aria-hidden="true" />
          <p className="text-sm text-fg-muted">{message ?? "No camera available."}</p>
        </>
      );
    case "error":
      return (
        <>
          <CameraOff className="h-8 w-8 text-amber-400" aria-hidden="true" />
          <p className="text-sm text-fg-muted">{message ?? "Camera error."}</p>
        </>
      );
    case "idle":
    case "active":
    default:
      return (
        <>
          <Camera className="h-8 w-8 text-fg-subtle" aria-hidden="true" />
          <p className="text-sm text-fg-muted">
            Click “Start camera” to try live recognition.
          </p>
        </>
      );
  }
}
