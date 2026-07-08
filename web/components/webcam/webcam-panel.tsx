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
 *
 * All loop/inference logic lives in ./use-webcam-loop.ts; this file is purely
 * the render layer.
 */

import { motion, useReducedMotion } from "framer-motion";
import { Camera, CameraOff, ShieldCheck, VideoOff } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ConfidenceBars } from "@/components/webcam/confidence-bars";
import { ConfidenceTimeseries } from "@/components/webcam/confidence-timeseries";
import { CAPTURE_LETTERS } from "@/components/webcam/capture";
import { backspace } from "@/components/webcam/word-builder";
import { useWebcamLoop, type CameraState } from "@/components/webcam/use-webcam-loop";
import { scaleIn, tapScale } from "@/lib/motion";
import { useKeyboardShortcuts } from "@/lib/use-keyboard-shortcuts";
import { ShareButton, buildShareUrl, shareResult } from "@/components/share-button";
import { cn } from "@/lib/utils";

export function WebcamPanel() {
  const reduceMotion = useReducedMotion();
  const {
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
  } = useWebcamLoop();

  // Keyboard shortcuts (mapping + help dialog live in Stream B's modules; the
  // "?" help trigger is wired globally in the site header). Here we bind the
  // panel-scoped actions to the existing callbacks. Space toggles the camera,
  // C copies the spelled word, R resets it, S shares the current prediction.
  useKeyboardShortcuts({
    camera: () => {
      if (cameraState === "active") stopCamera();
      else void startCamera();
    },
    copy: () => {
      if (word) void navigator.clipboard?.writeText(word);
    },
    reset: () => setWord(""),
    share: () => {
      if (result) void shareResult(buildShareUrl(result));
    },
  });

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
                {warmSlow && (
                  <div className="mt-1 flex flex-col gap-1">
                    <span className="max-w-[16rem] text-pretty text-xs text-fg-subtle">
                      Still loading the model… (first visit downloads ~9 MB; it&apos;s
                      cached after).
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      className="self-start"
                      onClick={warmUp}
                    >
                      Retry
                    </Button>
                  </div>
                )}
              </div>
            )}
            {warmStatus === "error" && (
              <div className="flex flex-col gap-1" role="status" aria-live="polite">
                <span className="text-sm text-amber-400">
                  {warmError ?? "Model failed to load."}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  className="self-start"
                  onClick={warmUp}
                >
                  Retry
                </Button>
              </div>
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

          {result && (
            <ShareButton result={result} className="self-start" />
          )}

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
            Click &ldquo;Start camera&rdquo; to try live recognition.
          </p>
        </>
      );
  }
}
