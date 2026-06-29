"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, ImageUp, Loader2, UploadCloud } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";

import { Card, CardContent } from "@/components/ui/card";
import { interpret, type ConfidenceVerdict } from "@/lib/confidence";
import { revealVariants, scaleIn, transition } from "@/lib/motion";
import {
  cropToCanvas,
  detectHandInImage,
  getImageHandLandmarker,
} from "@/lib/handcrop";
import { classifyImage, type InferenceResult } from "@/lib/inference";
import { IMAGE_SIZE } from "@/lib/preprocess";
import { cn } from "@/lib/utils";

import { ShareButton } from "@/components/share-button";

import { ResultBars } from "./result-bars";

/** Click-to-try example images served from public/examples, with true labels. */
const EXAMPLES: ReadonlyArray<{ src: string; label: string }> = [
  { src: "/examples/A.png", label: "A" },
  { src: "/examples/B.png", label: "B" },
  { src: "/examples/C.png", label: "C" },
  { src: "/examples/L.png", label: "L" },
  { src: "/examples/W.png", label: "W" },
  { src: "/examples/Y.png", label: "Y" },
];

/**
 * Labels with a pre-computed Grad-CAM overlay in public/gradcam/. Grad-CAM needs
 * backward gradients, which the in-browser engine can't provide, so we precompute
 * (make gradcam-web) only for the fixed bundled examples and show them on click.
 */
const GRADCAM_LABELS = new Set(EXAMPLES.map((e) => e.label));

type Status = "idle" | "loading" | "done" | "error";

interface Outcome {
  result: InferenceResult;
  verdict: ConfidenceVerdict;
  /** Whether a hand was detected and the image was cropped before classifying. */
  handFound: boolean;
}

/**
 * Load an image URL into a fully-decoded HTMLImageElement.
 *
 * `crossOrigin` is set so same-origin object URLs and public assets draw to a
 * canvas without tainting it during preprocessing.
 */
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not load that image."));
    img.src = src;
  });
}

export function UploadPanel() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [trueLabel, setTrueLabel] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [showGradcam, setShowGradcam] = useState(false);
  /** Data URL of the cropped hand region actually classified (transparency). */
  const [cropPreview, setCropPreview] = useState<string | null>(null);

  const reduceMotion = useReducedMotion();

  const inputRef = useRef<HTMLInputElement>(null);
  // Track the most recent object URL so we can revoke the previous one.
  const objectUrlRef = useRef<string | null>(null);

  // Revoke any outstanding object URL on unmount to avoid leaks.
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, []);

  // Warm the IMAGE-mode hand landmarker so the first upload isn't slow. Browser
  // API access is confined to this effect (never module scope) to stay SSR-safe.
  useEffect(() => {
    void getImageHandLandmarker().catch(() => {
      // Warm-up failures are non-fatal: classifyFromUrl falls back to the
      // whole-image path, so swallow here and let the real run surface state.
    });
  }, []);

  /** Classify an already-loaded preview image and update result state. */
  const classifyFromUrl = useCallback(async (url: string, knownLabel: string | null) => {
    setStatus("loading");
    setError(null);
    setOutcome(null);
    setTrueLabel(knownLabel);
    setPreviewUrl(url);
    setShowGradcam(false);
    setCropPreview(null);
    try {
      const img = await loadImage(url);

      // Crop to the hand first (mirrors the webcam path) so the model sees a
      // tight, background-free hand instead of the whole squeezed photo. A
      // detection failure must never crash the UI — fall back to whole-image.
      let handFound = false;
      let source: CanvasImageSource = img;
      let cropUrl: string | null = null;
      try {
        const detection = await detectHandInImage(img);
        if (detection.found && detection.box) {
          const cropCanvas = cropToCanvas(
            img,
            img.naturalWidth,
            img.naturalHeight,
            detection.box,
            IMAGE_SIZE,
          );
          source = cropCanvas;
          handFound = true;
          // Keep the exact region the model saw, for a transparency preview.
          cropUrl = cropCanvas.toDataURL("image/png");
        }
      } catch {
        // Detection hiccup (e.g. WASM failed to load) — classify whole image.
      }

      const result = await classifyImage(source);
      setCropPreview(cropUrl);
      setOutcome({ result, verdict: interpret(result), handFound });
      setStatus("done");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Something went wrong running the model.",
      );
      setStatus("error");
    }
  }, []);

  /** Accept a user-selected File: validate it's an image, then classify. */
  const handleFile = useCallback(
    (file: File) => {
      const rejectWith = (message: string) => {
        if (objectUrlRef.current) {
          URL.revokeObjectURL(objectUrlRef.current);
          objectUrlRef.current = null;
        }
        setPreviewUrl(null);
        setOutcome(null);
        setStatus("error");
        setError(message);
      };
      if (!file.type.startsWith("image/")) {
        rejectWith(`“${file.name}” isn't an image. Please choose a PNG, JPG, or similar.`);
        return;
      }
      // Guard against oversized uploads that would stall a slow connection /
      // device. The classifier only needs a 128px crop, so this is generous.
      const MAX_BYTES = 10_000_000;
      if (file.size > MAX_BYTES) {
        rejectWith(
          `“${file.name}” is ${(file.size / 1_000_000).toFixed(1)} MB — please choose an image under 10 MB.`,
        );
        return;
      }
      // Revoke the previous object URL before minting a new one.
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      void classifyFromUrl(url, null);
    },
    [classifyFromUrl],
  );

  /** Load a bundled example; these are static assets, not object URLs. */
  const handleExample = useCallback(
    (src: string, label: string) => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
      void classifyFromUrl(src, label);
    },
    [classifyFromUrl],
  );

  const onInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) handleFile(file);
      // Reset so selecting the same file again re-triggers onChange.
      event.target.value = "";
    },
    [handleFile],
  );

  const openPicker = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const onDropzoneKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openPicker();
      }
    },
    [openPicker],
  );

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragActive(false);
      const file = event.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(true);
  }, []);

  const onDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
  }, []);

  const isLoading = status === "loading";

  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-6 md:grid-cols-2">
        {/* Dropzone — keyboard operable, ARIA-labelled. */}
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload an image. Press Enter or Space to choose a file, or drag and drop one here."
          aria-disabled={isLoading}
          onClick={openPicker}
          onKeyDown={onDropzoneKeyDown}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={cn(
            "flex min-h-[16rem] cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-6 text-center transition-[colors,transform] duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg",
            dragActive
              ? "border-accent bg-accent/10 motion-safe:scale-[1.01]"
              : "border-border bg-bg-card hover:border-accent/50",
            isLoading && "pointer-events-none opacity-60",
          )}
        >
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Uploaded hand sign preview"
              className="max-h-44 rounded-lg object-contain"
            />
          ) : (
            <>
              <UploadCloud className="h-10 w-10 text-fg-subtle" aria-hidden="true" />
              <p className="text-sm font-medium text-fg">
                Drop an image here, or click to choose
              </p>
              <p className="text-xs text-fg-subtle">
                PNG or JPG of a single hand sign. Runs entirely in your browser.
              </p>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="sr-only"
            onChange={onInputChange}
            tabIndex={-1}
          />
        </div>

        {/* Result / status panel. Each status animates its own entrance (a
            transform-only reveal); the panel has a fixed min-height so only one
            state shows at a time with no layout shift. We deliberately do NOT
            wrap this in AnimatePresence `mode="wait"` — the sequential
            exit-then-enter handoff stalled the state swap under automated
            interaction (the e2e example-classify test). Under reduced motion the
            variants are dropped so blocks swap instantly. */}
        <Card>
          <CardContent className="flex min-h-[16rem] flex-col justify-center p-6">
            <>
              {status === "idle" && (
                <motion.div
                  key="idle"
                  className="flex flex-col items-center gap-2 text-center"
                  variants={reduceMotion ? undefined : revealVariants}
                  initial={reduceMotion ? false : "hidden"}
                  animate={reduceMotion ? false : "visible"}
                  transition={transition}
                >
                  <ImageUp className="h-8 w-8 text-fg-subtle" aria-hidden="true" />
                  <p className="text-sm text-fg-muted">
                    Your prediction will appear here.
                  </p>
                </motion.div>
              )}

              {isLoading && (
                <motion.div
                  key="loading"
                  className="flex flex-col items-center gap-3 text-center"
                  role="status"
                  aria-live="polite"
                  variants={reduceMotion ? undefined : revealVariants}
                  initial={reduceMotion ? false : "hidden"}
                  animate={reduceMotion ? false : "visible"}
                  transition={transition}
                >
                  <Loader2
                    className="h-8 w-8 animate-spin text-accent"
                    aria-hidden="true"
                  />
                  <p className="text-sm text-fg-muted">
                    Warming up the model and classifying…
                  </p>
                </motion.div>
              )}

              {status === "error" && error && (
                <motion.div
                  key="error"
                  className="flex flex-col items-center gap-3 text-center"
                  role="alert"
                  variants={reduceMotion ? undefined : revealVariants}
                  initial={reduceMotion ? false : "hidden"}
                  animate={reduceMotion ? false : "visible"}
                  transition={transition}
                >
                  <AlertCircle className="h-8 w-8 text-amber-500" aria-hidden="true" />
                  <p className="text-sm text-fg">{error}</p>
                </motion.div>
              )}

              {status === "done" && outcome && (
                <motion.div
                  key="done"
                  className="flex flex-col gap-4"
                  aria-live="polite"
                  variants={reduceMotion ? undefined : revealVariants}
                  initial={reduceMotion ? false : "hidden"}
                  animate={reduceMotion ? false : "visible"}
                  transition={transition}
                >
                <div className="flex items-baseline justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-fg-subtle">
                      Predicted
                    </p>
                    <motion.p
                      className={cn(
                        "font-mono text-5xl font-bold leading-none",
                        outcome.verdict.unsure ? "text-amber-400" : "text-fg",
                      )}
                      variants={reduceMotion ? undefined : scaleIn}
                      initial={reduceMotion ? false : "hidden"}
                      animate={reduceMotion ? false : "visible"}
                    >
                      {outcome.result.top.label}
                    </motion.p>
                  </div>
                  {trueLabel && (
                    <p className="text-right text-xs text-fg-muted">
                      Example labelled{" "}
                      <span className="font-mono font-semibold text-fg">{trueLabel}</span>
                      {trueLabel === outcome.result.top.label
                        ? " — match"
                        : " — mismatch"}
                    </p>
                  )}
                </div>

                {!outcome.handFound && (
                  <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
                    No hand detected — classified the whole image; the result may
                    be unreliable. Try a clearer photo of a single hand.
                  </p>
                )}

                {outcome.handFound && cropPreview && (
                  <div className="flex items-center gap-3">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={cropPreview}
                      alt="The cropped hand region the model classified"
                      className="h-14 w-14 rounded-md border border-border-subtle object-cover"
                    />
                    <p className="text-xs text-fg-subtle">
                      Hand detected and cropped — this 128×128 region is what the
                      model actually classified.
                    </p>
                  </div>
                )}

                {outcome.verdict.unsure && (
                  <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
                    {outcome.verdict.hint}
                  </p>
                )}

                <ResultBars
                  ranked={outcome.result.ranked}
                  count={5}
                  unsure={outcome.verdict.unsure}
                />

                <ShareButton result={outcome.result} className="self-start" />

                {trueLabel && GRADCAM_LABELS.has(trueLabel) && (
                  <div className="border-t border-border-subtle pt-3">
                    <button
                      type="button"
                      onClick={() => setShowGradcam((s) => !s)}
                      aria-expanded={showGradcam}
                      className="text-xs font-medium text-accent hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                    >
                      {showGradcam ? "Hide" : "Show"} what the model looked at →
                    </button>
                    {showGradcam && (
                      <div className="mt-3 flex flex-col gap-2">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={`/gradcam/${trueLabel}.png`}
                          alt={`Grad-CAM heatmap for ${trueLabel}: warmer regions influenced the prediction more`}
                          className="max-h-44 self-start rounded-lg object-contain"
                        />
                        <p className="text-xs text-fg-subtle">
                          Grad-CAM saliency — red regions drove the prediction, blue
                          mattered least. Pre-computed offline (in-browser inference
                          can&apos;t expose the gradients Grad-CAM needs), so it&apos;s
                          shown for the bundled examples only.
                        </p>
                      </div>
                    )}
                  </div>
                )}
                </motion.div>
              )}
            </>
          </CardContent>
        </Card>
      </div>

      {/* Click-to-try examples. */}
      <div className="flex flex-col gap-3">
        <p className="text-xs uppercase tracking-wide text-fg-subtle">
          No image handy? Try an example
        </p>
        <ul className="flex flex-wrap gap-3">
          {EXAMPLES.map((example) => (
            <li key={example.src}>
              <button
                type="button"
                onClick={() => handleExample(example.src, example.label)}
                disabled={isLoading}
                aria-label={`Classify example image for the letter ${example.label}`}
                // CSS hover-lift + tap (transform-only, reduced-motion safe via
                // motion-safe:). NOT framer whileHover — a JS hover transform
                // keeps the element perpetually "unstable" and stalls
                // click-actionability (it broke the e2e example-classify test).
                className="group flex flex-col items-center gap-1 rounded-lg border border-border bg-bg-card p-2 transition-[colors,transform] duration-150 ease-out hover:border-accent/50 motion-safe:hover:-translate-y-0.5 motion-safe:active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg disabled:cursor-not-allowed disabled:opacity-50"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={example.src}
                  alt={`ASL sign for ${example.label}`}
                  className="h-14 w-14 rounded object-cover"
                />
                <span className="font-mono text-xs font-semibold text-fg-muted group-hover:text-fg">
                  {example.label}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
