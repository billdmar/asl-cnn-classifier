"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, ImageUp, Loader2, UploadCloud } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { interpret, type ConfidenceVerdict } from "@/lib/confidence";
import {
  cropToCanvas,
  detectHandInImage,
  getImageHandLandmarker,
} from "@/lib/handcrop";
import { classifyImage, type InferenceResult } from "@/lib/inference";
import { IMAGE_SIZE } from "@/lib/preprocess";
import { cn } from "@/lib/utils";

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
    try {
      const img = await loadImage(url);

      // Crop to the hand first (mirrors the webcam path) so the model sees a
      // tight, background-free hand instead of the whole squeezed photo. A
      // detection failure must never crash the UI — fall back to whole-image.
      let handFound = false;
      let source: CanvasImageSource = img;
      try {
        const detection = await detectHandInImage(img);
        if (detection.found && detection.box) {
          source = cropToCanvas(
            img,
            img.naturalWidth,
            img.naturalHeight,
            detection.box,
            IMAGE_SIZE,
          );
          handFound = true;
        }
      } catch {
        // Detection hiccup (e.g. WASM failed to load) — classify whole image.
      }

      const result = await classifyImage(source);
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
      if (!file.type.startsWith("image/")) {
        // Drop the rejected file; keep any prior preview cleared for clarity.
        if (objectUrlRef.current) {
          URL.revokeObjectURL(objectUrlRef.current);
          objectUrlRef.current = null;
        }
        setPreviewUrl(null);
        setOutcome(null);
        setStatus("error");
        setError(`“${file.name}” isn't an image. Please choose a PNG, JPG, or similar.`);
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
            "flex min-h-[16rem] cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-6 text-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg",
            dragActive
              ? "border-accent bg-accent/10"
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

        {/* Result / status panel. */}
        <Card>
          <CardContent className="flex min-h-[16rem] flex-col justify-center p-6">
            {status === "idle" && (
              <div className="flex flex-col items-center gap-2 text-center">
                <ImageUp className="h-8 w-8 text-fg-subtle" aria-hidden="true" />
                <p className="text-sm text-fg-muted">Your prediction will appear here.</p>
              </div>
            )}

            {isLoading && (
              <div
                className="flex flex-col items-center gap-3 text-center"
                role="status"
                aria-live="polite"
              >
                <Loader2
                  className="h-8 w-8 animate-spin text-accent"
                  aria-hidden="true"
                />
                <p className="text-sm text-fg-muted">
                  Warming up the model and classifying…
                </p>
              </div>
            )}

            {status === "error" && error && (
              <div className="flex flex-col items-center gap-3 text-center" role="alert">
                <AlertCircle className="h-8 w-8 text-amber-500" aria-hidden="true" />
                <p className="text-sm text-fg">{error}</p>
              </div>
            )}

            {status === "done" && outcome && (
              <div className="flex flex-col gap-4" aria-live="polite">
                <div className="flex items-baseline justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-fg-subtle">
                      Predicted
                    </p>
                    <p
                      className={cn(
                        "font-mono text-5xl font-bold leading-none",
                        outcome.verdict.unsure ? "text-amber-400" : "text-fg",
                      )}
                    >
                      {outcome.result.top.label}
                    </p>
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
              </div>
            )}
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
                className="group flex flex-col items-center gap-1 rounded-lg border border-border bg-bg-card p-2 transition-colors hover:border-accent/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg disabled:cursor-not-allowed disabled:opacity-50"
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
