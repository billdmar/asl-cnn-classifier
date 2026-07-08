"use client";

/**
 * ExplainerPanel — top-level orchestrator for the inference pipeline explainer.
 *
 * Self-contained: loads an example image via its own "Use an example" button,
 * runs hand detection (IMAGE mode) and the full pipeline, then renders a
 * step-by-step walk-through with animated transitions between steps.
 *
 * Designed to be lazy-loaded via next/dynamic so it doesn't bloat the initial
 * page bundle.
 */

import { useCallback, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ImageIcon, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { PipelineStepper } from "@/components/explainer/pipeline-stepper";
import { StepRawFrame } from "@/components/explainer/step-raw-frame";
import { StepHandDetection } from "@/components/explainer/step-hand-detection";
import { StepCrop } from "@/components/explainer/step-crop";
import { StepTensorChannels } from "@/components/explainer/step-tensor-channels";
import { StepPrediction } from "@/components/explainer/step-prediction";
import { tensorToChannelViz } from "@/lib/explainer/tensor-viz";
import {
  captureImageFrame,
  runFullPipeline,
  type FrozenFrame,
  type PipelineSnapshot,
} from "@/lib/explainer/pipeline-steps";
import { IMAGE_SIZE } from "@/lib/preprocess";

const STEP_LABELS = [
  "Raw frame",
  "Hand detection",
  "Crop",
  "Tensor channels",
  "Prediction",
];

const EXAMPLE_IMAGES = [
  "/examples/A.png",
  "/examples/B.png",
  "/examples/C.png",
  "/examples/L.png",
  "/examples/W.png",
  "/examples/Y.png",
];

/** Framer motion variants for step content transitions. */
const stepVariants = {
  enter: { opacity: 0, x: 20 },
  center: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -20 },
};

export function ExplainerPanel() {
  const [frozenFrame, setFrozenFrame] = useState<FrozenFrame | null>(null);
  const [snapshot, setSnapshot] = useState<PipelineSnapshot | null>(null);
  const [activeStep, setActiveStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** Load an example image, run hand detection + full pipeline. */
  const loadExample = useCallback(async (src?: string) => {
    const url = src ?? EXAMPLE_IMAGES[Math.floor(Math.random() * EXAMPLE_IMAGES.length)]!;
    setLoading(true);
    setError(null);

    try {
      // Load image.
      const img = new Image();
      img.crossOrigin = "anonymous";
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error("Failed to load example image."));
        img.src = url;
      });

      // Capture frame.
      const frame = captureImageFrame(img);
      setFrozenFrame(frame);

      // Run hand detection in IMAGE mode.
      const { getImageHandLandmarker } = await import("@/lib/handcrop");
      const landmarker = await getImageHandLandmarker();
      const handResult = landmarker.detect(img);

      // Run full pipeline.
      const result = await runFullPipeline(frame, handResult);
      setSnapshot(result);
      setActiveStep(1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Pipeline failed.");
    } finally {
      setLoading(false);
    }
  }, []);

  /** Reset the explainer to its initial prompt state. */
  const reset = useCallback(() => {
    setFrozenFrame(null);
    setSnapshot(null);
    setActiveStep(1);
    setError(null);
  }, []);

  // Compute channel visualizations when we have a snapshot.
  const channels = useMemo(() => {
    if (!snapshot) return null;
    return tensorToChannelViz(snapshot.tensor, IMAGE_SIZE);
  }, [snapshot]);

  // No frame loaded: show prompt.
  if (!frozenFrame || !snapshot) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center gap-4 p-8 text-center">
          <ImageIcon className="h-10 w-10 text-fg-muted" aria-hidden="true" />
          <p className="max-w-sm text-sm text-fg-muted">
            Freeze a webcam frame or upload an image above, then come back here
            to see the pipeline step by step.
          </p>
          <Button
            onClick={() => void loadExample()}
            disabled={loading}
          >
            {loading ? "Loading…" : "Use an example"}
          </Button>
          {/* Quick-pick grid */}
          <div className="flex flex-wrap justify-center gap-2">
            {EXAMPLE_IMAGES.map((src) => {
              const letter = src.split("/").pop()?.replace(".png", "") ?? "?";
              return (
                <button
                  key={src}
                  type="button"
                  onClick={() => void loadExample(src)}
                  disabled={loading}
                  className="flex h-8 w-8 items-center justify-center rounded-md border border-border-subtle font-mono text-sm text-fg-muted transition-colors hover:border-accent hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50"
                  aria-label={`Load example letter ${letter}`}
                >
                  {letter}
                </button>
              );
            })}
          </div>
          {error && (
            <p className="text-sm text-amber-400" role="alert">
              {error}
            </p>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Stepper */}
      <PipelineStepper
        activeStep={activeStep}
        onStepClick={setActiveStep}
        labels={STEP_LABELS}
      />

      {/* Step content with animated transitions */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeStep}
          variants={stepVariants}
          initial="enter"
          animate="center"
          exit="exit"
          transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
        >
          {activeStep === 1 && (
            <StepRawFrame frameCanvas={snapshot.frame.source} />
          )}
          {activeStep === 2 && (
            <StepHandDetection
              frameCanvas={snapshot.frame.source}
              cropBox={snapshot.handDetection.box ?? null}
              handFound={snapshot.handDetection.found}
            />
          )}
          {activeStep === 3 && (
            <StepCrop
              cropCanvas={snapshot.cropCanvas}
              imageData={snapshot.imageData}
            />
          )}
          {activeStep === 4 && channels && (
            <StepTensorChannels channels={channels} />
          )}
          {activeStep === 5 && (
            <StepPrediction
              logits={snapshot.logits}
              initialProbs={snapshot.probs}
            />
          )}
        </motion.div>
      </AnimatePresence>

      {/* Navigation + reset */}
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setActiveStep((s) => Math.max(1, s - 1))}
            disabled={activeStep === 1}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setActiveStep((s) => Math.min(5, s + 1))}
            disabled={activeStep === 5}
          >
            Next
          </Button>
        </div>
        <Button variant="ghost" size="sm" onClick={reset}>
          <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
          Reset
        </Button>
      </div>
    </div>
  );
}
