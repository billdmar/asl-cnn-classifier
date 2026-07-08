"use client";

import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PipelineStepper } from "./pipeline-stepper";
import { StepRawFrame } from "./step-raw-frame";
import { StepHandDetection } from "./step-hand-detection";
import { StepCrop } from "./step-crop";
import { StepTensorChannels } from "./step-tensor-channels";
import { StepPrediction } from "./step-prediction";
import {
  captureImageFrame,
  runFullPipeline,
  type FrozenFrame,
  type PipelineSnapshot,
} from "@/lib/explainer/pipeline-steps";
import { getImageHandLandmarker } from "@/lib/handcrop";
import { scaleIn } from "@/lib/motion";

const STEP_LABELS = ["Raw frame", "Hand detection", "Crop", "Tensor channels", "Prediction"];
const EXAMPLES = ["/examples/A.png", "/examples/B.png", "/examples/C.png", "/examples/L.png", "/examples/W.png", "/examples/Y.png"];

export function ExplainerPanel() {
  const [snapshot, setSnapshot] = useState<PipelineSnapshot | null>(null);
  const [activeStep, setActiveStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runExample = useCallback(async (src: string) => {
    setLoading(true);
    setError(null);
    try {
      const img = new Image();
      img.crossOrigin = "anonymous";
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error("Failed to load example image"));
        img.src = src;
      });
      const frame: FrozenFrame = captureImageFrame(img);
      const landmarker = await getImageHandLandmarker();
      const handResult = landmarker.detect(img);
      const snap = await runFullPipeline(frame, handResult);
      setSnapshot(snap);
      setActiveStep(1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Pipeline failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setSnapshot(null);
    setActiveStep(1);
    setError(null);
  }, []);

  useEffect(() => {
    const handler = (e: Event) => {
      const src = (e as CustomEvent<{ src: string }>).detail.src;
      if (src) runExample(src);
    };
    window.addEventListener("explain-image", handler);
    return () => window.removeEventListener("explain-image", handler);
  }, [runExample]);

  if (!snapshot) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center gap-4 p-6 text-center">
          <p className="text-fg-muted">
            Click an example to see the full inference pipeline step by step.
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            {EXAMPLES.map((src) => {
              const letter = src.split("/").pop()?.replace(".png", "") ?? "?";
              return (
                <button
                  key={src}
                  onClick={() => runExample(src)}
                  disabled={loading}
                  className="flex h-12 w-12 items-center justify-center rounded-lg border border-border-subtle bg-bg-card text-lg font-bold text-fg transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                  aria-label={`Explain letter ${letter}`}
                >
                  {letter}
                </button>
              );
            })}
          </div>
          {loading && <Badge>Loading model &amp; running pipeline&hellip;</Badge>}
          {error && <p className="text-sm text-red-400">{error}</p>}
        </CardContent>
      </Card>
    );
  }

  return (
    <motion.div className="space-y-6" variants={scaleIn} initial="hidden" animate="visible">
      <div className="flex items-center justify-between">
        <PipelineStepper activeStep={activeStep} onStepClick={setActiveStep} labels={STEP_LABELS} />
        <button
          onClick={reset}
          className="rounded-md border border-border-subtle px-3 py-1.5 text-xs text-fg-muted transition-colors hover:text-fg"
        >
          Reset
        </button>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={activeStep}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.2 }}
        >
          {activeStep === 1 && <StepRawFrame frame={snapshot.frame} />}
          {activeStep === 2 && <StepHandDetection frame={snapshot.frame} handDetection={snapshot.handDetection} />}
          {activeStep === 3 && <StepCrop cropCanvas={snapshot.cropCanvas} imageData={snapshot.imageData} />}
          {activeStep === 4 && <StepTensorChannels tensor={snapshot.tensor} />}
          {activeStep === 5 && <StepPrediction logits={snapshot.logits} initialProbs={snapshot.probs} />}
        </motion.div>
      </AnimatePresence>

      <div className="flex justify-center gap-2">
        <button
          onClick={() => setActiveStep((s) => Math.max(1, s - 1))}
          disabled={activeStep === 1}
          className="rounded-md border border-border-subtle px-4 py-1.5 text-sm text-fg-muted transition-colors hover:text-fg disabled:opacity-30"
        >
          &larr; Previous
        </button>
        <button
          onClick={() => setActiveStep((s) => Math.min(STEP_LABELS.length, s + 1))}
          disabled={activeStep === STEP_LABELS.length}
          className="rounded-md border border-border-subtle px-4 py-1.5 text-sm text-fg-muted transition-colors hover:text-fg disabled:opacity-30"
        >
          Next &rarr;
        </button>
      </div>
    </motion.div>
  );
}
