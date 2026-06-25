import { ArrowRight, ArrowDown } from "lucide-react";

import { cn } from "@/lib/utils";

/**
 * A styled, screen-reader-described diagram of the model architecture and the
 * in-browser inference flow. Pure CSS/SVG — no images — so it renders crisply
 * in the static export and scales with the type.
 */

interface FlowStepProps {
  label: string;
  sub?: string;
  /** Visually emphasize the terminal/model nodes. */
  accent?: boolean;
}

function FlowStep({ label, sub, accent = false }: FlowStepProps) {
  return (
    <div
      className={cn(
        "flex min-w-[7.5rem] flex-1 flex-col items-center justify-center rounded-lg border px-3 py-3 text-center",
        accent
          ? "border-accent/40 bg-accent/10 text-fg"
          : "border-border bg-bg-subtle text-fg",
      )}
    >
      <span className="text-sm font-medium">{label}</span>
      {sub ? <span className="mt-0.5 text-xs text-fg-subtle">{sub}</span> : null}
    </div>
  );
}

const MODEL_STEPS: ReadonlyArray<FlowStepProps> = [
  { label: "Input", sub: "3×128×128 RGB" },
  { label: "MobileNetV2 backbone", sub: "ImageNet-pretrained", accent: true },
  { label: "Classifier head", sub: "replaced, 26-way" },
  { label: "26 logits", sub: "A–Z", accent: true },
];

const INFERENCE_STEPS: ReadonlyArray<FlowStepProps> = [
  { label: "Webcam / upload", sub: "in the browser" },
  { label: "MediaPipe hand-crop", sub: "locate the hand" },
  { label: "Resize + normalize", sub: "ImageNet stats → CHW" },
  { label: "ONNX model", sub: "onnxruntime-web", accent: true },
  { label: "Softmax → output", sub: "predicted letter", accent: true },
];

function Arrow() {
  return (
    <>
      <ArrowRight
        className="hidden h-5 w-5 shrink-0 text-fg-subtle sm:block"
        aria-hidden="true"
      />
      <ArrowDown
        className="h-5 w-5 shrink-0 text-fg-subtle sm:hidden"
        aria-hidden="true"
      />
    </>
  );
}

function FlowRow({ steps }: { steps: ReadonlyArray<FlowStepProps> }) {
  return (
    <div
      className="flex flex-col items-stretch gap-2 sm:flex-row sm:items-center"
      aria-hidden="true"
    >
      {steps.map((step, i) => (
        <div
          key={step.label}
          className="flex flex-col items-stretch gap-2 sm:flex-1 sm:flex-row sm:items-center"
        >
          <FlowStep {...step} />
          {i < steps.length - 1 ? <Arrow /> : null}
        </div>
      ))}
    </div>
  );
}

export function ArchitectureDiagram() {
  return (
    <figure className="space-y-8">
      {/* Screen-reader text description of the whole diagram. */}
      <figcaption className="sr-only">
        Architecture diagram. The model is a MobileNetV2 backbone pretrained on ImageNet,
        with its classifier head replaced by a 26-way head, taking a 3 by 128 by 128 RGB
        image and producing 26 class logits for the letters A through Z. The inference
        flow runs entirely in the browser: a webcam frame or uploaded image is cropped to
        the hand with MediaPipe, resized and normalized with ImageNet statistics into
        channel-first layout, run through the ONNX model with onnxruntime-web, and passed
        through a softmax to produce the predicted letter.
      </figcaption>

      <div>
        <p className="mb-3 text-sm font-semibold text-fg">Model</p>
        <FlowRow steps={MODEL_STEPS} />
      </div>

      <div>
        <p className="mb-3 text-sm font-semibold text-fg">In-browser inference flow</p>
        <FlowRow steps={INFERENCE_STEPS} />
      </div>
    </figure>
  );
}
