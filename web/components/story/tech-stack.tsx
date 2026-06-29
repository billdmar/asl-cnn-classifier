import { Reveal, RevealItem } from "@/components/ui/reveal";
import { Badge } from "@/components/ui/badge";

interface StackGroup {
  heading: string;
  items: ReadonlyArray<string>;
}

const STACK: ReadonlyArray<StackGroup> = [
  {
    heading: "Modeling & training",
    items: [
      "PyTorch 2.x",
      "torchvision",
      "MobileNetV2 (transfer learning)",
      "AdamW + cosine LR",
      "CUDA / Apple-Silicon MPS / CPU",
    ],
  },
  {
    heading: "Evaluation & explainability",
    items: [
      "scikit-learn (confusion matrix, F1)",
      "Grad-CAM",
      "distribution-shift suite",
    ],
  },
  {
    heading: "Serving & in-browser inference",
    items: ["ONNX export", "onnxruntime-web (WASM / WebGPU)", "MediaPipe hand crop"],
  },
  {
    heading: "Web showcase",
    items: ["Next.js 15 (App Router)", "React 19", "TypeScript (strict)", "Tailwind CSS"],
  },
  {
    heading: "Engineering",
    items: ["GitHub Actions CI", "pytest + coverage gate", "ruff + black + mypy"],
  },
];

export function TechStack() {
  return (
    // Stagger each group in on scroll; each lifts subtly on hover (CSS
    // transform-only, so it composes with the reveal's translate without CLS).
    <Reveal stagger as="dl" className="grid gap-6 sm:grid-cols-2">
      {STACK.map((group) => (
        // RevealItem renders the single <div> group that the dl directly
        // contains (axe `dlitem`/`definition-list` require dt/dd to live in a
        // bare <div> child of the <dl> — no extra nesting). Hover-lift on it.
        <RevealItem
          key={group.heading}
          className="transition-transform duration-150 ease-out hover:-translate-y-0.5"
        >
          <dt className="text-sm font-semibold text-fg">{group.heading}</dt>
          <dd className="mt-3 flex flex-wrap gap-2">
            {group.items.map((item) => (
              <Badge key={item}>{item}</Badge>
            ))}
          </dd>
        </RevealItem>
      ))}
    </Reveal>
  );
}
