import Image from "next/image";

import { Card, CardContent } from "@/components/ui/card";

/**
 * Shows the single real Grad-CAM overlay committed to the repo (class V),
 * labelled honestly. More overlays can be generated locally with `make gradcam`.
 */
export function GradcamExplainer() {
  return (
    <div className="space-y-5">
      <p>
        Grad-CAM highlights the image regions the network leaned on for its prediction — a
        sanity check that the model attends to the hand, not the background. Below is the
        real saliency overlay for class <strong className="text-fg">V</strong>, produced
        by{" "}
        <code className="rounded bg-bg-subtle px-1 py-0.5 text-sm">src/gradcam.py</code>.
      </p>

      <Card className="overflow-hidden">
        <CardContent className="flex flex-col items-center gap-4 p-6">
          <figure className="flex flex-col items-center gap-3">
            <Image
              src="/gradcam/V.png"
              alt="Grad-CAM saliency overlay for the ASL letter V: a heatmap over the hand-sign image showing which regions most influenced the model's prediction, concentrated on the extended fingers."
              width={224}
              height={224}
              unoptimized
              className="rounded-lg border border-border"
            />
            <figcaption className="text-center text-sm text-fg-subtle">
              Grad-CAM saliency for class V
            </figcaption>
          </figure>
        </CardContent>
      </Card>

      <p className="text-sm text-fg-subtle">
        This is one real overlay, not a full gallery — overlays for any class can be
        generated locally with{" "}
        <code className="rounded bg-bg-subtle px-1 py-0.5">make gradcam</code>.
      </p>
    </div>
  );
}
