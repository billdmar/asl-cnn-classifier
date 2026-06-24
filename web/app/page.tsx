import { Footer } from "@/components/footer";
import { Hero } from "@/components/hero";
import { SiteHeader } from "@/components/site-header";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const SECTIONS = [
  {
    id: "live",
    title: "Live demo",
    description:
      "Point your webcam at an ASL alphabet sign and watch the prediction update in real time — all on-device.",
  },
  {
    id: "upload",
    title: "Upload an image",
    description:
      "Prefer not to use a webcam? Drop in a photo of a hand sign and get a classification.",
  },
  {
    id: "metrics",
    title: "Metrics",
    description:
      "The MobileNetV2 model reaches 96.8% held-out test accuracy and 97.8% validation accuracy. Confusion matrix and per-class breakdowns land next.",
  },
  {
    id: "how",
    title: "How it works",
    description:
      "Frames are preprocessed and run through an ONNX model with onnxruntime-web — entirely in the browser. Real-world webcam accuracy is lower than the benchmark; measured numbers coming.",
  },
] as const;

export default function HomePage() {
  return (
    <>
      <SiteHeader />
      <main>
        <Hero />

        <div className="mx-auto max-w-6xl space-y-16 px-6 py-20">
          {SECTIONS.map((section) => (
            <section
              key={section.id}
              id={section.id}
              aria-labelledby={`${section.id}-heading`}
              className="scroll-mt-24"
            >
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between gap-4">
                    <CardTitle id={`${section.id}-heading`}>{section.title}</CardTitle>
                    <Badge>Coming in the next PR</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-pretty">{section.description}</p>
                </CardContent>
              </Card>
            </section>
          ))}
        </div>
      </main>
      <Footer />
    </>
  );
}
