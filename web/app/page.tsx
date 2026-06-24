import { Footer } from "@/components/footer";
import { Hero } from "@/components/hero";
import { SiteHeader } from "@/components/site-header";
import { MetricsDashboard } from "@/components/metrics/metrics-dashboard";
import { UploadPanel } from "@/components/upload/upload-panel";
import { WebcamPanel } from "@/components/webcam/webcam-panel";

interface SectionShellProps {
  id: string;
  title: string;
  blurb: string;
  children: React.ReactNode;
}

function SectionShell({ id, title, blurb, children }: SectionShellProps) {
  return (
    <section id={id} aria-labelledby={`${id}-heading`} className="scroll-mt-24">
      <div className="mb-6">
        <h2 id={`${id}-heading`} className="text-2xl font-semibold tracking-tight">
          {title}
        </h2>
        <p className="mt-2 max-w-2xl text-pretty text-fg-muted">{blurb}</p>
      </div>
      {children}
    </section>
  );
}

export default function HomePage() {
  return (
    <>
      <SiteHeader />
      <main>
        <Hero />

        <div className="mx-auto max-w-6xl space-y-20 px-6 py-20">
          <SectionShell
            id="live"
            title="Live demo"
            blurb="Point your webcam at an ASL alphabet sign and watch the prediction update in real time. Your hand is detected and cropped, then classified — all on-device. Frames never leave your browser."
          >
            <WebcamPanel />
          </SectionShell>

          <SectionShell
            id="upload"
            title="Upload an image"
            blurb="Prefer not to use a webcam? Drop in a photo of a hand sign — or click an example — and get a classification with top-5 probabilities."
          >
            <UploadPanel />
          </SectionShell>

          <SectionShell
            id="metrics"
            title="Metrics"
            blurb="Every number below is produced by reproducible code in the repo and measured on the held-out test set — nothing is hardcoded."
          >
            <MetricsDashboard />
          </SectionShell>

          <SectionShell
            id="how"
            title="How it works"
            blurb="Hands are detected with MediaPipe, cropped, resized, and ImageNet-normalized, then run through a MobileNetV2 ONNX model with onnxruntime-web — entirely in the browser. Real-world webcam accuracy is lower than the benchmark; a measured number is in progress. Read the full story and model card on the About page."
          >
            <a
              href="/about"
              className="inline-flex items-center rounded-lg bg-accent-gradient px-5 py-2.5 font-medium text-bg transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
            >
              Read the project story &amp; model card →
            </a>
          </SectionShell>
        </div>
      </main>
      <Footer />
    </>
  );
}
