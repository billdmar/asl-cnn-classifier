import { ArrowRight, Upload } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

/**
 * Hero is the LCP content, so it is intentionally a Server Component with NO
 * JS-driven entrance animation. The earlier framer-motion version rendered the
 * hero text at opacity:0 until the framer-motion bundle hydrated and animated
 * it in — which pushed LCP out to ~10s. We now render the text opaque from the
 * server and use the JS-free CSS `fade-up` animation (tailwind keyframe, with
 * `both` fill and a short stagger) purely as progressive enhancement. The text
 * is paintable on first frame; the subtle motion runs without blocking LCP and
 * is automatically disabled under `prefers-reduced-motion` via globals.css.
 */
export function Hero() {
  return (
    <section
      id="top"
      className="relative overflow-hidden bg-accent-radial"
      aria-labelledby="hero-heading"
    >
      <div className="mx-auto flex max-w-3xl flex-col items-center px-6 py-24 text-center sm:py-32">
        <div className="animate-fade-up [animation-delay:0ms]">
          <Badge variant="accent">In-browser · privacy-first</Badge>
        </div>

        {/*
         * LCP-critical: no entrance animation and no transform on these two
         * elements, so they paint at the first frame and Largest Contentful
         * Paint fires immediately (transforms/opacity transitions on the LCP
         * node inflate "render delay" under Lighthouse's throttled trace).
         */}
        <h1
          id="hero-heading"
          className="mt-6 text-balance bg-accent-gradient bg-clip-text text-4xl font-bold leading-tight text-transparent sm:text-6xl"
        >
          Read sign language in your browser
        </h1>

        <p className="mt-6 max-w-2xl text-pretty text-lg text-fg-muted">
          100% in-browser inference · webcam frames never leave your device · MobileNetV2,
          59.8% honest cross-dataset accuracy (A–Y) vs 96.9% same-dataset.
        </p>

        <div className="mt-8 flex w-full animate-fade-up flex-col items-center gap-3 [animation-delay:240ms] sm:w-auto sm:flex-row">
          <a href="#live" className="w-full sm:w-auto">
            <Button size="lg" className="w-full sm:w-auto">
              Try the live demo
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Button>
          </a>
          <a href="#upload" className="w-full sm:w-auto">
            <Button variant="outline" size="lg" className="w-full sm:w-auto">
              <Upload className="h-4 w-4" aria-hidden="true" />
              Upload an image
            </Button>
          </a>
        </div>

        <p className="mt-6 animate-fade-up text-sm text-fg-subtle [animation-delay:320ms]">
          Real-world webcam accuracy is lower than the benchmark —{" "}
          <a
            href="#how"
            className="rounded text-fg-muted underline underline-offset-4 transition-colors hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
          >
            see How it works
          </a>
          .
        </p>
      </div>
    </section>
  );
}
