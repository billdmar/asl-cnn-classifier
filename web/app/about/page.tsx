import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft, Github } from "lucide-react";

import { Footer } from "@/components/footer";
import { SiteHeader } from "@/components/site-header";
import { AccuracyStory } from "@/components/story/accuracy-story";
import { ArchitectureDiagram } from "@/components/story/architecture-diagram";
import { EthicsLimitations } from "@/components/story/ethics-limitations";
import { GradcamExplainer } from "@/components/story/gradcam-explainer";
import { InferenceExplainer } from "@/components/story/inference-explainer";
import { StorySection } from "@/components/story/section";
import { TechStack } from "@/components/story/tech-stack";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const REPO_URL = "https://github.com/billdmar/asl-cnn-classifier";

export const metadata: Metadata = {
  title: "Project & model card — ASL Classifier",
  description:
    "How the in-browser ASL alphabet classifier was built: MobileNetV2 transfer learning, the honest benchmark-vs-real-world accuracy story, Grad-CAM explainability, in-browser ONNX inference, and ethics & limitations.",
};

export default function AboutPage() {
  return (
    <>
      <SiteHeader />
      <main className="mx-auto max-w-3xl px-6 py-16 sm:py-20">
        <header className="mb-16">
          <Badge variant="accent">Project & model card</Badge>
          {/*
           * LCP element for /about: transform-only `animate-rise-up` paints
           * fully opaque from the first frame (never opacity:0, never JS-wrapped),
           * so Largest Contentful Paint fires immediately.
           */}
          <h1 className="mt-5 animate-rise-up text-balance text-4xl font-bold leading-tight text-fg sm:text-5xl">
            Reading the ASL alphabet, in the browser
          </h1>
          <p className="mt-5 text-pretty text-lg leading-relaxed text-fg-muted">
            An end-to-end computer-vision project: a MobileNetV2 transfer model that
            classifies the 26 static letters of the American Sign Language alphabet,
            exported to ONNX and run entirely client-side. This page is the honest story
            of how it works and where it falls short.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/">
              <Button variant="outline">
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                Back to the demo
              </Button>
            </Link>
            <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
              <Button variant="ghost">
                <Github className="h-4 w-4" aria-hidden="true" />
                View source on GitHub
              </Button>
            </a>
          </div>
        </header>

        <div className="space-y-20">
          <StorySection id="problem" eyebrow="The problem" title="What this project is">
            <div className="space-y-4">
              <p>
                The American Sign Language alphabet is fingerspelled with 26 distinct
                static hand shapes (A–Z). The goal here is narrow and concrete: given a
                single image of a hand, predict which letter it is — and do it fast,
                privately, and on-device, with no server in the loop.
              </p>
              <p>
                It is built as a full ML-engineering lifecycle rather than a one-off
                notebook: dataset ingestion and stratified splitting, augmentation-aware
                training, rigorous confusion-matrix evaluation, explainability, ONNX
                export with a numerical parity test, and a static, in-browser demo. The
                point is to show the whole pipeline done carefully — including being
                honest about its limits.
              </p>
            </div>
          </StorySection>

          <StorySection
            id="architecture"
            eyebrow="Architecture"
            title="From pixels to a predicted letter"
          >
            <div className="space-y-6">
              <p>
                The classifier is a MobileNetV2 backbone pretrained on ImageNet with its
                final classifier head replaced by a 26-way head. The ImageNet backbone is
                frozen for a short warm-up, then the whole network is fine-tuned
                end-to-end at a 10× lower learning rate. One deliberate choice in the
                augmentation pipeline:{" "}
                <strong className="text-fg">no horizontal flip</strong> — ASL signs are
                not flip-invariant, since pairs like b/d and p/q are mirror images of each
                other.
              </p>
              <ArchitectureDiagram />
            </div>
          </StorySection>

          <StorySection
            id="accuracy"
            eyebrow="Accuracy, honestly"
            title="Benchmark vs. real world"
          >
            <AccuracyStory />
          </StorySection>

          <StorySection
            id="explainability"
            eyebrow="Explainability"
            title="What is the model looking at?"
          >
            <GradcamExplainer />
          </StorySection>

          <StorySection
            id="inference"
            eyebrow="In-browser inference"
            title="How it runs without a server"
          >
            <div className="space-y-6">
              <p>
                There is no inference backend. The trained model is exported to ONNX and
                executed in your browser tab, so every prediction happens locally.
              </p>
              <InferenceExplainer />
            </div>
          </StorySection>

          <StorySection id="stack" eyebrow="Under the hood" title="Tech stack">
            <TechStack />
          </StorySection>

          <StorySection id="ethics" eyebrow="Honesty" title="Ethics & limitations">
            <div className="space-y-6">
              <p>
                A model is only as trustworthy as the caveats shipped with it. These are
                drawn directly from the project&rsquo;s model card.
              </p>
              <EthicsLimitations />
            </div>
          </StorySection>

          <StorySection id="links" eyebrow="More" title="Explore further">
            <div className="flex flex-wrap gap-3">
              <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
                <Button>
                  <Github className="h-4 w-4" aria-hidden="true" />
                  Source on GitHub
                </Button>
              </a>
              <Link href="/">
                <Button variant="outline">
                  <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                  Try the live demo
                </Button>
              </Link>
            </div>
          </StorySection>
        </div>
      </main>
      <Footer />
    </>
  );
}
