import { Cpu, ShieldCheck, Server, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { Reveal, RevealItem } from "@/components/ui/reveal";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ExplainerPoint {
  icon: LucideIcon;
  title: string;
  body: string;
}

const POINTS: ReadonlyArray<ExplainerPoint> = [
  {
    icon: Cpu,
    title: "Runs with onnxruntime-web",
    body: "The model is exported to ONNX and executed by onnxruntime-web, which runs the network via a WebAssembly (WASM) backend — with a WebGPU path where the browser supports it.",
  },
  {
    icon: ShieldCheck,
    title: "Frames never leave your device",
    body: "Webcam frames and uploaded images are preprocessed and classified entirely in the browser tab. No pixels are uploaded anywhere — the inference is fully client-side and private.",
  },
  {
    icon: Server,
    title: "No server, static deploy",
    body: "The whole site is a static export: plain HTML, JS, and the .onnx weights served as files. There is no inference backend to run, scale, or pay for.",
  },
  {
    icon: Zap,
    title: "Low latency, on-device",
    body: "Because there is no network round-trip, predictions update locally as fast as the device can run the forward pass — no API calls per frame.",
  },
];

export function InferenceExplainer() {
  return (
    <Reveal stagger className="grid gap-4 sm:grid-cols-2">
      {POINTS.map((point) => {
        const Icon = point.icon;
        return (
          <RevealItem key={point.title} className="h-full">
            <Card>
              <CardHeader className="flex-row items-center gap-3 space-y-0">
                <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-accent/30 bg-accent/10">
                  <Icon className="h-5 w-5 text-accent" aria-hidden="true" />
                </span>
                <CardTitle className="text-base">{point.title}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm">
                <p>{point.body}</p>
              </CardContent>
            </Card>
          </RevealItem>
        );
      })}
    </Reveal>
  );
}
