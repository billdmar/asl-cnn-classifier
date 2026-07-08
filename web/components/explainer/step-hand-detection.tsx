"use client";

import { useEffect, useRef } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { FrozenFrame, HandDetection } from "@/lib/explainer/pipeline-steps";

interface StepHandDetectionProps {
  frame: FrozenFrame;
  handDetection: HandDetection;
}

export function StepHandDetection({ frame, handDetection }: StepHandDetectionProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const scale = Math.min(400 / frame.width, 300 / frame.height, 1);
    canvas.width = Math.round(frame.width * scale);
    canvas.height = Math.round(frame.height * scale);
    ctx.drawImage(frame.source, 0, 0, canvas.width, canvas.height);
    if (handDetection.found && handDetection.box) {
      const box = handDetection.box;
      ctx.strokeStyle = "#7c5cff";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(
        box.x * canvas.width,
        box.y * canvas.height,
        box.width * canvas.width,
        box.height * canvas.height,
      );
    }
  }, [frame, handDetection]);

  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 p-4">
        <Badge variant={handDetection.found ? "accent" : "default"}>
          {handDetection.found ? "Hand detected — crop region shown" : "No hand detected — using full frame"}
        </Badge>
        <canvas ref={canvasRef} className="max-w-full rounded-md border border-border-subtle" />
      </CardContent>
    </Card>
  );
}
