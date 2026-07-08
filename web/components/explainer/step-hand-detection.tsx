"use client";

/**
 * Step 2: Hand detection visualization.
 *
 * Renders the frame with a colored rectangle overlay showing the crop box.
 * Uses a <canvas> element: draws the frame image, then strokes the crop box
 * in accent color.
 */

import { useEffect, useRef } from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { CropBox } from "@/lib/handcrop";

export interface StepHandDetectionProps {
  frameCanvas: HTMLCanvasElement;
  cropBox: CropBox | null;
  handFound: boolean;
}

export function StepHandDetection({
  frameCanvas,
  cropBox,
  handFound,
}: StepHandDetectionProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = frameCanvas.width;
    const h = frameCanvas.height;
    canvas.width = w;
    canvas.height = h;

    // Draw the frame.
    ctx.drawImage(frameCanvas, 0, 0, w, h);

    // Draw the crop box overlay if a hand was found.
    if (cropBox) {
      const bx = cropBox.x * w;
      const by = cropBox.y * h;
      const bw = cropBox.width * w;
      const bh = cropBox.height * h;

      ctx.strokeStyle = "#7c5cff";
      ctx.lineWidth = Math.max(2, Math.round(w / 150));
      ctx.setLineDash([Math.round(w / 30), Math.round(w / 50)]);
      ctx.strokeRect(bx, by, bw, bh);

      // Corner accent marks.
      ctx.setLineDash([]);
      ctx.lineWidth = Math.max(3, Math.round(w / 100));
      const corner = Math.round(bw * 0.08);

      // Top-left
      ctx.beginPath();
      ctx.moveTo(bx, by + corner);
      ctx.lineTo(bx, by);
      ctx.lineTo(bx + corner, by);
      ctx.stroke();
      // Top-right
      ctx.beginPath();
      ctx.moveTo(bx + bw - corner, by);
      ctx.lineTo(bx + bw, by);
      ctx.lineTo(bx + bw, by + corner);
      ctx.stroke();
      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(bx, by + bh - corner);
      ctx.lineTo(bx, by + bh);
      ctx.lineTo(bx + corner, by + bh);
      ctx.stroke();
      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(bx + bw - corner, by + bh);
      ctx.lineTo(bx + bw, by + bh);
      ctx.lineTo(bx + bw, by + bh - corner);
      ctx.stroke();
    }
  }, [frameCanvas, cropBox]);

  return (
    <div className="mx-auto max-w-md">
      <Card>
        <CardContent className="flex flex-col gap-3 p-4">
          <Badge variant={handFound ? "accent" : "default"}>
            {handFound
              ? "Hand detected — crop region shown"
              : "No hand detected — using full frame"}
          </Badge>
          <canvas
            ref={canvasRef}
            className="w-full rounded-lg border border-border-subtle"
            aria-label="Frame with hand detection overlay"
          />
        </CardContent>
      </Card>
    </div>
  );
}
