"use client";

/**
 * Reusable component that renders a single ChannelViz's ImageData to a canvas.
 *
 * Uses pixelated rendering at a configurable display size so individual
 * normalized pixel values are visible as a heatmap.
 */

import { useEffect, useRef } from "react";

import type { ChannelViz } from "@/lib/explainer/tensor-viz";

export interface ChannelHeatmapProps {
  viz: ChannelViz;
  displaySize?: number; // CSS pixels (default 192)
}

export function ChannelHeatmap({ viz, displaySize = 192 }: ChannelHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = viz.imageData.width;
    canvas.height = viz.imageData.height;
    ctx.putImageData(viz.imageData, 0, 0);
  }, [viz]);

  return (
    <canvas
      ref={canvasRef}
      className="rounded-lg"
      style={{
        width: `${displaySize}px`,
        height: `${displaySize}px`,
        imageRendering: "pixelated",
      }}
      aria-label={`${viz.label} heatmap visualization`}
    />
  );
}
