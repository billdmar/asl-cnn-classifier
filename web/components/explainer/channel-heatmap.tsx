"use client";

import { useEffect, useRef } from "react";
import type { ChannelViz } from "@/lib/explainer/tensor-viz";

interface ChannelHeatmapProps {
  viz: ChannelViz;
  displaySize?: number;
}

export function ChannelHeatmap({ viz, displaySize = 192 }: ChannelHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(viz.imageData, 0, 0);
  }, [viz]);

  return (
    <canvas
      ref={canvasRef}
      width={viz.imageData.width}
      height={viz.imageData.height}
      className="rounded-md border border-border-subtle"
      style={{
        width: displaySize,
        height: displaySize,
        imageRendering: "pixelated",
      }}
    />
  );
}
