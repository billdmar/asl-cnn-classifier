"use client";

/**
 * Step 3: Crop visualization.
 *
 * Shows the 128x128 crop at a larger display size (256x256 CSS) with
 * pixelated rendering so individual pixels are visible. Falls back to
 * the full-frame resize when no hand was detected.
 */

import { useMemo } from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

export interface StepCropProps {
  cropCanvas: HTMLCanvasElement | null;
  imageData: ImageData;
}

export function StepCrop({ cropCanvas, imageData }: StepCropProps) {
  // If we have a cropCanvas, use it directly; otherwise render imageData.
  const dataUrl = useMemo(() => {
    if (cropCanvas) {
      return cropCanvas.toDataURL("image/png");
    }
    // Render imageData to a temp canvas.
    const tmp = document.createElement("canvas");
    tmp.width = imageData.width;
    tmp.height = imageData.height;
    const ctx = tmp.getContext("2d");
    if (ctx) ctx.putImageData(imageData, 0, 0);
    return tmp.toDataURL("image/png");
  }, [cropCanvas, imageData]);

  const size = cropCanvas ? cropCanvas.width : imageData.width;

  return (
    <div className="mx-auto max-w-md">
      <Card>
        <CardContent className="flex flex-col items-center gap-3 p-4">
          <Badge variant="accent">
            {`${size} × ${size} px — what the model sees`}
          </Badge>
          <img
            src={dataUrl}
            alt={`${size}x${size} cropped input that the CNN processes`}
            className="rounded-lg border border-border-subtle"
            style={{
              width: "256px",
              height: "256px",
              imageRendering: "pixelated",
            }}
          />
          {!cropCanvas && (
            <p className="text-xs text-fg-muted">
              No hand was detected, so the full frame was resized instead.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
