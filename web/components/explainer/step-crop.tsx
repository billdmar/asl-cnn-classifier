"use client";

import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { IMAGE_SIZE } from "@/lib/preprocess";

interface StepCropProps {
  cropCanvas: HTMLCanvasElement | null;
  imageData: ImageData;
}

export function StepCrop({ cropCanvas, imageData }: StepCropProps) {
  const dataUrl = useMemo(() => {
    if (cropCanvas) return cropCanvas.toDataURL("image/png");
    const c = document.createElement("canvas");
    c.width = imageData.width;
    c.height = imageData.height;
    c.getContext("2d")?.putImageData(imageData, 0, 0);
    return c.toDataURL("image/png");
  }, [cropCanvas, imageData]);

  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 p-4">
        <Badge variant="accent">{IMAGE_SIZE} &times; {IMAGE_SIZE} px &mdash; what the model sees</Badge>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={dataUrl}
          alt="Cropped 128x128 input to the model"
          className="rounded-md border border-border-subtle"
          style={{ width: 256, height: 256, imageRendering: "pixelated" }}
        />
      </CardContent>
    </Card>
  );
}
