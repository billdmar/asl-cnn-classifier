"use client";

/**
 * Step 1: Raw frame display.
 *
 * Renders the frozen frame canvas as an <img> via canvas.toDataURL().
 * Shows dimensions badge and wraps in a Card.
 */

import { useMemo } from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

export interface StepRawFrameProps {
  frameCanvas: HTMLCanvasElement;
}

export function StepRawFrame({ frameCanvas }: StepRawFrameProps) {
  const dataUrl = useMemo(() => frameCanvas.toDataURL("image/png"), [frameCanvas]);
  const width = frameCanvas.width;
  const height = frameCanvas.height;

  return (
    <div className="mx-auto max-w-md">
      <Card>
        <CardContent className="flex flex-col gap-3 p-4">
          <div className="flex items-center gap-2">
            <Badge variant="accent">Raw frame</Badge>
            <Badge>{`${width} × ${height}`}</Badge>
          </div>
          <img
            src={dataUrl}
            alt="Frozen camera frame before processing"
            className="w-full rounded-lg border border-border-subtle"
          />
        </CardContent>
      </Card>
    </div>
  );
}
