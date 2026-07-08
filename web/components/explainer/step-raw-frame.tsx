"use client";

import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { FrozenFrame } from "@/lib/explainer/pipeline-steps";

interface StepRawFrameProps {
  frame: FrozenFrame;
}

export function StepRawFrame({ frame }: StepRawFrameProps) {
  const dataUrl = useMemo(() => frame.source.toDataURL("image/png"), [frame]);

  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 p-4">
        <Badge>Raw frame</Badge>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={dataUrl}
          alt="Captured raw frame"
          className="max-h-64 max-w-full rounded-md border border-border-subtle object-contain"
        />
        <p className="text-xs text-fg-muted">{frame.width} &times; {frame.height} px</p>
      </CardContent>
    </Card>
  );
}
