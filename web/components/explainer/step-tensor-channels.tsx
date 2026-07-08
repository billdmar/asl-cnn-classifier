"use client";

import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ChannelHeatmap } from "./channel-heatmap";
import { tensorToChannelViz } from "@/lib/explainer/tensor-viz";
import { IMAGE_SIZE } from "@/lib/preprocess";

interface StepTensorChannelsProps {
  tensor: Float32Array;
}

export function StepTensorChannels({ tensor }: StepTensorChannelsProps) {
  const channels = useMemo(() => tensorToChannelViz(tensor, IMAGE_SIZE), [tensor]);

  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-4 p-4">
        <Badge>ImageNet-normalized tensor channels</Badge>
        <div className="flex flex-col gap-4 sm:flex-row sm:gap-6">
          {channels.map((ch) => (
            <div key={ch.label} className="flex flex-col items-center gap-1">
              <ChannelHeatmap viz={ch} displaySize={160} />
              <span className="text-xs font-medium text-fg-muted">{ch.label}</span>
              <span className="text-[10px] text-fg-muted">
                [{ch.range[0].toFixed(2)}, {ch.range[1].toFixed(2)}]
              </span>
            </div>
          ))}
        </div>
        <p className="max-w-md text-center text-xs text-fg-muted">
          After ImageNet normalization &mdash; negative values (dark) represent below-mean intensity, positive (bright) above-mean.
        </p>
      </CardContent>
    </Card>
  );
}
