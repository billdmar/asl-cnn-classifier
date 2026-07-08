"use client";

/**
 * Step 4: Tensor channel visualization.
 *
 * Renders three canvases side-by-side (stacked on mobile) showing the R, G, B
 * normalized channels after ImageNet normalization.
 */

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { ChannelHeatmap } from "@/components/explainer/channel-heatmap";
import type { ChannelViz } from "@/lib/explainer/tensor-viz";

export interface StepTensorChannelsProps {
  channels: [ChannelViz, ChannelViz, ChannelViz];
}

export function StepTensorChannels({ channels }: StepTensorChannelsProps) {
  return (
    <Card>
      <CardContent className="flex flex-col gap-4 p-4">
        <div className="flex flex-col gap-4 md:flex-row md:justify-center md:gap-6">
          {channels.map((ch) => (
            <div key={ch.label} className="flex flex-col items-center gap-2">
              <Badge variant="accent">{ch.label}</Badge>
              <ChannelHeatmap viz={ch} displaySize={192} />
              <span className="font-mono text-xs text-fg-muted">
                Range: [{ch.range[0].toFixed(2)}, {ch.range[1].toFixed(2)}]
              </span>
            </div>
          ))}
        </div>
        <p className="text-center text-xs text-fg-muted">
          After ImageNet normalization — negative values (dark) represent
          below-mean intensity
        </p>
      </CardContent>
    </Card>
  );
}
