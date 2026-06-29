"use client";

import { useReducedMotion } from "framer-motion";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  YAxis,
} from "recharts";

import type { ConfidencePoint } from "@/components/webcam/smoothing";

const ACCENT = "#2dd4bf";

/**
 * Live sparkline of the smoothed top-class confidence over recent frames. Makes
 * temporal stability visible: a steady high line = confident hold; a jagged or
 * low line = the model is flickering / unsure. Driven by a capped ring buffer in
 * the webcam panel (no extra inference — it reuses the smoothed result).
 */
export function ConfidenceTimeseries({ points }: { points: ConfidencePoint[] }) {
  const reduceMotion = useReducedMotion();
  const latest = points[points.length - 1];
  const summary = latest
    ? `Top prediction ${latest.label || "—"} confidence is ${(latest.prob * 100).toFixed(0)}% and trending across the last ${points.length} frames.`
    : "Awaiting predictions.";

  return (
    <figure role="img" aria-label={summary} className="w-full">
      <div className="h-16 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
            <YAxis domain={[0, 1]} hide />
            <Area
              type="monotone"
              dataKey="prob"
              stroke={ACCENT}
              fill={ACCENT}
              fillOpacity={0.18}
              strokeWidth={2}
              // Subtle line draw-in / fade, gated on reduced-motion.
              isAnimationActive={!reduceMotion}
              animationDuration={300}
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-1 text-[11px] text-fg-subtle">
        Confidence of the top letter, last {points.length} frames
      </figcaption>
    </figure>
  );
}
