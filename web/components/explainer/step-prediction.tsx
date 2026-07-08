"use client";

/**
 * Step 5: Prediction visualization.
 *
 * Shows a recharts BarChart with all 26 class probabilities, colored by rank,
 * with a reference line at the UNSURE_THRESHOLD. Includes the temperature
 * slider for interactive exploration.
 */

import { useCallback, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ReferenceLine,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

import { Card, CardContent } from "@/components/ui/card";
import { TemperatureSlider } from "@/components/explainer/temperature-slider";
import { UNSURE_THRESHOLD } from "@/lib/confidence";
import { CLASS_NAMES } from "@/lib/labels";

export interface StepPredictionProps {
  logits: Float32Array;
  initialProbs: Float32Array;
}

interface BarDatum {
  label: string;
  prob: number;
  index: number;
}

function probsToData(probs: Float32Array): BarDatum[] {
  return Array.from(probs, (prob, index) => ({
    label: CLASS_NAMES[index] ?? String(index),
    prob,
    index,
  })).sort((a, b) => b.prob - a.prob);
}

export function StepPrediction({ logits, initialProbs }: StepPredictionProps) {
  const [data, setData] = useState<BarDatum[]>(() => probsToData(initialProbs));

  const handleTemperatureChange = useCallback(
    (probs: Float32Array, _t: number) => {
      setData(probsToData(probs));
    },
    [],
  );

  const topLabel = data[0]?.label ?? "";

  return (
    <Card>
      <CardContent className="flex flex-col gap-4 p-4">
        {/* Bar chart */}
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              margin={{ top: 8, right: 8, bottom: 4, left: 0 }}
            >
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fill: "var(--color-fg-muted)" }}
                axisLine={{ stroke: "var(--color-border)" }}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "var(--color-fg-muted)" }}
                axisLine={false}
                tickLine={false}
                domain={[0, 1]}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                formatter={(value: number) => [
                  `${(value * 100).toFixed(1)}%`,
                  "Probability",
                ]}
                contentStyle={{
                  backgroundColor: "var(--color-bg-card)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
                labelStyle={{ color: "var(--color-fg)" }}
              />
              <ReferenceLine
                y={UNSURE_THRESHOLD}
                stroke="var(--color-amber-400, #fbbf24)"
                strokeDasharray="4 4"
                label={{
                  value: "Confidence threshold",
                  position: "insideTopRight",
                  fill: "var(--color-fg-muted)",
                  fontSize: 10,
                }}
              />
              <Bar dataKey="prob" radius={[2, 2, 0, 0]}>
                {data.map((entry) => (
                  <Cell
                    key={entry.label}
                    fill={
                      entry.label === topLabel
                        ? "var(--color-accent, #7c5cff)"
                        : "var(--color-accent-muted, rgba(124, 92, 255, 0.3))"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Temperature slider */}
        <TemperatureSlider logits={logits} onChange={handleTemperatureChange} />
      </CardContent>
    </Card>
  );
}
