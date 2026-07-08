"use client";

import { useCallback, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, ReferenceLine, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { TemperatureSlider } from "./temperature-slider";
import { CLASS_NAMES } from "@/lib/labels";

interface StepPredictionProps {
  logits: Float32Array;
  initialProbs: Float32Array;
}

interface BarDatum { label: string; prob: number; }

function probsToData(probs: Float32Array): BarDatum[] {
  return Array.from(probs, (prob, index) => ({
    label: CLASS_NAMES[index] ?? String(index), prob,
  })).sort((a, b) => b.prob - a.prob);
}

const UNSURE_THRESHOLD = 0.6;

export function StepPrediction({ logits, initialProbs }: StepPredictionProps) {
  const [data, setData] = useState<BarDatum[]>(() => probsToData(initialProbs));

  const handleChange = useCallback((probs: Float32Array) => {
    setData(probsToData(probs));
  }, []);

  const topLabel = data[0]?.label ?? "";

  return (
    <Card>
      <CardContent className="flex flex-col gap-4 p-4">
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fill: "var(--color-fg-muted)" }}
                axisLine={{ stroke: "var(--color-border)" }}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "var(--color-fg-muted)" }}
                axisLine={false} tickLine={false}
                domain={[0, 1]}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "Probability"]}
                contentStyle={{ backgroundColor: "var(--color-bg-card)", border: "1px solid var(--color-border)", borderRadius: "8px", fontSize: "12px" }}
              />
              <ReferenceLine
                y={UNSURE_THRESHOLD}
                stroke="#fbbf24"
                strokeDasharray="4 4"
                label={{ value: "Confidence threshold", position: "insideTopRight", fill: "var(--color-fg-muted)", fontSize: 10 }}
              />
              <Bar dataKey="prob" radius={[2, 2, 0, 0]}>
                {data.map((entry) => (
                  <Cell
                    key={entry.label}
                    fill={entry.label === topLabel ? "var(--color-accent, #7c5cff)" : "rgba(124, 92, 255, 0.3)"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <TemperatureSlider logits={logits} onChange={handleChange} />
      </CardContent>
    </Card>
  );
}
