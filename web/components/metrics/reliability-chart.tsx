"use client";

import { useReducedMotion } from "framer-motion";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";

import { toReliabilityRows, type CalibrationData } from "@/lib/metrics";

const ACCENT = "#7c5cff";
const DIAGONAL = "#6a6a7a";
const GRID = "#23232f";
const TEXT_MUTED = "#a0a0b0";

interface ChartPoint {
  /** Bin midpoint, shared x for both the model curve and the diagonal. */
  x: number;
  /** Empirical accuracy in this bin (model reliability curve). */
  acc: number;
  /** y = x perfect-calibration reference at this midpoint. */
  diagonal: number;
  lower: number;
  upper: number;
  conf: number;
  count: number;
}

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function ReliabilityTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || payload === undefined || payload.length === 0) return null;
  const point = payload[0]?.payload as ChartPoint | undefined;
  if (point === undefined) return null;
  return (
    <div className="rounded-lg border border-border bg-bg-card px-3 py-2 text-xs shadow-lg">
      <p className="mb-1 text-sm font-semibold text-fg">
        Confidence bin {pct(point.lower)}–{pct(point.upper)}
      </p>
      <p className="text-fg-muted">
        Accuracy: <span className="tabular-nums text-fg">{pct(point.acc)}</span>
      </p>
      <p className="text-fg-muted">
        Mean confidence: <span className="tabular-nums text-fg">{pct(point.conf)}</span>
      </p>
      <p className="text-fg-muted">
        Samples:{" "}
        <span className="tabular-nums text-fg">{point.count.toLocaleString()}</span>
      </p>
    </div>
  );
}

/**
 * Reliability diagram for the held-out test set. Plots per-bin empirical
 * accuracy against the bin's confidence midpoint (model curve, {@link ACCENT})
 * versus the y = x perfect-calibration diagonal ({@link DIAGONAL}, dashed).
 * Only bins with samples (`bin_count > 0`) are drawn. Data is REAL — computed on
 * the held-out real test split with the trained MobileNetV2 checkpoint.
 */
export function ReliabilityChart({ calibration }: { calibration: CalibrationData }) {
  const reduceMotion = useReducedMotion() ?? false;
  const rows = toReliabilityRows(calibration.bins);
  const data: ChartPoint[] = rows.map((r) => ({
    x: r.midpoint,
    acc: r.acc,
    diagonal: r.midpoint,
    lower: r.lower,
    upper: r.upper,
    conf: r.conf,
    count: r.count,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -16 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis
          dataKey="x"
          type="number"
          domain={[0, 1]}
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          stroke={TEXT_MUTED}
          tick={{ fill: TEXT_MUTED, fontSize: 11 }}
          tickLine={false}
          label={{
            value: "Confidence",
            position: "insideBottom",
            offset: -2,
            fill: TEXT_MUTED,
            fontSize: 11,
          }}
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          stroke={TEXT_MUTED}
          tick={{ fill: TEXT_MUTED, fontSize: 11 }}
          tickLine={false}
          width={48}
        />
        <Tooltip content={<ReliabilityTooltip />} />
        <Line
          type="linear"
          dataKey="diagonal"
          name="Perfect calibration"
          stroke={DIAGONAL}
          strokeWidth={1.5}
          strokeDasharray="4 4"
          dot={false}
          isAnimationActive={!reduceMotion}
        />
        <Line
          type="monotone"
          dataKey="acc"
          name="Model accuracy"
          stroke={ACCENT}
          strokeWidth={2}
          dot={{ r: 3, fill: ACCENT }}
          isAnimationActive={!reduceMotion}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
