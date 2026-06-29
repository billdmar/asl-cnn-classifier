"use client";

import { useReducedMotion } from "framer-motion";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";

import type { TrainingHistory } from "@/lib/metrics";

const ACCENT = "rgb(var(--chart-accent))";
const ACCENT_2 = "rgb(var(--chart-accent-2))";
const GRID = "rgb(var(--chart-grid))";
const TEXT_MUTED = "rgb(var(--fg-muted))";

interface CurvePoint {
  epoch: number;
  trainAcc: number;
  valAcc: number;
}

function TrainingTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || payload === undefined || payload.length === 0) return null;
  return (
    <div className="rounded-lg border border-border bg-bg-card px-3 py-2 text-xs shadow-lg">
      <p className="mb-1 text-sm font-semibold text-fg">Epoch {String(label)}</p>
      {payload.map((p) => (
        <p key={p.name} className="text-fg-muted">
          {p.name}:{" "}
          <span className="tabular-nums text-fg">
            {typeof p.value === "number" ? `${(p.value * 100).toFixed(1)}%` : "—"}
          </span>
        </p>
      ))}
    </div>
  );
}

/**
 * Train vs. validation accuracy over epochs. Data comes from
 * `training_history.json` (`train_acc` / `val_acc` per epoch).
 */
export function TrainingChart({ history }: { history: TrainingHistory }) {
  const reduceMotion = useReducedMotion() ?? false;
  const data: CurvePoint[] = history.map((e) => ({
    epoch: e.epoch,
    trainAcc: e.train_acc,
    valAcc: e.val_acc,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="epoch"
          stroke={TEXT_MUTED}
          tick={{ fill: TEXT_MUTED, fontSize: 11 }}
          tickLine={false}
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          stroke={TEXT_MUTED}
          tick={{ fill: TEXT_MUTED, fontSize: 11 }}
          tickLine={false}
          width={48}
        />
        <Tooltip content={<TrainingTooltip />} />
        <Legend wrapperStyle={{ fontSize: 12, color: TEXT_MUTED }} iconType="plainline" />
        <Line
          type="monotone"
          dataKey="trainAcc"
          name="Train accuracy"
          stroke={ACCENT}
          strokeWidth={2}
          dot={false}
          isAnimationActive={!reduceMotion}
        />
        <Line
          type="monotone"
          dataKey="valAcc"
          name="Validation accuracy"
          stroke={ACCENT_2}
          strokeWidth={2}
          dot={false}
          isAnimationActive={!reduceMotion}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
