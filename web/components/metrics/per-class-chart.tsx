"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";

import type { ClassRow } from "@/lib/metrics";

const ACCENT = "#7c5cff";
const ACCENT_LOW = "#2dd4bf";
const GRID = "#23232f";
const TEXT_MUTED = "#a0a0b0";

/** Letters with the very best F1 get the teal accent; the rest the purple. */
const HIGH_F1 = 0.99;

function PerClassTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || payload === undefined || payload.length === 0) return null;
  const row = payload[0]?.payload as ClassRow | undefined;
  if (row === undefined) return null;
  return (
    <div className="rounded-lg border border-border bg-bg-card px-3 py-2 text-xs shadow-lg">
      <p className="mb-1 text-sm font-semibold text-fg">Class {row.letter}</p>
      <p className="text-fg-muted">
        F1: <span className="tabular-nums text-fg">{(row.f1 * 100).toFixed(1)}%</span>
      </p>
      <p className="text-fg-muted">
        Precision:{" "}
        <span className="tabular-nums text-fg">{(row.precision * 100).toFixed(1)}%</span>
      </p>
      <p className="text-fg-muted">
        Recall:{" "}
        <span className="tabular-nums text-fg">{(row.recall * 100).toFixed(1)}%</span>
      </p>
      <p className="text-fg-muted">
        Support: <span className="tabular-nums text-fg">{row.support}</span>
      </p>
    </div>
  );
}

/**
 * Per-class F1 bar chart across all 26 letters. Data comes from
 * `metrics.json` → `per_class`; `rows` is built by `toClassRows`.
 */
export function PerClassChart({ rows }: { rows: ClassRow[] }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="letter"
          stroke={TEXT_MUTED}
          tick={{ fill: TEXT_MUTED, fontSize: 11 }}
          tickLine={false}
          interval={0}
        />
        <YAxis
          domain={[0.85, 1]}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          stroke={TEXT_MUTED}
          tick={{ fill: TEXT_MUTED, fontSize: 11 }}
          tickLine={false}
          width={48}
        />
        <Tooltip
          cursor={{ fill: "rgba(124,92,255,0.08)" }}
          content={<PerClassTooltip />}
        />
        <Bar dataKey="f1" radius={[3, 3, 0, 0]} isAnimationActive={false}>
          {rows.map((r) => (
            <Cell key={r.letter} fill={r.f1 >= HIGH_F1 ? ACCENT_LOW : ACCENT} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
