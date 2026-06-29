"use client";

import { useEffect, useState } from "react";
import { useReducedMotion } from "framer-motion";
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

const ACCENT = "rgb(var(--chart-accent))";
const ACCENT_2 = "rgb(var(--chart-accent-2))";
const TEXT_MUTED = "rgb(var(--fg-muted))";

/** The honest cross-dataset accuracy at each training-data milestone. */
const TRAJECTORY = [
  { step: "1 source", label: "Marxulia", acc: 33.4 },
  { step: "+ aliciiavs", label: "multi-signer", acc: 47.6 },
  { step: "+ Hemg", label: "3 sources (deployed)", acc: 55.5 },
];

const SUMMARY =
  "Honest cross-dataset accuracy rose from 33.4% (single source) to 47.6% (adding a multi-signer dataset) to 55.5% (a third source) — every gain came from data diversity, not model tweaks.";

/**
 * Visual of the accuracy trajectory — the core engineering insight (diversity is
 * the lever) told as a chart rather than prose. Client-only + mounted-gated so
 * the recharts ResponsiveContainer doesn't run during static prerender.
 */
export function AccuracyTrajectory() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  // Let the bars grow in once on mount, but stay static under reduced motion.
  const reduce = useReducedMotion();

  return (
    <figure role="img" aria-label={SUMMARY} className="w-full">
      <div className="h-56 w-full">
        {mounted ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={TRAJECTORY}
              margin={{ top: 24, right: 8, bottom: 0, left: -8 }}
            >
              <XAxis
                dataKey="step"
                stroke={TEXT_MUTED}
                tick={{ fill: TEXT_MUTED, fontSize: 12 }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 100]}
                tickFormatter={(v: number) => `${v}%`}
                stroke={TEXT_MUTED}
                tick={{ fill: TEXT_MUTED, fontSize: 11 }}
                tickLine={false}
                width={40}
              />
              <Bar
                dataKey="acc"
                radius={[4, 4, 0, 0]}
                isAnimationActive={!reduce}
                animationDuration={400}
                animationEasing="ease-out"
              >
                {TRAJECTORY.map((d, i) => (
                  <Cell key={d.step} fill={i === TRAJECTORY.length - 1 ? ACCENT_2 : ACCENT} />
                ))}
                <LabelList
                  dataKey="acc"
                  position="top"
                  formatter={(v: number) => `${v}%`}
                  fill="rgb(var(--fg))"
                  fontSize={13}
                  fontWeight={700}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : null}
      </div>
      <figcaption className="mt-2 text-xs text-fg-subtle">
        Honest cross-dataset accuracy (26-class) at each training-data milestone.
        Every jump came from <strong className="text-fg-muted">adding a diverse
        dataset</strong> — preprocessing, augmentation, calibration, and
        architecture swaps were all measured and found not to help.
      </figcaption>
    </figure>
  );
}
