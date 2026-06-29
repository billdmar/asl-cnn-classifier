"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Card } from "@/components/ui/card";
import { Reveal, RevealItem } from "@/components/ui/reveal";
import { hoverLift } from "@/lib/motion";
import { useCountUp } from "@/lib/use-count-up";

/** A single headline statistic. Value is pre-formatted from fetched JSON. */
export interface Stat {
  label: string;
  value: string;
  /** Where this number came from, shown as a small caption (honesty). */
  source: string;
}

/**
 * Parse a pre-formatted stat string into the numeric part to animate plus a
 * function that re-renders that number in the original format. Preserves the
 * exact formatting (thousands separators, decimal places, trailing `%`). If the
 * value has no leading number (e.g. the em-dash placeholder), `num` is null and
 * the string is rendered verbatim.
 */
function parseStat(value: string): {
  num: number | null;
  format: (n: number) => string;
} {
  // Leading number: optional digits with optional thousands commas + decimals.
  const match = value.match(/^(\d[\d,]*)(\.(\d+))?/);
  if (match === null) {
    return { num: null, format: () => value };
  }
  const matched = match[0];
  const intPart = match[1] ?? "";
  const fracPart = match[3];
  const suffix = value.slice(matched.length); // e.g. "%" or ""
  const usesGrouping = intPart.includes(",");
  const decimals = fracPart?.length ?? 0;
  const num = Number(matched.replace(/,/g, ""));

  return {
    num,
    format: (n: number) =>
      `${n.toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
        useGrouping: usesGrouping,
      })}${suffix}`,
  };
}

/** A single stat card with its headline number animated by `useCountUp`. */
function StatCard({ stat, reduceMotion }: { stat: Stat; reduceMotion: boolean }) {
  const { num, format } = parseStat(stat.value);
  const animated = useCountUp({ to: num ?? 0, reduceMotion });
  const display = num === null ? stat.value : format(animated);

  return (
    <RevealItem>
      <motion.div whileHover={reduceMotion ? undefined : hoverLift}>
        <Card className="p-5">
          <dt className="text-sm font-medium text-fg-muted">{stat.label}</dt>
          <dd className="mt-2 bg-accent-gradient bg-clip-text text-3xl font-bold tabular-nums text-transparent">
            {display}
          </dd>
          <p className="mt-2 text-xs text-fg-subtle">{stat.source}</p>
        </Card>
      </motion.div>
    </RevealItem>
  );
}

/** Headline statistic cards. Every value is passed in from fetched JSON. */
export function StatCards({ stats }: { stats: Stat[] }) {
  const reduceMotion = useReducedMotion() ?? false;
  return (
    <Reveal
      as="dl"
      stagger
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
    >
      {stats.map((s) => (
        <StatCard key={s.label} stat={s} reduceMotion={reduceMotion} />
      ))}
    </Reveal>
  );
}
