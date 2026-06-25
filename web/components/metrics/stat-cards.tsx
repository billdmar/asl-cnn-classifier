import { Card } from "@/components/ui/card";

/** A single headline statistic. Value is pre-formatted from fetched JSON. */
export interface Stat {
  label: string;
  value: string;
  /** Where this number came from, shown as a small caption (honesty). */
  source: string;
}

/** Headline statistic cards. Every value is passed in from fetched JSON. */
export function StatCards({ stats }: { stats: Stat[] }) {
  return (
    <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((s) => (
        <Card key={s.label} className="p-5">
          <dt className="text-sm font-medium text-fg-muted">{s.label}</dt>
          <dd className="mt-2 bg-accent-gradient bg-clip-text text-3xl font-bold tabular-nums text-transparent">
            {s.value}
          </dd>
          <p className="mt-2 text-xs text-fg-subtle">{s.source}</p>
        </Card>
      ))}
    </dl>
  );
}
