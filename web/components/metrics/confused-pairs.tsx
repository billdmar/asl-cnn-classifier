import { ArrowRight } from "lucide-react";

import { Reveal, RevealItem } from "@/components/ui/reveal";
import type { ConfusedPair } from "@/lib/metrics";

/**
 * Ranked list of the most-confused (true → predicted) class pairs. Data comes
 * from `metrics.json` → `most_confused_pairs`.
 */
export function ConfusedPairs({ pairs }: { pairs: ConfusedPair[] }) {
  return (
    <Reveal as="ol" stagger className="flex flex-col gap-2">
      {pairs.map((p, i) => (
        <RevealItem
          key={`${p.true}-${p.pred}`}
          as="li"
          className="flex items-center justify-between rounded-lg border border-border-subtle bg-bg-subtle px-3 py-2"
        >
          <span className="flex items-center gap-2 text-sm">
            <span className="w-5 text-right tabular-nums text-fg-subtle">{i + 1}.</span>
            <span className="font-semibold text-fg">{p.true}</span>
            <ArrowRight className="h-3.5 w-3.5 text-fg-subtle" aria-hidden="true" />
            <span className="font-semibold text-fg">{p.pred}</span>
          </span>
          <span className="text-sm tabular-nums text-fg-muted">
            {p.count} {p.count === 1 ? "image" : "images"}
          </span>
        </RevealItem>
      ))}
    </Reveal>
  );
}
