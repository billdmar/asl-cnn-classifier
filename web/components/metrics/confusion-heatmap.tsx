import { confusionCells, type RealworldEval } from "@/lib/metrics";

/**
 * Row-normalized confusion-matrix heatmap rendered as a CSS grid (no chart lib,
 * so it's light and screen-reader friendly). Color encodes recall (row
 * fraction): the diagonal should glow on a well-behaved class, off-diagonal
 * heat marks confusions. Data comes from `realworld_eval.json` — the HONEST
 * cross-dataset confusions (where the model actually struggles), not the
 * inflated same-dataset ones.
 */
export function ConfusionHeatmap({ data }: { data: RealworldEval }) {
  const labels = data.confusion_labels;
  const cells = confusionCells(data.confusion_matrix, labels);
  const n = labels.length;

  if (n === 0) return null;

  // Accessible summary: the worst off-diagonal confusion.
  const worst = data.most_confused_pairs[0];
  const summary = worst
    ? `Confusion matrix over ${n} classes. The most frequent confusion is true ${worst.true} predicted as ${worst.pred} (${worst.count} images).`
    : `Confusion matrix over ${n} classes.`;

  // grid: 1 corner + N column headers, then N rows of (header + N cells).
  return (
    <figure
      role="img"
      aria-label={summary}
      className="overflow-x-auto"
    >
      <div
        className="grid gap-px text-[10px]"
        style={{ gridTemplateColumns: `1.25rem repeat(${n}, 1fr)`, minWidth: `${n * 0.9 + 2}rem` }}
        aria-hidden="true"
      >
        <div />
        {labels.map((l) => (
          <div key={`col-${l}`} className="text-center font-medium text-fg-subtle">
            {l}
          </div>
        ))}
        {labels.map((rowLabel, i) => (
          <Row key={`row-${rowLabel}`} rowLabel={rowLabel} cells={cells.slice(i * n, i * n + n)} />
        ))}
      </div>
      <figcaption className="mt-3 text-xs text-fg-subtle">
        Row-normalized (recall): each row sums to 100%. Brighter = larger share of
        that letter&apos;s true examples landed in that column. The diagonal is correct;
        off-diagonal heat is a confusion. Measured on the held-out cross-dataset set.
      </figcaption>
    </figure>
  );
}

function Row({
  rowLabel,
  cells,
}: {
  rowLabel: string;
  cells: ReturnType<typeof confusionCells>;
}) {
  return (
    <>
      <div className="flex items-center justify-end pr-1 font-medium text-fg-subtle">
        {rowLabel}
      </div>
      {cells.map((c) => (
        <div
          key={`${c.trueLabel}-${c.predLabel}`}
          title={`true ${c.trueLabel} → predicted ${c.predLabel}: ${c.count} (${(c.fraction * 100).toFixed(0)}%)`}
          className="aspect-square rounded-[2px]"
          style={{
            backgroundColor: cellColor(c.fraction, c.isDiagonal),
          }}
        />
      ))}
    </>
  );
}

/** Map a 0–1 fraction to a color: teal for the diagonal, amber for confusions. */
function cellColor(fraction: number, isDiagonal: boolean): string {
  if (fraction <= 0) return "rgb(30 30 40)"; // empty cell, bg-subtle-ish
  const alpha = 0.15 + 0.85 * Math.min(1, fraction);
  // diagonal = accent teal (good); off-diagonal = amber (confusion).
  return isDiagonal
    ? `rgba(45, 212, 191, ${alpha})`
    : `rgba(251, 191, 36, ${alpha})`;
}
