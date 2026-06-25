"use client";

import { useEffect, useState } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  bestValEpoch,
  fetchCalibration,
  fetchMetrics,
  fetchRealworldEval,
  fetchTrainingHistory,
  topConfusedPairs,
  toClassRows,
  type CalibrationData,
  type Metrics,
  type RealworldEval,
  type TrainingHistory,
} from "@/lib/metrics";

import { ConfusedPairs } from "./confused-pairs";
import { PerClassChart } from "./per-class-chart";
import { ReliabilityChart } from "./reliability-chart";
import { StatCards, type Stat } from "./stat-cards";
import { TrainingChart } from "./training-chart";

const SOURCE = "measured on the held-out test set (1,631 images)";

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | {
      status: "ready";
      metrics: Metrics;
      history: TrainingHistory;
      calibration: CalibrationData;
      realworld: RealworldEval;
      gate2: RealworldEval | null;
    };

/** Visually-hidden text for screen readers (mirrors Tailwind's sr-only). */
function SrOnly({ children }: { children: React.ReactNode }) {
  return <span className="sr-only">{children}</span>;
}

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

/**
 * Interactive metrics dashboard. Fetches the committed evaluation JSON at
 * runtime and renders headline stats, a per-class F1 bar chart, training
 * curves, and the most-confused class pairs. Charts (recharts) render only
 * after mount so static prerender (`next build`) stays SSR-safe.
 */
export function MetricsDashboard() {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  // recharts' ResponsiveContainer needs the DOM; gate it behind mount so the
  // server-prerendered HTML doesn't try to measure a zero-size container.
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    let cancelled = false;
    Promise.all([
      fetchMetrics(),
      fetchTrainingHistory(),
      fetchCalibration(),
      fetchRealworldEval(),
      // The 2nd gate is optional — tolerate it being absent.
      fetchRealworldEval("realworld_eval_gate2").catch(() => null),
    ])
      .then(([metrics, history, calibration, realworld, gate2]) => {
        if (!cancelled)
          setState({
            status: "ready",
            metrics,
            history,
            calibration,
            realworld,
            gate2,
          });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setState({
            status: "error",
            message: err instanceof Error ? err.message : "Unknown error",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (state.status === "loading") {
    return (
      <div
        className="flex min-h-[20rem] items-center justify-center text-fg-muted"
        role="status"
      >
        Loading measured metrics…
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <Card className="p-6">
        <p className="text-sm text-fg-muted" role="alert">
          Couldn&apos;t load the metrics data: {state.message}
        </p>
      </Card>
    );
  }

  const { metrics, history, calibration, realworld, gate2 } = state;
  const best = bestValEpoch(history);
  const classRows = toClassRows(metrics.per_class);
  const pairs = topConfusedPairs(metrics.most_confused_pairs, 10);

  const stats: Stat[] = [
    {
      label: "Test accuracy",
      value: pct(metrics.overall_accuracy),
      source: SOURCE,
    },
    {
      label: "Macro F1",
      value: pct(metrics.macro_f1),
      source: "unweighted mean over 26 classes",
    },
    {
      label: "Test samples",
      value: metrics.num_test_samples.toLocaleString(),
      source: "held-out evaluation images",
    },
    {
      label: "Best val accuracy",
      value: best !== undefined ? pct(best.val_acc) : "—",
      source: best !== undefined ? `peak during training (epoch ${best.epoch})` : "—",
    },
  ];

  // Worst/best classes for the chart's text summary (no hardcoded values).
  const byF1 = toClassRows(metrics.per_class, "f1");
  const worst = byF1[0];
  const bestClass = byF1[byF1.length - 1];

  return (
    <div className="flex flex-col gap-8">
      <Card>
        <CardHeader>
          <CardTitle>Cross-dataset generalization — the honest number</CardTitle>
          <p className="text-sm text-fg-muted">
            Measured on a <strong>different</strong> dataset the model never trained
            on (different signers, real backgrounds). This is how it performs on a
            stranger&apos;s hand — far below the {pct(metrics.overall_accuracy)}{" "}
            same-dataset benchmark, and the only number that reflects real-world use.
          </p>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div>
              <dt className="text-xs text-fg-subtle">Accuracy (A–Y headline)</dt>
              <dd className="text-2xl font-bold tabular-nums text-fg">
                {pct(realworld.accuracy_ay)}
              </dd>
              <dd className="text-xs text-fg-subtle">
                24 static letters (J, Z are dynamic motion signs)
              </dd>
            </div>
            <div>
              <dt className="text-xs text-fg-subtle">Accuracy (all 26)</dt>
              <dd className="text-2xl font-bold tabular-nums text-fg">
                {pct(realworld.accuracy)}
              </dd>
              <dd className="text-xs text-fg-subtle">
                macro-F1 {realworld.macro_f1.toFixed(3)}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-fg-subtle">Eval images</dt>
              <dd className="text-2xl font-bold tabular-nums text-fg">
                {realworld.num_samples.toLocaleString()}
              </dd>
              <dd className="text-xs text-fg-subtle">held-out, never trained on</dd>
            </div>
            <div>
              <dt className="text-xs text-fg-subtle">Trajectory</dt>
              <dd className="text-2xl font-bold tabular-nums text-fg">
                33 → 47 → 55
              </dd>
              <dd className="text-xs text-fg-subtle">
                each step added a diverse dataset
              </dd>
            </div>
          </dl>
          {gate2 ? (
            <p className="mt-4 text-xs text-fg-subtle">
              Second (directional) gate: {pct(gate2.accuracy_ay)} A–Y on{" "}
              {gate2.num_samples.toLocaleString()} images — small and imbalanced, a
              sanity check rather than a precise measure.
            </p>
          ) : null}
          <p className="mt-3 text-xs text-fg-subtle">{realworld.note}</p>
        </CardContent>
      </Card>

      <StatCards stats={stats} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Per-class F1 score</CardTitle>
            <p className="text-sm text-fg-muted">
              All 26 letters, A–Z. Hover a bar for precision, recall, F1, and support —{" "}
              {SOURCE}.
            </p>
          </CardHeader>
          <CardContent>
            {mounted ? <PerClassChart rows={classRows} /> : null}
            <SrOnly>
              Bar chart of per-class F1 scores across all 26 letters, measured on the
              held-out test set.
              {worst !== undefined && bestClass !== undefined
                ? ` Lowest F1 is class ${worst.letter} at ${pct(worst.f1)}; ${bestClass.letter} reaches ${pct(bestClass.f1)}.`
                : ""}
            </SrOnly>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Training curves</CardTitle>
            <p className="text-sm text-fg-muted">
              Train vs. validation accuracy over {history.length} epochs.
            </p>
          </CardHeader>
          <CardContent>
            {mounted ? <TrainingChart history={history} /> : null}
            <SrOnly>
              Line chart of training and validation accuracy across {history.length}{" "}
              epochs.
              {best !== undefined
                ? ` Validation accuracy peaks at ${pct(best.val_acc)} on epoch ${best.epoch}.`
                : ""}
            </SrOnly>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Most-confused pairs</CardTitle>
            <p className="text-sm text-fg-muted">
              Where the model most often substitutes one letter for another — {SOURCE}.
            </p>
          </CardHeader>
          <CardContent>
            <ConfusedPairs pairs={pairs} />
            <SrOnly>
              Ranked list of the most-confused true-to-predicted class pairs on the
              held-out test set.
              {pairs[0] !== undefined
                ? ` The top pair is true ${pairs[0].true} predicted as ${pairs[0].pred}, ${pairs[0].count} times.`
                : ""}
            </SrOnly>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Calibration &amp; reliability</CardTitle>
            <p className="text-sm text-fg-muted">
              Per-bin accuracy vs. confidence against the perfect-calibration diagonal —{" "}
              {SOURCE}.
            </p>
          </CardHeader>
          <CardContent>
            <dl className="mb-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
              <div>
                <dt className="text-xs text-fg-subtle">ECE</dt>
                <dd className="font-semibold tabular-nums text-fg">
                  {calibration.ece.toFixed(3)}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-fg-subtle">Test images</dt>
                <dd className="font-semibold tabular-nums text-fg">
                  {calibration.num_test_samples.toLocaleString()}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-fg-subtle">Accuracy</dt>
                <dd className="font-semibold tabular-nums text-fg">
                  {pct(calibration.accuracy)}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-fg-subtle">Mean confidence</dt>
                <dd className="font-semibold tabular-nums text-fg">
                  {pct(calibration.mean_confidence)}
                </dd>
              </div>
            </dl>
            {mounted ? <ReliabilityChart calibration={calibration} /> : null}
            <SrOnly>
              Reliability diagram measured on the held-out test set of{" "}
              {calibration.num_test_samples.toLocaleString()} images. Expected calibration
              error is {calibration.ece.toFixed(3)}. Aggregate mean confidence (
              {pct(calibration.mean_confidence)}) is slightly below overall accuracy (
              {pct(calibration.accuracy)}), so the model is mildly under-confident on
              average; the diagram shows where per-bin confidence and accuracy diverge.
            </SrOnly>
          </CardContent>
        </Card>
      </div>

      <p className="text-xs text-fg-subtle">
        All figures above are {SOURCE}. Checkpoint:{" "}
        <code className="text-fg-muted">{metrics.checkpoint}</code>.
      </p>
    </div>
  );
}
