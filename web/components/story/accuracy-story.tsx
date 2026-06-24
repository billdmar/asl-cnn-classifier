import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface BenchmarkMetric {
  label: string;
  value: string;
  note: string;
}

const BENCHMARK_METRICS: ReadonlyArray<BenchmarkMetric> = [
  {
    label: "Held-out test accuracy",
    value: "96.8%",
    note: "26 classes A–Z, 1,631 test images",
  },
  {
    label: "Macro F1",
    value: "0.968",
    note: "averaged equally across all 26 classes",
  },
  {
    label: "Validation accuracy",
    value: "97.8%",
    note: "best epoch during fine-tuning",
  },
];

export function AccuracyStory() {
  return (
    <div className="space-y-6">
      <p>
        Two numbers, told honestly. The headline figures below are{" "}
        <strong className="text-fg">benchmark</strong> numbers, measured on the held-out
        test split of the{" "}
        <code className="rounded bg-bg-subtle px-1 py-0.5 text-sm">
          Marxulia/asl_sign_languages_alphabets_v03
        </code>{" "}
        dataset (~10.9k images, 26 classes A–Z). They are real and reproducible with{" "}
        <code className="rounded bg-bg-subtle px-1 py-0.5 text-sm">make eval-real</code> —
        but they describe performance on a relatively uniform dataset, not the messy real
        world.
      </p>

      <div className="grid gap-4 sm:grid-cols-3">
        {BENCHMARK_METRICS.map((metric) => (
          <Card key={metric.label}>
            <CardHeader className="pb-2">
              <Badge variant="accent" className="w-fit">
                Benchmark
              </Badge>
              <CardTitle className="mt-2 text-3xl">{metric.value}</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-sm font-medium text-fg">{metric.label}</p>
              <p className="mt-1 text-xs text-fg-subtle">{metric.note}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="border-accent/30">
        <CardHeader>
          <div className="flex flex-wrap items-center gap-3">
            <CardTitle>Real-world webcam accuracy</CardTitle>
            <Badge>Measurement in progress</Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <p>
            Real-world accuracy is <strong className="text-fg">lower</strong> than the
            benchmark, and it varies with lighting, skin tone, background clutter, and
            camera angle. We deliberately do not show a single headline real-world number
            yet, because we do not have a trustworthy measured one — a robustness
            workstream is producing it. When it lands, it will appear here as a measured
            value, not an estimate.
          </p>
          <p>
            The benchmark dataset is also relatively homogeneous, so a random train/test
            split risks placing near-duplicate frames on both sides — a leakage-style
            optimism. A group-aware split (by signer or session) would likely lower the
            headline number. We report the benchmark with this caveat rather than
            presenting it as real-world capability.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
