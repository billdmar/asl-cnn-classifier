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
            <CardTitle>The honest cross-dataset number</CardTitle>
            <Badge variant="accent">Measured</Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <p>
            The benchmark above is inflated: train and test images come from the same
            uniform dataset, so the test set looks like the training set. The number
            that actually matters is{" "}
            <strong className="text-fg">cross-dataset</strong> accuracy — evaluated on a{" "}
            <em>different</em> dataset (<code className="rounded bg-bg-subtle px-1 py-0.5 text-sm">EitanG98/asl_letters</code>,
            different signers and real backgrounds) the model never trained on. It is{" "}
            <strong className="text-fg">59.8% on the 24-letter A–Y headline</strong>{" "}
            (J and Z are dynamic motion signs a single frame can&apos;t capture), or{" "}
            <strong className="text-fg">55.5% across all 26 classes</strong>. Reproduce
            with{" "}
            <code className="rounded bg-bg-subtle px-1 py-0.5 text-sm">
              make eval-realworld-diverse-hemg
            </code>
            .
          </p>
          <p>
            That number was <strong className="text-fg">earned by data diversity</strong>,
            the only lever that moved it: single-source{" "}
            <strong className="text-fg">33.4%</strong> → add a multi-signer dataset{" "}
            <strong className="text-fg">47.6%</strong> → add a third source{" "}
            <strong className="text-fg">55.5%</strong>. Preprocessing tricks, augmentation,
            calibration, and architecture swaps were all measured and found <em>not</em> to
            help — documented as honest negatives in the repo&apos;s{" "}
            <code className="rounded bg-bg-subtle px-1 py-0.5 text-sm">docs/</code>.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            A robustness experiment, reported honestly
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <p>
            We retrained the same MobileNetV2 with{" "}
            <strong className="text-fg">aggressive domain augmentation</strong> (wide
            crops/rotations, strong lighting and contrast jitter, random grayscale, blur,
            and erasing) aimed at the cluttered-webcam case — reproducible with{" "}
            <code className="rounded bg-bg-subtle px-1 py-0.5 text-xs">
              make train-robust
            </code>
            .
          </p>
          <p>
            Measured on the <strong className="text-fg">same</strong> held-out test split,
            the robust model scored <strong className="text-fg">92.3%</strong> versus the
            baseline&apos;s 96.8% — heavy augmentation cost ~4.5 points on the benchmark.
            Once we had a real cross-dataset test set, augmentation proved neutral there
            too: it was <strong className="text-fg">data diversity</strong>, not
            augmentation or architecture, that moved the honest number. The deployed model
            is now the diverse multi-source one — chosen because it measured better on the
            cross-dataset gate, not because we hoped it was better.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
