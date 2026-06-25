# ASL Classifier — in-browser showcase (`web/`)

A Next.js + TypeScript site that runs the project's **real** MobileNetV2 ASL
alphabet classifier **100% in the browser** via
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/). Webcam frames and
uploaded images never leave the device — there is no inference server.

This is the recruiter-facing front end for the ML pipeline in the repo root. The
model, metrics, and example images it serves are produced by reproducible code
in that pipeline; nothing here is hardcoded or fabricated.

## Stack

- **Next.js 15** (App Router, `output: "export"` → static site, deploys to Vercel)
- **TypeScript strict** (`noUncheckedIndexedAccess`, no `any`)
- **Tailwind CSS** + shadcn-style primitives + **Framer Motion** (dark "AI product" aesthetic)
- **onnxruntime-web** (WASM, WebGPU when available)
- **Vitest** + Testing Library (unit + cross-language parity) · **Playwright** (E2E)

## The single source of truth for preprocessing

The full preprocess pipeline (resize → ImageNet-normalize → CHW tensor) is
defined once in Python (`src/dataset.py::get_eval_transforms`) and mirrored in
`lib/preprocess.ts`. A **cross-language parity gate** guards against drift:

- **Strict parity** (`lib/__tests__/parity.strict.test.ts`): the SAME pre-resized
  128×128 pixels run through the TS normalize + ONNX path reproduce the Python
  pipeline's probabilities to **~5e-7** (asserted < 1e-3). This proves the
  arithmetic, tensor layout, and ONNX runtime are equivalent.
- **End-to-end parity** (`lib/__tests__/parity.e2e.test.ts`): the JS path also
  resizes the raw image; predicted class matches exactly, probabilities within a
  documented ~3e-2 tolerance (the only divergence is the resize kernel — PIL vs
  canvas/sharp bilinear — not a math bug).

Golden fixtures live in `test-fixtures/golden/` and are regenerated from the
trained checkpoint with `make export-onnx-web` (from the repo root). Run that
after any retrain so the live model and the parity gate both track the checkpoint.

## Commands

```bash
npm install
npm run dev            # http://localhost:3000
npm run typecheck      # tsc --noEmit (strict)
npm run lint           # next lint
npm test               # vitest: unit + parity
npm run test:e2e       # playwright (builds + serves the static export)
npm run build          # static export → out/
```

> Local Playwright note: if the bundled Chromium can't launch, run E2E against a
> system browser with `PW_CHANNEL=chrome npm run test:e2e`. CI uses the bundled
> Chromium on Linux.

## Honesty

Every displayed metric is produced by repo code on a real set and labeled with
its source (benchmark vs real-world). Headline numbers today: **96.8% held-out
test accuracy**, **97.8% validation** — both measured. Real-world webcam accuracy
is lower and is reported separately as it is measured.
