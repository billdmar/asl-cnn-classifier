# Deploying the web showcase to Vercel

The site is a **static export** (`output: "export"` → `web/out/`). It runs all
inference in the browser, so there is no server runtime — any static host works,
but the repo is wired for Vercel.

## One-time setup

The repo root `vercel.json` already points Vercel at the `web/` subdirectory:

```json
{
  "buildCommand": "cd web && npm run build",
  "outputDirectory": "web/out",
  "installCommand": "cd web && npm install",
  "framework": "nextjs"
}
```

### Option A — Vercel dashboard (recommended, auto-deploys on push)

1. Go to <https://vercel.com/new> and import `billdmar/asl-cnn-classifier`.
2. Leave the framework preset as **Next.js**; Vercel reads `vercel.json` for the
   build/output settings (root directory stays the repo root).
3. Deploy. Every push to `main` then auto-deploys; pushes to other branches get
   preview URLs.

### Option B — Vercel CLI (manual)

```bash
# From the repo root. Requires `vercel login` first.
vercel            # creates a preview deployment, prints a URL
vercel --prod     # promotes to production
```

> The CLI needs your authenticated Vercel account — that's a human step. The
> agent does not deploy or hold your Vercel credentials.

## What gets served

- `index.html` + `_next/` static chunks (lazy-loaded — onnxruntime/recharts load
  on demand).
- `model/model.onnx` — the real 26-class MobileNetV2 (~9 MB), committed as a
  deployment asset.
- `mediapipe/` — the hand-landmarker model + WASM (hand detection).
- `metrics/` — the real measured JSON the dashboard renders.
- `examples/`, `gradcam/` — sample and explainability images.

## Verify a deployment

After deploy, the live site should:

- load the dark landing page with the hero and four sections,
- classify the bundled **A** example to **A** (real in-browser inference),
- render the metrics dashboard with 96.8% / 97.8% / 1,631 and the ECE 0.046
  reliability diagram,
- show the `/about` model card.

The CI `web` job gates merges on: strict TypeScript, ESLint, unit + cross-language
parity tests, the static build, Playwright E2E (including a real A-example
inference assertion), and a Lighthouse budget (performance + accessibility ≥ 90).

## Note on the legacy Hugging Face Space

The Gradio Space (`app.py`, repo root) is independent and still works; it is the
optional legacy backend. The website does **not** depend on it. Redeploying the
Space needs the Hugging Face token — that remains a human-only step
(`make deploy-hf`, see `docs/DEPLOY.md`).
