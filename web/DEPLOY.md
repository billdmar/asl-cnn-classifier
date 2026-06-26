# Deploying the web showcase to Vercel

The site is a **static export** (`output: "export"` → `web/out/`). It runs all
inference in the browser, so there is no server runtime — any static host works,
but the repo is wired for Vercel.

## Live deployment

**Production:** <https://asl-cnn-classifier.vercel.app>

The Vercel project is configured with **Root Directory = `web`**, so Vercel
detects Next.js in `web/package.json` and builds natively — no `vercel.json` and
no `cd web` indirection. A repo-root `.vercelignore` keeps the dataset, venv, and
caches out of the upload.

### Option A — Vercel dashboard Git integration (auto-deploy on push)

1. Go to <https://vercel.com/new> and import `billdmar/asl-cnn-classifier`.
2. Set **Root Directory** to `web` (Settings → General → Root Directory). Vercel
   then auto-detects the **Next.js** preset.
3. With the Git integration connected, every push to `main` auto-deploys and other
   branches get preview URLs.

> **Note:** the repo is linked to the Vercel project via the CLI
> (`.vercel/project.json`), but the **dashboard Git integration is not connected**,
> so pushes do NOT auto-deploy today. Use Option B (manual) or Option C (CI action).

### Option C — GitHub Action (`.github/workflows/deploy.yml`)

A workflow deploys to production on push to `main`, **inert until you add three repo
secrets** (Settings → Secrets and variables → Actions):

| Secret | Value |
|---|---|
| `VERCEL_TOKEN` | a token from <https://vercel.com/account/tokens> |
| `VERCEL_ORG_ID` | `team_6xaQFg6rP37GuNDynoQPVfyc` |
| `VERCEL_PROJECT_ID` | `prj_c8EyEmscWD8eIVJSqW1J7MqcrQy5` |

Without `VERCEL_TOKEN` the job logs a notice and skips (no failure). Once set,
merges to `main` deploy automatically.

### Option B — Vercel CLI (manual)

```bash
# From the repo root (the project is already linked). Requires `vercel login`.
vercel deploy --prod --yes --archive=tgz   # --archive avoids the free-tier
                                           # per-file upload rate limit
```

> The CLI needs your authenticated Vercel account. The project's Root Directory
> is `web`; if you re-link a fresh project, set it (see Option A step 2) or the
> build fails with "No Next.js version detected."

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
