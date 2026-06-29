# HANDOFF — ASL CNN Classifier

_Last updated: 2026-06-26. Branch: `main` @ `b142e40` (clean, in sync; PRs #12/#13/#18 merged)._

## ⏭️ NEXT ROUND — planned, NOT yet executed (start here)
A two-track round was scoped (audits done) but not built. Resume by executing it.

**Track A — one honest, gated accuracy experiment (likely null; document either way).**
Accuracy is sourcing-bound, but three genuinely-UNTRIED zero-deploy-cost levers remain.
Bundle the two cheapest into ONE training run, gated:
- **SWA** (`torch.optim.swa_utils`: AveragedModel + SWALR for the last ~25% of epochs +
  `update_bn` over the train loader; save the averaged weights). Still one model → no
  deploy cost.
- **Label smoothing** — `nn.CrossEntropyLoss(label_smoothing=0.1)` (src/train.py criterion
  ~line 360, currently plain CE).
- Both gated behind config flags, **default-off = byte-identical** to today's training.
  Clone `configs/train_real_mobilenet_diverse_hemg.yaml` → `..._swa.yaml` with a SEPARATE
  `checkpoint_dir`. Train (~45min MPS), eval on the gate.
- **Decision rule:** ship ONLY if cross-dataset 26-class AND A-Y beat 55.5%/59.8% by ≥+2pt
  (beyond ±3.7 n=712 noise) with no strong-class regression. Else record a negative in a new
  `docs/EXPERIMENT_*.md` and keep the deployed model. (Ensemble was assessed and SKIPPED —
  2× web model size/latency fails the Lighthouse perf gate for ~+1pt. Mention, don't build.)

**Track B — aesthetic UI overhaul (the main event; parallelizable).** Site is lean but
visually plain. framer-motion 11.18 + recharts available; MUST keep reduced-motion respect,
transform/opacity-only (no layout thrash), SSR-safe (charts behind `mounted`), and
Lighthouse perf+a11y ≥0.90 (the #1 risk for an animation round). Streams with DISJOINT file
ownership:
- **S1 (barrier, do FIRST):** shared motion primitives — a scroll-reveal `useInView` wrapper
  + a count-up hook (both reduced-motion aware), in `web/components/ui/` or `web/lib/`; new
  tailwind keyframes (animated gradient/shimmer) in `tailwind.config.ts`/`globals.css`.
- Then **S2–S5 in parallel** (each owns its files, consumes S1 read-only):
  - S2 landing choreography — `web/app/page.tsx` section reveals/stagger + hero gradient.
  - S3 metrics motion — `web/components/metrics/*`: stat-card count-ups, recharts entrance
    (flip `isAnimationActive` on, gated by reduced-motion), confusion-heatmap cell fade,
    card stagger.
  - S4 live-demo polish (**do NOT touch inference logic**) — `web/components/webcam/*`:
    predicted-letter AnimatePresence exit, confidence-bars stagger, hand-detect glow,
    word-builder letter pop-in.
  - S5 upload micro-interactions — `web/components/upload/*`: dropzone hover/drag border,
    example-button hover scale+glow, result/Grad-CAM fade-in.
- **S6 (optional, last, abortable):** page transitions via AnimatePresence in layout — flag
  static-export/SSR + Lighthouse risk; drop if it regresses anything.
Taste bar: subtle, fast, premium — not busy/garish (this is a credibility piece).
Full audit detail is in the chat transcript of the session ending 2026-06-26; the file
ownership above is disjoint so streams won't collide.

**Ship:** branch `feat/aesthetic-overhaul-and-swa` → per-stream commits (all gates green) →
integration pass (E2E + lighthouse + axe) → user pushes (git push is agent-blocked) → I
open PR, watch CI, merge, `vercel --prod`. Track A runs in the background (different
lang/dir) while Track B builds — fully independent.

---

_Earlier state (still accurate below):_

## Goal
A portfolio-grade American Sign Language **alphabet (A–Z) image classifier** with an
**honest, measured** real-world accuracy story and a polished in-browser showcase.
Python ML pipeline (PyTorch → ONNX) + a Next.js static site running 100% client-side
inference (onnxruntime-web + MediaPipe), deployed on Vercel.

## Current state (all working unless noted)
- **Deployed model:** MobileNetV2, 26-class A–Z, trained on a **3-source union**
  (Marxulia + aliciiavs + Hemg). **Honest cross-dataset accuracy = 55.5% (A–Y headline
  59.8%)** on a held-out 4th dataset (EitanG98); same-dataset benchmark 96.9%. This is
  the ONLY number used to decide what ships.
- **Accuracy is sourcing-bound.** Diversity is the only lever that ever moved it
  (33.4 → 47.6 → 55.5). Preprocessing, augmentation, calibration, class-balancing, and
  arch swaps (mobilenet_v3_small, efficientnet_b0) were all measured and **rejected** —
  documented as honest negatives in `docs/EXPERIMENT_*.md`. The clean public supply of
  diverse A–Z datasets is exhausted (atalaydenknalbant rejected: 6.2% source-overlap with
  the eval gate).
- **Live site** (asl-cnn-classifier.vercel.app) is current as of `7691c19`. Shows the
  honest cross-dataset number, a confusion-matrix explorer, Grad-CAM on bundled examples,
  a live webcam confidence chart, and (this session) a fingerspelling word-builder +
  accuracy-trajectory chart + model-download progress.
- **Shipped THIS session (PR #13, merged + deployed):**
  - Fingerspelling **word-builder** (hold a letter ~1.5s to spell words; pure logic in
    `web/components/webcam/word-builder.ts`).
  - **Accuracy-trajectory chart** on /about (`web/components/story/accuracy-trajectory.tsx`).
  - **Model-download progress bar** (streamed fetch in `web/lib/inference.ts`).
  - **Error boundary** (`web/app/error.tsx`) + custom **404** (`web/app/not-found.tsx`).
  - **Model↔ONNX drift guard** (`tests/test_model_onnx_sync.py`).
  - **axe a11y E2E** (`web/tests-e2e/a11y.spec.ts`) — caught + fixed a real WCAG contrast
    bug (`fg-subtle` 3.72:1 → `#828292` 5.2:1 in `web/tailwind.config.ts`).
  - **dependabot.yml**, **SECURITY.md**, and an **inert** Vercel deploy workflow
    (`.github/workflows/deploy.yml`).
- **Tests green:** 238 Python + 73 web vitest + 10 Playwright E2E; ruff/black/mypy/tsc/
  eslint clean; ONNX↔PyTorch↔browser parity preserved.

## What's left (priority order)
1. **(Optional, user-only) Wire true auto-deploy:** add repo secrets `VERCEL_TOKEN`,
   `VERCEL_ORG_ID` (`team_6xaQFg6rP37GuNDynoQPVfyc`), `VERCEL_PROJECT_ID`
   (`prj_c8EyEmscWD8eIVJSqW1J7MqcrQy5`). The workflow is committed but inert until then;
   meanwhile deploys are manual `vercel --prod`. See `web/DEPLOY.md`.
2. **Accuracy is CLOSED on current assets** — see
   `docs/EXPERIMENT_supply_exhausted_closure.md`. An exhaustive dataset search found nothing
   clean+diverse+anonymous left (the one candidate, `NAM27/sign-language`, is the grassknoted
   single-signer overfit set), and the temporal J/Z path is unverifiable because the gate's
   J/Z are 60 static frames. Two explicit **unblock conditions** to reopen it: (a) a genuinely
   multi-signer A–Z dataset (or Kaggle creds), or (b) a held-out **video** benchmark that makes
   a temporal J/Z model measurable. Until one lands, 55.5% / A–Y 59.8% is the honest ceiling.
3. Lower-value, previously deferred (not rabbit holes, just low ROI): dark/light toggle,
   PWA manifest, service worker, per-class FastAPI confidence, OpenCV-path smoothing.

## Key decisions (don't re-litigate)
- **Cross-dataset gate is the only deploy decider**; never ship on the inflated
  same-dataset benchmark. Eval set (EitanG98) is NEVER trained/calibrated on.
- **A–Y is the headline metric** (J/Z are dynamic motion signs; mainstream convention).
- **T=1.0 calibration kept** — a same-dataset temperature fit sharpens wrongly.
- **Grad-CAM is precompute-only** (onnxruntime-web has no autograd) — examples only.
- **J/Z motion-tracking deliberately NOT built** (audit: unclear payoff, risks other letters).

## Gotchas / do-not-touch
- **`git push` is harness-blocked for the agent.** The user must run `git push` from a
  REAL terminal (the in-session `! cmd` prefix does NOT execute). The agent CAN run `gh`
  (PR/merge) and `vercel` (deploy) — only push is blocked.
- **The real deployed checkpoint is GITIGNORED** (`artifacts/checkpoints/best_model.pth`,
  ~9MB). It lives only locally + as the exported `web/public/model/model.onnx`. The real
  one is recoverable from `artifacts/checkpoints_diverse_hemg/best_model.pth`.
- **CI's `sample-train` step OVERWRITES `artifacts/checkpoints/best_model.pth`** with a
  throwaway 29-class custom_cnn model. Any test loading that path in CI gets the SAMPLE
  model, not the deployed one (this bit the drift guard — now it skips when class-count
  ≠ ONNX dim). After running sample-train locally, restore:
  `cp artifacts/checkpoints_diverse_hemg/best_model.pth artifacts/checkpoints/best_model.pth`
- **CI runs ruff AND black** — always `black --check src tests`, not just ruff, before
  committing. `make install-hooks` wires pre-commit to enforce this.
- **Vercel GitHub auto-deploy is NOT connected** — merging to main does NOT deploy.
  Deploy manually with `vercel --prod --yes` (root-directory=web is set in Vercel).
- **Web is `output: export`** (static) — `next start` fails; serve `out/`. Dynamic routes
  (opengraph-image) need `export const dynamic = "force-static"`.
- **Playwright local browser mismatch:** use `PW_CHANNEL=chrome npx playwright test` if the
  bundled headless build is missing; CI installs its own.

## Key files
- `src/train.py` — multi-dir union training; `src/eval_realworld.py` — the honest gate
  (+ confusion matrix, A–Y metric, decision-policy/TTA scaffolding).
- `src/download_hf_data.py` — dataset registry (`DATASETS` + `FILENAME_DATASETS` for
  Roboflow/YOLO-style); `scripts/check_eval_overlap.py` — the contamination guard.
- `web/components/webcam/{webcam-panel,word-builder,confidence-timeseries}.tsx` — live demo.
- `web/components/metrics/{metrics-dashboard,confusion-heatmap}.tsx` — dashboard.
- `web/lib/{inference,use-classifier,metrics}.ts` — in-browser inference + data.
- `tests/test_model_onnx_sync.py` — deployed-model drift guard.
- `docs/EXPERIMENT_*.md` — the honest experiment record (all negatives included).
- `web/DEPLOY.md` — deploy options + the secrets needed for auto-deploy.

## How to build / test / run
```bash
make install           # uv venv + dev deps + sample fixture
make install-hooks     # pre-commit (ruff+black+mypy, mirrors CI) — recommended

# Python gate (what CI runs)
.venv/bin/ruff check src tests && .venv/bin/black --check src tests && .venv/bin/mypy src
.venv/bin/python -m pytest -q --cov=src --cov-fail-under=80

# Web gate (cd web)
npm run typecheck && npm run lint && npm test && npm run build
PW_CHANNEL=chrome npx playwright test        # local; CI uses bundled chromium

# Reproduce the deployed model (~50 min MPS)
make download-real && make download-diverse && make download-hemg && \
  make check-overlap-hemg && make train-diverse-hemg && make eval-realworld-diverse-hemg

# Deploy (manual)
vercel --prod --yes                          # root-directory=web set in Vercel project
```

## Next action
Project is in a **clean, fully-shipped state** with accuracy **honestly closed** (see #2 +
`docs/EXPERIMENT_supply_exhausted_closure.md`). Nothing is pending or broken. If continuing:
the only remaining code-side item is the optional Vercel auto-deploy secrets (#1). Any
further **accuracy** work is gated on an external input (a new diverse dataset or a video
benchmark) — do not manufacture experiments; the levers on current assets are exhausted and
documented. Confirm the live site reflects the latest `main` before starting anything new.
