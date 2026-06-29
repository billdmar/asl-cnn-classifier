# HANDOFF — ASL CNN Classifier

_Last updated: 2026-06-29. Branch: `feat/web-product-features` (off `main` @ `34e5932`);
4 commits, all gates green locally, awaiting user push + PR._

## ✅ LATEST ROUND — dark/light theme toggle, DONE
Branch `feat/theme-toggle` (off `main` @ `2e1a021`); 3 commits, all gates green locally,
awaiting user push + PR. The previously-twice-deferred high-risk feature, now safe to do.

- **CSS-var migration**: 11 hardcoded-hex Tailwind tokens → `rgb(var(--x)/<alpha-value>)`;
  dark on `:root` (byte-identical default), contrast-verified light on `[data-theme=light]`
  in globals.css. Token-based components auto-themed (no edits). Gradients special-cased
  (can't use the alpha slot → literal alpha).
- **Light accent darkened** `#7c5cff`→`#5b3df5` (brand purple fails AA on white ~3.6:1; new
  is 6.12:1). Audit computed every fg/bg pair ≥AA before writing.
- **No-FOUC** inline script in layout.tsx sets `data-theme` pre-paint (default dark);
  `suppressHydrationWarning` on `<html>`. Toggle (`components/theme-toggle.tsx`) seeds from
  the live attribute (not localStorage) → no hydration mismatch.
- **Non-token colors** (recharts, heatmap) themed via `rgb(var(--chart-*))` in SVG props;
  heatmap uses `color-mix` over themed card bg. Canvas ROI stroke + OG/favicon left fixed
  (intentional — drawn over video / standalone branded assets).
- **Dual-theme a11y gate**: `a11y.spec.ts` now loops dark+light, waits 500ms for
  `transition-colors` to settle (WCAG applies to settled state), runs axe per theme. Caught
  + fixed a real light-only `link-in-text-block` (footer SHA link → underline).

Gates: tsc, lint, **127 unit** (+6 theme), build, **18 e2e** (a11y green in BOTH themes),
Lighthouse perf/a11y ≥0.90 on `/` and `/result`. Lesson: a programmatic `data-theme` swap
samples mid-`transition-colors` (axe saw a transient blended 1.02:1) — wait for settle, or
real toggles fade fine. Theme work is the last substantial unbuilt feature; the product is
now feature-complete pending only the deploy.

---

## Earlier round — web product features, MERGED as PR #20
Four static-export web features (IndexedDB model cache, share permalinks, keyboard shortcuts,
freshness footer), built by parallel agent teams (disjoint file ownership), all gates green —
**merged as PR #20** (squash `2e1a021`).

- **IndexedDB model cache + slow-network resilience** (Stream A) — ~9 MB model cached keyed by
  build SHA; repeat visits zero-refetch; 12 s slow-load hint + retry. Both webcam AND upload
  paths cache. Best-effort/SSR-safe. Doc: `docs/EXPERIMENT_model_caching.md`.
- **Shareable result permalinks** (Stream C) — base64url hash → `/result` renders the letter
  client-side; ShareButton (navigator.share→clipboard). Static OG is ONE generic card (honest:
  per-result preview needs a server we lack).
- **Keyboard shortcuts + help dialog** (Stream B) — Space/C/R/S/?; native `<dialog>` (axe-clean).
- **Deploy-freshness footer** (Stream D) — commit SHA + build date baked at build time.

Gates: tsc strict, eslint, **121 unit tests** (was 73; +48), static build (now exports `/result`),
**18 Playwright e2e** (incl. cache no-refetch, dialog-open axe, /result axe), Lighthouse median
**home perf 0.99 / a11y 0.90, result perf 0.95 / a11y 0.98** (all ≥0.90), on system Chrome.
Two integration fixes worth remembering: (1) caching had to be lifted OUT of the `onProgress`
gate so the upload path (no progress callback) also caches — else the reload e2e refetched;
(2) `text-accent` on `bg-bg-card` is 4.24:1 (under AA) — `/result` links use `text-fg` + accent
underline instead. Accuracy remains CLOSED — this round adds zero accuracy by design.

Lighthouse-on-`/result` gotcha: static export emits `out/result.html` (not `result/index.html`),
so `lighthouserc.json` lists `http://localhost/result.html`.

---

## Earlier round — product-overhaul, MERGED as PR #19 (both tracks)
Branch `feat/product-overhaul-round`. The two-track round below was executed; nothing is
left to build. **Next action: user pushes the branch, then I open the PR (CI is green
locally) → review → merge → `vercel --prod`.**

**Track A — gated accuracy experiment: NEGATIVE, not shipped.** SWA + label-smoothing 0.1
(`configs/train_real_mobilenet_diverse_hemg_swa.yaml`, both config-gated default-off in
`src/train.py`) trained + evaluated on the cross-dataset gate → **53.65% 26-class / 57.82%
A-Y**, i.e. **−1.8 / −2.0 pt vs the deployed 55.5%/59.8%** (label smoothing flattens the
already-thin class margins under shift; SWA basin-averaging compounds it). The +2pt-on-both
bar was not met, so the **deployed model + ONNX + fixtures + web metrics are UNCHANGED**.
Documented in `docs/EXPERIMENT_swa_label_smoothing.md`. The gated wiring is kept (default-off,
byte-identical) so it's reproducible and reusable if a more-diverse dataset reopens accuracy.

**Track B — animated UI overhaul: SHIPPED.** S0 motion primitives (`web/lib/motion.ts`,
`web/components/ui/reveal.tsx`, `web/lib/use-count-up.ts` + shimmer/gradient-pan keyframes)
consumed across landing / metrics / live-demo / upload / about. Subtle/fast/premium,
reduced-motion respected, transform/opacity-only. **All gates green:** tsc strict, eslint,
73 unit/parity, static build, 10/10 Playwright e2e (smoke+inference+axe a11y on / and
/about), Lighthouse median **perf 0.93 / a11y 1.0 / best-practices 1.0 / seo 1.0** (all ≥0.90).

**3 integration-pass lessons (root-caused, not papered over) — heed these for any future
motion work:**
1. **Reveal entrance is TRANSFORM-ONLY (translateY, no opacity).** An opacity fade leaves
   text at sub-AA contrast mid-animation, which **axe catches at page load on /about** (it
   scans before scroll). Transform-only = same `rise-up` discipline as the hero; axe + LCP safe.
2. **Don't wrap interactive panels in a scroll-reveal.** A reveal transform on the
   upload/webcam controls keeps their bounding box unstable, which **stalls Playwright
   click-actionability** (broke the example-classify e2e — `bbox.y` flips from undefined to
   settled). Reveal only the heading; the panels still enter via `LazyVisible`.
3. **framer `whileHover` + `AnimatePresence mode="wait"` interfere with automated clicks.**
   Use CSS hover/active (`motion-safe:hover:-translate-y-0.5`) for interactive controls;
   drop `mode="wait"` on fixed-height panels (no CLS benefit, real interaction cost).

**Local-env gotcha:** this sandbox can't fetch Playwright's pinned `chromium_headless_shell`
build (CDN download silently no-ops). Ran e2e + Lighthouse against **system Google Chrome**
via `channel: "chrome"` / `CHROME_PATH`. CI installs the shell normally and will pass.

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
