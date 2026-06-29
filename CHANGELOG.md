# Changelog

All notable changes to this project. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Accuracy figures are the
**honest cross-dataset** numbers (a dataset the model never trains on) unless
labelled "same-dataset".

## [Unreleased]

_Nothing yet._

## [1.0.0] — 2026-06-29

Headline accuracy is **55.5% cross-dataset (26-class)** / **59.8% A–Y**; the
same-dataset benchmark of **96.9% (3,170 held-out samples)** appears only as a
labelled leakage-inflated contrast, never as a headline.

### Added

- **Installable + offline PWA** — typed web manifest (standalone, 192/512/maskable
  icons generated from the brand SVG via `scripts/gen-icons.mjs`, apple-icon) plus a
  hand-written service worker (`public/sw.js`): network-first HTML with offline-shell
  fallback, stale-while-revalidate for `_next/static`, cache-first for MediaPipe. It
  explicitly bypasses `/model/*` (IndexedDB owns it), `/metrics/*`, and the
  cross-origin ORT CDN, and is SHA-keyed. Prod-only registration via a hostname guard
  (not `NODE_ENV`) so Playwright/Lighthouse run SW-free and the "no-refetch" inference
  test stays honest. Zero new deps (icons via transitively-present sharp).
- **SEO completeness** — `sitemap.ts`, `robots.ts`, and a schema.org `WebApplication`
  JSON-LD carrying the honest 59.8% / 96.9% numbers (no fabricated `aggregateRating`).
- **Dark/light theme toggle** — a sun/moon header toggle persisted to localStorage with
  a no-FOUC pre-paint script (SSR-safe), migrating 11 hardcoded-hex Tailwind tokens to
  CSS variables (`rgb(var(--x) / <alpha-value>)`). Dark remains the default and is
  byte-identical to before. Ships a WCAG-AA-verified light palette (accent darkened
  `#7c5cff → #5b3df5` because the brand purple fails AA on white) and theme-aware
  charts/heatmap.
- **IndexedDB model caching** — the ~9 MB `model.onnx` is downloaded once and cached
  (keyed by build SHA, with old-version eviction) across both the webcam and upload
  paths; best-effort with a fetch fallback, SSR-safe, plus a 12 s slow-warmup hint and
  a Retry control.
- **Shareable result permalinks** — a prediction (letter + top-5 + timestamp) encoded
  into a base64url URL hash, decoded client-side on a new `/result` page with a
  `ShareButton` (`navigator.share` + clipboard fallback). Honest static-export caveat:
  the OG preview is one generic static card (per-result images need a server this deploy
  lacks); the exact letter still renders on load.
- **Keyboard shortcuts + help** — Space = camera, C = copy word, R = reset, S = share,
  ? = help (native `<dialog>`), and a deploy-freshness footer baking the commit SHA +
  build date into the static export.
- **Confusion-matrix explorer** — the dense 26×26 cross-dataset confusion matrix
  rendered as a recall-normalized heatmap + top-confusions table, surfacing the honest
  failure map (closed-hand signs S/N/M/T collapse into each other).
- **Grad-CAM explainability** on the bundled examples — "what the model looked at"
  saliency overlays, pre-computed offline (in-browser inference can't expose gradients)
  and shown on click with an honest caption.
- **Live confidence time-series** on the webcam — a sparkline of the smoothed top-class
  confidence over recent frames, so temporal stability is visible.
- **OpenGraph share image** (generated via `next/og`) + social metadata, a skip-to-main
  link, immutable cache headers (`vercel.json`), an upload cropped-region preview + a
  10 MB file guard.
- **Reproducibility & contributor tooling** — `make reproduce-deployed` chains the six
  existing targets (3× download + gate, overlap check, train, eval) so the deployed
  checkpoint reproduces in one command; pre-commit hooks (`.pre-commit-config.yaml` +
  `make install-hooks`) mirror the CI ruff/black/mypy gate; added `CONTRIBUTING.md` and
  this `CHANGELOG.md`.

### Changed

- **Animated UI overhaul** — scroll reveals, stat count-ups (initialized at real
  values), and micro-interactions (predicted-letter scaleIn, confident-letter glow, tap
  feedback, sparkline draw-in) built on new shared motion primitives (`lib/motion.ts`,
  `components/ui/reveal.tsx`, `lib/use-count-up.ts`). Fully reduced-motion-respecting;
  reveals are transform-only (translateY, no opacity) to stay AA-contrast/LCP/axe-safe;
  interactive panels are intentionally not reveal-wrapped to preserve click
  actionability.
- **Recruiter-polish README** — leads with the live in-browser ML web app (live-demo
  link first + a real deployed-app screenshot `docs/web-hero.png`), an honest
  55.5% / 59.8% headline (96.9% only as the labelled leakage contrast), and a new
  "honest-accuracy story" section with a metrics-dashboard screenshot. Updated badges
  (added Live-demo / Next.js / TypeScript / ONNX-Runtime-web, dropped legacy OpenCV),
  reframed the legacy CNN/OpenCV work as a baseline (29×29 → 26×26), moved the legacy
  Gradio demo down, and updated `web/README.md` to the 55.5% / 59.8% honesty headline.
  Removed the misleading old Gradio screenshot (an untrained 2.86% checkpoint) and the
  unreferenced `docs/demo.png`.
- **Honest MODEL_CARD** — `MODEL_CARD.md` now states the deployed truth (MobileNetV2
  26-class 3-source, honest cross-dataset 55.5% / A–Y 59.8% as the only deploy metric;
  same-dataset 96.9% labelled as leakage-inflated), sourced from the metric JSON files,
  removing the stale "≥98% target, not yet reproduced" 29-class framing.
- **CI Playwright cache** — caches `~/.cache/ms-playwright` keyed by the resolved
  Playwright version (install-deps only on miss), saving ~2 min per web job; plus
  Dependabot GitHub-Actions bumps (setup-uv 5→7, checkout 4→7, setup-node 4→6).
- Repointed two dangling "Reproducing 98%" README anchors to the deployed-model section
  and dropped the stale "≥98% (29-class Kaggle) aspirational" row.

### Fixed

- The live site advertised stale single-source numbers (96.8% / 1,631 samples) while
  its own dashboard computed the deployed 3-source numbers (96.9% / 3,170) from JSON —
  corrected across hero/story/metadata; the dashboard provenance string is now derived
  from the data. CI lint (black) failures resolved.

### Notes — honest negatives & investigation closure

- **Accuracy investigation closed** — `docs/EXPERIMENT_supply_exhausted_closure.md`
  records the 33.4% → 47.6% → 55.5% trajectory and documents that the one remaining
  large dataset candidate (NAM27, 121k) is the same single-signer grassknoted overfit
  set already rejected, and that the temporal J/Z path is deferred (the honest gate's
  J/Z are 60 static frames with no motion to measure). Two explicit unblock conditions
  are recorded: a genuinely multi-signer dataset, or a held-out video gate.
- **Gated accuracy experiment was negative and NOT shipped** — config-gated SWA +
  label-smoothing (0.1) training levers were added to `src/train.py` (both default-off;
  the training path is byte-identical) with a separate config/checkpoint dir, but the
  experiment **regressed** the honest cross-dataset gate (−1.8 pt 26-class / −2.0 pt A–Y
  vs the deployed 55.5% / 59.8%), failing the required +2 pt-on-both bar. The deployed
  model, ONNX, fixtures, and web metrics are **unchanged**; only the default-off wiring
  is kept so the negative is reproducible.

---

The sections below are the pre-1.0 journey, preserved as faithful historical records.
The accuracy figures in each are the values measured **at that time** (e.g. the 96.8% /
1,631-sample same-dataset eval predates the current 96.9% / 3,170-sample benchmark).

## [3-source] — diverse multi-source training

- Added a 3rd diverse training source (Hemg, incl. J/Z) → **47.6% → 55.5%**
  (A–Y headline 59.8%). 23/26 classes improved.
- Surfaced the honest cross-dataset number on the live site (it had been measured
  but never displayed).
- Flexible filename-label dataset loader (`snapshot_download` + regex) for
  Roboflow/YOLO-style datasets.
- Rejected on evidence: class-balanced loss (neutral), mobilenet_v3_small (−20pt),
  efficientnet_b0 (−1pt) — architecture is not the lever; diversity is.

## [diverse] — first diversity win

- Added a 2nd, genuinely diverse training dataset (aliciiavs) → **33.4% → 47.6%**
  (+14.2 pts). Established that data diversity is the only lever that moves the
  honest number.
- Perceptual-hash eval-overlap guard (`check_eval_overlap.py`) to prevent training
  data from leaking into the cross-dataset gate.

## [honest-baseline] — measuring real-world accuracy

- Cross-dataset evaluation harness (`eval_realworld.py`) on a held-out dataset
  (EitanG98) with MediaPipe hand-crop — exposed the same-dataset (96.8% at the time,
  1,631 samples) vs 33.4% cross-dataset gap.
- Near-duplicate-aware dataset split, sketch-contamination filter, MediaPipe
  Python hand-crop, real ECE calibration.
- Proven neutral and documented as negatives: crop-consistency, augmentation
  (medium/heavy), per-class thresholds, TTA, temperature calibration.

## [showcase] — in-browser product

- Next.js showcase: live webcam + image-upload classification running 100% in the
  browser via onnxruntime-web; metrics dashboard; model card.
- Cross-language parity gate (Python ↔ ONNX ↔ browser preprocessing) at ~5e-7
  (strict) / 3e-2 (end-to-end) tolerances.
- Deployed to Vercel; MobileNetV2 transfer model on the real 26-class dataset.

[1.0.0]: https://github.com/billdmar/asl-cnn-classifier/releases/tag/v1.0.0
