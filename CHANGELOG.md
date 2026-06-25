# Changelog

All notable changes to this project. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Accuracy figures are the
**honest cross-dataset** numbers (a dataset the model never trains on) unless
labelled "same-dataset".

## [Unreleased] — product-quality & explainability round

### Added
- **Confusion-matrix explorer** — the dense 26×26 cross-dataset confusion matrix
  rendered as a recall-normalized heatmap + top-confusions table, surfacing the
  honest failure map (closed-hand signs S/N/M/T collapse into each other).
- **Grad-CAM explainability** on the bundled examples — "what the model looked
  at" saliency overlays, pre-computed offline (in-browser inference can't expose
  gradients) and shown on click with an honest caption.
- **Live confidence time-series** on the webcam — a sparkline of the smoothed
  top-class confidence over recent frames, so temporal stability is visible.
- **OpenGraph share image** (generated via `next/og`) + social metadata, a
  skip-to-main link, immutable cache headers (`vercel.json`), an upload
  cropped-region preview + 10 MB file guard.
- **Pre-commit hooks** (`.pre-commit-config.yaml` + `make install-hooks`) mirroring
  the CI ruff/black/mypy gate; `CONTRIBUTING.md` and this `CHANGELOG.md`.

### Fixed
- The live site advertised stale single-source numbers (96.8% / 1,631 samples)
  while its own dashboard computed the deployed 3-source numbers (96.9% / 3,170)
  from JSON — corrected across hero/story/metadata; the dashboard provenance
  string is now derived from the data. CI lint (black) failures resolved.

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
  (EitanG98) with MediaPipe hand-crop — exposed the 96.8% same-dataset vs 33.4%
  cross-dataset gap.
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
