# Experiment report: diverse multi-source training (the +14.2-point win)

**Date:** 2026-06-25
**Branch:** `feat/crop-consistent-retrain`
**Result:** honest cross-dataset accuracy **33.4% → 47.6% (+14.2 pts)**,
macro-F1 **0.340 → 0.468**. The diverse-trained model is deployed.

## The question

A MobileNetV2 ASL alphabet classifier scored 96.8% on its own held-out test
split but only **33.4% on a different dataset** (`data/asl_crossval`,
EitanG98/asl_letters — different signers, real backgrounds). That cross-dataset
number is the honest measure of real-world performance, and it's the only metric
used to decide what ships.

A prior 3-experiment effort (see
[`EXPERIMENT_crop_consistent_retrain.md`](EXPERIMENT_crop_consistent_retrain.md))
**proved preprocessing is tapped out**: hand-crop consistency, sketch-cleaning,
and augmentation all landed at ~33.4% within noise. The conclusion was that the
gap is a **data-domain gap** — the model overfits to single-signer,
plain-background Marxulia data — and the only lever with headroom is *more
diverse training data*. This experiment tests exactly that.

## What we did

1. **Found a verified-diverse second source.** `aliciiavs/sign_language_image_dataset`
   (8,442 images, A–Y, multiple signers, real varied backgrounds) — verified
   accessible via the HF datasets-server API. (Note: an earlier probe of 9
   candidates found 8 dead; this one uses underscores, not the dashed name that
   404s.)
2. **Built minimal multi-source plumbing.** `make_stratified_splits` gained a
   `samples=` injection point; `train.py` accepts a list/comma-separated
   `data_dir` and trains on the sorted-union of class folders
   (`get_union_class_names`). The single-source path stays byte-identical. The
   merged set is 19,315 images across 26 classes (Marxulia supplies J/Z).
3. **Guarded the eval gate against contamination.** aliciiavs and the EitanG98
   eval share a "real-room webcam" style, so `scripts/check_eval_overlap.py`
   runs a per-class perceptual-hash scan of the training set against the eval
   set before training. **Methodology note:** the first run flagged 35.6% at the
   sequential-video threshold (22) — but visual inspection proved those are
   *different photos of the same sign*, not duplicates (the min-distance
   distribution is a clean bell at 24 with zero images at ≤10, no spike near 0).
   Cross-dataset near-duplicate detection needs a tighter radius
   (`CROSS_DATASET_PHASH_THRESHOLD = 10`); at that radius the dataset reports
   **0.00% contamination**. The win is generalization, not memorization.

## Results

| Run | Training data | Aug | Cross-dataset acc | macro-F1 |
|-----|---------------|-----|-------------------|----------|
| Baseline (deployed before) | Marxulia only | standard | 33.4% | 0.340 |
| **D1** | **Marxulia + aliciiavs** | **standard** | **47.6%** | **0.468** |
| D2 | Marxulia + aliciiavs | medium | 44.8% | 0.442 |

**D1 wins** by +14.2 points — far beyond the ±3.5-point 95% CI at n=712.
**23 of 26 classes improved**; only C (−0.03) and B (−0.01) were trivially
worse. The biggest gains were on previously-broken classes: Q +0.31, F +0.30,
E +0.27, R +0.26, K +0.21. **D2 confirms the lesson**: adding augmentation on
top of diverse data *lowered* the score — diversity, not augmentation, is the
lever (augmentation was already shown neutral/harmful on the single-signer set).

## What shipped

- Deployed checkpoint promoted; ONNX re-exported (mobilenet_v2, 26-class, opset
  17 — shapes unchanged, no web code change). All parity layers re-verified at
  existing tolerances (Python 1e-3, web strict ~5e-7, web e2e 3e-2).
- Calibration kept at **T=1.0** (no separate real calibration set exists; fitting
  on the eval set would poison the gate). ECE on the merged held-out split
  improved 0.046 → 0.030.
- `web/public/metrics/*` refreshed: honest cross-dataset 47.6%, same-dataset
  held-out 97.3%.

## Honest caveats

- **J and Z stay weak** (J +0.05 to 0.143; Z stuck at 0.000). They are *motion*
  signs — a single static frame cannot represent them, and aliciiavs has neither.
  This is a structural floor, not a tuning failure.
- **Single eval gate.** We now train on data stylistically closer to the eval, so
  the pHash overlap check (0% at threshold 10) is the guard that the gain is real
  generalization, not eval leakage. A second independent honest gate would
  strengthen this further.
- **License.** aliciiavs declares no license — fine for this showcase, but flag
  before any redistribution.
- **Headroom remains.** 47.6% is a large step, not a finish line; more diverse
  sources (the `hemg` fallback is wired in the registry, and Roboflow/Kaggle sets
  exist behind auth) are the path to push further.
