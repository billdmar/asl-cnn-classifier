# Experiment report: crop-consistent retrain (honest negative result)

**Date:** 2026-06-24
**Branch:** `feat/crop-consistent-retrain`
**Goal:** raise the model's honest cross-dataset accuracy on `data/asl_crossval`
(EitanG98/asl_letters, 712 imgs, different signers + backgrounds) above the
deployed baseline of **33.4% accuracy / 0.34 macro-F1**.
**Outcome:** the central hypothesis was **disproven**. Crop-consistent training
and data-cleaning are **neutral** on the gate. The deployed baseline is unchanged.
Along the way we found a real **data-quality defect** (15% sketch contamination).

## Hypothesis

Both serving paths (webcam, upload) feed the model a MediaPipe **hand-crop**
(`web/lib/handcrop.ts`, `CROP_MARGIN=0.35`), but the model was **trained on raw
uncropped frames**. We hypothesised that closing this train/serve preprocessing
mismatch — training on the same hand-crop the browser produces — was the #1
fixable cause of the 96.8% (same-dataset) → 33.4% (cross-dataset) collapse.

## What we ran

Same recipe each time (MobileNetV2, 26-class, 128px, AdamW 1e-3 cosine, warmup 3,
12 epochs, seed 42); only the training data varied. Each candidate written to a
**separate** checkpoint dir to protect the deployed baseline. All gated on
`make eval-realworld-*` (cross-dataset). Baseline re-measured this session for an
apples-to-apples comparison.

| Run | Training data | Cross-dataset acc | macro-F1 | Same-dataset val |
|-----|---------------|-------------------|----------|------------------|
| Baseline | raw `asl_real` (uncropped, dirty) | **0.3343** | 0.3400 | 0.978 |
| A | hand-cropped, dirty | 0.3216 | 0.3281 | 0.9755 |
| A-clean | hand-cropped, **sketch-filtered** | **0.3343** | 0.3320 | **0.9798** |

**All three sit within one ±3.5-point 95% confidence band** (binomial, n=712).
The differences are noise. A-clean produced the best same-dataset val (cleaner
data trains better) but moved the real-world number not at all.

## The data-quality discovery

Investigating why Run A came out *neutral* (and why class **H** collapsed,
F1 0.636 → 0.125), we found the Marxulia training set is **15.1% pencil-sketch
line drawings, not photographs** — **exactly 63 per class, 1638 total**, uniform
across all 26 classes. They are near-colorless (RGB max−min < 12 over >97% of
pixels), **100% fail MediaPipe hand detection**, and account for ~69% of the
22.8% crop "no-hand" fallbacks. So the first cropped set was polluted: 15%
non-photographic data fed through uncropped.

We added a saturation-based sketch filter (`scripts/precrop_dataset.py
--drop-sketches`, `is_line_drawing()`). Filtering dropped the no-hand rate from
**22.8% → 7.7%** (now matching the cross-val set's 8.6%) and is what enabled the
clean Run A-clean. It recovered H partially (0.125 → 0.270) — but the headline
stayed flat.

## Why preprocessing can't move this number

A model that scores 98% same-dataset and 33% cross-dataset is **overfit to its
training domain** (single signer, plain pink background). Crop-consistency,
sketch-cleaning, and augmentation are all *preprocessing* — they reshape the
pixels but **cannot inject signer/background diversity that isn't in the training
data**. The 63-point gap is a data-domain gap, not a preprocessing gap. The only
lever that can move it is **more diverse training data**.

That lever (a distinct 3rd dataset) was probed (`scripts/probe_asl_datasets.py`)
and found blocked: 8 of 9 candidates don't exist / aren't accessible on the HF
Hub, and the one that loads is from the same Marxulia author (no real diversity).
Finding a genuinely multi-signer dataset is a separate research effort.

## A class that is fundamentally unsolvable here

**Z is F1 0.000 in all three runs.** Z is a *motion* sign (the index finger
traces a "Z"). The cross-val Z images are mid-gesture poses (fist / pointing)
that no single static training frame represents. A static-frame classifier cannot
do Z regardless of preprocessing — this is a label-scope issue, not tunable.

## What shipped (kept — useful infrastructure)

- `scripts/precrop_dataset.py` — MediaPipe pre-crop reusing `src/handcrop.py`
  (browser-identical geometry); `--drop-sketches`, `--drop-no-hand`; writes a
  `_precrop_report.json` with no-hand / sketch rates.
- `src/dataset.py` — a `medium` augmentation tier (standard/heavy kept
  byte-identical); `regime` arg on `get_train_transforms`.
- `src/train.py` — reads an `augmentation` config key (back-compat preserved).
- `src/eval_realworld.py` — `--output` so candidate evals don't overwrite the
  baseline artifact.
- Configs + Makefile targets: `precrop[-clean]`, `train-cropped[-midaug|-clean]`,
  `eval-realworld-cropped[-midaug|-clean]`.
- Tests: medium-tier, eval `--output`, sketch detector + drop pipeline. 211 pass.

## Recommendation

Keep the deployed baseline (nothing beat it). The infrastructure stays — it makes
the next, correct experiment (training on a genuinely diverse dataset) a
one-config change. **Do not** spend more compute on preprocessing/augmentation
variants of the Marxulia data; the ceiling is the data, not the pixels.
