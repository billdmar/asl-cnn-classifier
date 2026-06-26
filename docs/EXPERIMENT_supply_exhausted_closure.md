# Experiment closure: accuracy is sourcing-bound and the clean supply is exhausted

**Date:** 2026-06-26
**Status:** Accuracy investigation **closed** on current assets. Deployed model unchanged
(MobileNetV2, 26-class, **55.5% / A–Y 59.8%** honest cross-dataset).

## Why this is a closure, not another experiment

Across five rounds, exactly one lever ever moved the honest cross-dataset number, and it
was **training-data diversity**:

```
33.4%   single source (Marxulia)         — overfit to one signer / plain background
47.6%   + aliciiavs (multi-signer, real bg)   +14.2 pts
55.5%   + Hemg (3rd source, incl. J/Z)        +7.9 pts   ← deployed
```

Everything else was measured and **rejected** (see the other `docs/EXPERIMENT_*.md`):
crop-consistency, sketch-cleaning, augmentation (medium/heavy), per-class decision
thresholds, TTA, temperature calibration, class-balanced loss, and two architecture swaps
(mobilenet_v3_small −20pt, efficientnet_b0 −1pt). Diversity is the only thing that works —
and this round establishes that **the accessible supply of diverse data is exhausted.**

## Finding 1 — no new clean, diverse, anonymously-accessible A–Z dataset exists

An exhaustive Hugging Face Hub search (~12 query terms, ~50 datasets triaged, every plausible
candidate verified against the datasets-server API) found nothing usable that we don't
already train on. Constraints: A–Z static hand-sign images, genuine multi-signer/background
diversity, anonymous access (no Kaggle account — none is configured; no HF-gated/token sets),
and distinct from our four known datasets.

The one large new candidate is worth recording because it *looks* promising and isn't:

- **`NAM27/sign-language`** — 121,769 images, `image` + `ClassLabel` A–Z **including J/Z**,
  anonymous, viewer-enabled. **Rejected:** it is an anonymous mirror of the **grassknoted
  "ASL Alphabet"** Kaggle set — the original single-signer, plain-background overfit source
  this project deliberately climbed *away* from (it's `KAGGLE_DATASET` in `src/download_data.py`,
  and the README/MODEL_CARD already cite it). Visual inspection confirmed: one signer,
  identical room/window-frame background, lighting-only variation between near-duplicate
  frames. Adding its 24 A–Y classes would swamp the genuinely-diverse `aliciiavs` source and
  pull the model back toward single-signer bias — the opposite of the lever that works.

Everything else checked and ruled out: re-uploads of datasets we already use
(`thels07/...v03` = Marxulia), landmark-overlay sets (`raulit04/ASL_Dataset1` — MediaPipe
skeleton drawn on the hand), metadata-only manifests with no image bytes
(`cornelismus/Keithsel asl_signs` CSVs, `dnth/...vl-enriched` → 403 URIs), preprocessed
Sign-Language-MNIST tensors (`louiecerv/...`, 28×28 grayscale), word-level **video**
datasets, object-detection (YOLO bbox) sets, and many empty/tiny repos.
(`atalaydenknalbant/asl-dataset` was already rejected in a prior round for 6.2% source-overlap
with the EitanG98 eval gate.)

**Conclusion:** the clean, diverse, anonymously-accessible supply is exhausted.

## Finding 2 — the temporal J/Z path is real, but unverifiable on the current gate

J and Z are the two **dynamic motion signs** and the model's worst classes on the gate:

| class | precision | recall | f1 | gate support |
|------:|----------:|-------:|----|-------------:|
| J | 0.24 | 0.13 | 0.17 | 30 |
| Z | 0.20 | 0.03 | 0.057 | 30 |

They drag the headline ~4 pts (A–Y 59.8% vs 26-class 55.5%). The natural fix is a
**temporal / multi-frame model** (the live webcam already buffers 5 frames). But there is a
hard blocker: **the honest eval gate's J/Z are 60 static frames** (`data/asl_crossval/{J,Z}`
= 30 PNGs each), and `src/eval_realworld.py` has no video path. A temporal model has no
motion to consume at eval time, so its benefit **cannot be measured against the only deploy
decider.** Building a multi-week new-architecture model whose payoff is unverifiable on
current assets is not justifiable now.

This is a **deferral, not a dismissal** — temporal is the right idea once the missing piece
(a video benchmark) exists.

Also ruled out as busywork: adding *more single-signer* static J/Z frames (e.g. capped NAM27
J/Z). Current J/Z training signal is already single-signer (Marxulia 420/class + Hemg
70/class); more of the same adds volume, not diversity. Hemg's static J/Z frames already
delivered the one small lift available (J 0.143→0.170, Z 0.000→0.057), and at 30 gate
images/class any further delta sits inside the ±~3.5 pt (n=712) sampling-noise band — it
cannot clear a meaningful bar honestly. The gate's Z images are mid-gesture poses no single
static training frame represents (see `EXPERIMENT_crop_consistent_retrain.md`), a
label-scope mismatch that more static frames don't resolve.

## Unblock conditions (what would reopen accuracy work)

1. **A genuinely multi-signer, real-background A–Z image dataset** that is anonymously
   accessible (or Kaggle credentials to reach the gated ones) — wire it via
   `src/download_hf_data.py`, guard it with `scripts/check_eval_overlap.py` against the eval
   gate, then run the merged retrain (gate on cross-dataset, separate checkpoint dir). This
   is the proven lever; it's purely supply-blocked.
2. **A held-out *video* (or multi-frame) benchmark for J/Z** — this makes a temporal model
   measurable, at which point the deferred temporal path becomes a justifiable project.

Until one of those lands, the deployed 55.5% / A–Y 59.8% is the honest ceiling on current
assets, and the engineering value is the disciplined, fully-documented record of how it was
reached and what was ruled out.
