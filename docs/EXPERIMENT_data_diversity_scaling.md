# Experiment report: scaling data diversity + the inference-lever sweep

**Date:** 2026-06-25
**Branch:** `feat/crop-consistent-retrain`
**Result:** honest cross-dataset accuracy **47.6% → 55.5%** (26-class) / **59.8%**
(A–Y headline), macro-F1 **0.468 → 0.548**. A 3rd diverse training source
(Hemg) deployed; four zero-compute inference levers measured and (mostly) rejected.

## Starting point

The deployed model (MobileNetV2, 26-class, trained on Marxulia + aliciiavs)
scored **47.6% / 0.468** on the honest cross-dataset gate (EitanG98, 712 imgs).
The prior reports established diversity as the only proven lever and preprocessing
as tapped out. This round tested two things: (1) cheap inference-time levers
against the *current* model, and (2) adding a 3rd diverse dataset.

## Phase 0 — zero-compute inference levers (measured, mostly rejected)

All measured against the deployed model on the EitanG98 gate (n=712, CI ±3.7 pts).

| Lever | 26-class acc | Verdict |
|-------|-------------|---------|
| Baseline | 0.4761 | — |
| Per-class decision thresholds (fit on union val) | 0.4593 | **REJECTED** — hurt the gate |
| Multi-scale centre-crop TTA | 0.4874 | **NOT SHIPPED** — +1.1pt within noise, regressed a strong class (U −0.08) |
| Margin-based abstain | 0.4761 | no-op offline (only affects web "unsure" UX) |
| **A–Y headline metric** | — | **KEPT** — reporting honesty (51.4% A–Y vs 47.6% 26-class, same model) |
| Temperature calibration | — | **T=1.0 kept** — a same-dataset fit gives T=0.637, which *sharpens* and would make the model overconfident on hard real-world inputs |

**Lesson:** per-class thresholds fit on the training distribution's val split do
NOT transfer to a different eval distribution — the confusion structure differs.
This is the same overfitting trap, one level up. The levers are kept as opt-in,
default-off infrastructure (they may help once a real-world validation set exists),
but none ship enabled. The single keeper is the honest **A–Y headline metric**:
J and Z are dynamic motion signs, so excluding them from the headline (and
reporting them separately) is the defensible, mainstream convention.

## Phase 1 — a 3rd diverse training source (the win)

Added `Hemg/sign_language_dataset` (1,815 A–Z imgs incl. static J/Z frames,
plain-background, single-signer) to the training union → 21,130 images, 26 classes.

**Eval-gate integrity finding (the guard earned its keep):** `check_eval_overlap.py`
showed Hemg is **0% overlapping with the EitanG98 gate** (safe to train on), but a
random Hemg train/gate split is **75.9% internally near-duplicated** (138 exact
duplicates at pHash distance 0) — Hemg is single-signer with repeated poses, so it
**cannot serve as a held-out 2nd gate**. The planned second gate was shelved with
evidence; Hemg is training-only and EitanG98 remains the sole honest gate.

| Run | Training sources | Cross-dataset 26 | A–Y | macro-F1 (26) |
|-----|------------------|------------------|-----|---------------|
| D1 (prior deploy) | Marxulia + aliciiavs | 47.6% | 51.4% | 0.468 |
| **D3 (deployed)** | **+ Hemg** | **55.5%** | **59.8%** | **0.548** |

**+7.9 pts (26-class), +8.4 pts (A–Y)** — far beyond the ±3.7 noise band.
**23 of 26 classes improved**, no strong-class regression (worst: X −0.07). The
static J/Z frames even nudged the motion signs off the floor: **J 0.143 → 0.170,
Z 0.000 → 0.057**.

## The trajectory

```
33.4%  single-source (Marxulia only)        — overfit to one signer/background
47.6%  + aliciiavs   (diverse real photos)  — +14.2 pts
55.5%  + Hemg        (more diversity + J/Z) — +7.9 pts   ← deployed
```

Each diverse dataset added real generalization. Diminishing returns are visible
(+14 then +8), consistent with approaching the single-model ceiling on the
available data — but the lever remains the right one.

## What shipped

- Deployed checkpoint promoted (D3); ONNX re-exported (mobilenet_v2, 26-class,
  opset 17 — shapes unchanged, no web code change). Parity re-verified at existing
  tolerances (Python 1e-3, web strict ~5e-7, web e2e 3e-2).
- Calibration kept at **T=1.0**; ECE on merged held-out split 0.030 → 0.025.
- `web/public/metrics/*` refreshed (A–Y headline 59.8%, 26-class 55.5%,
  same-dataset 96.9%). Dashboard test pins updated.
- New opt-in infra: `apply_decision_policy` + `scripts/fit_thresholds.py`,
  `predict_with_tta` (`--tta`), per-class/margin in `web/lib/confidence.ts`,
  `scripts/split_hemg.py`.

## Honest caveats

- **J/Z floor is structural** (motion signs); the small lift from static frames is
  real but they stay the weakest classes. A–Y is the headline for this reason.
- **Single eval gate.** Hemg couldn't be a 2nd gate (internal near-duplication);
  EitanG98 remains the sole honest gate. A genuinely multi-signer held-out set
  would strengthen confidence further.
- **No declared license** on aliciiavs/Hemg — showcase use; flag for redistribution.
- **Inference levers fit on the training distribution don't transfer** — documented
  so future work fits them on real-world data or not at all.
- Both gates are upload-style stills — a proxy for the live-webcam product; confirm
  the operating point with human spot-checks.
