# Experiment: 224×224 input resolution (native MobileNetV2 resolution)

**Date:** 2026-07-08
**Status:** REJECTED — deploys WORSE on the cross-dataset gate (−3.7 pts).
**Deployed model unchanged** (MobileNetV2, 128×128, 55.5% / A–Y 59.8%).

## Hypothesis

MobileNetV2 was pretrained on 224×224 inputs. Using the project's 128×128
resolution discards spatial detail critical for distinguishing confusable hand
signs that differ by a single finger position (S/T, M/N, A/E). Increasing to
224×224 should preserve this detail and improve cross-dataset accuracy.

This was the single largest untried lever identified in the
[supply-exhausted closure](EXPERIMENT_supply_exhausted_closure.md).

## Method

Identical to the deployed D3 config except:
- `image_size: 128` → `224` (isolate the resolution variable)
- `batch_size: 64` → `32` (3× more memory per image at higher resolution)
- All other hyperparams unchanged (same 3-source data, same seed, same schedule)

Config: `configs/train_real_mobilenet_diverse_hemg_224.yaml`

## Training results

| Metric | 128×128 (deployed) | 224×224 (this experiment) |
|--------|-------------------|--------------------------|
| Val accuracy (same-dataset) | 96.8% | **99.15%** (+2.35 pts) |
| Training time (MPS, 12 epochs) | ~35 min | ~50 min |
| Checkpoint size | 9.0 MB | 9.3 MB |

The higher resolution model achieved a **significantly better** same-dataset
val accuracy (99.15% vs 96.8%), confirming that resolution helps the model
learn finer features from the training data.

## Cross-dataset evaluation (the honest gate)

| Metric | 128×128 (deployed) | 224×224 (this experiment) | Delta |
|--------|-------------------|--------------------------|-------|
| Accuracy (26-class) | **55.5%** | 51.8% | **−3.7 pts** |
| Accuracy (A–Y) | **59.8%** | 56.1% | **−3.7 pts** |
| Macro F1 (26-class) | 0.533 | 0.501 | −0.032 |
| Macro F1 (A–Y) | 0.577 | 0.556 | −0.021 |
| No-hand fallbacks | 61/712 | 61/712 | same |

## Interpretation

The 224×224 model **overfits more aggressively** to the training domain's visual
characteristics (texture, lighting, backgrounds) and generalizes WORSE to
out-of-domain signers. The higher resolution gives the model more fine-grained
texture cues to memorize — but those cues are domain-specific (e.g., the specific
skin texture, nail polish, background gradients in the training images) rather than
hand-shape-invariant.

This is consistent with the project's central finding: **the cross-dataset
bottleneck is data diversity, not model capacity or resolution.** The model
already has enough capacity (and now, resolution) to perfectly fit the training
domain. What it lacks is exposure to the visual diversity of real-world hands.

## Confusion analysis

Top confused pairs shifted slightly but the S-sink pattern persists:
- Z→T (14), W→V (11), V→K (10) — similar finger configurations
- M→S (8), E→S (7), T→S (7) — the familiar S-sink
- New: B→E (7), M→E (7), N→E (7) — the E-sink appeared (fist-like signs)

## Conclusion

**Resolution increase is not a useful lever for this problem.** It helps same-dataset
performance (the model fits training data better) but hurts cross-dataset
generalization (it overfits domain-specific textures more). The deployed 128×128
model remains superior on the honest gate.

This further confirms the supply-exhausted closure: within the current 3-source
training data, no architectural or preprocessing lever improves the honest number.
The only path to higher accuracy remains acquiring genuinely diverse additional
training data.

## Replication run (2026-07-08, resumed weights)

A second run starting from the epoch-3 warm-up checkpoint (to verify the result
is stable, not a random-seed artifact) produced slightly different numbers but
the same verdict:

| Metric | 128×128 (deployed) | 224×224 (run 2) | Delta |
|--------|-------------------|-----------------|-------|
| Accuracy (26-class) | **55.5%** | 53.2% | **−2.3 pts** |
| Accuracy (A–Y) | **59.8%** | 57.7% | **−2.1 pts** |
| Val accuracy | 96.9% | 98.5% | +1.6 pts |

Same pattern: higher same-dataset val, lower cross-dataset. **Verdict confirmed.**

## Artifacts

- Config: `configs/train_real_mobilenet_diverse_hemg_224.yaml`
- Checkpoint: `artifacts/checkpoints_224/best_model.pth` (NOT deployed)
- Eval JSON: `artifacts/realworld_eval_224.json`
