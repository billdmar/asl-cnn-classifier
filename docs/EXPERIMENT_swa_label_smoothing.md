# Experiment report: SWA + label smoothing (gated, NEGATIVE — not shipped)

**Date:** 2026-06-28
**Branch:** `feat/product-overhaul-round`
**Result:** **REJECTED.** Stochastic Weight Averaging + label smoothing 0.1
*regressed* the honest cross-dataset gate by **−1.8 pt** (26-class) and **−2.0 pt**
(A–Y) versus the deployed model. The decision rule required **both** metrics to
*beat* the baseline by **≥+2 pt**; the result moved the wrong direction by ~2 pt.
The deployed model, ONNX export, parity fixtures, and `web/public/metrics/*` are
**unchanged**.

## Why this experiment

The accuracy investigation was closed as sourcing-bound (see
`EXPERIMENT_supply_exhausted_closure.md`): training-data diversity is the only
lever that ever moved the honest number (33.4 → 47.6 → 55.5%). But two
generalization levers had never actually been *run* — they cost no new data and
keep a single deployable model, so they were worth one honest, gated attempt:

- **SWA** (`torch.optim.swa_utils`): average the weights over the fine-tune tail
  at a constant LR, then recompute BatchNorm stats. Known to flatten the loss
  basin and sometimes improve generalization/calibration.
- **Label smoothing 0.1** (`nn.CrossEntropyLoss(label_smoothing=0.1)`): softens
  one-hot targets; can reduce over-confidence.

Both were wired into `src/train.py` **config-gated, default-off** (byte-identical
to the prior training path when absent — verified by the existing train tests),
and combined into one run via
`configs/train_real_mobilenet_diverse_hemg_swa.yaml` (same 3-source D3 recipe +
the two levers; SWA averaging epochs 9–12 of 12 at LR 1e-4; separate checkpoint
dir, never clobbering the deployed baseline).

## Result

Trained on the full 3-source union (21,130 images, 26 classes) on MPS; SWA
averaged 4 tail epochs; **val accuracy 97.41%** (same-dataset, as expected — the
benchmark is not the decider). Evaluated on the sole honest gate (EitanG98
`data/asl_crossval`, n=712, CI ±~3.5 pt).

| Metric | Deployed D3 | SWA + LS | Δ |
|--------|------------:|---------:|---:|
| Cross-dataset accuracy (26-class) | **0.5548** | 0.5365 | **−0.0183** |
| A–Y headline accuracy | **0.5982** | 0.5782 | **−0.0199** |
| Macro-F1 (26-class) | **0.5476** | 0.5272 | −0.0204 |
| J f1 | 0.170 | 0.208 | +0.038 |
| Z f1 | 0.057 | 0.000 | −0.057 |

Both headline metrics regressed by ~2 pt — the opposite of the +2 pt bar. The
small J gain is washed out by a Z collapse (0.057 → 0.000) and a broad softening
across the alphabet; the two dynamic motion signs remain unfixable from static
frames regardless (the standing temporal-J/Z deferral is unaffected).

## Why it lost (interpretation)

The cross-dataset gap is driven by genuinely narrow inter-class separations under
distribution shift (the confusion pairs are the usual hand-shape neighbours:
T→S, E→S, N→S, V↔K). Label smoothing deliberately flattens the target
distribution, which **shrinks already-thin margins** between those neighbours,
and SWA's basin-averaging compounds the softening rather than sharpening the
decision boundaries that the shifted test set needs. On a same-dataset benchmark
this would likely help (and val did stay ~97%); on the honest cross-dataset gate
it hurts. Consistent with every prior round: only *more diverse data* moves this
number — regularization/optimization tricks sit at or below the deployed model.

## Disposition

- **Not shipped.** Deployed model and all web assets unchanged; the drift guard
  (`tests/test_model_onnx_sync.py`) and parity fixtures continue to track the
  deployed checkpoint.
- **Kept (no cost):** the config-gated `label_smoothing` + SWA wiring in
  `src/train.py` (default-off), the experiment config, and the
  `train/eval-realworld-diverse-hemg-swa` Makefile targets — so the negative is
  reproducible (`make train-diverse-hemg-swa && make eval-realworld-diverse-hemg-swa`)
  and the levers are available if a future, more-diverse dataset reopens the work.
- **Unblock conditions unchanged** from `EXPERIMENT_supply_exhausted_closure.md`:
  a genuinely multi-signer A–Z dataset, or a held-out video gate for temporal J/Z.
