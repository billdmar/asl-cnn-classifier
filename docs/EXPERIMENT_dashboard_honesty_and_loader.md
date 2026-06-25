# Experiment report: honest-number dashboard, flexible loader, and a contamination wall

**Date:** 2026-06-25
**Branch:** `feat/crop-consistent-retrain`
**Outcome:** two shipped wins (the live site now shows the honest cross-dataset
number; a reusable filename-label dataset loader) and one honest negative (the
only available new dataset is source-contaminated against the eval gate, so the
4th-source retrain and 2nd-gate were correctly NOT done).

## Context

After three rounds, the deployed model sits at **55.5% / A–Y 59.8%** honest
cross-dataset accuracy, reached purely by stacking diverse training data. Data
diversity is the only proven lever; every preprocessing/inference/architecture
lever was measured neutral-or-worse. This round pursued: (B) surface the honest
number on the site, (A) a loader for datasets that don't fit `image`+`label`,
(C) a 2nd eval gate, (D) a 4th training source.

## B — the honest number is now on the site (shipped)

The measured cross-dataset number existed in `web/public/metrics/realworld_eval.json`
for two deploys but was **never read by the dashboard**, and the site still said
*"a measured number is in progress"* and *"we deliberately do not show a single
headline real-world number yet."* That was false. Fixed:
- `web/lib/metrics.ts`: `RealworldEval` interface + `fetchRealworldEval`.
- Dashboard: a top card showing **A–Y 59.8% / 26-class 55.5%**, eval n, the
  33→47→55 diversity trajectory, and the honest note.
- `accuracy-story.tsx` + `page.tsx`: replaced the false "in progress" copy with
  the measured number; corrected the stale "deployed model is the baseline" claim.
- 56 web tests + eslint + `next build` green.

This is the round's highest-value deliverable: it corrects a false statement on
the employer-facing live site and surfaces real work that was hidden.

## A — flexible filename-label loader (shipped)

`atalaydenknalbant/asl-dataset` (the only candidate new diverse source) is a
Roboflow/YOLO export: `load_dataset` returns image-only, with the class encoded in
the filename (`A11_jpg.rf.<hash>.jpg`). Added `download_from_filenames()` using
`huggingface_hub.snapshot_download` + a filename regex, routed via a new
`FILENAME_DATASETS` registry, reusing `_normalize_class_name` and the
`<CLASS>/<i>.png` layout. Backward-compatible (existing tuple registry untouched);
tested with `snapshot_download` mocked. **Durable infra — unlocks any
filename/Roboflow-style ASL dataset in future.**

## C + D — blocked by a contamination wall (honest negative)

Materialized atalaydenknalbant via the new loader and ran the integrity guard
(`check_eval_overlap.py`, cross-dataset threshold 10) **before** trusting anything:

| Scan | Overlap | Finding |
|------|---------|---------|
| atalay_train vs **EitanG98** (primary gate) | **6.2%** (incl. a distance-0 exact match) | Roboflow ASL **shares source images with EitanG98** → training on it leaks into the gate |
| atalay_train vs atalay_test | 5.7% (10 exact dups) | Roboflow splits are internally near-duplicated |
| atalay_valid vs atalay_test | **29.6%** | same — augmented copies scattered across splits |

Consequences, decided honestly:
- **D (4th-source retrain) NOT run.** Training on a source that overlaps the eval
  gate would inflate the gate dishonestly — the exact failure the guard exists to
  prevent. Cleaning removes 371 obvious overlaps but threshold-10 cleaning is
  imperfect at the margin, and the payoff (a small, A–Y-only, no-J/Z source) does
  not justify the contamination risk.
- **C (2nd gate) is dead.** atalaydenknalbant's test split is internally leaky
  (can't be a clean held-out gate), and it overlaps EitanG98 anyway. As with Hemg,
  no truly independent multi-signer A–Z gate exists anonymously on HF today.
- The contaminated data dirs were deleted (nothing trainable left behind).

## Conclusion

The diversity lever is now **sourcing-bound, not code-bound**: the loader and
multi-source training are ready for more data, but the public HF supply of
*clean, independent, diverse* A–Z ASL image datasets is exhausted (every
remaining candidate is a duplicate, video, landmark-only, or — like
atalaydenknalbant — source-overlapping with our gate). The deployed model stays
at the honest 55.5% / 59.8%. The round's real value is the dashboard honesty fix
(B) and the reusable loader (A); the guard doing its job on C/D is a feature, not
a failure.

**Next-round headroom** would require either a genuinely new diverse dataset
(likely needing Kaggle credentials or manual collection), or moving beyond a
single-frame model (a temporal/video model for J/Z) — both larger efforts than a
config change.
