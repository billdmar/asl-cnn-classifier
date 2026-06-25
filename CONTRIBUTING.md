# Contributing

Thanks for your interest. This is a portfolio project, but it's built to
production standards — the workflow below keeps it that way.

## Setup

```bash
make install          # uv venv + dev deps + regenerate the sample fixture
make install-hooks    # git pre-commit hooks (ruff + black + mypy, mirrors CI)
```

`make install-hooks` is strongly recommended — the hooks run the **exact** lint /
format / type gate CI runs, so you never push a red build. (They call the
project's own `.venv` binaries, so hook behavior can't drift from CI.)

## The quality gate (what CI enforces)

Run these before pushing; all must pass:

```bash
# Python
.venv/bin/ruff check src tests
.venv/bin/black --check src tests
.venv/bin/mypy src
.venv/bin/pytest -q --cov=src --cov-fail-under=80     # 237+ tests, ≥80% coverage

# Web (cd web)
npm run typecheck && npm run lint && npm test && npm run build
npm run test:e2e        # Playwright (needs `npx playwright install` once)
npm run lighthouse      # gates ≥0.90 perf / a11y / best-practices / SEO
```

`make lint`, `make format`, `make mypy`, `make test` wrap the Python side.

## The honesty rule (important)

**Every number shown on the site comes from committed JSON in
`web/public/metrics/`, never a hardcoded literal.** The dashboard reads
`metrics.json` / `calibration.json` / `realworld_eval.json`; the E2E asserts the
rendered values against those files. If you change the model, regenerate the
JSON (below) — don't hand-edit displayed numbers. The deploy decider is the
**cross-dataset** gate (a dataset the model never trains on), never the inflated
same-dataset benchmark.

## Adding a training dataset

Data diversity is the only lever that has moved real-world accuracy
(33.4% → 47.6% → 55.5%). To add a source:

1. Register it in `src/download_hf_data.py` (`DATASETS` for image+label sets, or
   `FILENAME_DATASETS` for Roboflow/YOLO-style sets where the class is in the
   filename) and add a `download-<name>` Makefile target.
2. **Guard the eval gate:** `python scripts/check_eval_overlap.py --train_dir
   data/<name> --eval_dir data/asl_crossval`. Must report ~0% contamination —
   a source that overlaps the eval set is disqualified (this has caught real
   problems: Hemg's internal near-duplication, atalaydenknalbant's source
   overlap with EitanG98).
3. Add a `configs/train_*.yaml` listing the merged `data_dir` (multi-dir training
   is supported) with a **separate `checkpoint_dir`** so the deployed baseline is
   never clobbered.
4. `make train-<name>` then `make eval-realworld-<name>`. Promote only if it beats
   the deployed cross-dataset number by a margin beyond the n=712 noise band
   (±~3.5 pts).
5. Document the result — win or loss — in `docs/EXPERIMENT_*.md`.

## Regenerating committed model artifacts (after a retrain)

```bash
make export-onnx-web        # ONNX + parity fixtures → web/public/model/
make gradcam-web            # Grad-CAM overlays for examples → web/public/gradcam/
# copy the refreshed realworld_eval / metrics / calibration JSON into web/public/metrics/
```

Then update the pinned values in `web/lib/__tests__/metrics.test.ts`. The parity
tests (Python ↔ ONNX ↔ browser) must stay green at their existing tolerances.

## Nice-to-haves (open if you want to contribute)

- An `@axe-core/playwright` a11y E2E gate.
- A genuinely independent second cross-dataset eval gate (none exists anonymously
  on HF today — see the experiment docs).
