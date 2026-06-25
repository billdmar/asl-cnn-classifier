"""Fit per-class acceptance thresholds to curb over-predicted "sink" classes.

The cross-dataset model over-predicts a few closed-hand classes (notably S, and
to a lesser degree Q): S has high recall but low precision because T/N/E/M get
swallowed into it. A per-class *acceptance* threshold raises the bar before the
argmax is allowed to be one of these sinks — when the sink's probability is below
its threshold, the decision falls through to the next class that clears its own
threshold (see ``src.eval_realworld.apply_decision_policy``). This trades a little
sink recall for sink precision and recovers recall on the swallowed classes,
raising macro-F1.

CRITICAL: thresholds are fit on the **validation split of the TRAINING union**,
never on the eval gate. Fitting on the gate would be leakage. The eval harness
only *measures* the resulting policy on the gate.

Reuses src.calibration.collect_logits + the multi-dir val-split machinery, so the
data path is identical to training/calibration. Writes a decision-policy JSON:

    {"class_thresholds": {"S": 0.62, "Q": 0.55, ...}, "fit_on": "...", "margin": null}

Usage:
    python scripts/fit_thresholds.py \
        --checkpoint artifacts/checkpoints_diverse/best_model.pth \
        --data_dir "data/asl_real,data/asl_diverse" \
        --output artifacts/decision_policy.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import collect_logits  # noqa: E402
from src.dataset import (  # noqa: E402
    ASLDataset,
    _list_samples,
    get_class_names,
    get_eval_transforms,
    get_union_class_names,
    make_stratified_splits,
)
from src.infer_camera import load_checkpoint  # noqa: E402
from src.train import _normalize_data_dirs  # noqa: E402
from src.utils import get_device, save_json  # noqa: E402

# Candidate per-class thresholds to sweep. 0.0 == accept always (current argmax).
THRESHOLD_GRID = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _val_split(data_dir: str, seed: int) -> tuple[list[tuple[str, int]], list[str]]:
    """Recreate the VAL split of the (possibly merged) training union."""
    data_dirs = _normalize_data_dirs(data_dir)
    if len(data_dirs) == 1:
        class_names = get_class_names(data_dirs[0])
        _train, val, _test = make_stratified_splits(
            data_dirs[0], seed=seed, class_names=class_names
        )
    else:
        class_names = get_union_class_names(data_dirs)
        merged: list[tuple[str, int]] = []
        for d in data_dirs:
            merged.extend(_list_samples(d, class_names))
        _train, val, _test = make_stratified_splits(
            samples=merged, seed=seed, class_names=class_names
        )
    return val, class_names


def fit_thresholds(probs: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    """Greedily pick a per-class acceptance threshold that maximizes macro-F1.

    Argmax (all-zero thresholds) is the baseline. For each class, sweep its
    threshold over the grid and keep the value that most improves macro-F1 on the
    val set, holding the others fixed. One coordinate-ascent pass is enough to
    catch the dominant sink classes without overfitting the grid.
    """
    from src.eval_realworld import apply_decision_policy

    n = len(class_names)
    thresholds = {name: 0.0 for name in class_names}

    def macro_f1(th: dict) -> float:
        preds = [
            apply_decision_policy(probs[i], class_thresholds=th, class_names=class_names)
            for i in range(len(probs))
        ]
        return float(f1_score(labels, preds, labels=list(range(n)),
                              average="macro", zero_division=0))

    base = macro_f1(thresholds)
    best = base
    for name in class_names:
        best_t = 0.0
        for t in THRESHOLD_GRID:
            thresholds[name] = t
            score = macro_f1(thresholds)
            if score > best + 1e-9:
                best, best_t = score, t
        thresholds[name] = best_t  # keep the best for this class, then move on
    return {
        "class_thresholds": {k: v for k, v in thresholds.items() if v > 0.0},
        "macro_f1_val_argmax": base,
        "macro_f1_val_policy": best,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True, help="Training union (val split fit).")
    parser.add_argument("--output", default="artifacts/decision_policy.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device(args.device)
    model, ckpt_class_names = load_checkpoint(args.checkpoint, device)

    val, class_names = _val_split(args.data_dir, args.seed)
    class_names = ckpt_class_names or class_names
    ds = ASLDataset(samples=val, transform=get_eval_transforms(), class_names=class_names)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    logits, labels = collect_logits(model, loader, device)
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float64)), dim=1).numpy()

    result = fit_thresholds(probs, labels, class_names)
    result["fit_on"] = str(args.data_dir)
    result["margin"] = None
    save_json(args.output, result)

    print(f"Fit on VAL split of {args.data_dir} ({len(val)} samples)")
    print(f"Macro-F1 (val): argmax {result['macro_f1_val_argmax']:.4f} "
          f"-> policy {result['macro_f1_val_policy']:.4f}")
    print(f"Per-class thresholds set: {result['class_thresholds']}")
    print(f"Saved decision policy to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
