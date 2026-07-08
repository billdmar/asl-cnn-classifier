"""Cross-dataset real-world generalization harness for the ASL classifier.

The deployed model scores ~96.8% on its OWN held-out split, but that number is
inflated: a single signer, plain backgrounds, and sequential-frame leakage. This
script measures how the same checkpoint generalizes to a *different* dataset
(different signers, real photo backgrounds) so we have an honest number.

For each image it (optionally) detects + crops the hand with the same MediaPipe
geometry the browser uses (:func:`src.handcrop.detect_and_crop`, ``CROP_MARGIN``
mirrored), falling back to the whole image when no hand is found (and counting
those). It then applies the canonical :func:`src.dataset.get_eval_transforms`
and runs the model, accumulating predictions to compute overall accuracy,
macro-F1, per-class precision/recall/F1, and the most-confused pairs.

The result is written to ``artifacts/realworld_eval.json`` with an explicit,
honest note: this is CROSS-DATASET generalization, NOT the 96.8% benchmark, and
must NOT be merged with it.

Run, e.g.::

    python -m src.eval_realworld \\
        --checkpoint artifacts/checkpoints/best_model.pth \\
        --data_dir data/asl_crossval --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import _list_samples, get_class_names, get_eval_transforms
from src.eval import most_confused_pairs
from src.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
from src.utils import get_device, load_json, save_json

ARTIFACTS = Path("artifacts")
OUTPUT_PATH = ARTIFACTS / "realworld_eval.json"

HONEST_NOTE = (
    "CROSS-DATASET generalization: the checkpoint was trained on a different "
    "(single-signer, plain-background) dataset and is evaluated here on a "
    "DIFFERENT dataset (different signers/backgrounds). This is the honest "
    "real-world number — it is NOT the ~96.8% same-dataset held-out benchmark "
    "and MUST NOT be averaged or merged with it. Lower accuracy here is "
    "expected and reflects the domain gap, not a regression."
)


def apply_decision_policy(
    probs: np.ndarray,
    class_thresholds: dict[str, float] | None = None,
    class_names: list[str] | None = None,
    margin: float | None = None,
) -> int:
    """Pick a class index from softmax ``probs`` under an optional decision policy.

    Pure and array-only (unit-testable). With no policy this is plain argmax, so
    the default behavior is byte-identical to before. The policy targets the
    over-predicted "sink" classes (S, Q): a class is only *accepted* as the top
    prediction when its probability clears its per-class threshold AND the
    top1−top2 margin clears ``margin``; otherwise we fall through to the best
    class that does clear its threshold. This raises sink-class precision and
    recovers recall on the classes they were swallowing (T, N, E, M).

    Args:
        probs: 1-D softmax vector over ``class_names``.
        class_thresholds: optional ``{class_name: min_prob}``; classes absent get
            no floor (threshold 0).
        class_names: index→name map, required when ``class_thresholds`` is given.
        margin: optional minimum top1−top2 gap for the winner to be accepted.

    Returns:
        The chosen class index.
    """
    order = np.argsort(probs)[::-1]  # high→low
    top = int(order[0])
    if class_thresholds is None and margin is None:
        return top

    top2_gap = float(probs[order[0]] - probs[order[1]]) if probs.size > 1 else 1.0
    margin_ok = margin is None or top2_gap >= margin

    def floor_for(idx: int) -> float:
        if not class_thresholds or class_names is None:
            return 0.0
        return float(class_thresholds.get(class_names[idx], 0.0))

    # If the argmax clears its floor and the margin is satisfied, accept it.
    if float(probs[top]) >= floor_for(top) and margin_ok:
        return top
    # Otherwise fall through to the highest-ranked class that clears its floor.
    for idx in order:
        if float(probs[idx]) >= floor_for(int(idx)):
            return int(idx)
    return top  # nothing clears its floor → keep the argmax


def _predict_image(
    path: str,
    model: torch.nn.Module,
    transform: Any,
    device: torch.device,
    use_hand_crop: bool,
    landmarker: Any | None,
    *,
    class_thresholds: dict[str, float] | None = None,
    class_names: list[str] | None = None,
    margin: float | None = None,
    tta: bool = False,
) -> tuple[int, bool]:
    """Predict one image; return ``(pred_index, used_whole_image_fallback)``.

    When ``use_hand_crop`` is set we crop to the detected hand; if no hand is
    found we fall back to the whole image and flag it so the caller can count
    no-hand fallbacks. The optional decision policy (``class_thresholds`` /
    ``margin``) and ``tta`` default off, so the base path is unchanged.
    """
    image = Image.open(path).convert("RGB")
    fell_back = False
    if use_hand_crop:
        from src.handcrop import detect_and_crop

        cropped = detect_and_crop(image, landmarker=landmarker)
        if cropped is None:
            fell_back = True
        else:
            image = cropped
    probs = _infer_probs(image, model, transform, device, tta=tta)
    pred = apply_decision_policy(
        probs,
        class_thresholds=class_thresholds,
        class_names=class_names,
        margin=margin,
    )
    return pred, fell_back


def _infer_probs(
    image: Image.Image,
    model: torch.nn.Module,
    transform: Any,
    device: torch.device,
    *,
    tta: bool = False,
) -> np.ndarray:
    """Return the softmax probability vector for one image.

    With ``tta`` set, average softmax over multi-scale centre/corner-ish crops
    (NO horizontal flip — ASL signs are not flip-invariant: b/d, p/q are mirror
    images). Without it, a single deterministic forward pass.
    """
    views = _tta_views(image) if tta else [image]
    acc: np.ndarray | None = None
    with torch.no_grad():
        for view in views:
            tensor = transform(view).unsqueeze(0).to(device)
            logits = model(tensor)
            p = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            acc = p if acc is None else acc + p
    assert acc is not None
    return acc / len(views)


def _tta_views(image: Image.Image) -> list[Image.Image]:
    """Multi-scale centre crops for test-time augmentation (NO flips).

    Returns the original plus a few centre crops at 0.9 and 0.8 of each side.
    Centre-only + scale-only keeps it cheap and avoids the mirror-letter hazard.
    """
    w, h = image.size
    views = [image]
    for frac in (0.9, 0.8):
        cw, ch = int(w * frac), int(h * frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        views.append(image.crop((left, top, left + cw, top + ch)))
    return views


def evaluate(
    data_dir: str,
    checkpoint: str,
    device: torch.device,
    use_hand_crop: bool,
    *,
    class_thresholds: dict[str, float] | None = None,
    margin: float | None = None,
    tta: bool = False,
) -> dict[str, Any]:
    """Run the cross-dataset evaluation and return the metrics dict.

    Predictions use the checkpoint's recorded class names as the canonical
    index↔label map; on-disk folders are mapped to those same labels so a
    dataset missing a class (or containing extra ones) still aligns.
    """
    model, class_names = load_checkpoint(checkpoint, device)
    transform = get_eval_transforms()

    # Map on-disk folders to the checkpoint's label space. Folders not in the
    # checkpoint's class list are ignored (no label to score against).
    folder_names = get_class_names(data_dir)
    scorable = [n for n in folder_names if n in class_names]
    samples = _list_samples(data_dir, scorable)
    if not samples:
        raise RuntimeError(
            f"No scorable images found under {data_dir} for classes {class_names}."
        )
    label_of = {name: idx for idx, name in enumerate(class_names)}
    folder_label_of = {name: idx for idx, name in enumerate(scorable)}
    inv_folder = {idx: name for name, idx in folder_label_of.items()}

    # Build a single landmarker once and reuse it across all images (the heavy
    # MediaPipe graph build happens once, not per-image).
    landmarker = None
    if use_hand_crop:
        from src.handcrop import _build_landmarker
        from src.handcrop import DEFAULT_MODEL_PATH

        landmarker = _build_landmarker(DEFAULT_MODEL_PATH)

    y_true: list[int] = []
    y_pred: list[int] = []
    num_no_hand = 0
    try:
        for filepath, folder_idx in samples:
            true_idx = label_of[inv_folder[folder_idx]]
            pred_idx, fell_back = _predict_image(
                filepath,
                model,
                transform,
                device,
                use_hand_crop,
                landmarker,
                class_thresholds=class_thresholds,
                class_names=class_names,
                margin=margin,
                tta=tta,
            )
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            num_no_hand += int(fell_back)
    finally:
        if landmarker is not None:
            landmarker.close()

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    return _build_metrics(
        y_true_arr,
        y_pred_arr,
        class_names,
        checkpoint=checkpoint,
        data_dir=data_dir,
        use_hand_crop=use_hand_crop,
        num_no_hand=num_no_hand,
    )


# The two dynamic (motion) ASL letters. A single static frame cannot represent
# them, so the mainstream convention (e.g. Sign Language MNIST) reports a 24-class
# A–Y headline and treats J/Z separately. We surface BOTH numbers honestly.
MOTION_LETTERS = ("J", "Z")


def _subset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    exclude: tuple[str, ...],
) -> dict[str, float]:
    """Accuracy + macro-F1 over the classes NOT in ``exclude`` (e.g. drop J/Z).

    Pure. The headline A–Y metric: samples whose TRUE label is an excluded class
    are removed (you can't be scored on a class you don't report), and accuracy
    is measured on the remaining samples against the same predictions.
    """
    excluded_idx = {i for i, n in enumerate(class_names) if n in exclude}
    keep = np.array([t not in excluded_idx for t in y_true], dtype=bool)
    if not keep.any():
        return {"accuracy": 0.0, "macro_f1": 0.0, "num_samples": 0}
    yt, yp = y_true[keep], y_pred[keep]
    kept_labels = [i for i in range(len(class_names)) if i not in excluded_idx]
    kept_names = [class_names[i] for i in kept_labels]
    report = classification_report(
        yt,
        yp,
        labels=kept_labels,
        target_names=kept_names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float((yt == yp).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "num_samples": int(yt.size),
    }


def _build_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    *,
    checkpoint: str,
    data_dir: str,
    use_hand_crop: bool,
    num_no_hand: int,
) -> dict[str, Any]:
    """Aggregate predictions into the honest cross-dataset metrics dict.

    Pure given its array inputs (no model / disk), so it is unit-testable
    directly. Reuses :func:`src.eval.most_confused_pairs` for the confusion
    analysis so there is no second copy of that logic.
    """
    labels = list(range(len(class_names)))
    overall_accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0

    # classification_report rejects empty input; with no samples every per-class
    # metric is 0.0 by definition, so build that directly.
    per_class: dict[str, dict[str, float]]
    if y_true.size == 0:
        macro = {"f1-score": 0.0, "precision": 0.0, "recall": 0.0}
        per_class = {
            name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            for name in class_names
        }
        confused: list[dict[str, Any]] = []
        confusion: list[list[int]] = []
    else:
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
        macro = report["macro avg"]
        per_class = {}
        for name in class_names:
            stats = report.get(name, {})
            per_class[name] = {
                "precision": float(stats.get("precision", 0.0)),
                "recall": float(stats.get("recall", 0.0)),
                "f1": float(stats.get("f1-score", 0.0)),
                "support": int(stats.get("support", 0)),
            }
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        confused = most_confused_pairs(cm, class_names, top_k=10)
        confusion = cm.astype(int).tolist()

    # Headline A–Y metric (exclude the dynamic J/Z): the defensible mainstream
    # convention for a static-frame classifier. The full 26-class number stays
    # the primary `accuracy`/`macro_f1` fields for completeness.
    ay = _subset_metrics(y_true, y_pred, class_names, MOTION_LETTERS)

    return {
        "source": f"cross-dataset:{data_dir}",
        "num_samples": int(y_true.size),
        "hand_crop_used": bool(use_hand_crop),
        "num_no_hand_fallback": int(num_no_hand),
        "accuracy": overall_accuracy,
        "macro_f1": float(macro["f1-score"]),
        "macro_precision": float(macro["precision"]),
        "macro_recall": float(macro["recall"]),
        "accuracy_ay": ay["accuracy"],
        "macro_f1_ay": ay["macro_f1"],
        "num_samples_ay": ay["num_samples"],
        "per_class": per_class,
        "most_confused_pairs": confused,
        "confusion_labels": list(class_names),
        "confusion_matrix": confusion,
        "checkpoint": str(checkpoint),
        "note": HONEST_NOTE,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Honest cross-dataset generalization eval for the ASL model."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the model checkpoint (defaults to the deployed model).",
    )
    parser.add_argument(
        "--data_dir",
        default="data/asl_crossval",
        help="Class-folder dataset to evaluate (e.g. data/asl_crossval).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Compute device.",
    )
    parser.add_argument(
        "--hand_crop",
        dest="hand_crop",
        action="store_true",
        help="Detect+crop the hand before classifying (default).",
    )
    parser.add_argument(
        "--no-hand_crop",
        dest="hand_crop",
        action="store_false",
        help="Classify the whole image without hand detection.",
    )
    parser.set_defaults(hand_crop=True)
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help=(
            "Where to write the metrics JSON. Defaults to the deployed baseline "
            "path; pass a distinct file when evaluating a candidate so the "
            "baseline's realworld_eval.json is not overwritten."
        ),
    )
    parser.add_argument(
        "--thresholds_json",
        default=None,
        help=(
            "Optional decision-policy JSON (from scripts/fit_thresholds.py) with "
            "per-class acceptance thresholds. Fit on a VAL split, never the eval set."
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Optional minimum top1-top2 probability gap to accept the argmax.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Multi-scale centre-crop test-time augmentation (no flips).",
    )
    return parser.parse_args()


def main() -> int:
    """Load the deployed model, evaluate cross-dataset, and write the artifact."""
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Hand crop: {'ON' if args.hand_crop else 'OFF'}")

    class_thresholds = None
    if args.thresholds_json:
        policy = load_json(args.thresholds_json)
        class_thresholds = policy.get("class_thresholds", policy)
        print(f"Decision policy: per-class thresholds from {args.thresholds_json}")
    if args.margin is not None:
        print(f"Decision policy: top1-top2 margin >= {args.margin}")
    if args.tta:
        print("Test-time augmentation: ON (multi-scale centre crops)")

    metrics = evaluate(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        device=device,
        use_hand_crop=args.hand_crop,
        class_thresholds=class_thresholds,
        margin=args.margin,
        tta=args.tta,
    )

    output_path = Path(args.output)
    save_json(output_path, metrics)

    print("\n=== Cross-dataset generalization summary ===")
    print(f"Source           : {metrics['source']}")
    print(f"Samples          : {metrics['num_samples']}")
    print(f"Hand crop used   : {metrics['hand_crop_used']}")
    print(f"No-hand fallbacks: {metrics['num_no_hand_fallback']}")
    print(f"Accuracy (26)    : {metrics['accuracy']:.4f}")
    print(f"Macro F1 (26)    : {metrics['macro_f1']:.4f}")
    print(f"Accuracy (A-Y)   : {metrics['accuracy_ay']:.4f}  <- headline (no J/Z)")
    print(f"Macro F1 (A-Y)   : {metrics['macro_f1_ay']:.4f}")
    print(f"Macro precision  : {metrics['macro_precision']:.4f}")
    print(f"Macro recall     : {metrics['macro_recall']:.4f}")
    if metrics["most_confused_pairs"]:
        print("Top confused pairs (true -> pred: count):")
        for p in metrics["most_confused_pairs"]:
            print(f"  {p['true']} -> {p['pred']}: {p['count']}")
    print(f"\nSaved metrics to {output_path}")
    print(f"\nNOTE: {HONEST_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
