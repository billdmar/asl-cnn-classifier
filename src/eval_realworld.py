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
from src.infer_camera import DEFAULT_CHECKPOINT, load_checkpoint
from src.utils import get_device, save_json

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


def _predict_image(
    path: str,
    model: torch.nn.Module,
    transform: Any,
    device: torch.device,
    use_hand_crop: bool,
    landmarker: Any | None,
) -> tuple[int, bool]:
    """Predict one image; return ``(pred_index, used_whole_image_fallback)``.

    When ``use_hand_crop`` is set we crop to the detected hand; if no hand is
    found we fall back to the whole image and flag it so the caller can count
    no-hand fallbacks.
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
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = int(model(tensor).argmax(dim=1).item())
    return pred, fell_back


def evaluate(
    data_dir: str,
    checkpoint: str,
    device: torch.device,
    use_hand_crop: bool,
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
                filepath, model, transform, device, use_hand_crop, landmarker
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

    return {
        "source": f"cross-dataset:{data_dir}",
        "num_samples": int(y_true.size),
        "hand_crop_used": bool(use_hand_crop),
        "num_no_hand_fallback": int(num_no_hand),
        "accuracy": overall_accuracy,
        "macro_f1": float(macro["f1-score"]),
        "macro_precision": float(macro["precision"]),
        "macro_recall": float(macro["recall"]),
        "per_class": per_class,
        "most_confused_pairs": confused,
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
    return parser.parse_args()


def main() -> int:
    """Load the deployed model, evaluate cross-dataset, and write the artifact."""
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Hand crop: {'ON' if args.hand_crop else 'OFF'}")

    metrics = evaluate(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        device=device,
        use_hand_crop=args.hand_crop,
    )

    save_json(OUTPUT_PATH, metrics)

    print("\n=== Cross-dataset generalization summary ===")
    print(f"Source           : {metrics['source']}")
    print(f"Samples          : {metrics['num_samples']}")
    print(f"Hand crop used   : {metrics['hand_crop_used']}")
    print(f"No-hand fallbacks: {metrics['num_no_hand_fallback']}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"Macro precision  : {metrics['macro_precision']:.4f}")
    print(f"Macro recall     : {metrics['macro_recall']:.4f}")
    if metrics["most_confused_pairs"]:
        print("Top confused pairs (true -> pred: count):")
        for p in metrics["most_confused_pairs"]:
            print(f"  {p['true']} -> {p['pred']}: {p['count']}")
    print(f"\nSaved metrics to {OUTPUT_PATH}")
    print(f"\nNOTE: {HONEST_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
