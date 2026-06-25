"""Evaluate a trained ASL classifier on the held-out TEST split.

This script recreates the *exact* test split that ``train.py`` held out (the
stratified split is deterministic given the seed), runs inference over it, and
writes the full evaluation artifact set:

* ``artifacts/metrics.json`` — overall accuracy, macro precision/recall/F1,
  per-class metrics, and the most-confused class pairs.
* ``artifacts/confusion_matrix.png`` — an ``N×N`` seaborn heatmap (``N`` = the
  number of classes) with all class labels shown (even classes absent from the
  tiny sample test set).
* ``artifacts/per_class_errors.txt`` — the top-10 most-confused class pairs.

Preprocessing reuses :func:`src.dataset.get_eval_transforms` and checkpoint
loading reuses :func:`src.infer_camera.load_checkpoint`, so there is no second
copy of either piece of logic.

NOTE: accuracy on the tiny committed ``data/sample`` fixture is only a wiring
sanity check, not a meaningful measure of model quality.

Run, e.g.::

    python -m src.eval --checkpoint artifacts/checkpoints/best_model.pth \
        --data_dir data/sample --device cpu
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend — no display required.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

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
from src.utils import get_device, save_json, set_seed  # noqa: E402

DEFAULT_CHECKPOINT = "artifacts/checkpoints/best_model.pth"
ARTIFACTS = Path("artifacts")

SAMPLE_DATA_NOTE = (
    "Accuracy on the tiny data/sample fixture is a wiring sanity check, not a "
    "meaningful measure of model quality. Train on the full ASL Alphabet "
    "dataset for real numbers."
)
REAL_DATA_NOTE = (
    "Accuracy measured on the held-out test split of a real ASL hand-sign "
    "dataset (never seen during training or validation)."
)


def _accuracy_note(data_dir: str) -> str:
    """Pick an honest provenance note based on the dataset being evaluated.

    The synthetic ``data/sample`` fixture is a wiring sanity check; any other
    directory is treated as real data and gets the held-out-test note.
    """
    return SAMPLE_DATA_NOTE if "sample" in str(data_dir) else REAL_DATA_NOTE


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over ``loader`` and return ``(y_true, y_pred)`` arrays.

    CRITICAL: predictions and labels are moved to CPU via ``.detach().cpu()``
    before conversion to numpy — MPS tensors cannot be handed to numpy/sklearn
    directly.

    Returns:
        Two 1-D int numpy arrays of equal length: true labels and predictions.
    """
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for inputs, targets, _paths in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        all_pred.append(preds.detach().cpu().numpy())
        all_true.append(targets.detach().cpu().numpy())

    if not all_true:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(all_true), np.concatenate(all_pred)


def most_confused_pairs(
    cm: np.ndarray, class_names: list[str], top_k: int = 10
) -> list[dict[str, Any]]:
    """Return the ``top_k`` most-confused off-diagonal class pairs.

    Args:
        cm: Confusion matrix with ``cm[i, j]`` = count of true ``i`` predicted
            as ``j``.
        class_names: Index→label mapping.
        top_k: Number of pairs to return.

    Returns:
        A list of ``{"true", "pred", "count"}`` dicts sorted by count desc.
    """
    pairs: list[tuple[int, int, int]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((i, j, int(cm[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return [
        {"true": class_names[i], "pred": class_names[j], "count": count}
        for i, j, count in pairs[:top_k]
    ]


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    """Save an ``N×N`` seaborn heatmap with every class label legible."""
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("ASL classifier confusion matrix (held-out test split)")
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_per_class_errors(pairs: list[dict[str, Any]], path: Path) -> None:
    """Write the top confused pairs as lines like ``M -> N: 4``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Top-10 most-confused class pairs (true -> pred: count)", ""]
    if not pairs:
        lines.append("(no off-diagonal confusions on this test split)")
    else:
        lines.extend(f"{p['true']} -> {p['pred']}: {p['count']}" for p in pairs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _degrade(image: Image.Image, kind: str) -> Image.Image:
    """Apply a synthetic degradation to a clean RGB PIL image.

    Mirrors the five degradations used in ``benchmark.py`` so eval is
    self-contained; for the canonical study prefer ``python -m src.benchmark``.
    """
    if kind == "clean":
        return image
    if kind == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    if kind == "jpeg_q20":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=20)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "brightness_0.4":
        arr = np.asarray(image).astype(np.float32) * 0.4
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == "brightness_1.8":
        arr = np.asarray(image).astype(np.float32) * 1.8
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == "salt_pepper_5pct":
        arr = np.asarray(image).copy()
        rng = np.random.default_rng(0)
        mask = rng.random(arr.shape[:2])
        arr[mask < 0.025] = 0
        arr[mask > 0.975] = 255
        return Image.fromarray(arr)
    raise ValueError(f"Unknown degradation '{kind}'.")


@torch.no_grad()
def distribution_shift(
    model: nn.Module,
    test_samples: list[tuple[str, int]],
    device: torch.device,
) -> dict[str, float]:
    """Measure test accuracy under each of the five synthetic degradations.

    This is a self-contained basic version of the robustness study; the
    canonical implementation lives in ``benchmark.py``.

    Returns:
        Mapping of degradation name → accuracy in ``[0, 1]`` (``0.0`` if the
        test split is empty, matching ``zero_division=0`` semantics).
    """
    transform = get_eval_transforms()
    degradations = [
        "clean",
        "gaussian_blur",
        "jpeg_q20",
        "brightness_0.4",
        "brightness_1.8",
        "salt_pepper_5pct",
    ]
    results: dict[str, float] = {}
    for kind in degradations:
        correct = 0
        total = 0
        for filepath, label in test_samples:
            clean = Image.open(filepath).convert("RGB")
            tensor = transform(_degrade(clean, kind)).unsqueeze(0).to(device)
            pred = int(model(tensor).argmax(dim=1).item())
            correct += int(pred == label)
            total += 1
        results[kind] = (correct / total) if total > 0 else 0.0
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the ASL classifier on the test split."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional training-config YAML/JSON (unused for eval logic).",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--data_dir", default="data/sample", help="Path to class folders."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Compute device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed (must match training to recreate the split).",
    )
    parser.add_argument(
        "--distribution_shift",
        action="store_true",
        help="Also run the 5-degradation robustness analysis (basic version).",
    )
    return parser.parse_args()


def main() -> int:
    """Load the model, evaluate on the held-out test split, and write artifacts."""
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Recreate the SAME test split train.py held out (deterministic given seed).
    # Mirror train.py's single-vs-multi-dir handling so a model trained on a
    # merged union is evaluated on the matching merged held-out split.
    data_dirs = _normalize_data_dirs(args.data_dir)
    if len(data_dirs) == 1:
        class_names = get_class_names(data_dirs[0])
        _train, _val, test_samples = make_stratified_splits(
            data_dirs[0], seed=args.seed, class_names=class_names
        )
    else:
        class_names = get_union_class_names(data_dirs)
        merged: list[tuple[str, int]] = []
        for d in data_dirs:
            merged.extend(_list_samples(d, class_names))
        _train, _val, test_samples = make_stratified_splits(
            samples=merged, seed=args.seed, class_names=class_names
        )
    print(f"Held-out test split: {len(test_samples)} samples.")

    test_ds = ASLDataset(
        samples=test_samples,
        transform=get_eval_transforms(),
        class_names=class_names,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model, ckpt_class_names = load_checkpoint(args.checkpoint, device)
    # Prefer the checkpoint's recorded class names if present (label↔index map).
    class_names = ckpt_class_names or class_names

    y_true, y_pred = run_inference(model, test_loader, device)

    # Derive the label set from the actual class names (works for 26, 29, or any
    # count) instead of a hardcoded constant, so classification_report and
    # confusion_matrix never mismatch the model's output dimension.
    labels = list(range(len(class_names)))
    if y_true.size > 0:
        overall_accuracy = float((y_true == y_pred).mean())
    else:
        overall_accuracy = 0.0

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    macro = report["macro avg"]

    per_class: dict[str, dict[str, float]] = {}
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

    # --- Artifacts ---
    cm_path = ARTIFACTS / "confusion_matrix.png"
    save_confusion_matrix(cm, class_names, cm_path)

    errors_path = ARTIFACTS / "per_class_errors.txt"
    save_per_class_errors(confused, errors_path)

    metrics: dict[str, Any] = {
        "overall_accuracy": overall_accuracy,
        "macro_f1": float(macro["f1-score"]),
        "macro_precision": float(macro["precision"]),
        "macro_recall": float(macro["recall"]),
        "per_class": per_class,
        "most_confused_pairs": confused,
        "num_test_samples": int(y_true.size),
        "checkpoint": str(args.checkpoint),
        "note": _accuracy_note(args.data_dir),
    }

    if args.distribution_shift:
        print("Running distribution-shift analysis (basic, self-contained)...")
        shift = distribution_shift(model, test_samples, device)
        metrics["distribution_shift"] = shift
        for kind, acc in shift.items():
            print(f"  {kind:18s} accuracy={acc:.4f}")

    metrics_path = ARTIFACTS / "metrics.json"
    save_json(metrics_path, metrics)

    # --- Summary ---
    print("\n=== Evaluation summary ===")
    print(f"Test samples     : {metrics['num_test_samples']}")
    print(f"Overall accuracy : {overall_accuracy:.4f}")
    print(f"Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"Macro precision  : {metrics['macro_precision']:.4f}")
    print(f"Macro recall     : {metrics['macro_recall']:.4f}")
    if confused:
        print("Top confused pairs (true -> pred: count):")
        for p in confused:
            print(f"  {p['true']} -> {p['pred']}: {p['count']}")
    else:
        print("No off-diagonal confusions on this test split.")
    print(f"\nSaved metrics to            {metrics_path}")
    print(f"Saved confusion matrix to   {cm_path}")
    print(f"Saved per-class errors to   {errors_path}")
    print(f"\nNOTE: {_accuracy_note(args.data_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
