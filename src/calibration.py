"""Model calibration: Expected Calibration Error (ECE) and reliability diagram.

A well-calibrated classifier's predicted confidence should match its empirical
accuracy: among predictions made with ~70% confidence, ~70% should be correct.
This module quantifies that with the *Expected Calibration Error* and visualizes
it with a *reliability diagram*.

ECE (Guo et al., 2017) partitions the ``[0, 1]`` confidence range into ``M``
equal-width bins. For each bin ``B_m`` it computes the mean accuracy and mean
confidence of the predictions whose top-class confidence falls in that bin, then
takes the sample-weighted average of the gap::

    ECE = sum_m (|B_m| / N) * | acc(B_m) - conf(B_m) |

This is REAL math and is unit-tested against hand-constructed inputs with known
analytic ECE (see ``tests/test_calibration.py``).

Preprocessing reuses :func:`src.dataset.get_eval_transforms`, the held-out split
reuses :func:`src.dataset.make_stratified_splits`, and checkpoint loading reuses
:func:`src.infer_camera.load_checkpoint`.

CRITICAL HONESTY NOTE: with no trained checkpoint, ``load_checkpoint`` falls
back to a RANDOM-init model over the tiny synthetic ``data/sample`` fixture, so
the ECE value and reliability diagram WRITTEN TO DISK are wiring demonstrations,
not a meaningful calibration measurement. The ECE *computation itself* is
correct (and unit-tested); only the inputs here are synthetic. Train on the full
ASL Alphabet dataset for a real calibration assessment.

Run, e.g.::

    python -m src.calibration --checkpoint artifacts/checkpoints/best_model.pth \
        --data_dir data/sample --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — no display required.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.dataset import (  # noqa: E402
    ASLDataset,
    get_class_names,
    get_eval_transforms,
    make_stratified_splits,
)
from src.infer_camera import load_checkpoint  # noqa: E402
from src.utils import get_device, save_json, set_seed  # noqa: E402

DEFAULT_CHECKPOINT = "artifacts/checkpoints/best_model.pth"
ARTIFACTS = Path("artifacts")

SAMPLE_DATA_NOTE = (
    "ECE and reliability diagram on the tiny synthetic data/sample fixture with "
    "an untrained model are wiring demonstrations, not a meaningful calibration "
    "measurement. The ECE math is correct and unit-tested; only the inputs here "
    "are synthetic. Train on the full ASL Alphabet dataset for real numbers."
)


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, dict[str, list[float]]]:
    """Compute the Expected Calibration Error and per-bin statistics.

    Uses ``n_bins`` equal-width bins over ``[0, 1]``. Following the standard
    convention, a sample falls in bin ``m`` if its confidence is in
    ``(edge_{m-1}, edge_m]`` (the very first bin is closed on the left so a
    confidence of exactly ``0`` is still counted).

    Args:
        confidences: ``(N,)`` top-class confidence for each prediction, in
            ``[0, 1]``.
        predictions: ``(N,)`` predicted class indices.
        labels: ``(N,)`` ground-truth class indices.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ``(ece, bin_stats)`` where ``ece`` is the scalar calibration error and
        ``bin_stats`` maps ``"bin_lowers"``, ``"bin_uppers"``, ``"bin_acc"``,
        ``"bin_conf"``, ``"bin_count"`` to per-bin lists (empty bins report
        ``0.0`` accuracy/confidence and ``0`` count).
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    n = confidences.shape[0]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    correct = (predictions == labels).astype(np.float64)

    bin_lowers: list[float] = []
    bin_uppers: list[float] = []
    bin_acc: list[float] = []
    bin_conf: list[float] = []
    bin_count: list[float] = []

    ece = 0.0
    for m in range(n_bins):
        lo, hi = bin_edges[m], bin_edges[m + 1]
        # (lo, hi], with the first bin closed on the left to capture conf == 0.
        if m == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())

        if count > 0:
            acc = float(correct[in_bin].mean())
            conf = float(confidences[in_bin].mean())
            ece += (count / n) * abs(acc - conf) if n > 0 else 0.0
        else:
            acc = 0.0
            conf = 0.0

        bin_lowers.append(float(lo))
        bin_uppers.append(float(hi))
        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_count.append(count)

    bin_stats = {
        "bin_lowers": bin_lowers,
        "bin_uppers": bin_uppers,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_count": bin_count,
    }
    return float(ece), bin_stats


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over ``loader`` and return ``(confidences, preds, labels)``.

    Confidence is the softmax probability of the predicted (argmax) class. All
    tensors are moved to CPU before numpy conversion (MPS-safe).
    """
    model.eval()
    confs: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for inputs, targets, _paths in loader:
        inputs = inputs.to(device, non_blocking=True)
        probs = torch.softmax(model(inputs), dim=1)
        conf, pred = torch.max(probs, dim=1)
        confs.append(conf.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        trues.append(targets.detach().cpu().numpy())

    if not trues:
        empty = np.array([], dtype=float)
        return empty, empty.astype(int), empty.astype(int)
    return np.concatenate(confs), np.concatenate(preds), np.concatenate(trues)


def save_reliability_diagram(
    bin_stats: dict[str, list[float]], ece: float, path: Path
) -> None:
    """Save a reliability diagram (per-bin accuracy vs. confidence)."""
    lowers = np.asarray(bin_stats["bin_lowers"])
    uppers = np.asarray(bin_stats["bin_uppers"])
    accs = np.asarray(bin_stats["bin_acc"])
    counts = np.asarray(bin_stats["bin_count"])
    centers = (lowers + uppers) / 2.0
    width = float(uppers[0] - lowers[0]) if len(uppers) else 0.1

    fig, ax = plt.subplots(figsize=(6, 6))
    # Perfect-calibration reference line.
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    # Only draw bars for non-empty bins so empty bins don't read as 0% accuracy.
    nonempty = counts > 0
    ax.bar(
        centers[nonempty],
        accs[nonempty],
        width=width * 0.9,
        edgecolor="black",
        color="#3b78c2",
        alpha=0.8,
        label="Accuracy in bin",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.4f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute ECE and a reliability diagram for the ASL classifier."
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
        "--n_bins", type=int, default=10, help="Number of confidence bins."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed (must match training to recreate the split).",
    )
    return parser.parse_args()


def main() -> int:
    """Load the model, compute ECE on the test split, and write artifacts."""
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    class_names = get_class_names(args.data_dir)
    _train, _val, test_samples = make_stratified_splits(
        args.data_dir, seed=args.seed, class_names=class_names
    )
    print(f"Held-out test split: {len(test_samples)} samples.")

    test_ds = ASLDataset(
        samples=test_samples,
        transform=get_eval_transforms(),
        class_names=class_names,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model, _ = load_checkpoint(args.checkpoint, device)
    confidences, predictions, labels = collect_predictions(model, test_loader, device)

    ece, bin_stats = compute_ece(confidences, predictions, labels, n_bins=args.n_bins)

    diagram_path = ARTIFACTS / "reliability_diagram.png"
    save_reliability_diagram(bin_stats, ece, diagram_path)

    payload = {
        "ece": ece,
        "n_bins": args.n_bins,
        "num_test_samples": int(labels.size),
        "mean_confidence": (
            float(np.mean(confidences)) if confidences.size > 0 else 0.0
        ),
        "accuracy": (float(np.mean(predictions == labels)) if labels.size > 0 else 0.0),
        "bins": bin_stats,
        "checkpoint": str(args.checkpoint),
        "note": SAMPLE_DATA_NOTE,
    }
    calibration_path = ARTIFACTS / "calibration.json"
    save_json(calibration_path, payload)

    print("\n=== Calibration summary ===")
    print(f"Test samples    : {payload['num_test_samples']}")
    print(f"Accuracy        : {payload['accuracy']:.4f}")
    print(f"Mean confidence : {payload['mean_confidence']:.4f}")
    print(f"ECE ({args.n_bins} bins)    : {ece:.4f}")
    print(f"\nSaved calibration to       {calibration_path}")
    print(f"Saved reliability diagram  {diagram_path}")
    print(f"\nNOTE: {SAMPLE_DATA_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
