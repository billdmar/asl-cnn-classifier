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

Temperature scaling (Guo et al., 2017) is wired in via :func:`fit_temperature`,
but it is INERT by default: unless ``--fit_temperature`` is passed *and* the run
uses a real (non-sample) deployment-like data dir, the exported temperature is
``1.0`` (the identity, no change). This is deliberate — on the clean benchmark
the model is UNDER-confident (mean_confidence < accuracy), so a temperature fit
on that split would be ``T < 1`` (sharpening) and would make real-world false
positives MORE confident. The machinery is unit-tested; the actual fit waits for
deployment-like data in a later phase.

Run, e.g.::

    python -m src.calibration --checkpoint artifacts/checkpoints/best_model.pth \
        --data_dir data/sample --device cpu

    # Write the web inference calibration file with T=1.0 merged into the real ECE:
    python -m src.calibration --checkpoint artifacts/checkpoints/best_model.pth \
        --data_dir data/asl_real \
        --inference_out web/public/model/calibration.json
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
    _list_samples,
    get_class_names,
    get_eval_transforms,
    get_union_class_names,
    make_stratified_splits,
)
from src.infer_camera import load_checkpoint  # noqa: E402
from src.train import _normalize_data_dirs  # noqa: E402
from src.utils import get_device, load_json, save_json, set_seed  # noqa: E402

DEFAULT_CHECKPOINT = "artifacts/checkpoints/best_model.pth"
ARTIFACTS = Path("artifacts")

SAMPLE_DATA_NOTE = (
    "ECE and reliability diagram on the tiny synthetic data/sample fixture with "
    "an untrained model are wiring demonstrations, not a meaningful calibration "
    "measurement. The ECE math is correct and unit-tested; only the inputs here "
    "are synthetic. Train on the full ASL Alphabet dataset for real numbers."
)

REAL_DATA_NOTE = (
    "ECE measured on the held-out test split of the real ASL dataset with the "
    "trained MobileNetV2 checkpoint — a meaningful calibration assessment. "
    "Compare 'mean_confidence' against 'accuracy' to read the direction of "
    "miscalibration; the reliability diagram shows where per-bin confidence and "
    "accuracy diverge."
)


def _calibration_note(data_dir: str, trained: bool) -> str:
    """Pick an honest note based on the actual data and checkpoint used.

    The synthetic-fixture disclaimer only applies when the run genuinely used
    the ``data/sample`` wiring fixture or an untrained model; otherwise the
    numbers are a real measurement and must be described as such.
    """
    is_sample = "sample" in Path(data_dir).parts
    if is_sample or not trained:
        return SAMPLE_DATA_NOTE
    return REAL_DATA_NOTE


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
def collect_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over ``loader`` and return raw ``(logits, labels)``.

    ``logits`` is an ``(N, num_classes)`` array of the model's pre-softmax
    outputs — exactly what temperature scaling needs to fit ``T``. All tensors
    are moved to CPU before numpy conversion (MPS-safe).
    """
    model.eval()
    logits_chunks: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for inputs, targets, _paths in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        logits_chunks.append(logits.detach().cpu().numpy())
        trues.append(targets.detach().cpu().numpy())

    if not trues:
        return np.empty((0, 0), dtype=float), np.array([], dtype=int)
    return np.concatenate(logits_chunks, axis=0), np.concatenate(trues)


def _logits_to_predictions(
    logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce raw logits to ``(top1_confidence, predicted_index)`` via softmax."""
    if logits.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty.astype(int)
    t = torch.from_numpy(np.asarray(logits, dtype=np.float64))
    probs = torch.softmax(t, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return conf.numpy(), pred.numpy()


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over ``loader`` and return ``(confidences, preds, labels)``.

    Confidence is the softmax probability of the predicted (argmax) class. All
    tensors are moved to CPU before numpy conversion (MPS-safe). Implemented on
    top of :func:`collect_logits` so the logits are also available for fitting a
    calibration temperature without a second forward pass.
    """
    logits, labels = collect_logits(model, loader, device)
    if labels.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty.astype(int), empty.astype(int)
    confs, preds = _logits_to_predictions(logits)
    return confs, preds, labels


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.5,
) -> float:
    """Fit a single scalar temperature ``T`` by NLL minimization (Guo et al. 2017).

    Optimizes ``T`` so that ``softmax(logits / T)`` minimizes the cross-entropy
    against ``labels``. To keep ``T`` strictly positive we optimize an
    unconstrained ``log_T`` parameter and use ``T = exp(log_T)``; LBFGS handles
    the small smooth problem well.

    Args:
        logits: ``(N, num_classes)`` raw pre-softmax outputs.
        labels: ``(N,)`` ground-truth class indices.
        max_iter: Maximum LBFGS iterations.
        lr: LBFGS learning rate.

    Returns:
        The fitted temperature ``T > 0``. Returns ``1.0`` (the identity, no
        change) when there is nothing to fit (empty input).
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.size == 0 or labels.size == 0:
        return 1.0

    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(0) = 1
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(logits_t / torch.exp(log_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(torch.exp(log_t.detach()).item())


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


def _resolve_temperature(
    fit_requested: bool,
    data_dir: str,
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, str]:
    """Decide the exported temperature, defaulting INERT to ``1.0``.

    Only fits ``T`` when ``--fit_temperature`` was explicitly passed AND the data
    dir is a real (non-``sample``) deployment-like set. This guards against the
    known trap: fitting on the clean benchmark (where the model is
    under-confident) would learn ``T < 1`` and sharpen real-world false
    positives. See the module docstring.

    Returns ``(temperature, temperature_fit_on)`` where ``temperature_fit_on`` is
    ``"none"`` for the inert default or the data dir actually fit on.
    """
    is_sample = "sample" in Path(data_dir).parts
    if fit_requested and not is_sample and labels.size > 0:
        return fit_temperature(logits, labels), str(data_dir)
    return 1.0, "none"


def write_inference_calibration(
    path: Path, temperature: float, temperature_fit_on: str
) -> None:
    """Merge ``temperature`` into the web inference calibration file at ``path``.

    The web app fetches this file (``web/public/model/calibration.json``) and
    divides logits by ``temperature`` before softmax. We preserve any existing
    real-ECE payload already in the file and only set/refresh the temperature
    fields, so the measured ECE and reliability bins are not lost.
    """
    existing: dict[str, object] = {}
    if path.exists():
        loaded = load_json(path)
        if isinstance(loaded, dict):
            existing = loaded
    existing["temperature"] = temperature
    existing["temperature_fit_on"] = temperature_fit_on
    existing["temperature_note"] = (
        "Temperature scaling is WIRED but inert (T=1.0, identity) until it is fit "
        "on deployment-like data. A fit on the clean benchmark would sharpen "
        "(T<1) and is intentionally not shipped."
        if temperature == 1.0
        else f"Temperature fit on {temperature_fit_on} (Guo et al., 2017)."
    )
    save_json(path, existing)


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
    parser.add_argument(
        "--fit_temperature",
        action="store_true",
        help=(
            "Fit a calibration temperature T on this split and include it in the "
            "output. OFF by default; a T fit on the clean benchmark would sharpen "
            "(T<1) and is NOT shipped. Only meaningful on deployment-like data."
        ),
    )
    parser.add_argument(
        "--inference_out",
        default=None,
        help=(
            "Optional path to write/merge the web inference calibration file "
            "(e.g. web/public/model/calibration.json). The fitted temperature "
            "(or 1.0) is merged into the existing real ECE payload there."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Load the model, compute ECE on the test split, and write artifacts."""
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Mirror train.py's single-vs-multi-dir handling so a model trained on a
    # merged union is calibrated on the matching merged held-out split.
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

    trained = Path(args.checkpoint).exists()
    model, _ = load_checkpoint(args.checkpoint, device)
    logits, labels = collect_logits(model, test_loader, device)
    confidences, predictions = _logits_to_predictions(logits)
    note = _calibration_note(args.data_dir, trained)

    ece, bin_stats = compute_ece(confidences, predictions, labels, n_bins=args.n_bins)

    # Temperature scaling: INERT by default. Only fit when explicitly requested
    # AND on a real (non-sample) data dir; never auto-fit on the clean benchmark.
    temperature, temperature_fit_on = _resolve_temperature(
        args.fit_temperature, args.data_dir, logits, labels
    )

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
        "data_dir": str(args.data_dir),
        "temperature": temperature,
        "temperature_fit_on": temperature_fit_on,
        "note": note,
    }
    calibration_path = ARTIFACTS / "calibration.json"
    save_json(calibration_path, payload)

    if args.inference_out is not None:
        write_inference_calibration(
            Path(args.inference_out), temperature, temperature_fit_on
        )

    print("\n=== Calibration summary ===")
    print(f"Test samples    : {payload['num_test_samples']}")
    print(f"Accuracy        : {payload['accuracy']:.4f}")
    print(f"Mean confidence : {payload['mean_confidence']:.4f}")
    print(f"ECE ({args.n_bins} bins)    : {ece:.4f}")
    print(f"Temperature     : {temperature:.4f} (fit_on={temperature_fit_on})")
    print(f"\nSaved calibration to       {calibration_path}")
    print(f"Saved reliability diagram  {diagram_path}")
    if args.inference_out is not None:
        print(f"Wrote inference calibration {args.inference_out}")
    print(f"\nNOTE: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
