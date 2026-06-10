"""Latency/throughput benchmark, preprocessing ablation, and robustness study.

This script characterizes the deployed model along three axes:

1. **Latency & throughput** — end-to-end per-frame latency (resize + ToTensor +
   normalize + forward + argmax), reported as mean/p50/p95/p99 (ms) and FPS.
   Always benchmarked on CPU; if an accelerator (CUDA/MPS) is available it is
   benchmarked too and printed side-by-side.
2. **Preprocessing ablation** — mean latency for five progressively reduced
   pipelines, to attribute cost to each stage. Saved as a bar chart.
3. **Distribution-shift characterization** — held-out test accuracy under five
   synthetic degradations (blur, JPEG, dark, bright, salt-and-pepper) vs. a
   clean baseline.

All preprocessing reuses :func:`src.dataset.get_eval_transforms` (single source
of truth). Run::

    python -m src.benchmark --num_frames 1000 --device auto --test_dir data/sample
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless backend — no display required.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import transforms  # noqa: E402

from src.dataset import (  # noqa: E402
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ASLDataset,
    get_eval_transforms,
    make_stratified_splits,
)
from src.infer_camera import load_checkpoint  # noqa: E402
from src.utils import get_device, save_json  # noqa: E402

DEFAULT_CHECKPOINT = "artifacts/checkpoints/best_model.pth"
ARTIFACTS = Path("artifacts")
WARMUP_FRAMES = 50


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
def _load_frames(source: str | None, num_frames: int) -> list[np.ndarray]:
    """Return a list of RGB ``uint8`` frames.

    If ``source`` is a directory of images, frames are loaded (cycled to reach
    ``num_frames``). Otherwise synthetic random 128×128×3 ``uint8`` frames are
    generated.
    """
    if source:
        src_dir = Path(source)
        paths = sorted(
            p
            for p in src_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if paths:
            loaded = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
            frames = [loaded[i % len(loaded)] for i in range(num_frames)]
            print(
                f"Loaded {len(loaded)} source images from {src_dir} (cycled to {num_frames})."
            )
            return frames

    rng = np.random.default_rng(42)
    print(f"Synthesizing {num_frames} random 128x128x3 uint8 frames.")
    return [
        rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(num_frames)
    ]


# --------------------------------------------------------------------------- #
# Latency / throughput
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _benchmark_device(
    model: nn.Module,
    frames: list[np.ndarray],
    transform: transforms.Compose,
    device: torch.device,
) -> dict[str, float]:
    """Measure END-TO-END per-frame latency on ``device``.

    Each timed iteration covers PIL→transform (resize/ToTensor/normalize) +
    forward + argmax. The first :data:`WARMUP_FRAMES` iterations are discarded.

    Returns:
        Dict with ``mean_ms``, ``p50_ms``, ``p95_ms``, ``p99_ms``, ``fps``.
    """
    model.to(device).eval()
    n_warmup = min(WARMUP_FRAMES, max(0, len(frames) - 1))

    # Warm-up (discarded).
    for i in range(n_warmup):
        frame = frames[i % len(frames)]
        tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
        logits = model(tensor)
        _ = logits.argmax(dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
        logits = model(tensor)
        _ = logits.argmax(dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(latencies_ms)
    mean_ms = float(arr.mean())
    return {
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "fps": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Preprocessing ablation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _time_pipeline(
    model: nn.Module,
    frames: list[np.ndarray],
    device: torch.device,
    per_frame: Callable[[np.ndarray], torch.Tensor],
    run_model: bool = True,
) -> float:
    """Return mean per-frame latency (ms) for a custom preprocessing closure."""
    n_warmup = min(WARMUP_FRAMES, max(0, len(frames) - 1))
    for i in range(n_warmup):
        tensor = per_frame(frames[i % len(frames)]).to(device)
        if run_model:
            _ = model(tensor).argmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()

    latencies: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        tensor = per_frame(frame).to(device)
        if run_model:
            _ = model(tensor).argmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(latencies))


def _ablation(
    model: nn.Module, frames: list[np.ndarray], device: torch.device
) -> list[dict[str, str | float]]:
    """Measure mean latency for five progressively reduced pipelines.

    Variants:
        1. ``full`` — resize + ToTensor + normalize + forward.
        2. ``skip_colorjitter`` — same as full (ColorJitter is *not* part of the
           eval transform, so this is a no-op confirming there is nothing to
           skip; reported for transparency).
        3. ``skip_resize`` — feed the frame at native size. Because the CustomCNN
           uses global average pooling it accepts arbitrary spatial sizes, so we
           can run the model on the raw frame; this isolates the resize cost
           honestly rather than faking it.
        4. ``skip_normalize`` — resize + ToTensor + forward (no Normalize).
        5. ``model_only`` — preprocess once outside the timed loop; time the
           forward + argmax alone.
    """
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    full_tf = get_eval_transforms()

    def full(frame: np.ndarray) -> torch.Tensor:
        return full_tf(Image.fromarray(frame)).unsqueeze(0)

    def skip_resize(frame: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(frame)
        return normalize(to_tensor(pil)).unsqueeze(0)

    def skip_normalize(frame: np.ndarray) -> torch.Tensor:
        return to_tensor(resize(Image.fromarray(frame))).unsqueeze(0)

    # Pre-build a single batch for the model-only measurement.
    pre = full(frames[0]).to(device)

    @torch.no_grad()
    def _model_only() -> float:
        n_warmup = min(WARMUP_FRAMES, max(0, len(frames) - 1))
        for _ in range(n_warmup):
            _ = model(pre).argmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
        lat: list[float] = []
        for _ in frames:
            start = time.perf_counter()
            _ = model(pre).argmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - start) * 1000.0)
        return float(np.mean(lat))

    results: list[dict[str, str | float]] = [
        {"stage": "full", "mean_ms": _time_pipeline(model, frames, device, full)},
        # ColorJitter is absent at eval — identical to full (documented no-op).
        {
            "stage": "skip_colorjitter",
            "mean_ms": _time_pipeline(model, frames, device, full),
        },
        {
            "stage": "skip_resize",
            "mean_ms": _time_pipeline(model, frames, device, skip_resize),
        },
        {
            "stage": "skip_normalize",
            "mean_ms": _time_pipeline(model, frames, device, skip_normalize),
        },
        {"stage": "model_only", "mean_ms": _model_only()},
    ]
    return results


def _save_ablation_chart(ablation: list[dict[str, str | float]], path: Path) -> None:
    """Save a bar chart of mean latency per ablation stage."""
    stages = [str(a["stage"]) for a in ablation]
    values = [float(a["mean_ms"]) for a in ablation]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(stages)), values, color="#4C78A8")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Preprocessing ablation — mean per-frame latency")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=20, ha="right")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Distribution-shift characterization
# --------------------------------------------------------------------------- #
def _degrade(image: Image.Image, kind: str) -> Image.Image:
    """Apply a synthetic degradation to a clean RGB PIL image."""
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
def _distribution_shift(
    model: nn.Module,
    test_dir: str,
    class_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Measure test accuracy under each synthetic degradation.

    Loads the held-out test split via ``make_stratified_splits`` + ``ASLDataset``
    and classifies each test image after applying each degradation. Uses
    ``zero_division=0`` semantics: if the test split is empty, accuracies are
    reported as ``0.0`` rather than dividing by zero.

    Returns:
        Mapping of degradation name → accuracy in ``[0, 1]``.
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

    try:
        _, _, test = make_stratified_splits(test_dir, class_names=class_names)
    except RuntimeError as exc:
        print(f"WARNING: distribution-shift skipped — {exc}")
        return {k: 0.0 for k in degradations}

    dataset = ASLDataset(samples=test, transform=transform, class_names=class_names)
    print(f"Distribution-shift over {len(dataset)} held-out test images.")

    results: dict[str, float] = {}
    for kind in degradations:
        correct = 0
        total = 0
        for filepath, label in dataset.samples:
            clean = Image.open(filepath).convert("RGB")
            degraded = _degrade(clean, kind)
            tensor = transform(degraded).unsqueeze(0).to(device)
            pred = int(model(tensor).argmax(dim=1).item())
            correct += int(pred == label)
            total += 1
        results[kind] = (correct / total) if total > 0 else 0.0  # zero_division=0
    return results


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> int:
    """Parse args, run all three studies, and persist artifacts/JSON."""
    parser = argparse.ArgumentParser(
        description="ASL model latency / robustness benchmark."
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Accelerator preference.",
    )
    parser.add_argument(
        "--num_frames", type=int, default=1000, help="Number of timed frames."
    )
    parser.add_argument(
        "--source", default=None, help="Optional directory of images to benchmark on."
    )
    parser.add_argument(
        "--test_dir", default="data/sample", help="Dir for distribution-shift accuracy."
    )
    args = parser.parse_args()

    # CPU is always benchmarked; the accelerator (if any) is determined by the
    # user's preference via get_device.
    cpu_device = torch.device("cpu")
    accel_device = get_device(args.device)
    accel_name = accel_device.type if accel_device.type in ("cuda", "mps") else "none"

    # Model + class names (shared loader; random-init fallback if no checkpoint).
    model, class_names = load_checkpoint(args.checkpoint, cpu_device)
    transform = get_eval_transforms()
    frames = _load_frames(args.source, args.num_frames)

    print(f"\n=== Latency benchmark (num_frames={len(frames)}) ===")
    cpu_stats = _benchmark_device(model, frames, transform, cpu_device)
    print(
        f"CPU   | mean {cpu_stats['mean_ms']:.3f} ms | p50 {cpu_stats['p50_ms']:.3f} "
        f"| p95 {cpu_stats['p95_ms']:.3f} | p99 {cpu_stats['p99_ms']:.3f} "
        f"| {cpu_stats['fps']:.1f} FPS"
    )

    gpu_stats: dict[str, float] | None = None
    if accel_name != "none":
        gpu_stats = _benchmark_device(model, frames, transform, accel_device)
        print(
            f"{accel_name.upper():5s} | mean {gpu_stats['mean_ms']:.3f} ms | p50 "
            f"{gpu_stats['p50_ms']:.3f} | p95 {gpu_stats['p95_ms']:.3f} | p99 "
            f"{gpu_stats['p99_ms']:.3f} | {gpu_stats['fps']:.1f} FPS"
        )
    else:
        print("No CUDA/MPS accelerator available — CPU-only results.")

    # Ablation runs on CPU for a stable, comparable baseline.
    model.to(cpu_device).eval()
    print("\n=== Preprocessing ablation (CPU) ===")
    ablation = _ablation(model, frames, cpu_device)
    for stage in ablation:
        print(f"  {stage['stage']:18s} {stage['mean_ms']:.3f} ms")
    ablation_path = ARTIFACTS / "benchmark_ablation.png"
    _save_ablation_chart(ablation, ablation_path)
    print(f"Saved ablation chart to {ablation_path}")

    print("\n=== Distribution-shift characterization (CPU) ===")
    shift = _distribution_shift(model, args.test_dir, class_names, cpu_device)
    for kind, acc in shift.items():
        print(f"  {kind:18s} accuracy={acc:.4f}")
    shift_path = ARTIFACTS / "distribution_shift.json"
    save_json(shift_path, shift)
    print(f"Saved distribution-shift results to {shift_path}")

    results = {
        "cpu_mean_ms": cpu_stats["mean_ms"],
        "cpu_p50_ms": cpu_stats["p50_ms"],
        "cpu_p95_ms": cpu_stats["p95_ms"],
        "cpu_p99_ms": cpu_stats["p99_ms"],
        "cpu_fps": cpu_stats["fps"],
        "gpu_mean_ms": gpu_stats["mean_ms"] if gpu_stats else None,
        "gpu_fps": gpu_stats["fps"] if gpu_stats else None,
        "device_accel": accel_name,
        "ablation": ablation,
        "distribution_shift": shift,
    }
    results_path = ARTIFACTS / "benchmark_results.json"
    save_json(results_path, results)
    print(f"\nSaved benchmark results to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
