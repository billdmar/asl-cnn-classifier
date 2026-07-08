"""Multi-backend latency/throughput benchmark on CPU (and MPS if available).

Compares end-to-end per-frame inference latency across three serving backends on
the same preprocessed inputs:

1. **PyTorch FP32** — the eager ``nn.Module`` forward pass.
2. **ONNX Runtime** — the model exported via :mod:`src.export_onnx`, run through
   an ``onnxruntime`` ``CPUExecutionProvider`` session.
3. **INT8 quantized** — the dynamically quantized model from :mod:`src.quantize`.

PyTorch FP32 is additionally benchmarked on Apple-Silicon MPS *only* when
``torch.backends.mps.is_available()`` (it is skipped gracefully otherwise — e.g.
CPU-only CI). For every backend we report mean / p50 / p95 / p99 latency (ms) and
throughput (FPS). Preprocessing reuses :func:`src.dataset.get_eval_transforms`
(single source of truth), and the same frame set feeds every backend so the
comparison is apples-to-apples.

Results are written to ``artifacts/backend_benchmark.json`` and printed as a
markdown table. The model loads via :func:`src.infer_camera.load_checkpoint`
(random-init fallback when no checkpoint exists).

Run::

    python -m src.benchmark_backends --num_frames 200 --source data/sample
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.benchmark import WARMUP_FRAMES, _load_frames
from src.dataset import IMAGE_SIZE, get_eval_transforms
from src.export_onnx import export_to_onnx
from src.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
from src.quantize import quantize_dynamic_int8
from src.utils import save_json

ARTIFACTS = Path("artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "backend_benchmark.json"


def _summarize(latencies_ms: list[float]) -> dict[str, float]:
    """Reduce a list of per-frame latencies (ms) to summary statistics."""
    arr = np.asarray(latencies_ms)
    mean_ms = float(arr.mean())
    return {
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "fps": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


@torch.no_grad()
def _benchmark_torch(
    model: nn.Module,
    frames: list[np.ndarray],
    transform: transforms.Compose,
    device: torch.device,
) -> dict[str, float]:
    """End-to-end latency of the eager PyTorch forward pass on ``device``."""
    model.to(device).eval()
    n_warmup = min(WARMUP_FRAMES, max(0, len(frames) - 1))
    for i in range(n_warmup):
        tensor = transform(Image.fromarray(frames[i % len(frames)])).unsqueeze(0)
        _ = model(tensor.to(device)).argmax(dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()

    latencies: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
        _ = model(tensor).argmax(dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return _summarize(latencies)


def _benchmark_onnx(
    session: ort.InferenceSession,
    frames: list[np.ndarray],
    transform: transforms.Compose,
) -> dict[str, float]:
    """End-to-end latency of an ONNX Runtime session (CPU provider)."""
    input_name = session.get_inputs()[0].name
    n_warmup = min(WARMUP_FRAMES, max(0, len(frames) - 1))
    for i in range(n_warmup):
        tensor = transform(Image.fromarray(frames[i % len(frames)])).unsqueeze(0)
        _ = session.run(None, {input_name: tensor.numpy()})

    latencies: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        tensor = transform(Image.fromarray(frame)).unsqueeze(0)
        logits = session.run(None, {input_name: tensor.numpy()})[0]
        _ = int(np.argmax(logits, axis=1)[0])
        latencies.append((time.perf_counter() - start) * 1000.0)
    return _summarize(latencies)


def run_backends(
    model: nn.Module,
    frames: list[np.ndarray],
    transform: transforms.Compose,
    workdir: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Benchmark all backends and return ``{backend_name: stats}``.

    Always runs PyTorch FP32 (CPU), ONNX Runtime (CPU), and INT8 quantized
    (CPU). Adds ``pytorch_fp32_mps`` only when MPS is available; otherwise that
    key is simply absent (graceful skip). The intermediate ONNX export goes to a
    private temp dir (cleaned up automatically); ``workdir`` is accepted for
    backward compatibility but ignored.
    """
    cpu = torch.device("cpu")
    results: dict[str, dict[str, float]] = {}

    # 1) PyTorch FP32 on CPU.
    results["pytorch_fp32_cpu"] = _benchmark_torch(model, frames, transform, cpu)

    # 2) ONNX Runtime on CPU — export the current model, then run a session.
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "bench_model.onnx"
        export_to_onnx(model, onnx_path, cpu, image_size=IMAGE_SIZE)
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        results["onnxruntime_cpu"] = _benchmark_onnx(session, frames, transform)

    # 3) INT8 dynamically-quantized model on CPU.
    quantized = quantize_dynamic_int8(model)
    results["int8_dynamic_cpu"] = _benchmark_torch(quantized, frames, transform, cpu)

    # 4) PyTorch FP32 on MPS — only when truly available (skipped on CPU-only).
    mps_ok = bool(getattr(torch.backends, "mps", None)) and (
        torch.backends.mps.is_available()
    )
    if mps_ok:
        results["pytorch_fp32_mps"] = _benchmark_torch(
            model, frames, transform, torch.device("mps")
        )

    return results


def format_markdown_table(results: dict[str, dict[str, float]]) -> str:
    """Render the per-backend results as a markdown table string."""
    header = (
        "| Backend | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | FPS |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    rows = [
        f"| {name} | {s['mean_ms']:.3f} | {s['p50_ms']:.3f} | "
        f"{s['p95_ms']:.3f} | {s['p99_ms']:.3f} | {s['fps']:.1f} |"
        for name, s in results.items()
    ]
    return "\n".join([header, *rows])


def main() -> int:
    """Parse args, benchmark every backend, print a table, and persist JSON."""
    parser = argparse.ArgumentParser(
        description="Multi-backend (PyTorch / ONNX / INT8) latency benchmark."
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--num_frames", type=int, default=200, help="Number of timed frames."
    )
    parser.add_argument(
        "--source", default=None, help="Optional directory of images to benchmark on."
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Destination JSON path."
    )
    args = parser.parse_args()

    # Loading + all backends run on CPU (MPS added inside run_backends if present).
    model, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    transform = get_eval_transforms()
    frames = _load_frames(args.source, args.num_frames)

    out_path = Path(args.output)
    print(f"\n=== Multi-backend benchmark (num_frames={len(frames)}) ===")
    results = run_backends(model, frames, transform, out_path.parent)

    table = format_markdown_table(results)
    print(table)

    mps_available = bool(getattr(torch.backends, "mps", None)) and (
        torch.backends.mps.is_available()
    )
    payload = {
        "num_frames": len(frames),
        "mps_available": mps_available,
        "backends": results,
        "markdown_table": table,
    }
    save_json(out_path, payload)
    print(f"\nSaved backend benchmark results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
