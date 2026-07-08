"""Export a trained (or random-init fallback) model to ONNX.

The model is loaded via :func:`src.infer_camera.load_checkpoint`, so this script
runs even before a real checkpoint exists (it falls back to an UNTRAINED
``custom_cnn`` and prints a clear warning — see ``load_checkpoint``). The export
uses a **dynamic batch axis** with a fixed ``3×128×128`` spatial input, matching
:data:`src.dataset.IMAGE_SIZE`, so the resulting graph accepts any batch size at
inference time while preprocessing stays identical to training/eval.

Example::

    python -m src.export_onnx --checkpoint artifacts/checkpoints/best_model.pth \
        --output artifacts/model.onnx --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.dataset import IMAGE_SIZE
from src.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
from src.utils import get_device

DEFAULT_OUTPUT = "artifacts/model.onnx"
OPSET_VERSION = 17


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    device: torch.device,
    image_size: int = IMAGE_SIZE,
) -> Path:
    """Export ``model`` to ONNX with a dynamic batch axis.

    A single dummy ``(1, 3, image_size, image_size)`` input drives the trace.
    The ``batch`` dimension of both the input and output is marked dynamic so
    the exported graph serves arbitrary batch sizes; the spatial dims stay fixed
    at ``image_size``.

    Args:
        model: Module in eval mode (callers should pass a loaded model).
        output_path: Destination ``.onnx`` path (parents are created).
        device: Device the dummy input is allocated on.
        image_size: Fixed spatial side length (default :data:`IMAGE_SIZE`).

    Returns:
        The resolved output :class:`~pathlib.Path`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        # Use the stable TorchScript-based exporter (the newer dynamo path
        # pulls in an extra ``onnxscript`` dependency we don't need here).
        dynamo=False,
    )
    return output_path


def main() -> int:
    """Parse CLI args, load the model, and export it to ONNX."""
    parser = argparse.ArgumentParser(description="Export an ASL model to ONNX.")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Destination .onnx path."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device used to trace the export (CPU is fine for ONNX export).",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    model, class_names = load_checkpoint(args.checkpoint, device)
    out = export_to_onnx(model, args.output, device)
    size_bytes = out.stat().st_size
    print(
        f"Exported ONNX model to {out} "
        f"({size_bytes} bytes, {len(class_names)} classes, opset {OPSET_VERSION})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
