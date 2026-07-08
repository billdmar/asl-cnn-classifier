"""Checkpoint loading: the single canonical way to materialize a trained model.

Every script that needs a model — evaluation, benchmarking, serving, inference,
explainability — loads it through :func:`load_checkpoint`. This module is the
sole owner of:

* The checkpoint file schema:
  ``{"model_state_dict", "arch", "class_names", "config", "val_accuracy"}``.
* The random-init fallback (so scripts remain runnable before any real
  checkpoint has been produced).
* :data:`DEFAULT_CHECKPOINT` — the canonical path ``artifacts/checkpoints/best_model.pth``.

Extracting checkpoint logic here (rather than keeping it inside
``infer_camera.py``) breaks a dependency-graph coupling: evaluation,
calibration, and serving no longer depend on the camera module.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.dataset import get_class_names
from src.model import build_model

DEFAULT_CHECKPOINT = "artifacts/checkpoints/best_model.pth"


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[nn.Module, list[str]]:
    """Load a model + class names from a training checkpoint.

    The checkpoint schema is
    ``{"model_state_dict", "arch", "class_names", "config", "val_accuracy"}``.
    The model is rebuilt from the checkpoint's recorded ``arch`` and moved to
    ``device`` in eval mode.

    If ``path`` does not exist, this falls back to an **untrained**
    ``custom_cnn`` with random weights and prints a clear warning, so the
    inference and benchmark scripts remain runnable before any real checkpoint
    has been produced.

    Args:
        path: Path to the ``.pth`` checkpoint.
        device: Target compute device.

    Returns:
        A tuple of ``(model, class_names)`` with the model in eval mode on
        ``device``.
    """
    path = Path(path)
    if not path.exists():
        print(
            f"WARNING: checkpoint '{path}' not found — falling back to an "
            "UNTRAINED custom_cnn with random weights. Predictions will be "
            "meaningless; train a model to produce real results."
        )
        class_names = get_class_names()
        model = build_model(
            "custom_cnn", num_classes=len(class_names), pretrained=False
        )
        model.to(device).eval()
        return model, class_names

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    arch = checkpoint["arch"]
    class_names = checkpoint.get("class_names") or get_class_names()
    model = build_model(arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    val_acc = checkpoint.get("val_accuracy")
    acc_str = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else "n/a"
    print(f"Loaded checkpoint '{path}' (arch={arch}, val_accuracy={acc_str}).")
    return model, class_names
