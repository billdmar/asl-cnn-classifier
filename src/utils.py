"""Shared utilities: reproducible seeding, device selection, and JSON I/O.

This module is the single source of truth for two cross-cutting concerns that
every other script depends on:

* ``set_seed`` — deterministic runs (seeds ``random``, ``numpy``, ``torch``).
* ``get_device`` — picks CUDA, then Apple-Silicon MPS, then CPU.

Keeping these here avoids subtle drift between training, evaluation, and
benchmarking (e.g. one script seeding differently than another).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for reproducible runs.

    On CUDA this also enables deterministic cuDNN. On Apple-Silicon MPS, full
    bit-exact reproducibility is *not* guaranteed (several MPS kernels are
    nondeterministic), so the strict "identical accuracy on rerun" check is
    scoped to CPU in the test suite.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make hash-based ops (e.g. set ordering) deterministic across processes.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuDNN determinism only matters on CUDA; harmless no-ops elsewhere.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prefer deterministic algorithms where available; warn (don't crash) when
    # a deterministic implementation is missing (common on MPS).
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(prefer: str = "auto") -> torch.device:
    """Return the best available compute device.

    Args:
        prefer: ``"auto"`` (default) picks CUDA → MPS → CPU. ``"cpu"`` forces
            CPU. ``"cuda"``/``"mps"`` request a specific accelerator, falling
            back to CPU if it is unavailable.
    """
    prefer = prefer.lower()
    if prefer == "cpu":
        return torch.device("cpu")

    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if prefer == "cuda":
        return torch.device("cuda") if cuda_ok else torch.device("cpu")
    if prefer == "mps":
        return torch.device("mps") if mps_ok else torch.device("cpu")

    # auto
    if cuda_ok:
        return torch.device("cuda")
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


def save_json(path: str | Path, data: Any) -> None:
    """Write ``data`` to ``path`` as pretty-printed JSON, creating parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)


def load_json(path: str | Path) -> Any:
    """Load and return JSON content from ``path``."""
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)
