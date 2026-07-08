"""Synthetic image degradations for robustness evaluation.

Applies controlled corruptions to clean images so that accuracy under
distribution shift (blur, compression, lighting, noise) can be measured
reproducibly. Used by both :mod:`src.eval` (the self-contained basic version)
and :mod:`src.benchmark` (the canonical full study).

Each degradation mimics a realistic failure mode:

* ``gaussian_blur`` — camera out of focus or motion blur.
* ``jpeg_q20`` — heavy lossy compression (e.g. low-bandwidth video stream).
* ``brightness_0.4`` — underexposure / low-light environment.
* ``brightness_1.8`` — overexposure / bright window behind the signer.
* ``salt_pepper_5pct`` — sensor noise or transmission errors.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

DEGRADATION_KINDS: tuple[str, ...] = (
    "clean",
    "gaussian_blur",
    "jpeg_q20",
    "brightness_0.4",
    "brightness_1.8",
    "salt_pepper_5pct",
)


def degrade(image: Image.Image, kind: str) -> Image.Image:
    """Apply a synthetic degradation to a clean RGB PIL image.

    Args:
        image: An RGB PIL image.
        kind: One of :data:`DEGRADATION_KINDS`.

    Returns:
        The degraded image (same size, RGB mode).

    Raises:
        ValueError: If ``kind`` is not recognized.
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
