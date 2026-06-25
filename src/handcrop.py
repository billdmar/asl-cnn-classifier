"""MediaPipe hand detection + crop — the Python half of the crop contract.

This mirrors the browser implementation in ``web/lib/handcrop.ts`` so the crop
geometry is identical on both sides: detect the hand, take the landmarks'
bounding square, expand it by :data:`CROP_MARGIN` on each side, clamp to the
image, and crop. Cropping to the hand before classifying removes most of the
background dependence that hurts the model (trained on uniform images) on
cluttered real-world photos.

The crop math (:func:`crop_box_from_landmarks`) is pure and unit-tested without
the MediaPipe runtime; :func:`detect_and_crop` adds the actual detection. The
``mediapipe`` import is deferred so importing this module (and the rest of the
pipeline) does not hard-require the heavy dependency.

The shared geometry contract — square box, :data:`CROP_MARGIN` fraction — MUST
stay in sync with ``web/lib/handcrop.ts`` (``CROP_MARGIN`` there).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Fraction of the hand's bounding-box size added as margin on each side. Must
# match CROP_MARGIN in web/lib/handcrop.ts.
CROP_MARGIN = 0.35

# Default location of the MediaPipe hand-landmarker model (shared with the web
# app, which serves it as a static asset).
DEFAULT_MODEL_PATH = "web/public/mediapipe/hand_landmarker.task"


@dataclass(frozen=True)
class CropBox:
    """A normalized crop box in ``[0, 1]`` image coordinates."""

    x: float
    y: float
    width: float
    height: float


def crop_box_from_landmarks(
    landmarks: list[tuple[float, float]] | np.ndarray,
    margin: float = CROP_MARGIN,
) -> CropBox | None:
    """Compute a square, margined crop box from normalized hand landmarks.

    Pure and dependency-free (no MediaPipe needed) so it is directly testable.
    Returns ``None`` when there are no landmarks.

    Args:
        landmarks: Sequence of ``(x, y)`` points in ``[0, 1]`` image coords.
        margin: Margin fraction (defaults to :data:`CROP_MARGIN`).

    Returns:
        A square :class:`CropBox` clamped to the unit image, or ``None``.
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.size == 0:
        return None

    min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
    max_x, max_y = pts[:, 0].max(), pts[:, 1].max()

    w = max_x - min_x
    h = max_y - min_y
    side = max(w, h) * (1 + 2 * margin)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    x = max(0.0, cx - side / 2)
    y = max(0.0, cy - side / 2)
    box_side = side
    if x + box_side > 1:
        box_side = min(box_side, 1 - x)
    if y + box_side > 1:
        box_side = min(box_side, 1 - y)

    return CropBox(
        x=float(x), y=float(y), width=float(box_side), height=float(box_side)
    )


def crop_image(image: Image.Image, box: CropBox) -> Image.Image:
    """Crop a PIL image to a normalized :class:`CropBox` (returns the region)."""
    w, h = image.size
    left = int(round(box.x * w))
    top = int(round(box.y * h))
    right = int(round((box.x + box.width) * w))
    bottom = int(round((box.y + box.height) * h))
    # Guard against a degenerate zero-area box.
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return image.crop((left, top, right, bottom))


def _build_landmarker(model_path: str | Path) -> Any:
    """Create a MediaPipe HandLandmarker for still images (deferred import)."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_and_crop(
    image: Image.Image,
    landmarker: Any | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    margin: float = CROP_MARGIN,
) -> Image.Image | None:
    """Detect the hand in ``image`` and return the cropped region, or ``None``.

    Args:
        image: An RGB PIL image.
        landmarker: A pre-built MediaPipe HandLandmarker to reuse across calls;
            if ``None``, one is created from ``model_path`` for this call.
        model_path: Path to ``hand_landmarker.task`` (when ``landmarker`` is None).
        margin: Crop margin fraction.

    Returns:
        The cropped hand region as a PIL image, or ``None`` if no hand is found.
    """
    import mediapipe as mp

    own = landmarker is None
    if landmarker is None:
        landmarker = _build_landmarker(model_path)
    try:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        pts = [(lm.x, lm.y) for lm in result.hand_landmarks[0]]
        box = crop_box_from_landmarks(pts, margin=margin)
        if box is None:
            return None
        return crop_image(image, box)
    finally:
        if own:
            landmarker.close()
