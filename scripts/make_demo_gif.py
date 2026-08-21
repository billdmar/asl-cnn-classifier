"""Render a polished, HONEST demo GIF of the ASL inference pipeline.

The GIF cycles through several synthetic sample images, running each through the
*real* inference path used by :mod:`src.infer_camera` — centered ROI crop →
:func:`get_eval_transforms` preprocess → CNN forward → softmax top-1 — and draws
the ROI box, predicted letter, and confidence onto every frame.

Honesty notes (deliberate, do not remove):

* The committed ``data/sample/`` images are **synthetic** sanity fixtures, not
  real hands (see :mod:`src.make_sample_data`). A persistent banner says so.
* If no trained checkpoint exists, :func:`src.infer_camera.load_checkpoint`
  falls back to UNTRAINED random weights, so the predictions shown are
  *meaningless*. This script therefore demonstrates the **interface/pipeline**,
  not accuracy. A footer states this.

The result is a self-contained artifact regenerated from source — not a magic
binary. Run from the repo root::

    python -m scripts.make_demo_gif
    python -m scripts.make_demo_gif --device cpu --fps 1.2

Output defaults to ``docs/demo.gif``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from src.dataset import get_eval_transforms
from src.infer_camera import (
    _annotate,
    _center_roi,
    load_checkpoint,
    predict_roi,
)
from src.utils import get_device

# A small, varied set of sample letters so the GIF shows different inputs.
DEFAULT_LETTERS = ["A", "B", "C", "D", "L", "Y"]
SAMPLE_ROOT = Path("data/sample")
DEFAULT_OUTPUT = Path("docs/demo.gif")

# Frame composition. The annotated 200x200 sample is upscaled to this size so
# the overlays are legible in a README; banners frame it top and bottom.
CANVAS_W = 480
PANEL_H = 360  # the upscaled, annotated sample image
BANNER_H = 44  # top "synthetic input" banner
FOOTER_H = 56  # bottom honesty footer
CANVAS_H = BANNER_H + PANEL_H + FOOTER_H

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_centered(
    img: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw ``text`` horizontally centered on ``img`` at baseline ``y``."""
    (tw, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    x = max(0, (img.shape[1] - tw) // 2)
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def _title_card() -> np.ndarray:
    """Build the opening title-card frame (BGR uint8)."""
    card = np.full((CANVAS_H, CANVAS_W, 3), 18, dtype=np.uint8)  # near-black
    _put_centered(card, "ASL Classifier", 150, 1.3, (80, 220, 120), 3)
    _put_centered(card, "real-time inference pipeline", 200, 0.7, (220, 220, 220), 1)
    _put_centered(
        card, "ROI crop -> preprocess -> CNN -> top-1", 250, 0.55, (160, 160, 160), 1
    )
    return card


def _compose_frame(
    annotated_panel: np.ndarray, label: str, confidence: float
) -> np.ndarray:
    """Stack the top banner, upscaled annotated panel, and honesty footer."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 18, dtype=np.uint8)

    # Top banner: make the synthetic nature unmissable (amber on dark).
    canvas[:BANNER_H] = (12, 40, 60)
    _put_centered(
        canvas, "SYNTHETIC DEMO INPUT (not a real hand)", 28, 0.55, (60, 200, 255), 1
    )

    # Center the upscaled annotated panel horizontally.
    panel = cv2.resize(
        annotated_panel, (PANEL_H, PANEL_H), interpolation=cv2.INTER_NEAREST
    )
    x0 = (CANVAS_W - PANEL_H) // 2
    canvas[BANNER_H : BANNER_H + PANEL_H, x0 : x0 + PANEL_H] = panel

    # Footer: prediction line + the meaningfulness caveat.
    footer_top = BANNER_H + PANEL_H
    _put_centered(
        canvas,
        f"pred: {label}   conf: {confidence * 100:.1f}%",
        footer_top + 24,
        0.62,
        (120, 230, 140),
        2,
    )
    _put_centered(
        canvas,
        "untrained weights -> labels illustrative only",
        footer_top + 46,
        0.42,
        (150, 150, 150),
        1,
    )
    return canvas


def build_frames(
    letters: list[str],
    checkpoint: str,
    device_str: str,
    roi_size: int,
) -> list[np.ndarray]:
    """Run inference on one sample per letter and return composed RGB frames.

    Args:
        letters: Class folder names under ``data/sample/`` to demo.
        checkpoint: Path to a ``.pth`` checkpoint (falls back to untrained).
        device_str: Device selector passed to :func:`get_device`.
        roi_size: Centered-ROI side length (matches the camera default).

    Returns:
        A list of RGB ``uint8`` frames (title card first), ready for imageio.
    """
    device = get_device(device_str)
    model, class_names = load_checkpoint(checkpoint, device)
    transform = get_eval_transforms()

    frames: list[np.ndarray] = [_title_card()]
    for letter in letters:
        img_path = SAMPLE_ROOT / letter / "0.png"
        if not img_path.exists():
            print(f"WARNING: sample '{img_path}' missing — skipping.")
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"WARNING: could not read '{img_path}' — skipping.")
            continue

        h, w = frame.shape[:2]
        roi_box = _center_roi(h, w, roi_size)
        x1, y1, x2, y2 = roi_box
        label, confidence = predict_roi(
            frame[y1:y2, x1:x2], model, transform, device, class_names
        )
        annotated = _annotate(frame.copy(), roi_box, label, confidence)
        frames.append(_compose_frame(annotated, label, confidence))
        print(f"{letter}/0.png -> pred={label} conf={confidence:.4f}")

    # Convert BGR (OpenCV) -> RGB (imageio) once at the end.
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]


def main() -> int:
    """Parse args, build frames, and write the looping GIF."""
    parser = argparse.ArgumentParser(description="Render the demo GIF.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/best_model.pth",
        help="Checkpoint path (untrained fallback if missing).",
    )
    parser.add_argument(
        "--letters",
        nargs="+",
        default=DEFAULT_LETTERS,
        help="Sample class folders to include.",
    )
    parser.add_argument("--device", default="cpu", help="Compute device.")
    parser.add_argument(
        "--roi_size", type=int, default=300, help="Centered ROI side (px)."
    )
    parser.add_argument("--fps", type=float, default=1.2, help="GIF playback rate.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output .gif path."
    )
    args = parser.parse_args()

    frames = build_frames(args.letters, args.checkpoint, args.device, args.roi_size)
    if len(frames) < 2:
        print("ERROR: no sample frames were produced.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Hold the title card a little longer than the per-letter frames.
    durations = [1.6] + [1.0 / args.fps] * (len(frames) - 1)
    # imageio's mimsave stub types `ims` as an invariant list; our list of
    # ndarray frames is valid at runtime, so ignore the variance complaint.
    imageio.mimsave(str(args.output), frames, duration=durations, loop=0)  # type: ignore[arg-type]
    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {args.output} ({len(frames)} frames, {size_kb:.0f} KB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
