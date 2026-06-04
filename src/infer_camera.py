"""Real-time (and static-image) ASL inference with OpenCV.

This script loads a trained checkpoint and classifies a hand sign cropped from
a centered region of interest (ROI). It supports three sources:

* ``--source 0`` — live webcam (interactive window with FPS overlay).
* ``--source path/to/video.mp4`` — a video file (same interactive window).
* ``--source path/to/image.png`` — a single image, run once and saved
  annotated, with **no GUI calls at all** (CI / headless safe).

Preprocessing reuses :func:`src.dataset.get_eval_transforms` so that camera
frames are preprocessed *identically* to training/validation — there is no
second copy of the resize/normalize logic anywhere.

Examples::

    python -m src.infer_camera --source 0
    python -m src.infer_camera --source data/sample/A/0.png --device cpu
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from src.dataset import get_class_names, get_eval_transforms
from src.model import build_model
from src.utils import get_device, load_json

# File extensions we treat as still images vs. video containers.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

DEFAULT_CHECKPOINT = "artifacts/checkpoints/best_model.pth"
SNAPSHOT_DIR = Path("artifacts/camera_snapshots")


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


def _center_roi(frame_h: int, frame_w: int, roi_size: int) -> tuple[int, int, int, int]:
    """Return ``(x1, y1, x2, y2)`` for a centered ROI clamped to the frame."""
    roi = min(roi_size, frame_h, frame_w)
    x1 = (frame_w - roi) // 2
    y1 = (frame_h - roi) // 2
    return x1, y1, x1 + roi, y1 + roi


@torch.no_grad()
def predict_roi(
    bgr_roi: np.ndarray,
    model: nn.Module,
    transform,
    device: torch.device,
    class_names: list[str],
) -> tuple[str, float]:
    """Classify a single BGR ROI crop.

    The exact eval-time pipeline is applied: BGR→RGB → PIL → ``transform``
    (resize/ToTensor/normalize from :func:`get_eval_transforms`) → batch dim →
    device → forward → softmax.

    Args:
        bgr_roi: ROI crop in OpenCV BGR ``uint8`` layout.
        model: Model in eval mode.
        transform: The eval transform (from ``get_eval_transforms()``).
        device: Compute device.
        class_names: Index→label mapping.

    Returns:
        ``(predicted_class, confidence)`` where confidence is in ``[0, 1]``.
    """
    rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    conf, idx = torch.max(probs, dim=1)
    return class_names[int(idx.item())], float(conf.item())


def _annotate(
    frame: np.ndarray,
    roi_box: tuple[int, int, int, int],
    label: str,
    confidence: float,
    fps: float | None = None,
) -> np.ndarray:
    """Draw the ROI rectangle, predicted letter, confidence, and FPS in place."""
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{confidence * 100:.1f}%",
        (x1, y2 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def run_static_image(
    source: Path,
    model: nn.Module,
    transform,
    device: torch.device,
    class_names: list[str],
    roi_size: int,
) -> int:
    """Run ONE inference pass on a still image. Fully headless (no GUI calls).

    Loads the image, classifies a centered ROI, prints the prediction, and
    saves an annotated copy under :data:`SNAPSHOT_DIR`.

    Returns:
        Process exit code (always ``0`` on success).
    """
    frame = cv2.imread(str(source))
    if frame is None:
        print(f"ERROR: could not read image '{source}'.")
        return 0

    h, w = frame.shape[:2]
    roi_box = _center_roi(h, w, roi_size)
    x1, y1, x2, y2 = roi_box
    roi = frame[y1:y2, x1:x2]
    label, confidence = predict_roi(roi, model, transform, device, class_names)

    print(f"predicted_class: {label}")
    print(f"confidence: {confidence:.4f}")

    annotated = _annotate(frame.copy(), roi_box, label, confidence)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"{source.stem}_pred_{label}.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image to {out_path}")
    return 0


def run_camera(
    source: int | str,
    model: nn.Module,
    transform,
    device: torch.device,
    class_names: list[str],
    roi_size: int,
) -> int:
    """Interactive webcam/video loop with a live overlay.

    Draws a centered green ROI, the predicted letter, confidence, and a rolling
    30-frame-average FPS. ``q`` quits; ``s`` saves the current frame +
    prediction to :data:`SNAPSHOT_DIR`. These ``imshow``/``waitKey`` calls are
    reachable only in this interactive branch (never in CI).

    If the capture device cannot be opened, prints a clear message and returns
    ``0`` (so CI without a camera does not crash).

    Returns:
        Process exit code.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(
            f"WARNING: could not open video source '{source}'. "
            "If you intended a webcam, ensure a camera is connected and "
            "accessible. Exiting cleanly."
        )
        return 0

    frame_times: deque[float] = deque(maxlen=30)
    window = "ASL Inference (q=quit, s=save)"
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting live inference. Press 'q' to quit, 's' to save a snapshot.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of stream / failed to read frame. Exiting.")
                break

            start = time.perf_counter()
            h, w = frame.shape[:2]
            roi_box = _center_roi(h, w, roi_size)
            x1, y1, x2, y2 = roi_box
            roi = frame[y1:y2, x1:x2]
            label, confidence = predict_roi(roi, model, transform, device, class_names)
            frame_times.append(time.perf_counter() - start)

            avg = sum(frame_times) / len(frame_times)
            fps = (1.0 / avg) if avg > 0 else 0.0
            annotated = _annotate(frame, roi_box, label, confidence, fps)

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = SNAPSHOT_DIR / f"snapshot_{ts}_{label}_{confidence:.2f}.png"
                cv2.imwrite(str(out_path), annotated)
                print(f"Saved snapshot to {out_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def _classify_source(source: str) -> tuple[str, int | str]:
    """Resolve the ``--source`` string into a mode and capture argument.

    Returns:
        ``("webcam", index)``, ``("video", path)``, or ``("image", path)``.
    """
    if source == "0" or source.isdigit():
        return "webcam", int(source)
    ext = Path(source).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video", source
    if ext in IMAGE_EXTS:
        return "image", source
    # Unknown extension: assume image (single-pass, headless) to stay CI-safe.
    return "image", source


def main() -> int:
    """Parse CLI args, load the model, and dispatch to the right run mode."""
    parser = argparse.ArgumentParser(description="Real-time ASL OpenCV inference.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional training-config YAML/JSON (unused for inference logic).",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="'0' for webcam, or a path to an image/video file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Compute device.",
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        default=300,
        help="Side length (px) of the centered ROI.",
    )
    args = parser.parse_args()

    # The config is accepted for symmetry with the other scripts; inference
    # preprocessing is fully determined by get_eval_transforms(), so we only
    # touch the config to validate it is loadable when provided.
    if args.config is not None and Path(args.config).exists():
        try:
            load_json(args.config)
        except Exception:  # noqa: BLE001 - YAML configs aren't JSON; ignore.
            pass

    device = get_device(args.device)
    print(f"Using device: {device}")

    model, class_names = load_checkpoint(args.checkpoint, device)
    transform = get_eval_transforms()

    mode, capture_arg = _classify_source(args.source)
    if mode == "image":
        return run_static_image(
            Path(args.source), model, transform, device, class_names, args.roi_size
        )
    return run_camera(capture_arg, model, transform, device, class_names, args.roi_size)


if __name__ == "__main__":
    raise SystemExit(main())
