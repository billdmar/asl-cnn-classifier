"""Pre-crop a class-folder ASL dataset to the MediaPipe hand region, once.

The deployed model is *served* MediaPipe hand-crops (both the webcam and upload
paths in the web app crop to the hand before classifying, via
``web/lib/handcrop.ts``), but it was *trained* on raw uncropped frames. That
train/serve distribution mismatch is the dominant driver of the model's
cross-dataset accuracy collapse. This script closes the gap on the training
side: it crops every image in ``--in_dir`` to the same hand region the browser
produces and writes the result to ``--out_dir`` with an identical class-folder
layout, so training on the cropped set matches what the model sees at serve time.

Crop geometry is delegated to :func:`src.handcrop.detect_and_crop` — the exact
same code path eval uses — so the crop is pixel-consistent with the browser
(square box, ``CROP_MARGIN`` fraction, clamped to the image). Filenames are
preserved so frame-sequence ordering (and the dedup split) stays valid.

No-hand handling mirrors the serve fallback: by default an image where MediaPipe
finds no hand is copied through whole (the upload path and ``eval_realworld`` do
the same), so the model trains on the same fallback distribution it will face.
Pass ``--drop-no-hand`` to skip those images instead (for A/B comparison).

A ``_precrop_report.json`` is written to ``--out_dir`` with per-class and global
no-hand counts so a pathological detection rate is visible, not silently absorbed.

Usage:
    python scripts/precrop_dataset.py --in_dir data/asl_real \\
        --out_dir data/asl_real_cropped --margin 0.35
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_class_names  # noqa: E402
from src.handcrop import (  # noqa: E402
    CROP_MARGIN,
    DEFAULT_MODEL_PATH,
    _build_landmarker,
    detect_and_crop,
)

# A per-class no-hand rate above this fraction is surfaced as a warning — it
# signals the detector is failing on a class, which would quietly degrade the
# crop-consistent training set rather than improve it.
NO_HAND_WARN_RATE = 0.20

# The Marxulia training set bundles ~63 pencil-sketch line drawings per class
# (15% of the data) that are not photographs — grayscale diagrams with near-zero
# color saturation. They look nothing like the served (photographic) distribution
# and 100% fail hand detection, so they pollute crop-consistent training. A pixel
# is "colorless" when its RGB max-min spread is below SKETCH_SAT_THRESHOLD; an
# image is a line drawing when at least SKETCH_PIXEL_FRACTION of pixels are.
SKETCH_SAT_THRESHOLD = 12
SKETCH_PIXEL_FRACTION = 0.97


def is_line_drawing(
    image: Image.Image,
    sat_threshold: int = SKETCH_SAT_THRESHOLD,
    pixel_fraction: float = SKETCH_PIXEL_FRACTION,
) -> bool:
    """True if ``image`` is a near-grayscale line drawing (not a color photo).

    Pure and dependency-light so it is unit-testable. Measures per-pixel color
    saturation as ``max(R,G,B) - min(R,G,B)``; a sketch/B&W diagram has almost
    every pixel near-colorless, while a skin-tone photo does not.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    return bool((saturation < sat_threshold).mean() > pixel_fraction)


def precrop_dataset(
    in_dir: str | Path,
    out_dir: str | Path,
    margin: float = CROP_MARGIN,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    drop_no_hand: bool = False,
    drop_sketches: bool = False,
) -> dict:
    """Crop every image in ``in_dir`` to its hand region; write to ``out_dir``.

    Returns a report dict (also written to ``out_dir/_precrop_report.json``) with
    per-class totals, no-hand counts, sketch counts, and the global rates.

    Args:
        drop_sketches: When ``True``, skip non-photographic line drawings
            (see :func:`is_line_drawing`) — they pollute crop-consistent
            training and all fail hand detection.
    """
    in_root = Path(in_dir)
    out_root = Path(out_dir)
    class_names = get_class_names(in_root)

    landmarker = _build_landmarker(model_path)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    per_class: dict[str, dict[str, int]] = {}
    total = 0
    total_no_hand = 0
    total_sketch = 0
    total_written = 0
    try:
        for name in class_names:
            class_in = in_root / name
            if not class_in.is_dir():
                continue
            class_out = out_root / name
            class_out.mkdir(parents=True, exist_ok=True)

            n_total = 0
            n_no_hand = 0
            n_sketch = 0
            n_written = 0
            for img_path in sorted(class_in.iterdir()):
                if img_path.suffix.lower() not in exts:
                    continue
                n_total += 1
                image = Image.open(img_path).convert("RGB")
                if is_line_drawing(image):
                    n_sketch += 1
                    if drop_sketches:
                        continue
                cropped = detect_and_crop(image, landmarker=landmarker, margin=margin)
                if cropped is None:
                    n_no_hand += 1
                    if drop_no_hand:
                        continue
                    cropped = image  # fallback to whole image, matching serve
                # Preserve the original filename so frame order / dedup stays valid.
                cropped.save(class_out / f"{img_path.stem}.png", format="PNG")
                n_written += 1

            per_class[name] = {
                "total": n_total,
                "no_hand": n_no_hand,
                "sketches": n_sketch,
                "written": n_written,
            }
            total += n_total
            total_no_hand += n_no_hand
            total_sketch += n_sketch
            total_written += n_written

            rate = (n_no_hand / n_total) if n_total else 0.0
            flag = "  <-- HIGH no-hand rate" if rate > NO_HAND_WARN_RATE else ""
            print(
                f"  {name}: {n_total} imgs, {n_no_hand} no-hand "
                f"({rate:.1%}), {n_sketch} sketches, {n_written} written{flag}"
            )
    finally:
        landmarker.close()

    report = {
        "in_dir": str(in_root),
        "out_dir": str(out_root),
        "margin": margin,
        "drop_no_hand": drop_no_hand,
        "drop_sketches": drop_sketches,
        "num_classes": len(per_class),
        "total_images": total,
        "total_no_hand": total_no_hand,
        "total_sketches": total_sketch,
        "total_written": total_written,
        "global_no_hand_rate": (total_no_hand / total) if total else 0.0,
        "global_sketch_rate": (total_sketch / total) if total else 0.0,
        "per_class": per_class,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_precrop_report.json").write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_dir", default="data/asl_real", help="Source dataset.")
    parser.add_argument(
        "--out_dir", default="data/asl_real_cropped", help="Cropped output dataset."
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=CROP_MARGIN,
        help=f"Crop margin fraction (default {CROP_MARGIN}, matches the browser).",
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="Path to hand_landmarker.task.",
    )
    parser.add_argument(
        "--drop-no-hand",
        dest="drop_no_hand",
        action="store_true",
        help="Skip images with no detectable hand (default: copy whole image).",
    )
    parser.add_argument(
        "--drop-sketches",
        dest="drop_sketches",
        action="store_true",
        help="Skip non-photographic line drawings (the ~15%% sketch contamination).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Pre-cropping {args.in_dir} -> {args.out_dir} (margin={args.margin})")
    report = precrop_dataset(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        margin=args.margin,
        model_path=args.model_path,
        drop_no_hand=args.drop_no_hand,
        drop_sketches=args.drop_sketches,
    )
    print("\n=== Pre-crop summary ===")
    print(f"Classes        : {report['num_classes']}")
    print(f"Total images   : {report['total_images']}")
    print(f"No-hand images : {report['total_no_hand']} "
          f"({report['global_no_hand_rate']:.1%})")
    print(f"Sketches       : {report['total_sketches']} "
          f"({report['global_sketch_rate']:.1%})"
          f"{' (dropped)' if report['drop_sketches'] else ''}")
    print(f"Written        : {report['total_written']}")
    print(f"Report         : {Path(report['out_dir']) / '_precrop_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
