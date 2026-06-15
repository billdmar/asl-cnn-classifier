"""Extract a few real ASL example images from a class-folder dataset.

The Gradio demo (``app.py``) wires up click-to-try examples. Rather than ship
the synthetic colored-square CI fixtures (``data/sample/<CLASS>/0.png``), this
script copies a representative *real* image per requested class out of a
class-folder dataset directory into ``docs/examples/<CLASS>.png``, which the demo
prefers when present.

A "class-folder dataset" is any directory whose immediate children are named
after the classes, each holding image files::

    <src>/A/img001.png
    <src>/A/img002.png
    <src>/C/...
    ...

This matches the layout produced by the dataset-ingestion path
(``data/asl_real/<CLASS>/*.png``), so the two stay decoupled: this script only
relies on the on-disk folder contract, not on any ingestion code.

The extraction is **deterministic** (the first image per class in sorted order)
and headless. A requested class that is missing or empty in ``--src`` is skipped
with a warning rather than crashing, so a partial dataset still yields a partial
example set.

Typical use::

    python scripts/extract_examples.py --src data/asl_real --out docs/examples
    python scripts/extract_examples.py --src data/asl_real --classes A C L W Y space
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

# A handful of visually distinct hand shapes (open palm, fist, fingers
# extended, etc.) so a visitor can sample a spread of signs by default.
DEFAULT_CLASSES: tuple[str, ...] = ("A", "B", "C", "L", "W", "Y")

# Square edge length (pixels) the extracted examples are resized to. Small
# enough to commit, large enough to read on the demo.
DEFAULT_SIZE = 200

# Image file extensions considered when picking a representative image.
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _first_image(class_dir: Path) -> Path | None:
    """Return the first image file in ``class_dir`` (sorted), or ``None``.

    Sorting by name keeps the choice deterministic across runs and machines.
    """
    if not class_dir.is_dir():
        return None
    candidates = sorted(
        p
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    return candidates[0] if candidates else None


def extract_examples(
    src: Path,
    out: Path,
    classes: tuple[str, ...],
    per_class: int = 1,
    size: int = DEFAULT_SIZE,
) -> list[Path]:
    """Copy/resize a representative image per class into ``out``.

    Args:
        src: A class-folder dataset directory (``<src>/<CLASS>/<image>``).
        out: Output directory; created if missing. Images are written as
            ``<out>/<CLASS>.png`` (and ``<CLASS>_<n>.png`` for ``per_class > 1``).
        classes: The class names to extract.
        per_class: How many images to extract per class (first ``n`` sorted).
        size: Square edge length the images are resized to.

    Returns:
        The list of written output paths (only for classes that were found).

    A class missing from ``src`` (or with no images) is skipped with a warning
    on stderr; it does not raise.
    """
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for cls in classes:
        class_dir = src / cls
        if not class_dir.is_dir():
            print(f"warning: no folder for class {cls!r} in {src}", file=sys.stderr)
            continue

        images = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not images:
            print(
                f"warning: no images for class {cls!r} in {class_dir}",
                file=sys.stderr,
            )
            continue

        for idx, image_path in enumerate(images[:per_class]):
            suffix = "" if per_class == 1 else f"_{idx}"
            dest = out / f"{cls}{suffix}.png"
            try:
                with Image.open(image_path) as img:
                    resized = img.convert("RGB").resize(
                        (size, size), Image.Resampling.LANCZOS
                    )
                    resized.save(dest, format="PNG")
            except (OSError, ValueError) as exc:
                print(
                    f"warning: could not read {image_path} for class {cls!r}: {exc}",
                    file=sys.stderr,
                )
                continue
            written.append(dest)

    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract real ASL example images from a class-folder dataset into "
            "docs/examples for the Gradio demo."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Class-folder dataset dir (e.g. data/asl_real with A/, B/, ... subdirs).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/examples"),
        help="Output directory for the extracted examples (default: docs/examples).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        metavar="CLASS",
        help=f"Class names to extract (default: {' '.join(DEFAULT_CLASSES)}).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=1,
        help="Number of images to extract per class (default: 1).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Square edge length in pixels (default: {DEFAULT_SIZE}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    if not args.src.is_dir():
        print(f"error: --src is not a directory: {args.src}", file=sys.stderr)
        return 2
    if args.per_class < 1:
        print("error: --per-class must be >= 1", file=sys.stderr)
        return 2

    written = extract_examples(
        src=args.src,
        out=args.out,
        classes=tuple(args.classes),
        per_class=args.per_class,
        size=args.size,
    )

    if not written:
        print(
            f"error: no examples extracted from {args.src} "
            f"(none of {args.classes} had usable images).",
            file=sys.stderr,
        )
        return 1

    print(f"Extracted {len(written)} example image(s) to {args.out}:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
