"""Generate the committed ``data/sample/`` subset (8 images × 29 classes = 232).

The real Kaggle dataset (~87k images) is too large to commit, but CI and a
fresh clone still need *something* to run the full train → eval → benchmark
pipeline end-to-end. We therefore generate a tiny, **deterministic** synthetic
subset where each class is visually distinguishable, so a model can actually
overfit it in a couple of epochs (a sanity signal that the training loop works).

Each class gets a unique base hue plus a class-index glyph and per-sample noise,
so the 8 images within a class vary slightly while remaining separable from
other classes. Everything is seeded, so regenerating produces identical bytes.

This is a SANITY FIXTURE, not real data — accuracy numbers on it are meaningless
and the README/MODEL_CARD say so explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.dataset import CLASS_NAMES

IMG_SIZE = 200  # match the real dataset's native 200×200 resolution
PER_CLASS = 8  # 8 × 29 = 232 images ("200+" per the resume framing)


def _class_color(class_idx: int) -> tuple[int, int, int]:
    """Map a class index to a distinct RGB base color via evenly spaced hues."""
    hue = (class_idx / len(CLASS_NAMES)) * 360.0
    # Simple HSV→RGB at full saturation/value.
    h = hue / 60.0
    c = 255
    x = int(c * (1 - abs(h % 2 - 1)))
    table = [
        (c, x, 0),
        (x, c, 0),
        (0, c, x),
        (0, x, c),
        (x, 0, c),
        (c, 0, x),
    ]
    return table[int(h) % 6]


def generate(out_dir: str | Path, per_class: int = PER_CLASS, seed: int = 42) -> int:
    """Write ``per_class`` deterministic images for each of the 29 classes."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    total = 0

    for class_idx, name in enumerate(CLASS_NAMES):
        class_dir = out_dir / name
        class_dir.mkdir(parents=True, exist_ok=True)
        base = np.array(_class_color(class_idx), dtype=np.float32)

        for i in range(per_class):
            # Per-class base color + small per-sample Gaussian noise → 8 distinct
            # but related images per class.
            noise = rng.normal(0, 18, size=(IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
            arr = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

            # Draw the class index as a glyph so classes are clearly separable.
            draw = ImageDraw.Draw(img)
            label_color = (0, 0, 0) if sum(base) > 350 else (255, 255, 255)
            draw.text((20, 20), f"{class_idx:02d}", fill=label_color)
            draw.text((20, 80), name, fill=label_color)
            # A class-dependent rectangle adds a second separable feature.
            x0 = 10 + (class_idx % 5) * 30
            y0 = 130 + (class_idx % 3) * 15
            draw.rectangle([x0, y0, x0 + 40, y0 + 40], outline=label_color, width=3)

            img.save(class_dir / f"{i}.png")
            total += 1

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the sample data subset.")
    parser.add_argument("--out_dir", default="data/sample")
    parser.add_argument("--per_class", type=int, default=PER_CLASS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    n = generate(args.out_dir, args.per_class, args.seed)
    print(f"Generated {n} images across {len(CLASS_NAMES)} classes in {args.out_dir}")
