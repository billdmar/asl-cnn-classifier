"""Pre-compute Grad-CAM overlays for the web app's bundled example images.

Grad-CAM needs backward gradients, which the in-browser inference engine
(onnxruntime-web) does not expose — so it CANNOT run client-side. Instead we
pre-compute overlays here for the fixed set of bundled examples and commit them
as static assets; the web UI shows the heatmap when a user clicks an example,
with an explicit "pre-computed offline" caption.

Reuses :func:`src.gradcam.run_gradcam` verbatim. Each overlay is written as
``<out_dir>/<EXAMPLE_LABEL>.png`` (named by the example's known label, not the
predicted class, so the UI can look it up by the example it rendered).

Usage:
    python scripts/gradcam_web.py \
        --checkpoint artifacts/checkpoints/best_model.pth \
        --examples_dir web/public/examples --out_dir web/public/gradcam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gradcam import run_gradcam  # noqa: E402
from src.checkpoint import load_checkpoint  # noqa: E402
from src.utils import get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best_model.pth")
    parser.add_argument("--examples_dir", default="web/public/examples")
    parser.add_argument("--out_dir", default="web/public/gradcam")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device(args.device)
    model, class_names = load_checkpoint(args.checkpoint, device)

    examples_dir = Path(args.examples_dir)
    out_dir = Path(args.out_dir)
    images = sorted(p for p in examples_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No example PNGs under {examples_dir}")

    print(f"Pre-computing Grad-CAM for {len(images)} examples → {out_dir}")
    for img in images:
        example_label = img.stem  # e.g. "A.png" → "A"
        # run_gradcam writes <out_dir>/<predicted>.png; we want it keyed by the
        # example label, so render to a temp dir then move into place.
        tmp_path, predicted = run_gradcam(
            img, model, device, class_names, out_dir=out_dir
        )
        final = out_dir / f"{example_label}.png"
        if tmp_path != final:
            tmp_path.replace(final)
        flag = "" if predicted == example_label else f"  (predicted {predicted})"
        print(f"  {example_label}: saved{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
