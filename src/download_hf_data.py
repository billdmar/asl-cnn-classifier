"""Download a real ASL hand-sign dataset from the Hugging Face Hub.

Unlike :mod:`src.download_data` (Kaggle, needs ``~/.kaggle/kaggle.json``), this
helper pulls the public, credential-free dataset
``Marxulia/asl_sign_languages_alphabets_v03`` (~74 MB, 10,873 images across 26
classes labeled ``0..25`` = ``A..Z``) and materializes it into the **same
class-folder layout** the rest of the pipeline already understands::

    data/asl_real/<CLASS>/<i>.png

so :func:`src.dataset._list_samples` / :func:`~src.dataset.get_class_names` /
:func:`~src.dataset.make_stratified_splits` consume it unchanged. Each image is
written as RGB PNG. ``--max_per_class`` caps the images written per class, which
is handy for fast smoke runs.

No token is ever hardcoded; the dataset is public so anonymous access works. If
the ``datasets`` library is missing or the download fails, the error is reported
clearly rather than producing a partial/silent result.

Run ``python -m src.download_hf_data --out_dir data/asl_real`` to fetch it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

HF_DATASET = "Marxulia/asl_sign_languages_alphabets_v03"
HF_SPLIT = "train"


def _resolve_class_names(dataset: Any) -> list[str]:
    """Return the ``index -> class name`` mapping for the loaded dataset.

    Prefers the dataset's ``ClassLabel`` feature names (e.g. ``["A", ..., "Z"]``)
    so the on-disk folder names match the published label order. Falls back to
    stringified integer labels if the feature carries no names.
    """
    label_feature = dataset.features.get("label")
    names = getattr(label_feature, "names", None)
    if names:
        return [str(n) for n in names]
    # Fallback: derive a stable, sorted name set from the observed labels
    # (handles datasets that yield raw string or integer labels directly).
    observed = sorted({str(lbl) for lbl in dataset["label"]})
    return observed


def _label_to_name(label: Any, class_names: list[str]) -> str:
    """Map a raw row label (int index or already-a-string) to a folder name."""
    if isinstance(label, str):
        return label
    return class_names[int(label)]


def download(
    out_dir: str = "data/asl_real",
    max_per_class: int | None = None,
) -> dict[str, int]:
    """Download the HF ASL dataset and write it as ``out_dir/<CLASS>/<i>.png``.

    Args:
        out_dir: Destination root. One subfolder per class is created.
        max_per_class: If given, stop after writing this many images per class
            (useful for fast runs). ``None`` writes every image.

    Returns:
        A mapping of ``class name -> number of images written``.

    Raises:
        RuntimeError: If the ``datasets`` library is unavailable or the dataset
            cannot be loaded.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError(
            "The 'datasets' package is required. Install it with "
            "`pip install datasets` (or `pip install -r requirements.txt`)."
        ) from exc

    print(f"Loading '{HF_DATASET}' (split='{HF_SPLIT}') from the Hugging Face Hub...")
    try:
        dataset = load_dataset(HF_DATASET, split=HF_SPLIT)
    except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
        raise RuntimeError(
            f"Failed to load dataset '{HF_DATASET}' from the Hugging Face Hub: {exc}"
        ) from exc

    class_names = _resolve_class_names(dataset)
    print(f"Resolved {len(class_names)} classes: {class_names}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written: Counter[str] = Counter()
    for row in dataset:
        name = _label_to_name(row["label"], class_names)
        if max_per_class is not None and written[name] >= max_per_class:
            continue
        class_dir = out_root / name
        class_dir.mkdir(parents=True, exist_ok=True)
        image = row["image"].convert("RGB")
        image.save(class_dir / f"{written[name]}.png", format="PNG")
        written[name] += 1

    counts = {name: written.get(name, 0) for name in class_names}
    total = sum(counts.values())
    print(f"\nWrote {total} images to {out_root}/")
    print("Per-class counts:")
    for name in class_names:
        print(f"  {name}: {counts[name]}")
    return counts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download a real ASL dataset from the Hugging Face Hub."
    )
    parser.add_argument(
        "--out_dir",
        default="data/asl_real",
        help="Destination root for the class-folder layout.",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Cap images written per class (useful for fast runs).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    download(out_dir=args.out_dir, max_per_class=args.max_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
