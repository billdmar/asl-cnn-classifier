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

A second, *more diverse* ASL alphabet dataset (different signers/backgrounds)
can be pulled with ``--dataset``/``--split`` (or one of the friendly names in
:data:`DATASETS`) so we can honestly measure cross-dataset generalization. Class
labels are normalized to the canonical uppercase ``A..Z`` and any class outside
``A..Z`` is dropped (see :func:`_normalize_class_name`), so a dataset that uses
lowercase ``a..z`` or carries extra non-letter classes maps cleanly onto the
deployed 26-class model.

Run ``python -m src.download_hf_data --out_dir data/asl_real`` for the default
Marxulia set, or e.g.::

    python -m src.download_hf_data --dataset asl_letters \\
        --out_dir data/asl_crossval --max_per_class 30
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

HF_DATASET = "Marxulia/asl_sign_languages_alphabets_v03"
HF_SPLIT = "train"

# Friendly names → (hf_dataset_id, split). The default Marxulia set is the
# uniform single-signer training distribution; ``asl_letters`` is a SECOND,
# more diverse 26-class A–Z dataset (different signers, real photo backgrounds)
# used for the honest cross-dataset generalization measurement.
DATASETS: dict[str, tuple[str, str]] = {
    "marxulia": (HF_DATASET, HF_SPLIT),
    "asl_letters": ("EitanG98/asl_letters", "train"),
    # A genuinely diverse TRAINING source (multiple signers, skin tones, real
    # varied backgrounds/lighting) — the antidote to Marxulia's single-signer,
    # plain-background overfit. 8,442 imgs, A–Y (no J/Z, both motion signs).
    "diverse": ("aliciiavs/sign_language_image_dataset", "train"),
    # Fallback diversity source: balanced ~70/class, includes J/Z, dark
    # cropped-hand backgrounds (a different regime than Marxulia/aliciiavs).
    "hemg": ("Hemg/sign_language_dataset", "train"),
}

# Datasets whose class label is NOT in an HF `label` column but is encoded in
# the image FILENAME (e.g. Roboflow/YOLO exports: ``A11_jpg.rf.<hash>.jpg``).
# `datasets.load_dataset` drops the filename, so these are pulled with
# `huggingface_hub.snapshot_download` and the class is parsed from the name via
# ``regex`` (first capture group → :func:`_normalize_class_name`). Each split is
# a subdir under the repo root. See :func:`download_from_filenames`.
FILENAME_DATASETS: dict[str, dict[str, Any]] = {
    "atalaydenknalbant": {
        "hf_id": "atalaydenknalbant/asl-dataset",
        "regex": r"^([A-Za-z])",
        "split_subdirs": {
            "train": "train/images",
            "valid": "valid/images",
            "test": "test/images",
        },
        "default_split": "train",
    },
}


def _normalize_class_name(name: str) -> str | None:
    """Map a raw class name to a canonical ``A..Z`` folder name, or ``None``.

    Datasets vary: some use uppercase ``A``, some lowercase ``a``, some include
    extra non-letter classes (``del``, ``space``, ``nothing``, digits). We
    uppercase single-letter names and keep only ``A..Z``; anything else returns
    ``None`` so the caller drops it (the deployed model has no class for it).
    """
    upper = name.strip().upper()
    if len(upper) == 1 and "A" <= upper <= "Z":
        return upper
    return None


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


def _resolve_dataset_spec(dataset: str | None, split: str | None) -> tuple[str, str]:
    """Resolve a friendly name or raw HF id (+ split) into ``(hf_id, split)``.

    ``None`` defaults to the Marxulia constant (backward-compatible). A value
    matching a :data:`DATASETS` key expands to its ``(id, split)``; otherwise
    the value is treated as a raw HF dataset id and paired with ``split`` (or
    :data:`HF_SPLIT` if unspecified).
    """
    if dataset is None:
        return HF_DATASET, split or HF_SPLIT
    if dataset in DATASETS:
        hf_id, default_split = DATASETS[dataset]
        return hf_id, split or default_split
    return dataset, split or HF_SPLIT


def download(
    out_dir: str = "data/asl_real",
    max_per_class: int | None = None,
    dataset: str | None = None,
    split: str | None = None,
) -> dict[str, int]:
    """Download an HF ASL dataset and write it as ``out_dir/<CLASS>/<i>.png``.

    Class names are normalized to canonical uppercase ``A..Z`` and any class
    outside that set is dropped (see :func:`_normalize_class_name`), so a second
    dataset using lowercase or extra classes still maps onto the deployed
    26-class model.

    Args:
        out_dir: Destination root. One subfolder per class is created.
        max_per_class: If given, stop after writing this many images per class
            (useful for fast runs). ``None`` writes every image.
        dataset: A friendly name from :data:`DATASETS` (e.g. ``"asl_letters"``)
            or a raw HF dataset id. ``None`` uses the default Marxulia set.
        split: HF split to pull. ``None`` uses the dataset's default split.

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

    hf_id, hf_split = _resolve_dataset_spec(dataset, split)
    print(f"Loading '{hf_id}' (split='{hf_split}') from the Hugging Face Hub...")
    try:
        loaded = load_dataset(hf_id, split=hf_split)
    except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
        raise RuntimeError(
            f"Failed to load dataset '{hf_id}' from the Hugging Face Hub: {exc}"
        ) from exc

    class_names = _resolve_class_names(loaded)
    print(f"Resolved {len(class_names)} raw classes: {class_names}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written: Counter[str] = Counter()
    dropped = 0
    for row in loaded:
        raw_name = _label_to_name(row["label"], class_names)
        name = _normalize_class_name(raw_name)
        if name is None:
            dropped += 1
            continue
        if max_per_class is not None and written[name] >= max_per_class:
            continue
        class_dir = out_root / name
        class_dir.mkdir(parents=True, exist_ok=True)
        image = row["image"].convert("RGB")
        image.save(class_dir / f"{written[name]}.png", format="PNG")
        written[name] += 1

    counts = {name: written[name] for name in sorted(written)}
    total = sum(counts.values())
    print(f"\nWrote {total} images to {out_root}/")
    if dropped:
        print(f"Dropped {dropped} images in non-A–Z classes.")
    print("Per-class counts:")
    for name in sorted(counts):
        print(f"  {name}: {counts[name]}")
    return counts


def download_from_filenames(
    hf_id: str,
    out_dir: str,
    split_subdir: str,
    regex: str,
    max_per_class: int | None = None,
) -> dict[str, int]:
    """Download a filename-labeled HF dataset and write the class-folder layout.

    For datasets where the class is encoded in the image filename (not an HF
    ``label`` column) — e.g. Roboflow/YOLO exports. Files are fetched with
    ``huggingface_hub.snapshot_download`` (``load_dataset`` would drop the
    filename), the leading capture group of ``regex`` is run through
    :func:`_normalize_class_name`, and each image is written as
    ``out_dir/<CLASS>/<i>.png`` — the SAME layout the rest of the pipeline
    consumes, so training/eval need no changes.

    Args:
        hf_id: HF dataset repo id.
        out_dir: Destination root (one subfolder per class).
        split_subdir: Path under the repo root holding this split's images
            (e.g. ``"train/images"``).
        regex: Pattern whose first group is the raw class token in the filename.
        max_per_class: Optional per-class cap.

    Returns:
        A mapping of ``class name -> number of images written``.

    Raises:
        RuntimeError: If ``huggingface_hub`` is unavailable or the download fails.
    """
    import re

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError(
            "The 'huggingface_hub' package is required for filename-labeled "
            "datasets. Install it with `pip install huggingface_hub`."
        ) from exc

    print(f"Snapshotting '{hf_id}' split subdir '{split_subdir}' from the HF Hub...")
    try:
        local_root = snapshot_download(
            repo_id=hf_id,
            repo_type="dataset",
            allow_patterns=[f"{split_subdir}/*"],
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure clearly
        raise RuntimeError(f"Failed to snapshot dataset '{hf_id}': {exc}") from exc

    pattern = re.compile(regex)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    split_dir = Path(local_root) / split_subdir
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written: Counter[str] = Counter()
    dropped = 0
    for img_path in sorted(split_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        match = pattern.match(img_path.name)
        name = _normalize_class_name(match.group(1)) if match else None
        if name is None:
            dropped += 1
            continue
        if max_per_class is not None and written[name] >= max_per_class:
            continue
        class_dir = out_root / name
        class_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(img_path).convert("RGB")
        image.save(class_dir / f"{written[name]}.png", format="PNG")
        written[name] += 1

    counts = {name: written[name] for name in sorted(written)}
    total = sum(counts.values())
    print(f"\nWrote {total} images to {out_root}/")
    if dropped:
        print(f"Dropped {dropped} images with no parseable A–Z class.")
    print("Per-class counts:")
    for name in sorted(counts):
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
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Friendly name (one of: "
            f"{', '.join(sorted({*DATASETS, *FILENAME_DATASETS}))}) or a raw HF "
            "dataset id. Defaults to the Marxulia training set."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help="HF split to pull (defaults to the dataset's default split).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    # Route filename-labeled datasets (class in the filename, no `label` column)
    # to the snapshot-based loader; everything else uses the standard path.
    if args.dataset in FILENAME_DATASETS:
        spec = FILENAME_DATASETS[args.dataset]
        split = args.split or spec["default_split"]
        if split not in spec["split_subdirs"]:
            raise ValueError(
                f"Unknown split '{split}' for '{args.dataset}'. "
                f"Expected one of {sorted(spec['split_subdirs'])}."
            )
        download_from_filenames(
            hf_id=spec["hf_id"],
            out_dir=args.out_dir,
            split_subdir=spec["split_subdirs"][split],
            regex=spec["regex"],
            max_per_class=args.max_per_class,
        )
        return 0
    download(
        out_dir=args.out_dir,
        max_per_class=args.max_per_class,
        dataset=args.dataset,
        split=args.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
