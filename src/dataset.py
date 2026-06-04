"""Dataset, stratified splitting, and the canonical image transforms.

This module is the **single source of truth** for the preprocessing transforms.
``train.py``, ``eval.py``, ``infer_camera.py``, and ``benchmark.py`` all import
``get_train_transforms`` / ``get_eval_transforms`` from here so that test-time
preprocessing is provably identical everywhere (no silent drift between
training and inference).

Run ``python -m src.dataset --data_dir data/sample`` to print dataset stats.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

# The 29 ASL Alphabet classes (A–Z, plus the three control signs). Sorted so the
# integer label assignment is deterministic and matches folder order on disk.
CLASS_NAMES: list[str] = sorted(
    [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]
)

# Input size fed to the network. Source images are 200×200; we resize to 128.
IMAGE_SIZE = 128

# ImageNet statistics — required for the MobileNetV2 transfer variant and a fine
# default for the custom CNN as well.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Augmentation pipeline for training.

    NOTE: deliberately **no horizontal flip** — ASL signs are not
    flip-invariant (e.g. b/d, p/q are mirror images, and several letters differ
    only by orientation). Flipping would create mislabeled training data.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic test-time pipeline (resize → tensor → normalize).

    Used for validation, test, live camera inference, and benchmarking so that
    every code path preprocesses frames identically.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_class_names(root_dir: str | Path | None = None) -> list[str]:
    """Return the 29 class names.

    If ``root_dir`` is given and contains class folders, the on-disk folder
    names are used (sorted); otherwise the canonical :data:`CLASS_NAMES` list is
    returned. This keeps label↔index mapping stable across machines.
    """
    if root_dir is not None:
        root = Path(root_dir)
        found = sorted(p.name for p in root.iterdir() if p.is_dir())
        if found:
            return found
    return list(CLASS_NAMES)


def _list_samples(root_dir: str | Path, class_names: list[str]) -> list[tuple[str, int]]:
    """Walk class folders and return ``(filepath, label_int)`` for every image."""
    root = Path(root_dir)
    label_of = {name: idx for idx, name in enumerate(class_names)}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    samples: list[tuple[str, int]] = []
    for name in class_names:
        class_dir = root / name
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in exts:
                samples.append((str(img_path), label_of[name]))
    return samples


class ASLDataset(Dataset):
    """ASL Alphabet image dataset.

    Can be constructed either from a ``root_dir`` (walks class folders) or from
    a pre-computed list of ``(filepath, label)`` samples (used after a
    stratified split, so the split happens once at the file level — no leakage).

    Each item is ``(image_tensor, label_int, filepath)``.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        samples: list[tuple[str, int]] | None = None,
        transform: transforms.Compose | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        if samples is None and root_dir is None:
            raise ValueError("Provide either root_dir or samples.")
        self.class_names = class_names or get_class_names(root_dir)
        if samples is None:
            samples = _list_samples(root_dir, self.class_names)
        self.samples = samples
        self.transform = transform or get_eval_transforms()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        tensor = self.transform(image)
        return tensor, label, filepath


def make_stratified_splits(
    root_dir: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    class_names: list[str] | None = None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """Split dataset files into train/val/test, stratified by class label.

    Splitting happens at the **file** level (not batch level) so no augmented
    view of an image can leak across splits. Returns three lists of
    ``(filepath, label)`` tuples.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    class_names = class_names or get_class_names(root_dir)
    samples = _list_samples(root_dir, class_names)
    if not samples:
        raise RuntimeError(f"No images found under {root_dir}")

    files = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # First split: train vs (val+test).
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
    train_idx, rest_idx = next(sss1.split(files, labels))

    rest_files = [files[i] for i in rest_idx]
    rest_labels = [labels[i] for i in rest_idx]

    # Second split: divide the remainder into val vs test.
    test_share = test_frac / (val_frac + test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_share, random_state=seed)
    val_idx, test_idx = next(sss2.split(rest_files, rest_labels))

    train = [(files[i], labels[i]) for i in train_idx]
    val = [(rest_files[i], rest_labels[i]) for i in val_idx]
    test = [(rest_files[i], rest_labels[i]) for i in test_idx]
    return train, val, test


def _print_stats(data_dir: str) -> None:
    """CLI helper: print class counts and split sizes for a quick sanity check."""
    class_names = get_class_names(data_dir)
    samples = _list_samples(data_dir, class_names)
    counts = Counter(label for _, label in samples)
    print(f"Data dir: {data_dir}")
    print(f"Classes found: {len(class_names)}")
    print(f"Total images: {len(samples)}")
    per_class = {class_names[lbl]: counts.get(lbl, 0) for lbl in range(len(class_names))}
    print(f"Per-class counts: {per_class}")

    train, val, test = make_stratified_splits(data_dir)
    print(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print ASL dataset statistics.")
    parser.add_argument("--data_dir", default="data/sample", help="Path to class folders.")
    args = parser.parse_args()
    _print_stats(args.data_dir)
