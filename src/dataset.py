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


def get_train_transforms(
    image_size: int = IMAGE_SIZE, heavy: bool = False
) -> transforms.Compose:
    """Augmentation pipeline for training.

    NOTE: deliberately **no horizontal flip** — ASL signs are not
    flip-invariant (e.g. b/d, p/q are mirror images, and several letters differ
    only by orientation). Flipping would create mislabeled training data.

    Args:
        image_size: Output side length.
        heavy: When ``True``, use the aggressive domain-augmentation pipeline
            intended to close the benchmark→real-world gap — wider
            crop/rotation/affine ranges, stronger lighting/contrast jitter,
            random Gaussian blur, grayscale (a crude skin-tone invariance), and
            random erasing (occlusion robustness). These deliberately make the
            *training* distribution look more like a cluttered webcam; the
            eval transform stays untouched so the held-out metric is comparable.
    """
    if heavy:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)
                ),
                transforms.RandomRotation(25),
                transforms.RandomAffine(
                    degrees=0, translate=(0.18, 0.18), scale=(0.8, 1.2), shear=12
                ),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.4, hue=0.08
                ),
                transforms.RandomGrayscale(p=0.15),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
            ),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
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


def _list_samples(
    root_dir: str | Path, class_names: list[str]
) -> list[tuple[str, int]]:
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
            assert root_dir is not None  # guaranteed by the check above
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


# Default perceptual-hash Hamming distance under which two frames are treated as
# near-duplicates of one another. Chosen from the real dataset (sequential video
# frames of the same signer/session): consecutive frames of one recording fall
# within this radius, while distinct signs/sessions sit well above it. Connected
# components of the near-duplicate graph become atomic groups that never straddle
# splits — this is what makes the held-out metric honest.
DEDUP_PHASH_THRESHOLD = 22

# Each frame is only compared against this many subsequent frames within the same
# class. The dataset is sequential video, so a recording's near-duplicates are
# always neighbours — this keeps clustering O(n·window) instead of O(n²) on the
# ~11k-image real set while still linking every contiguous near-duplicate run.
DEDUP_WINDOW = 8


def _phash_groups(
    samples: list[tuple[str, int]],
    threshold: int = DEDUP_PHASH_THRESHOLD,
    window: int = DEDUP_WINDOW,
) -> list[int]:
    """Cluster near-duplicate frames into atomic groups via perceptual hashing.

    Builds a near-duplicate graph: within each class, every frame is linked to a
    later frame when their perceptual-hash (pHash) Hamming distance is ``<=
    threshold``. The connected components of that graph are the groups returned
    here (one integer group id per input sample, aligned to ``samples`` order).

    Because the real dataset is sequential video — each class is a concatenation
    of recording sessions whose frames are mutual near-duplicates — comparing
    only the next ``window`` frames inside a class is enough to chain a whole
    session into a single component without an O(n^2) all-pairs scan.

    Args:
        samples: ``(filepath, label)`` tuples in a deterministic order.
        threshold: Max pHash Hamming distance counted as a near-duplicate edge.
        window: Number of subsequent same-class frames each frame is compared to.

    Returns:
        A list of group ids, ``group_ids[i]`` for ``samples[i]``. Frames in the
        same near-duplicate cluster share an id; isolated frames get unique ids.
    """
    import imagehash  # local import: only needed on the opt-in dedup path.

    n = len(samples)
    hashes = [
        imagehash.phash(Image.open(filepath).convert("RGB"))
        for filepath, _label in samples
    ]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Indices grouped by class, preserving sample order (sequential frames).
    by_label: dict[int, list[int]] = {}
    for idx, (_filepath, label) in enumerate(samples):
        by_label.setdefault(label, []).append(idx)

    for indices in by_label.values():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 : pos + 1 + window]:
                if hashes[i] - hashes[j] <= threshold:
                    union(i, j)

    # Re-map roots to small contiguous group ids for readability/reproducibility.
    root_to_group: dict[int, int] = {}
    group_ids: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        group_ids.append(root_to_group[root])
    return group_ids


def make_stratified_splits(
    root_dir: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    class_names: list[str] | None = None,
    dedup: bool = False,
    dedup_threshold: int = DEDUP_PHASH_THRESHOLD,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """Split dataset files into train/val/test, stratified by class label.

    Splitting happens at the **file** level (not batch level) so no augmented
    view of an image can leak across splits. Returns three lists of
    ``(filepath, label)`` tuples.

    Leakage caveat (why ``dedup`` exists): the file-level random split is
    *honest only when files are independent*. The real dataset is sequential
    video — a class folder holds long runs of near-duplicate frames from the
    same signer/session — so a random split scatters a frame into train and its
    near-twin into test, inflating the test metric. Set ``dedup=True`` to make
    near-duplicate frames an atomic group (clustered by perceptual hash) that is
    assigned to a single split, so no frame and its near-twin ever straddle
    train/test. This lowers the headline number but makes it trustworthy.

    Args:
        dedup: When ``True``, cluster near-duplicate frames (perceptual hash) and
            keep every cluster wholly within one split (group-aware split). When
            ``False`` (default) the behavior is the original file-level random
            split — byte-identical to before, so existing repro is unchanged.
        dedup_threshold: pHash Hamming distance under which two frames are
            near-duplicates. Only used when ``dedup=True``.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    class_names = class_names or get_class_names(root_dir)
    samples = _list_samples(root_dir, class_names)
    if not samples:
        raise RuntimeError(f"No images found under {root_dir}")

    files = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    if dedup:
        return _make_grouped_splits(
            samples,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            dedup_threshold=dedup_threshold,
        )

    # First split: train vs (val+test).
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac + test_frac, random_state=seed
    )
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


def _make_grouped_splits(
    samples: list[tuple[str, int]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    dedup_threshold: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """Group-aware train/val/test split: no near-duplicate cluster straddles splits.

    Near-duplicate frames are clustered with :func:`_phash_groups` and the whole
    cluster is assigned to one split via :class:`GroupShuffleSplit` (two stages,
    mirroring the stratified path). Splitting on groups — not files — is what
    removes the train/test leakage. Stratification by class is best-effort: the
    grouped split is keyed on clusters, but groups are class-pure here (clusters
    only link same-class frames), so class balance is preserved at the group
    level.

    Returns the same ``(filepath, label)`` contract as
    :func:`make_stratified_splits`.
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = _phash_groups(samples, threshold=dedup_threshold)

    # First split: train-groups vs (val+test)-groups.
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=val_frac + test_frac, random_state=seed
    )
    train_idx, rest_idx = next(gss1.split(samples, groups=groups))

    rest_samples = [samples[i] for i in rest_idx]
    rest_groups = [groups[i] for i in rest_idx]

    # Second split: divide the remainder groups into val vs test.
    test_share = test_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest_samples, groups=rest_groups))

    train = [samples[i] for i in train_idx]
    val = [rest_samples[i] for i in val_idx]
    test = [rest_samples[i] for i in test_idx]
    return train, val, test


def _print_stats(data_dir: str) -> None:
    """CLI helper: print class counts and split sizes for a quick sanity check."""
    class_names = get_class_names(data_dir)
    samples = _list_samples(data_dir, class_names)
    counts = Counter(label for _, label in samples)
    print(f"Data dir: {data_dir}")
    print(f"Classes found: {len(class_names)}")
    print(f"Total images: {len(samples)}")
    per_class = {
        class_names[lbl]: counts.get(lbl, 0) for lbl in range(len(class_names))
    }
    print(f"Per-class counts: {per_class}")

    train, val, test = make_stratified_splits(data_dir)
    print(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print ASL dataset statistics.")
    parser.add_argument(
        "--data_dir", default="data/sample", help="Path to class folders."
    )
    args = parser.parse_args()
    _print_stats(args.data_dir)
