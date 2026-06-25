"""Tests for the eval-overlap guard (scripts/check_eval_overlap.py).

Synthetic, no network: plant a near-duplicate of an eval image in the train set
and assert it is flagged, while random images are not.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from scripts import check_eval_overlap as ceo


def _save(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def test_planted_duplicate_is_flagged_randoms_are_not(tmp_path):
    rng = np.random.default_rng(0)
    train = tmp_path / "train"
    eval_ = tmp_path / "eval"

    # One eval image in class A.
    base = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    _save(eval_ / "A" / "0.png", base)

    # Train class A: a near-duplicate of the eval image (tiny noise) + a random.
    noise = rng.integers(-3, 4, size=base.shape)
    near = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    _save(train / "A" / "0.png", near)  # should flag
    _save(train / "A" / "1.png",
          rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))  # should not

    report = ceo.check_overlap(train_dir=train, eval_dir=eval_, threshold=22)

    assert report["total_train_images"] == 2
    assert report["total_flagged"] == 1
    flagged = {p["train"] for p in report["closest_pairs"]}
    assert any(f.endswith("A/0.png") for f in flagged)
    assert not any(f.endswith("A/1.png") for f in flagged)


def test_no_overlap_reports_clean(tmp_path):
    rng = np.random.default_rng(1)
    train = tmp_path / "train"
    eval_ = tmp_path / "eval"
    _save(eval_ / "A" / "0.png",
          rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    _save(train / "A" / "0.png",
          rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))

    report = ceo.check_overlap(train_dir=train, eval_dir=eval_, threshold=22)
    assert report["total_flagged"] == 0
    assert report["contaminated"] is False


def test_only_same_class_compared(tmp_path):
    """A train image identical to an eval image of a DIFFERENT class isn't flagged."""
    rng = np.random.default_rng(2)
    train = tmp_path / "train"
    eval_ = tmp_path / "eval"
    base = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    _save(eval_ / "A" / "0.png", base)
    _save(train / "B" / "0.png", base)  # same pixels, different class
    # Give A and B both folders in each root so the class intersection includes both.
    _save(eval_ / "B" / "0.png",
          rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    _save(train / "A" / "0.png",
          rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))

    report = ceo.check_overlap(train_dir=train, eval_dir=eval_, threshold=22)
    # The identical pair is cross-class (train B vs eval A) → not compared → not flagged.
    assert report["total_flagged"] == 0
