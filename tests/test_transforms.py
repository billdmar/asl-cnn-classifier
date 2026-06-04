"""Tests for the canonical preprocessing transforms.

Guards two correctness properties:
* The TRAIN pipeline must NOT contain a horizontal flip — ASL signs are not
  flip-invariant, so flipping would create mislabeled data.
* The EVAL pipeline must be deterministic and apply normalization.
"""

from __future__ import annotations

from PIL import Image

import torch

from src.dataset import get_eval_transforms, get_train_transforms

SAMPLE_IMAGE = "data/sample/A/0.png"


def test_train_transforms_have_no_flip():
    pipeline = get_train_transforms().transforms
    class_names = [t.__class__.__name__ for t in pipeline]
    assert not any(
        "Flip" in name for name in class_names
    ), f"Train pipeline must not flip ASL images; found: {class_names}"


def test_eval_transforms_are_deterministic():
    transform = get_eval_transforms()
    image = Image.open(SAMPLE_IMAGE).convert("RGB")
    first = transform(image)
    second = transform(image)
    assert torch.equal(first, second)


def test_eval_transform_shape_and_normalization():
    transform = get_eval_transforms()
    image = Image.open(SAMPLE_IMAGE).convert("RGB")
    tensor = transform(image)

    assert tuple(tensor.shape) == (3, 128, 128)

    # Normalization (ImageNet mean/std) shifts values outside [0, 1]; a raw
    # ToTensor output would be bounded to [0, 1], so a negative min proves the
    # Normalize step ran.
    assert tensor.min().item() < 0.0
