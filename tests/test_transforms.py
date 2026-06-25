"""Tests for the canonical preprocessing transforms.

Guards two correctness properties:
* The TRAIN pipeline must NOT contain a horizontal flip — ASL signs are not
  flip-invariant, so flipping would create mislabeled data.
* The EVAL pipeline must be deterministic and apply normalization.
"""

from __future__ import annotations

from PIL import Image

import torch

import pytest

from src.dataset import AUG_REGIMES, get_eval_transforms, get_train_transforms

SAMPLE_IMAGE = "data/sample/A/0.png"


def _class_names(pipeline) -> list[str]:
    return [t.__class__.__name__ for t in pipeline.transforms]


def test_train_transforms_have_no_flip():
    pipeline = get_train_transforms().transforms
    class_names = [t.__class__.__name__ for t in pipeline]
    assert not any(
        "Flip" in name for name in class_names
    ), f"Train pipeline must not flip ASL images; found: {class_names}"


@pytest.mark.parametrize("regime", AUG_REGIMES)
def test_no_regime_flips_asl_images(regime):
    """No augmentation regime may flip — ASL signs aren't flip-invariant."""
    names = _class_names(get_train_transforms(regime=regime))
    assert not any("Flip" in n for n in names), names


def test_default_and_heavy_back_compat_unchanged():
    """regime=None must reproduce the legacy standard/heavy pipelines exactly."""
    assert _class_names(get_train_transforms()) == _class_names(
        get_train_transforms(regime="standard")
    )
    assert _class_names(get_train_transforms(heavy=True)) == _class_names(
        get_train_transforms(regime="heavy")
    )


def test_medium_regime_is_between_standard_and_heavy():
    """Medium adds erasing over standard but omits heavy's grayscale/blur."""
    standard = _class_names(get_train_transforms(regime="standard"))
    medium = _class_names(get_train_transforms(regime="medium"))
    heavy = _class_names(get_train_transforms(regime="heavy"))

    # Medium has RandomErasing (standard does not).
    assert "RandomErasing" not in standard
    assert "RandomErasing" in medium
    # Medium deliberately omits the most crop-destructive ops heavy uses
    # (grayscale, and Gaussian blur wrapped in RandomApply).
    assert "RandomGrayscale" in heavy and "RandomApply" in heavy
    assert "RandomGrayscale" not in medium and "RandomApply" not in medium


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        get_train_transforms(regime="bogus")


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
