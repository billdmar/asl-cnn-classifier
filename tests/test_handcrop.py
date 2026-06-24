"""Tests for the Python hand-crop geometry (mirrors web/lib/handcrop.ts).

The pure box math is tested directly (no MediaPipe runtime). The crop contract
— square box expanded by CROP_MARGIN, clamped to the unit image — must stay in
sync with the browser implementation so train/eval and live inference crop
identically.
"""

from __future__ import annotations

from PIL import Image

from src.handcrop import CROP_MARGIN, crop_box_from_landmarks, crop_image


def test_empty_landmarks_returns_none() -> None:
    assert crop_box_from_landmarks([]) is None


def test_box_is_square() -> None:
    # A non-square landmark spread must still yield a square box.
    box = crop_box_from_landmarks([(0.3, 0.4), (0.5, 0.9)])
    assert box is not None
    assert abs(box.width - box.height) < 1e-9


def test_margin_expands_box() -> None:
    # Two points 0.2 apart in x, 0.0 in y → larger dim 0.2, side = 0.2*(1+2m).
    box = crop_box_from_landmarks([(0.4, 0.5), (0.6, 0.5)], margin=0.35)
    assert box is not None
    expected_side = 0.2 * (1 + 2 * 0.35)
    assert abs(box.width - expected_side) < 1e-9


def test_box_clamped_to_unit_image() -> None:
    # Landmarks near a corner: the box must not extend past [0,1].
    box = crop_box_from_landmarks([(0.95, 0.95), (0.99, 0.99)])
    assert box is not None
    assert box.x >= 0.0
    assert box.y >= 0.0
    assert box.x + box.width <= 1.0 + 1e-9
    assert box.y + box.height <= 1.0 + 1e-9


def test_default_margin_matches_web_contract() -> None:
    # The shared geometry constant must match web/lib/handcrop.ts CROP_MARGIN.
    assert CROP_MARGIN == 0.35


def test_crop_image_returns_subregion() -> None:
    img = Image.new("RGB", (100, 100), "white")
    box = crop_box_from_landmarks([(0.25, 0.25), (0.75, 0.75)])
    assert box is not None
    cropped = crop_image(img, box)
    # The crop should be a non-empty sub-image no larger than the original.
    assert 0 < cropped.size[0] <= 100
    assert 0 < cropped.size[1] <= 100
