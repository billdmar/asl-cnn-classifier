"""Tests for the pre-crop dataset script.

Covers the pure sketch detector (:func:`is_line_drawing`) and the pipeline's
drop-sketches / drop-no-hand behaviour. MediaPipe detection is monkeypatched so
the test does not require the runtime or model asset.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from scripts import precrop_dataset as pc


def _gray_sketch(size=(64, 64)) -> Image.Image:
    """A near-colorless line drawing: gray background with darker gray strokes."""
    arr = np.full((*size, 3), 200, dtype=np.uint8)  # light gray, R==G==B
    arr[20:40, 20:40] = 60  # a darker gray "stroke" — still colorless
    return Image.fromarray(arr)


def _color_photo(size=(64, 64)) -> Image.Image:
    """A skin-tone photo: clearly non-gray (R > G > B)."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    arr[..., 0] = 220  # R
    arr[..., 1] = 150  # G
    arr[..., 2] = 120  # B
    return Image.fromarray(arr)


def test_is_line_drawing_flags_grayscale_sketch():
    assert pc.is_line_drawing(_gray_sketch()) is True


def test_is_line_drawing_passes_color_photo():
    assert pc.is_line_drawing(_color_photo()) is False


def test_drop_sketches_excludes_line_drawings(tmp_path, monkeypatch):
    # Build a 1-class dataset: 2 color photos + 1 sketch.
    cls = tmp_path / "A"
    cls.mkdir()
    _color_photo().save(cls / "0.png")
    _color_photo().save(cls / "1.png")
    _gray_sketch().save(cls / "2.png")

    # Stub out MediaPipe: pretend every image has a detectable hand (identity crop).
    monkeypatch.setattr(pc, "_build_landmarker", lambda _p: _FakeLandmarker())
    monkeypatch.setattr(pc, "detect_and_crop", lambda image, **kw: image, raising=True)

    out = tmp_path / "cropped"
    report = pc.precrop_dataset(in_dir=tmp_path, out_dir=out, drop_sketches=True)

    assert report["total_images"] == 3
    assert report["total_sketches"] == 1
    assert report["total_written"] == 2  # sketch dropped
    # The sketch file must not be written.
    assert not (out / "A" / "2.png").exists()
    assert (out / "A" / "0.png").exists()


def test_keeps_sketches_when_flag_off(tmp_path, monkeypatch):
    cls = tmp_path / "A"
    cls.mkdir()
    _color_photo().save(cls / "0.png")
    _gray_sketch().save(cls / "1.png")

    monkeypatch.setattr(pc, "_build_landmarker", lambda _p: _FakeLandmarker())
    monkeypatch.setattr(pc, "detect_and_crop", lambda image, **kw: image)

    out = tmp_path / "cropped"
    report = pc.precrop_dataset(in_dir=tmp_path, out_dir=out, drop_sketches=False)

    assert report["total_sketches"] == 1
    assert report["total_written"] == 2  # sketch kept


class _FakeLandmarker:
    def close(self):
        pass
