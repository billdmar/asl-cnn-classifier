"""Tests for the Gradio demo app's headless prediction path.

These tests import ``app`` and call :func:`app.predict` directly — they never
launch the Gradio server (``demo.launch()`` is guarded behind ``__main__``), so
the suite stays fast and CI-safe. With no checkpoint present, ``predict`` runs
the untrained random-init fallback; we assert only on the *structure* of the
output (valid labels, a probability distribution), never on accuracy.
"""

from __future__ import annotations

import os

import pytest
from PIL import Image

import app
from src.dataset import get_class_names


def _repo_path(rel):
    repo_root = app.__file__.rsplit("/app.py", 1)[0]
    return os.path.join(repo_root, rel)


def _sample_image() -> Image.Image:
    return Image.open(_repo_path("data/sample/A/0.png"))


def test_predict_returns_valid_label_and_probs():
    top_probs, summary = app.predict(_sample_image())
    class_names = get_class_names()

    # Top-K dict: valid labels, probabilities in [0, 1].
    assert isinstance(top_probs, dict)
    assert 0 < len(top_probs) <= app.TOP_K
    for label, prob in top_probs.items():
        assert label in class_names
        assert 0.0 <= prob <= 1.0

    # Summary names the most-likely class.
    assert isinstance(summary, str)
    best_label = max(top_probs, key=top_probs.get)
    assert best_label in summary


def test_predict_top_probs_are_descending():
    top_probs, _ = app.predict(_sample_image())
    values = list(top_probs.values())
    assert values == sorted(values, reverse=True)


def test_predict_converts_non_rgb_image():
    # A grayscale ("L") image must be accepted (converted to RGB internally).
    gray = _sample_image().convert("L")
    top_probs, _ = app.predict(gray)
    assert len(top_probs) > 0


def test_predict_none_raises():
    with pytest.raises(ValueError):
        app.predict(None)


def test_is_using_trained_checkpoint_is_bool():
    assert isinstance(app.is_using_trained_checkpoint(), bool)
