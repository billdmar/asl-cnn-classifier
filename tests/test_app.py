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
import torch
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


def test_predict_non_image_raises():
    # A non-image input (e.g. a stray string) must raise a clean ValueError
    # rather than an opaque AttributeError deep in the transform.
    with pytest.raises(ValueError):
        app.predict("not-an-image")


def test_is_using_trained_checkpoint_is_bool():
    assert isinstance(app.is_using_trained_checkpoint(), bool)


def test_model_is_loaded_once_as_singleton():
    # The model bundle is built once at import; repeated accessors return the
    # identical object (no per-request reload of weights).
    first = app.get_model()
    second = app.get_model()
    assert first is second
    assert first is app.MODEL
    # The singleton was constructed at import time, before any predict() call.
    assert isinstance(first.model, torch.nn.Module)


def test_full_probability_distribution_sums_to_one():
    # The top-K view only sums to ~1.0 when K covers every class, so verify the
    # underlying softmax over ALL classes is a valid distribution using the
    # shared singleton's model + transform (no second copy of preprocessing).
    bundle = app.get_model()
    tensor = bundle.transform(_sample_image().convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(bundle.model(tensor.to(bundle.device)), dim=1)
    assert probs.shape[1] == len(bundle.class_names)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-4)
    assert torch.all(probs >= 0.0)


def test_topk_size_matches_config():
    top_probs, _ = app.predict(_sample_image())
    # With far more than TOP_K classes, predict surfaces exactly TOP_K of them.
    assert len(top_probs) == min(app.TOP_K, len(app.MODEL.class_names))


def test_example_paths_exist_and_are_distinct():
    rows = app._example_paths()
    assert len(rows) >= 1
    seen = set()
    for row in rows:
        assert len(row) == 1
        path = row[0]
        assert os.path.exists(_repo_path(path))
        assert path not in seen
        seen.add(path)


def test_banner_is_honest_about_untrained_model():
    # On a fresh checkout (no checkpoint) the banner must flag the random-init
    # fallback and call its predictions meaningless. If a checkpoint IS present
    # the banner instead reports it as trained — assert whichever is accurate.
    banner = app._banner_markdown(app.get_model())
    assert app.REPO_URL in banner
    if app.is_using_trained_checkpoint():
        assert "trained checkpoint is loaded" in banner
    else:
        assert "untrained" in banner
        assert "meaningless" in banner


def test_build_demo_constructs_blocks_without_launching():
    # build_demo() must construct a Gradio Blocks UI without starting a server.
    gr = pytest.importorskip("gradio")
    demo = app.build_demo()
    assert isinstance(demo, gr.Blocks)
    # No server should have been started by construction.
    assert getattr(demo, "is_running", False) is False
