"""Tests for Grad-CAM explainability.

CPU-only and headless. The model is the random-init ``custom_cnn`` fallback
(no trained checkpoint required); these tests verify the Grad-CAM *plumbing and
shapes/normalization*, not saliency quality.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from PIL import Image

from src import gradcam as gcmod
from src.dataset import IMAGE_SIZE, get_eval_transforms
from src.model import build_model

DATA_DIR = "data/sample"


def _repo_path(rel):
    repo_root = gcmod.__file__.rsplit("/src/", 1)[0]
    return os.path.join(repo_root, rel)


def _toy_model():
    model = build_model("custom_cnn", num_classes=29)
    model.eval()
    return model


def test_find_target_layer_custom_cnn():
    model = _toy_model()
    layer = gcmod.find_target_layer(model)
    assert isinstance(layer, torch.nn.Conv2d)
    # Last conv in CustomCNN.features outputs 256 channels.
    assert layer.out_channels == 256


def test_find_target_layer_mobilenet():
    model = build_model("mobilenet_v2", num_classes=29, pretrained=False)
    layer = gcmod.find_target_layer(model)
    assert isinstance(layer, torch.nn.Conv2d)


def test_find_target_layer_no_conv_raises():
    model = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError):
        gcmod.find_target_layer(model)


def test_gradcam_shape_and_normalization():
    model = _toy_model()
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with gcmod.GradCAM(model) as engine:
        cam, class_idx = engine(x)
    # Heatmap is upsampled to input spatial size.
    assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE)
    # Normalized to [0, 1].
    assert cam.min() >= 0.0 - 1e-6
    assert cam.max() <= 1.0 + 1e-6
    assert 0 <= class_idx < 29


def test_gradcam_explicit_class_idx():
    model = _toy_model()
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with gcmod.GradCAM(model) as engine:
        _cam, class_idx = engine(x, class_idx=7)
    assert class_idx == 7


def test_gradcam_rejects_batch():
    model = _toy_model()
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    engine = gcmod.GradCAM(model)
    with pytest.raises(ValueError):
        engine(x)
    engine.remove()


def test_gradcam_hooks_removed():
    model = _toy_model()
    engine = gcmod.GradCAM(model)
    assert len(engine._handles) == 2
    engine.remove()
    assert engine._handles == []


def test_overlay_shape_and_dtype():
    cam = np.linspace(0, 1, IMAGE_SIZE * IMAGE_SIZE, dtype=np.float32).reshape(
        IMAGE_SIZE, IMAGE_SIZE
    )
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (100, 120, 140))
    overlay = gcmod.overlay_cam_on_image(image, cam)
    assert overlay.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert overlay.mode == "RGB"
    arr = np.asarray(overlay)
    assert arr.dtype == np.uint8


def test_run_gradcam_writes_file(tmp_path):
    model = _toy_model()
    # Use a real sample image so the eval transform path is exercised.
    sample = os.path.join(_repo_path(DATA_DIR), "A", "0.png")
    class_names = [str(i) for i in range(29)]
    out_path, label = gcmod.run_gradcam(
        sample, model, torch.device("cpu"), class_names, out_dir=tmp_path
    )
    assert out_path.exists() and out_path.stat().st_size > 0
    assert out_path.name == f"{label}.png"


def test_gradcam_main_end_to_end(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.chdir(tmp_path)
    sample = os.path.join(_repo_path(DATA_DIR), "A", "0.png")
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            checkpoint="missing.pth", source=sample, device="cpu"
        ),
    )
    rc = gcmod.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Saved Grad-CAM overlay" in out
    # Artifact written under cwd (tmp_path)/artifacts/gradcam/.
    overlays = list((tmp_path / "artifacts" / "gradcam").glob("*.png"))
    assert len(overlays) == 1


def test_transform_input_is_correct_size():
    # Sanity: the eval transform produces the spatial size Grad-CAM upsamples to.
    image = Image.new("RGB", (200, 200), (10, 20, 30))
    tensor = get_eval_transforms()(image)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
