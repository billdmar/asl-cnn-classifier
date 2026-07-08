"""Tests for the OpenCV inference script (static-image + helpers).

Only the STATIC-IMAGE and non-interactive paths are tested — no cv2 GUI calls
(imshow/waitKey) are ever triggered. The webcam loop's "cannot open device"
branch is exercised by mocking ``cv2.VideoCapture``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src import infer_camera as ic
from src.dataset import get_eval_transforms

DATA_DIR = "data/sample"


def _repo_path(rel):
    repo_root = ic.__file__.rsplit("/src/", 1)[0]
    return os.path.join(repo_root, rel)


def _sample_image():
    return _repo_path(os.path.join(DATA_DIR, "A", "0.png"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_center_roi_clamps():
    x1, y1, x2, y2 = ic._center_roi(100, 200, 300)
    # ROI clamped to min(300, 100, 200) = 100.
    assert (x2 - x1) == (y2 - y1) == 100
    assert x1 >= 0 and y1 >= 0


def test_classify_source_modes():
    assert ic._classify_source("0") == ("webcam", 0)
    assert ic._classify_source("5") == ("webcam", 5)
    assert ic._classify_source("clip.mp4")[0] == "video"
    assert ic._classify_source("pic.png")[0] == "image"
    # Unknown extension defaults to image (headless-safe).
    assert ic._classify_source("weird.xyz")[0] == "image"


def test_predict_roi():
    model, class_names = ic.load_checkpoint("missing.pth", torch.device("cpu"))
    roi = np.zeros((64, 64, 3), dtype=np.uint8)
    label, conf = ic.predict_roi(
        roi, model, get_eval_transforms(), torch.device("cpu"), class_names
    )
    assert label in class_names
    assert 0.0 <= conf <= 1.0


def test_annotate_returns_frame():
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    out = ic._annotate(frame, (10, 10, 90, 90), "A", 0.5, fps=12.3)
    assert out.shape == frame.shape


# --------------------------------------------------------------------------- #
# load_checkpoint: fallback + real checkpoint
# --------------------------------------------------------------------------- #
def test_load_checkpoint_fallback(capsys):
    model, class_names = ic.load_checkpoint("nope.pth", torch.device("cpu"))
    assert len(class_names) == 29
    assert "UNTRAINED" in capsys.readouterr().out
    out = model(torch.randn(1, 3, 128, 128))
    assert out.shape == (1, 29)


def test_load_checkpoint_real(tmp_path, capsys):
    from src.dataset import get_class_names
    from src.model import build_model

    class_names = get_class_names()
    model = build_model("custom_cnn", num_classes=len(class_names))
    ckpt = tmp_path / "m.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": "custom_cnn",
            "class_names": class_names,
            "config": {},
            "val_accuracy": 0.5,
        },
        ckpt,
    )
    loaded, loaded_names = ic.load_checkpoint(ckpt, torch.device("cpu"))
    assert loaded_names == class_names
    assert "Loaded checkpoint" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Static image run + main dispatch
# --------------------------------------------------------------------------- #
def test_run_static_image(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    model, class_names = ic.load_checkpoint("missing.pth", torch.device("cpu"))
    from pathlib import Path

    rc = ic.run_static_image(
        Path(_sample_image()),
        model,
        get_eval_transforms(),
        torch.device("cpu"),
        class_names,
        roi_size=128,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "predicted_class" in out
    snaps = list((tmp_path / "artifacts" / "camera_snapshots").glob("*.png"))
    assert len(snaps) == 1


def test_run_static_image_unreadable(tmp_path, capsys):
    from pathlib import Path

    model, class_names = ic.load_checkpoint("missing.pth", torch.device("cpu"))
    rc = ic.run_static_image(
        Path(tmp_path / "no_such.png"),
        model,
        get_eval_transforms(),
        torch.device("cpu"),
        class_names,
        roi_size=64,
    )
    assert rc == 0
    assert "could not read image" in capsys.readouterr().out


def test_main_image_dispatch(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            config=None,
            checkpoint="missing.pth",
            source=_sample_image(),
            device="cpu",
            roi_size=128,
        ),
    )
    rc = ic.main()
    assert rc == 0
    assert "predicted_class" in capsys.readouterr().out


def test_main_config_load_branch(tmp_path, monkeypatch, capsys):
    """A provided (loadable) config exercises the config-validation branch."""
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"a": 1}')
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            config=str(cfg),
            checkpoint="missing.pth",
            source=_sample_image(),
            device="cpu",
            roi_size=128,
        ),
    )
    assert ic.main() == 0


def test_run_camera_cannot_open(monkeypatch, capsys):
    class FakeCap:
        def __init__(self, *a, **k):
            pass

        def isOpened(self):
            return False

    monkeypatch.setattr(ic.cv2, "VideoCapture", FakeCap)
    model, class_names = ic.load_checkpoint("missing.pth", torch.device("cpu"))
    rc = ic.run_camera(
        0, model, get_eval_transforms(), torch.device("cpu"), class_names, 128
    )
    assert rc == 0
    assert "could not open video source" in capsys.readouterr().out


def test_main_webcam_dispatch_cannot_open(monkeypatch, capsys):
    """main() with source='0' dispatches to run_camera (mocked, cannot open)."""

    class FakeCap:
        def __init__(self, *a, **k):
            pass

        def isOpened(self):
            return False

    monkeypatch.setattr(ic.cv2, "VideoCapture", FakeCap)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            config=None,
            checkpoint="missing.pth",
            source="0",
            device="cpu",
            roi_size=128,
        ),
    )
    assert ic.main() == 0
    assert "could not open video source" in capsys.readouterr().out
