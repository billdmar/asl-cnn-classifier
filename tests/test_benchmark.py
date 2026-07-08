"""Tests for the latency/robustness benchmark script.

Runs on CPU with a tiny ``num_frames`` so the timing loops are fast. Headless
(matplotlib Agg). The checkpoint-missing fallback path is used so no trained
model is required.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src import benchmark as bench
from src.dataset import get_eval_transforms
from src.checkpoint import load_checkpoint

DATA_DIR = "data/sample"


def _repo_path(rel):
    repo_root = bench.__file__.rsplit("/src/", 1)[0]
    return os.path.join(repo_root, rel)


def test_load_frames_synthetic():
    frames = bench._load_frames(None, 5)
    assert len(frames) == 5
    assert frames[0].shape == (128, 128, 3)


def test_load_frames_from_dir():
    frames = bench._load_frames(_repo_path(DATA_DIR), 10)
    assert len(frames) == 10
    assert frames[0].ndim == 3


def test_benchmark_device():
    model, _ = load_checkpoint("missing.pth", torch.device("cpu"))
    frames = bench._load_frames(None, 8)
    stats = bench._benchmark_device(
        model, frames, get_eval_transforms(), torch.device("cpu")
    )
    assert set(stats) == {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "fps"}
    assert stats["mean_ms"] >= 0.0


def test_ablation_and_chart(tmp_path):
    model, _ = load_checkpoint("missing.pth", torch.device("cpu"))
    frames = bench._load_frames(None, 6)
    ablation = bench._ablation(model, frames, torch.device("cpu"))
    stages = {a["stage"] for a in ablation}
    assert stages == {
        "full",
        "skip_colorjitter",
        "skip_resize",
        "skip_normalize",
        "model_only",
    }
    chart = tmp_path / "ablation.png"
    bench._save_ablation_chart(ablation, chart)
    assert chart.exists() and chart.stat().st_size > 0


def test_degrade_and_unknown():
    from PIL import Image

    import pytest

    from src.degradations import degrade

    img = Image.new("RGB", (16, 16), (90, 100, 110))
    for kind in [
        "clean",
        "gaussian_blur",
        "jpeg_q20",
        "brightness_0.4",
        "brightness_1.8",
        "salt_pepper_5pct",
    ]:
        assert degrade(img, kind).size == (16, 16)
    with pytest.raises(ValueError):
        degrade(img, "bogus")


def test_distribution_shift_with_data():
    from src.dataset import get_class_names

    data = _repo_path(DATA_DIR)
    model, _ = load_checkpoint("missing.pth", torch.device("cpu"))
    class_names = get_class_names(data)
    results = bench._distribution_shift(model, data, class_names, torch.device("cpu"))
    assert all(0.0 <= v <= 1.0 for v in results.values())


def test_distribution_shift_empty_dir(tmp_path):
    model, _ = load_checkpoint("missing.pth", torch.device("cpu"))
    # Empty dir -> make_stratified_splits raises RuntimeError -> all zeros.
    results = bench._distribution_shift(
        model, str(tmp_path), ["A", "B"], torch.device("cpu")
    )
    assert all(v == 0.0 for v in results.values())


def test_benchmark_main_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data = _repo_path(DATA_DIR)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            checkpoint="missing.pth",
            device="cpu",
            num_frames=12,
            source=data,
            test_dir=data,
        ),
    )
    rc = bench.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Latency benchmark" in out
    assert (tmp_path / "artifacts" / "benchmark_results.json").exists()
    assert (tmp_path / "artifacts" / "benchmark_ablation.png").exists()
    assert (tmp_path / "artifacts" / "distribution_shift.json").exists()


def test_main_with_synthetic_frames(tmp_path, monkeypatch):
    """source=None exercises the synthetic-frame branch in main."""
    monkeypatch.chdir(tmp_path)
    data = _repo_path(DATA_DIR)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            checkpoint="missing.pth",
            device="cpu",
            num_frames=8,
            source=None,
            test_dir=data,
        ),
    )
    assert bench.main() == 0
    assert isinstance(np.float64(1.0), np.floating)  # numpy import sanity
