"""Tests for the evaluation script and its helpers.

Runs end-to-end on the committed ``data/sample`` fixture, CPU-only and headless
(matplotlib Agg backend). The checkpoint-missing fallback path is exercised so
no trained model is required.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src import eval as evalmod

DATA_DIR = "data/sample"


def _repo_path(rel):
    repo_root = evalmod.__file__.rsplit("/src/", 1)[0]
    return os.path.join(repo_root, rel)


# --------------------------------------------------------------------------- #
# Helper-level tests
# --------------------------------------------------------------------------- #
def test_most_confused_pairs():
    cm = np.array([[5, 2, 0], [1, 4, 0], [0, 3, 6]])
    pairs = evalmod.most_confused_pairs(cm, ["A", "B", "C"], top_k=2)
    assert len(pairs) == 2
    # Largest off-diagonal is (C->B: 3).
    assert pairs[0] == {"true": "C", "pred": "B", "count": 3}


def test_run_inference_empty_loader():
    model = evalmod.run_inference  # noqa: F841 - referenced for clarity
    y_true, y_pred = evalmod.run_inference(
        torch.nn.Linear(2, 2), [], torch.device("cpu")
    )
    assert y_true.size == 0 and y_pred.size == 0


def test_save_per_class_errors_empty(tmp_path):
    path = tmp_path / "errors.txt"
    evalmod.save_per_class_errors([], path)
    assert "no off-diagonal" in path.read_text().lower()


def test_save_per_class_errors_with_pairs(tmp_path):
    path = tmp_path / "errors.txt"
    evalmod.save_per_class_errors([{"true": "A", "pred": "B", "count": 3}], path)
    assert "A -> B: 3" in path.read_text()


def test_save_confusion_matrix(tmp_path):
    cm = np.eye(3, dtype=int)
    path = tmp_path / "cm.png"
    evalmod.save_confusion_matrix(cm, ["A", "B", "C"], path)
    assert path.exists() and path.stat().st_size > 0


def test_degrade_variants():
    from PIL import Image

    from src.degradations import degrade

    img = Image.new("RGB", (16, 16), (120, 130, 140))
    for kind in [
        "clean",
        "gaussian_blur",
        "jpeg_q20",
        "brightness_0.4",
        "brightness_1.8",
        "salt_pepper_5pct",
    ]:
        out = degrade(img, kind)
        assert out.size == (16, 16)


def test_degrade_unknown_raises():
    from PIL import Image

    import pytest

    from src.degradations import degrade

    img = Image.new("RGB", (8, 8))
    with pytest.raises(ValueError):
        degrade(img, "nope")


def test_distribution_shift_function():
    from src.dataset import get_class_names, make_stratified_splits
    from src.checkpoint import load_checkpoint

    data = _repo_path(DATA_DIR)
    class_names = get_class_names(data)
    _train, _val, test = make_stratified_splits(data, class_names=class_names)
    model, _ = load_checkpoint("does_not_exist.pth", torch.device("cpu"))
    results = evalmod.distribution_shift(model, test[:3], torch.device("cpu"))
    assert set(results) >= {"clean", "gaussian_blur", "jpeg_q20"}
    assert all(0.0 <= v <= 1.0 for v in results.values())


# --------------------------------------------------------------------------- #
# Full main() run (checkpoint-missing fallback + distribution shift)
# --------------------------------------------------------------------------- #
def test_eval_main_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data = _repo_path(DATA_DIR)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            config=None,
            checkpoint="missing.pth",
            data_dir=data,
            device="cpu",
            seed=42,
            distribution_shift=True,
        ),
    )
    rc = evalmod.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Evaluation summary" in out
    # Artifacts written under cwd (tmp_path).
    assert (tmp_path / "artifacts" / "metrics.json").exists()
    assert (tmp_path / "artifacts" / "confusion_matrix.png").exists()
    assert (tmp_path / "artifacts" / "per_class_errors.txt").exists()
