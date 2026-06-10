"""Tests for calibration (ECE) and the per-class F1 plot.

The ECE tests are the rigorous part: they use hand-constructed inputs whose
correct ECE is known *analytically*, so we verify the math itself rather than
any model behavior. CPU-only and headless.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src import calibration as calmod
from src import plot_per_class as ppc

DATA_DIR = "data/sample"


def _repo_path(rel):
    repo_root = calmod.__file__.rsplit("/src/", 1)[0]
    return os.path.join(repo_root, rel)


# --------------------------------------------------------------------------- #
# ECE math — analytic ground-truth cases (the rigorous part).
# --------------------------------------------------------------------------- #
def test_ece_perfectly_calibrated_is_zero():
    """A perfectly calibrated set: in each used bin, accuracy == confidence.

    Bin (0.4, 0.5]: 2 samples @ conf 0.5, exactly 1 correct -> acc 0.5 == conf,
        gap 0.
    Bin (0.9, 1.0]: 2 samples @ conf 1.0, both correct -> acc 1.0 == conf, gap 0.
    ECE = 0 exactly.
    """
    # Bin (0.4, 0.5]: 2 samples @ conf 0.5, exactly 1 correct -> acc 0.5 == conf.
    # Bin (0.9, 1.0]: 2 samples @ conf 1.0, both correct -> acc 1.0 == conf.
    confidences = np.array([0.5, 0.5, 1.0, 1.0])
    predictions = np.array([0, 0, 0, 0])
    labels = np.array([0, 1, 0, 0])  # first bin: 1/2 correct; second: 2/2 correct
    ece, _ = calmod.compute_ece(confidences, predictions, labels, n_bins=10)
    assert ece == 0.0


def test_ece_known_miscalibrated_value():
    """A single bin, fully analytic.

    All 4 samples have confidence 0.8 -> they land in bin (0.7, 0.8].
    Suppose 2 of 4 are correct -> accuracy 0.5, confidence 0.8.
    ECE = (4/4) * |0.5 - 0.8| = 0.3.
    """
    confidences = np.array([0.8, 0.8, 0.8, 0.8])
    predictions = np.array([0, 0, 0, 0])
    labels = np.array([0, 0, 1, 1])  # 2 correct of 4
    ece, stats = calmod.compute_ece(confidences, predictions, labels, n_bins=10)
    assert abs(ece - 0.3) < 1e-12
    # The (0.7, 0.8] bin (index 7) holds all 4 samples.
    assert stats["bin_count"][7] == 4
    assert stats["bin_acc"][7] == 0.5
    assert abs(stats["bin_conf"][7] - 0.8) < 1e-12


def test_ece_two_bin_weighted_average():
    """Two unequally populated bins -> sample-weighted average gap.

    Bin (0.5, 0.6]: 1 sample @ conf 0.6, incorrect -> acc 0.0, gap 0.6.
    Bin (0.9, 1.0]: 3 samples @ conf 1.0, all correct -> acc 1.0, gap 0.0.
    ECE = (1/4)*0.6 + (3/4)*0.0 = 0.15.
    """
    confidences = np.array([0.6, 1.0, 1.0, 1.0])
    predictions = np.array([0, 1, 1, 1])
    labels = np.array([9, 1, 1, 1])  # first wrong, rest correct
    ece, _ = calmod.compute_ece(confidences, predictions, labels, n_bins=10)
    assert abs(ece - 0.15) < 1e-12


def test_ece_zero_confidence_lands_in_first_bin():
    """conf == 0.0 must be counted (first bin is closed on the left)."""
    confidences = np.array([0.0, 0.0])
    predictions = np.array([0, 0])
    labels = np.array([0, 1])  # acc 0.5, conf 0.0 -> gap 0.5
    ece, stats = calmod.compute_ece(confidences, predictions, labels, n_bins=10)
    assert stats["bin_count"][0] == 2
    assert abs(ece - 0.5) < 1e-12


def test_ece_bin_boundaries_use_upper_inclusive():
    """A confidence exactly on a bin edge goes to the lower bin (upper-inclusive)."""
    # 0.1 -> bin 0 (0.0, 0.1]; 0.2 -> bin 1 (0.1, 0.2].
    confidences = np.array([0.1, 0.2])
    predictions = np.array([0, 0])
    labels = np.array([0, 0])
    _ece, stats = calmod.compute_ece(confidences, predictions, labels, n_bins=10)
    assert stats["bin_count"][0] == 1
    assert stats["bin_count"][1] == 1


# --------------------------------------------------------------------------- #
# collect_predictions + diagram + end-to-end main()
# --------------------------------------------------------------------------- #
def test_collect_predictions_empty_loader():
    confs, preds, trues = calmod.collect_predictions(
        torch.nn.Linear(2, 2), [], torch.device("cpu")
    )
    assert confs.size == 0 and preds.size == 0 and trues.size == 0


def test_save_reliability_diagram(tmp_path):
    _ece, stats = calmod.compute_ece(
        np.array([0.8, 0.8, 0.8, 0.8]),
        np.array([0, 0, 0, 0]),
        np.array([0, 0, 1, 1]),
        n_bins=10,
    )
    path = tmp_path / "reliability.png"
    calmod.save_reliability_diagram(stats, 0.3, path)
    assert path.exists() and path.stat().st_size > 0


def test_calibration_main_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data = _repo_path(DATA_DIR)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            checkpoint="missing.pth",
            data_dir=data,
            device="cpu",
            n_bins=10,
            seed=42,
        ),
    )
    rc = calmod.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Calibration summary" in out
    cal_json = tmp_path / "artifacts" / "calibration.json"
    assert cal_json.exists()
    assert (tmp_path / "artifacts" / "reliability_diagram.png").exists()
    from src.utils import load_json

    payload = load_json(cal_json)
    assert 0.0 <= payload["ece"] <= 1.0
    assert "demonstration" in payload["note"].lower()


# --------------------------------------------------------------------------- #
# Per-class F1 plot
# --------------------------------------------------------------------------- #
def test_plot_per_class_f1_writes_file(tmp_path):
    metrics = {
        "macro_f1": 0.5,
        "per_class": {
            "A": {"f1": 0.9, "support": 3},
            "B": {"f1": 0.1, "support": 2},
            "C": {"f1": 0.5, "support": 1},
        },
    }
    out = tmp_path / "per_class_f1.png"
    result = ppc.plot_per_class_f1(metrics, out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_plot_per_class_f1_missing_section_raises(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        ppc.plot_per_class_f1({"macro_f1": 0.5}, tmp_path / "x.png")


def test_plot_per_class_main_end_to_end(tmp_path, monkeypatch, capsys):
    from src.utils import save_json

    metrics_path = tmp_path / "metrics.json"
    save_json(
        metrics_path,
        {
            "macro_f1": 0.4,
            "note": "demonstration on synthetic fixture",
            "per_class": {"A": {"f1": 0.4}, "B": {"f1": 0.2}},
        },
    )
    out_path = tmp_path / "per_class_f1.png"
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(metrics=str(metrics_path), out=str(out_path)),
    )
    rc = ppc.main()
    assert rc == 0
    assert out_path.exists()
    assert "Saved per-class F1" in capsys.readouterr().out
