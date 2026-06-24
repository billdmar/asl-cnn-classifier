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
            fit_temperature=False,
            inference_out=None,
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
    # Temperature scaling is wired but inert by default (no --fit_temperature).
    assert payload["temperature"] == 1.0
    assert payload["temperature_fit_on"] == "none"


# --------------------------------------------------------------------------- #
# Temperature scaling (Guo et al., 2017) — the calibration plumbing.
# --------------------------------------------------------------------------- #
def _synthetic_logits(n: int = 2000, num_classes: int = 5, seed: int = 0):
    """Perfectly-calibrated logits + labels SAMPLED from their softmax.

    Drawing each label from ``Categorical(softmax(logit_row))`` makes the base
    logits genuinely calibrated: ``T = 1`` minimizes the NLL by construction. So
    scaling the logits by a factor ``c`` is exactly equivalent to those same
    calibrated logits at temperature ``c``, and ``fit_temperature`` on the scaled
    logits must recover ``~c``. A large ``n`` keeps the sampling noise small.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=2.0, size=(n, num_classes))
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.array(
        [rng.choice(num_classes, p=probs[i]) for i in range(n)], dtype=np.int64
    )
    return logits.astype(np.float64), labels


def test_fit_temperature_recovers_known_scaling():
    """Scaling calibrated logits by 2.0 should be recovered as T ~= 2.0.

    If logits L are well-fit at T=1, then logits (L * 2) are equivalent to L at
    T=2, so fitting temperature on the scaled logits must recover ~2.0.
    """
    logits, labels = _synthetic_logits()
    scaled = logits * 2.0
    t = calmod.fit_temperature(scaled, labels)
    assert abs(t - 2.0) < 0.15


def test_fit_temperature_near_one_on_calibrated_logits():
    """On already-calibrated synthetic logits, the fitted T is ~1.0."""
    logits, labels = _synthetic_logits(seed=7)
    t = calmod.fit_temperature(logits, labels)
    assert abs(t - 1.0) < 0.2


def test_fit_temperature_is_positive_and_finite():
    logits, labels = _synthetic_logits(seed=3)
    t = calmod.fit_temperature(logits * 0.5, labels)
    assert t > 0.0 and np.isfinite(t)


def test_fit_temperature_empty_returns_identity():
    assert calmod.fit_temperature(np.empty((0, 0)), np.array([])) == 1.0


def test_collect_logits_empty_loader():
    logits, labels = calmod.collect_logits(
        torch.nn.Linear(2, 2), [], torch.device("cpu")
    )
    assert logits.size == 0 and labels.size == 0


def test_resolve_temperature_inert_without_flag():
    """No --fit_temperature -> identity, regardless of data dir."""
    logits, labels = _synthetic_logits()
    t, fit_on = calmod._resolve_temperature(False, "data/asl_real", logits, labels)
    assert t == 1.0 and fit_on == "none"


def test_resolve_temperature_refuses_sample_dir_even_with_flag():
    """Even with the flag, the synthetic sample fixture must not be fit on."""
    logits, labels = _synthetic_logits()
    t, fit_on = calmod._resolve_temperature(True, "data/sample", logits, labels)
    assert t == 1.0 and fit_on == "none"


def test_resolve_temperature_fits_on_real_dir_with_flag():
    """Flag + real dir -> a fitted T tagged with the dir it was fit on."""
    logits, labels = _synthetic_logits()
    t, fit_on = calmod._resolve_temperature(True, "data/asl_real", logits * 2.0, labels)
    assert fit_on == "data/asl_real"
    assert abs(t - 2.0) < 0.15


def test_write_inference_calibration_merges_temperature(tmp_path):
    """The inference file keeps existing real ECE and gains temperature fields."""
    from src.utils import load_json, save_json

    path = tmp_path / "calibration.json"
    save_json(path, {"ece": 0.0464, "accuracy": 0.968, "bins": {"bin_count": [1]}})
    calmod.write_inference_calibration(path, 1.0, "none")
    payload = load_json(path)
    # Real ECE preserved.
    assert payload["ece"] == 0.0464
    assert payload["accuracy"] == 0.968
    assert payload["bins"] == {"bin_count": [1]}
    # Temperature merged in, inert.
    assert payload["temperature"] == 1.0
    assert payload["temperature_fit_on"] == "none"
    assert "inert" in payload["temperature_note"].lower()


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
