"""Supplementary dataset tests covering error branches and the CLI helper."""

from __future__ import annotations

import os
import sys

import pytest

from src import dataset
from src.dataset import ASLDataset, make_stratified_splits

DATA_DIR = "data/sample"


def _repo_path(rel):
    repo_root = dataset.__file__.rsplit("/src/", 1)[0]
    return os.path.join(repo_root, rel)


def test_asldataset_requires_root_or_samples():
    with pytest.raises(ValueError):
        ASLDataset()


def test_make_stratified_splits_bad_fractions():
    with pytest.raises(ValueError):
        make_stratified_splits(
            _repo_path(DATA_DIR), train_frac=0.5, val_frac=0.3, test_frac=0.3
        )


def test_make_stratified_splits_empty_dir(tmp_path):
    with pytest.raises(RuntimeError):
        make_stratified_splits(str(tmp_path))


def test_get_class_names_empty_dir_returns_canonical(tmp_path):
    names = dataset.get_class_names(str(tmp_path))
    assert names == list(dataset.CLASS_NAMES)


def test_print_stats_cli(monkeypatch, capsys):
    import runpy

    monkeypatch.setattr(sys, "argv", ["dataset", "--data_dir", _repo_path(DATA_DIR)])
    runpy.run_module("src.dataset", run_name="__main__")
    out = capsys.readouterr().out
    assert "Total images" in out
    assert "Split sizes" in out


def test_build_model_transfer_unsupported_arch():
    from src.model import TransferModel

    with pytest.raises(ValueError):
        TransferModel(arch="vgg16")
