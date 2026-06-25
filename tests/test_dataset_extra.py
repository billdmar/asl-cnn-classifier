"""Supplementary dataset tests covering error branches and the CLI helper."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PIL import Image

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


# --- Near-duplicate-aware (dedup) split -------------------------------------


def _make_duplicate_dataset(tmp_path, classes=("A", "B"), groups_per_class=6):
    """Build a synthetic dataset: each class has several groups of 3 near-twins.

    Within a group the three frames are tiny pixel-noise variants of one base
    image (perceptually near-identical → must cluster together); different groups
    use distinct random base images (perceptually far apart → separate clusters).
    Returns the dataset root and a ``filepath -> intended_group_key`` mapping.
    """
    rng = np.random.default_rng(0)
    group_of: dict[str, str] = {}
    for cls in classes:
        cdir = tmp_path / cls
        cdir.mkdir()
        for g in range(groups_per_class):
            base = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            for k in range(3):
                noise = rng.integers(-3, 4, size=base.shape)
                arr = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
                # Zero-pad so lexical sort == sequential frame order, matching
                # the real dataset where one session's frames are contiguous.
                fname = f"{g * 3 + k:04d}.png"
                fpath = cdir / fname
                Image.fromarray(arr).save(fpath)
                group_of[str(fpath)] = f"{cls}-{g}"
    return str(tmp_path), group_of


def test_dedup_default_path_unchanged():
    """dedup=False must remain byte-identical to the original split."""
    a = make_stratified_splits(_repo_path(DATA_DIR))
    b = make_stratified_splits(_repo_path(DATA_DIR), dedup=False)
    assert a == b


def test_dedup_no_cluster_straddles_splits(tmp_path):
    root, group_of = _make_duplicate_dataset(tmp_path)
    train, val, test = make_stratified_splits(root, dedup=True, seed=42)

    # Coverage: every file accounted for exactly once, no file leaks across splits.
    all_files = [f for split in (train, val, test) for f, _ in split]
    assert len(all_files) == len(set(all_files))
    assert len(all_files) == sum(len(s) for s in (train, val, test))

    # The core guarantee: no near-duplicate group appears in more than one split.
    def groups(split):
        return {group_of[f] for f, _ in split}

    g_tr, g_va, g_te = groups(train), groups(val), groups(test)
    assert g_tr.isdisjoint(g_te)
    assert g_tr.isdisjoint(g_va)
    assert g_va.isdisjoint(g_te)


def test_dedup_clusters_near_duplicates_together(tmp_path):
    """Each intended group of 3 near-twins should collapse into one cluster."""
    root, group_of = _make_duplicate_dataset(tmp_path)
    samples = dataset._list_samples(root, dataset.get_class_names(root))
    group_ids = dataset._phash_groups(samples)

    # Map intended-group-key -> set of assigned cluster ids; near-twins must share.
    by_intended: dict[str, set[int]] = {}
    for (fpath, _label), gid in zip(samples, group_ids):
        by_intended.setdefault(group_of[fpath], set()).add(gid)
    assert all(len(ids) == 1 for ids in by_intended.values())


def test_dedup_is_deterministic(tmp_path):
    root, _ = _make_duplicate_dataset(tmp_path)
    first = make_stratified_splits(root, dedup=True, seed=42)
    second = make_stratified_splits(root, dedup=True, seed=42)
    assert first == second
