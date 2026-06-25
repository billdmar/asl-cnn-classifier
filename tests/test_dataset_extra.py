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


def test_dedup_clustering_invariant_to_input_order(tmp_path):
    """Clustering must be frame-sequence based, not input-order based.

    The real dataset names frames numerically but _list_samples sorts them
    lexically (0,1,10,100,…), scattering true neighbours. _phash_groups
    natural-sorts internally, so shuffling the input must not change which
    frames cluster together (only the group-id labels may be renumbered).
    """
    root, group_of = _make_duplicate_dataset(tmp_path)
    samples = dataset._list_samples(root, dataset.get_class_names(root))

    shuffled = list(reversed(samples))
    base_ids = dataset._phash_groups(samples)
    shuf_ids = dataset._phash_groups(shuffled)

    def partition(sample_list, ids):
        # frozenset of co-clustered files per group — order-independent identity.
        members: dict[int, set[str]] = {}
        for (fpath, _label), gid in zip(sample_list, ids):
            members.setdefault(gid, set()).add(fpath)
        return {frozenset(v) for v in members.values()}

    assert partition(samples, base_ids) == partition(shuffled, shuf_ids)


# --- Multi-source merge (diverse training) ----------------------------------


def _make_class_dataset(root, classes, n_per_class=6):
    """Write n tiny random RGB PNGs per class under root/<CLASS>/."""
    rng = np.random.default_rng(0)
    for cls in classes:
        cdir = root / cls
        cdir.mkdir(parents=True)
        for i in range(n_per_class):
            arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cdir / f"{i:03d}.png")


def test_make_splits_samples_param_matches_root_dir():
    """samples= must produce the byte-identical split to the root_dir path."""
    names = dataset.get_class_names(_repo_path(DATA_DIR))
    samples = dataset._list_samples(_repo_path(DATA_DIR), names)

    from_root = make_stratified_splits(_repo_path(DATA_DIR), class_names=names)
    from_samples = make_stratified_splits(samples=samples, class_names=names)
    assert from_root == from_samples


def test_make_splits_requires_root_or_samples():
    with pytest.raises(ValueError):
        make_stratified_splits()  # neither root_dir nor samples


def test_get_union_class_names_unions_across_dirs(tmp_path):
    """A-Z dir unioned with A-Y dir yields the full sorted A-Z (incl. J, Z)."""
    az = tmp_path / "az"
    ay = tmp_path / "ay"
    _make_class_dataset(az, [chr(c) for c in range(ord("A"), ord("Z") + 1)])
    _make_class_dataset(ay, [c for c in
                             (chr(x) for x in range(ord("A"), ord("Z") + 1))
                             if c not in ("J", "Z")])
    union = dataset.get_union_class_names([az, ay])
    assert union == [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    assert "J" in union and "Z" in union


def test_merged_two_dir_split_no_leak_both_sources(tmp_path):
    """Merging two dirs: union classes, no file leak, both sources represented."""
    d1 = tmp_path / "src1"
    d2 = tmp_path / "src2"
    _make_class_dataset(d1, ["A", "B"], n_per_class=10)
    _make_class_dataset(d2, ["B", "C"], n_per_class=10)  # overlapping + new class

    class_names = dataset.get_union_class_names([d1, d2])
    assert class_names == ["A", "B", "C"]

    merged = dataset._list_samples(d1, class_names) + dataset._list_samples(
        d2, class_names
    )
    train, val, test = make_stratified_splits(
        samples=merged, class_names=class_names, seed=42
    )

    all_files = [f for split in (train, val, test) for f, _ in split]
    assert len(all_files) == len(set(all_files))  # no file in two splits
    assert len(all_files) == len(merged)  # every file placed once
    # Both source dirs contribute to the training split.
    train_files = [f for f, _ in train]
    assert any("src1" in f for f in train_files)
    assert any("src2" in f for f in train_files)
