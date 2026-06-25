"""Tests for the dataset train/gate splitter (scripts/split_hemg.py).

Synthetic, no network: build a tiny class-folder dataset, split it, and assert
the two partitions are disjoint, class-stratified, deterministic, and cover every
file exactly once.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from scripts import split_hemg


def _make(root, classes, n_per_class=10):
    rng = np.random.default_rng(0)
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n_per_class):
            arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i:03d}.png")


def test_split_is_disjoint_and_complete(tmp_path):
    src = tmp_path / "src"
    _make(src, ["A", "B", "C"], n_per_class=10)
    m = split_hemg.split_dataset(src, tmp_path / "out", gate_frac=0.2, seed=42)

    train_files = list((tmp_path / "out_train").rglob("*.png"))
    gate_files = list((tmp_path / "out_gate").rglob("*.png"))
    assert m["num_train"] == len(train_files) == 24  # 80% of 30
    assert m["num_gate"] == len(gate_files) == 6  # 20% of 30

    # Disjoint by filename within each class (copies preserve names).
    train_keys = {(f.parent.name, f.name) for f in train_files}
    gate_keys = {(f.parent.name, f.name) for f in gate_files}
    assert train_keys.isdisjoint(gate_keys)
    assert len(train_keys) + len(gate_keys) == 30


def test_split_is_stratified(tmp_path):
    src = tmp_path / "src"
    _make(src, ["A", "B", "C"], n_per_class=10)
    split_hemg.split_dataset(src, tmp_path / "out", gate_frac=0.2, seed=42)
    # Each class contributes 2 of its 10 to the gate (stratified).
    for cls in ("A", "B", "C"):
        assert len(list((tmp_path / "out_gate" / cls).glob("*.png"))) == 2


def test_split_is_deterministic(tmp_path):
    src = tmp_path / "src"
    _make(src, ["A", "B"], n_per_class=10)
    m1 = split_hemg.split_dataset(src, tmp_path / "a", gate_frac=0.2, seed=42)
    m2 = split_hemg.split_dataset(src, tmp_path / "b", gate_frac=0.2, seed=42)
    a_gate = sorted(f.name for f in (tmp_path / "a_gate").rglob("*.png"))
    b_gate = sorted(f.name for f in (tmp_path / "b_gate").rglob("*.png"))
    assert a_gate == b_gate
    assert m1["num_gate"] == m2["num_gate"]
