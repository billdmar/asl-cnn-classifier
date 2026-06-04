"""Tests for ASLDataset, stratified splitting, and class-name resolution.

All tests run against the committed ``data/sample`` fixture (232 images across
29 classes) so they're fast and require no external data download.
"""

from __future__ import annotations

from collections import Counter

from src.dataset import (
    CLASS_NAMES,
    ASLDataset,
    get_class_names,
    get_eval_transforms,
    make_stratified_splits,
)

DATA_DIR = "data/sample"
TOTAL = 232
NUM_CLASSES = 29


def test_dataset_length_and_item_shape():
    ds = ASLDataset(root_dir=DATA_DIR)
    assert len(ds) == TOTAL

    item = ds[0]
    assert isinstance(item, tuple)
    assert len(item) == 3

    tensor, label, filepath = item
    assert tuple(tensor.shape) == (3, 128, 128)
    assert isinstance(label, int)
    assert 0 <= label <= 28
    assert isinstance(filepath, str)


def test_get_class_names_matches_canonical():
    names = get_class_names(DATA_DIR)
    assert len(names) == NUM_CLASSES
    assert names == CLASS_NAMES


def test_stratified_splits_sum_and_coverage():
    train, val, test = make_stratified_splits(DATA_DIR)
    assert len(train) + len(val) + len(test) == TOTAL

    # Every class must be represented in the train split.
    train_labels = {label for _, label in train}
    assert len(train_labels) == NUM_CLASSES

    # Proportions roughly 70/15/15 — generous tolerance for the tiny fixture.
    assert abs(len(train) / TOTAL - 0.70) < 0.10
    assert abs(len(val) / TOTAL - 0.15) < 0.10
    assert abs(len(test) / TOTAL - 0.15) < 0.10

    # No file leaks across splits.
    train_files = {f for f, _ in train}
    val_files = {f for f, _ in val}
    test_files = {f for f, _ in test}
    assert train_files.isdisjoint(val_files)
    assert train_files.isdisjoint(test_files)
    assert val_files.isdisjoint(test_files)


def test_dataset_from_samples_list():
    train, _val, _test = make_stratified_splits(DATA_DIR)
    ds = ASLDataset(samples=train, transform=get_eval_transforms())
    assert len(ds) == len(train)

    tensor, label, _filepath = ds[0]
    assert tuple(tensor.shape) == (3, 128, 128)
    assert isinstance(label, int)

    # Sanity: labels in the split are a subset of the valid class index range.
    counts = Counter(label for _, label in train)
    assert all(0 <= lbl <= 28 for lbl in counts)
