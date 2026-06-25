"""Tests for the Hugging Face dataset ingestion path and class-count-agnostic eval.

No real network is used: ``datasets.load_dataset`` is monkeypatched to return a
tiny in-memory fake dataset (a few PIL images across 2-3 classes) and we assert
the class-folder layout + per-class counts are written correctly. A separate
test confirms ``src.eval`` derives its label set from ``class_names`` (so a
non-29 class count evaluates without a length mismatch).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

from src import download_hf_data as dl


class _FakeClassLabel:
    """Minimal stand-in for ``datasets.ClassLabel`` carrying ``.names``."""

    def __init__(self, names: list[str]) -> None:
        self.names = names


class _FakeDataset:
    """Tiny iterable that mimics the bits of a HF ``Dataset`` we rely on."""

    def __init__(self, rows: list[dict], names: list[str]) -> None:
        self._rows = rows
        self.features = {"label": _FakeClassLabel(names)}

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key: str) -> list:
        return [row[key] for row in self._rows]


def _make_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (8, 8), color)


def _install_fake_datasets(monkeypatch, fake_ds: _FakeDataset) -> None:
    """Inject a fake ``datasets`` module so the import inside download() works."""
    module = types.ModuleType("datasets")
    module.load_dataset = lambda *a, **k: fake_ds  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


def test_download_writes_class_folder_layout(monkeypatch, tmp_path, capsys):
    rows = [
        {"image": _make_image((10, 20, 30)), "label": 0},
        {"image": _make_image((40, 50, 60)), "label": 0},
        {"image": _make_image((70, 80, 90)), "label": 1},
        {"image": _make_image((11, 22, 33)), "label": 2},
        {"image": _make_image((44, 55, 66)), "label": 2},
        {"image": _make_image((77, 88, 99)), "label": 2},
    ]
    fake_ds = _FakeDataset(rows, names=["A", "B", "C"])
    _install_fake_datasets(monkeypatch, fake_ds)

    out_dir = tmp_path / "asl_real"
    counts = dl.download(out_dir=str(out_dir), max_per_class=None)

    assert counts == {"A": 2, "B": 1, "C": 3}
    # One folder per class, populated with PNGs.
    for name, expected in counts.items():
        folder = out_dir / name
        assert folder.is_dir()
        pngs = sorted(folder.glob("*.png"))
        assert len(pngs) == expected
        # Files named 0.png, 1.png, ... and readable as RGB.
        assert {p.name for p in pngs} == {f"{i}.png" for i in range(expected)}
        with Image.open(pngs[0]) as im:
            assert im.mode == "RGB"

    out = capsys.readouterr().out
    assert "Per-class counts" in out


def test_download_respects_max_per_class(monkeypatch, tmp_path):
    rows = [{"image": _make_image((i, i, i)), "label": 0} for i in range(5)]
    rows += [{"image": _make_image((i, i, i)), "label": 1} for i in range(4)]
    fake_ds = _FakeDataset(rows, names=["A", "B"])
    _install_fake_datasets(monkeypatch, fake_ds)

    out_dir = tmp_path / "capped"
    counts = dl.download(out_dir=str(out_dir), max_per_class=2)

    assert counts == {"A": 2, "B": 2}
    assert len(list((out_dir / "A").glob("*.png"))) == 2
    assert len(list((out_dir / "B").glob("*.png"))) == 2


def test_download_handles_string_labels(monkeypatch, tmp_path):
    """A dataset that already yields string labels (no ClassLabel names)."""
    rows = [
        {"image": _make_image((1, 2, 3)), "label": "X"},
        {"image": _make_image((4, 5, 6)), "label": "Y"},
        {"image": _make_image((7, 8, 9)), "label": "X"},
    ]
    # No names on the feature -> _resolve_class_names falls back to observed
    # string labels (sorted), then images are routed by the string label.
    fake_ds = _FakeDataset(rows, names=[])
    _install_fake_datasets(monkeypatch, fake_ds)

    out_dir = tmp_path / "strlabels"
    counts = dl.download(out_dir=str(out_dir))

    assert counts.get("X") == 2
    assert counts.get("Y") == 1


def test_resolve_class_names_prefers_feature_names():
    fake_ds = _FakeDataset([], names=["A", "B", "Z"])
    assert dl._resolve_class_names(fake_ds) == ["A", "B", "Z"]


def test_label_to_name_int_and_str():
    names = ["A", "B", "C"]
    assert dl._label_to_name(1, names) == "B"
    assert dl._label_to_name("Q", names) == "Q"


def test_download_missing_datasets_raises(monkeypatch, tmp_path):
    """If importing ``datasets`` fails, a clear RuntimeError is raised."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("no datasets")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    import pytest

    with pytest.raises(RuntimeError, match="datasets"):
        dl.download(out_dir=str(tmp_path / "x"))


def test_download_load_failure_raises(monkeypatch, tmp_path):
    """A failure inside load_dataset is surfaced as a RuntimeError."""

    def boom(*a, **k):
        raise ValueError("network down")

    module = types.ModuleType("datasets")
    module.load_dataset = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)

    import pytest

    with pytest.raises(RuntimeError, match="Failed to load dataset"):
        dl.download(out_dir=str(tmp_path / "x"))


# --------------------------------------------------------------------------- #
# eval.py is class-count-agnostic (no hardcoded 29)
# --------------------------------------------------------------------------- #
def test_eval_labels_derive_from_class_names(monkeypatch, tmp_path, capsys):
    """Run eval.main on a synthetic 3-class dataset; it must not assume 29.

    We build a tiny 3-folder dataset on disk and a 3-class checkpoint, then run
    the real eval pipeline. A hardcoded 29-label set would make
    classification_report/confusion_matrix mismatch the 3-class model.
    """
    import torch

    from src import eval as evalmod
    from src.model import build_model

    # Build a tiny 3-class dataset: data_dir/<A|B|C>/<n>.png
    data_dir = tmp_path / "tiny"
    class_names = ["A", "B", "C"]
    rng = np.random.default_rng(0)
    for name in class_names:
        d = data_dir / name
        d.mkdir(parents=True)
        # Enough per class so the 70/15/15 stratified split keeps >=2 each.
        for i in range(12):
            arr = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")

    # Save a real 3-class checkpoint so load_checkpoint builds a 3-output model.
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "model.pth"
    model = build_model("custom_cnn", num_classes=len(class_names))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": "custom_cnn",
            "class_names": class_names,
            "config": {"num_classes": len(class_names)},
            "val_accuracy": 0.0,
        },
        ckpt_path,
    )

    monkeypatch.chdir(tmp_path)
    import argparse

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            config=None,
            checkpoint=str(ckpt_path),
            data_dir=str(data_dir),
            device="cpu",
            seed=42,
            distribution_shift=False,
        ),
    )

    rc = evalmod.main()
    assert rc == 0

    # The written metrics report exactly the 3 classes — no 29-class artifacts.
    import json

    metrics = json.loads(Path("artifacts/metrics.json").read_text())
    assert set(metrics["per_class"]) == set(class_names)
    assert len(metrics["per_class"]) == 3


# --- Filename-labeled ingestion (snapshot_download path) --------------------


def _install_fake_snapshot(monkeypatch, snapshot_root: Path) -> None:
    """Inject a fake huggingface_hub whose snapshot_download returns a local dir."""
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = lambda *a, **k: str(snapshot_root)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_download_from_filenames_parses_class_from_name(monkeypatch, tmp_path):
    """Class is the leading filename letter; layout + counts must be correct."""
    # Build a fake snapshot: <root>/train/images/<LETTER>..._jpg.rf.<hash>.jpg
    images_dir = tmp_path / "snap" / "train" / "images"
    images_dir.mkdir(parents=True)
    specs = {"A": 3, "B": 2, "C": 1}
    for letter, n in specs.items():
        for i in range(n):
            _make_image((10, 20, 30)).save(
                images_dir / f"{letter}1{i}_jpg.rf.deadbeef{i}.jpg"
            )
    # A non-letter-prefixed file must be dropped.
    _make_image((0, 0, 0)).save(images_dir / "9x_jpg.rf.nope.jpg")

    _install_fake_snapshot(monkeypatch, tmp_path / "snap")
    out = tmp_path / "out"
    counts = dl.download_from_filenames(
        hf_id="atalaydenknalbant/asl-dataset",
        out_dir=str(out),
        split_subdir="train/images",
        regex=r"^([A-Za-z])",
        max_per_class=None,
    )
    assert counts == {"A": 3, "B": 2, "C": 1}
    assert sorted(p.name for p in (out / "A").iterdir()) == ["0.png", "1.png", "2.png"]
    assert not (out / "9").exists()  # non-letter dropped


def test_download_from_filenames_respects_max_per_class(monkeypatch, tmp_path):
    images_dir = tmp_path / "snap" / "valid" / "images"
    images_dir.mkdir(parents=True)
    for i in range(5):
        _make_image((1, 2, 3)).save(images_dir / f"A1{i}_jpg.rf.h{i}.jpg")

    _install_fake_snapshot(monkeypatch, tmp_path / "snap")
    counts = dl.download_from_filenames(
        hf_id="x/y",
        out_dir=str(tmp_path / "out"),
        split_subdir="valid/images",
        regex=r"^([A-Za-z])",
        max_per_class=2,
    )
    assert counts == {"A": 2}


def test_filename_dataset_registry_has_atalay():
    spec = dl.FILENAME_DATASETS["atalaydenknalbant"]
    assert spec["hf_id"] == "atalaydenknalbant/asl-dataset"
    assert set(spec["split_subdirs"]) == {"train", "valid", "test"}
