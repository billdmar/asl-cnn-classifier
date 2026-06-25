"""Tests for the cross-dataset real-world eval harness and label normalization.

No network, model, or MediaPipe runtime is required: we test the pure metric
aggregation (:func:`src.eval_realworld._build_metrics`), the JSON schema it
produces, and the second-dataset label normalization / drop logic in
``download_hf_data``. A tiny synthetic on-disk dataset exercises the full
``evaluate`` path with hand-crop OFF and a stubbed model.
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import torch
from PIL import Image

from src import download_hf_data as dl
from src import eval_realworld as erw


# --------------------------------------------------------------------------- #
# Second-dataset label normalization / drop logic
# --------------------------------------------------------------------------- #
def test_normalize_class_name_uppercases_and_keeps_letters():
    assert dl._normalize_class_name("a") == "A"
    assert dl._normalize_class_name("Z") == "Z"
    assert dl._normalize_class_name(" m ") == "M"


def test_normalize_class_name_drops_non_letters():
    for raw in ["del", "space", "nothing", "1", "AA", ""]:
        assert dl._normalize_class_name(raw) is None


def test_resolve_dataset_spec_defaults_and_registry():
    # None -> default Marxulia constant (backward-compatible).
    assert dl._resolve_dataset_spec(None, None) == (dl.HF_DATASET, dl.HF_SPLIT)
    # Friendly name expands to its registered (id, split).
    assert dl._resolve_dataset_spec("asl_letters", None) == dl.DATASETS["asl_letters"]
    # Raw id passthrough with an explicit split override.
    assert dl._resolve_dataset_spec("foo/bar", "test") == ("foo/bar", "test")


class _FakeClassLabel:
    def __init__(self, names: list[str]) -> None:
        self.names = names


class _FakeDataset:
    def __init__(self, rows: list[dict], names: list[str]) -> None:
        self._rows = rows
        self.features = {"label": _FakeClassLabel(names)}

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key: str) -> list:
        return [row[key] for row in self._rows]


def _img(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (8, 8), color)


def test_download_lowercases_and_drops_extra_classes(monkeypatch, tmp_path):
    """Lowercase labels uppercase to A..Z; non-letter classes are dropped."""
    rows = [
        {"image": _img((1, 1, 1)), "label": 0},  # "a" -> A
        {"image": _img((2, 2, 2)), "label": 1},  # "b" -> B
        {"image": _img((3, 3, 3)), "label": 2},  # "space" -> dropped
        {"image": _img((4, 4, 4)), "label": 2},  # "space" -> dropped
    ]
    fake = _FakeDataset(rows, names=["a", "b", "space"])
    module = types.ModuleType("datasets")
    module.load_dataset = lambda *a, **k: fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)

    out_dir = tmp_path / "crossval"
    counts = dl.download(out_dir=str(out_dir), dataset="asl_letters")

    assert counts == {"A": 1, "B": 1}
    assert not (out_dir / "space").exists()
    assert (out_dir / "A" / "0.png").exists()


# --------------------------------------------------------------------------- #
# Metric aggregation (pure) + JSON schema
# --------------------------------------------------------------------------- #
def test_build_metrics_schema_and_values():
    class_names = ["A", "B", "C"]
    # true: A A B B C C ; pred: A B B B C A  -> 4/6 correct.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    m = erw._build_metrics(
        y_true,
        y_pred,
        class_names,
        checkpoint="ckpt.pth",
        data_dir="data/asl_crossval",
        use_hand_crop=True,
        num_no_hand=3,
    )

    # Required keys / honest schema.
    expected_keys = {
        "source",
        "num_samples",
        "hand_crop_used",
        "num_no_hand_fallback",
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "per_class",
        "most_confused_pairs",
        "confusion_labels",
        "confusion_matrix",
        "checkpoint",
        "note",
    }
    assert expected_keys <= set(m)
    assert m["source"] == "cross-dataset:data/asl_crossval"
    assert m["num_samples"] == 6
    assert m["hand_crop_used"] is True
    assert m["num_no_hand_fallback"] == 3
    assert m["accuracy"] == 4 / 6
    assert set(m["per_class"]) == set(class_names)
    # Dense confusion matrix is square, aligned to labels, and row-sums = support.
    assert m["confusion_labels"] == class_names
    assert len(m["confusion_matrix"]) == len(class_names)
    for i, name in enumerate(class_names):
        assert len(m["confusion_matrix"][i]) == len(class_names)
        assert sum(m["confusion_matrix"][i]) == m["per_class"][name]["support"]
    # Honest provenance note must NOT claim the same-dataset benchmark.
    assert "CROSS-DATASET" in m["note"]
    assert "96.8" in m["note"]
    # The confusion pairs are off-diagonal only.
    for p in m["most_confused_pairs"]:
        assert p["true"] != p["pred"]


def test_build_metrics_empty_is_safe():
    m = erw._build_metrics(
        np.array([], dtype=int),
        np.array([], dtype=int),
        ["A", "B"],
        checkpoint="c",
        data_dir="d",
        use_hand_crop=False,
        num_no_hand=0,
    )
    assert m["num_samples"] == 0
    assert m["accuracy"] == 0.0
    assert m["most_confused_pairs"] == []


# --------------------------------------------------------------------------- #
# End-to-end evaluate() with a stubbed checkpoint and hand-crop OFF
# --------------------------------------------------------------------------- #
class _ConstModel(torch.nn.Module):
    """Always predicts class index 0 (so accuracy == fraction of class A)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        logits = torch.zeros(n, self.num_classes)
        logits[:, 0] = 1.0
        return logits


def test_evaluate_end_to_end_no_hand_crop(monkeypatch, tmp_path):
    class_names = ["A", "B"]
    data_dir = tmp_path / "tiny"
    for name in class_names:
        d = data_dir / name
        d.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (16, 16), (i * 10, i * 10, i * 10)).save(d / f"{i}.png")

    monkeypatch.setattr(
        erw,
        "load_checkpoint",
        lambda ckpt, device: (_ConstModel(len(class_names)).eval(), class_names),
    )

    m = erw.evaluate(
        data_dir=str(data_dir),
        checkpoint="unused.pth",
        device=torch.device("cpu"),
        use_hand_crop=False,
    )

    # Model always says A; with 3 A and 3 B, accuracy is 3/6.
    assert m["num_samples"] == 6
    assert m["accuracy"] == 0.5
    assert m["hand_crop_used"] is False
    assert m["num_no_hand_fallback"] == 0


def test_main_writes_artifact(monkeypatch, tmp_path):
    """main() wires args -> evaluate -> save_json with the honest schema."""
    fake_metrics = {
        "source": "cross-dataset:x",
        "num_samples": 0,
        "hand_crop_used": True,
        "num_no_hand_fallback": 0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "accuracy_ay": 0.0,
        "macro_f1_ay": 0.0,
        "num_samples_ay": 0,
        "most_confused_pairs": [],
    }
    monkeypatch.setattr(erw, "evaluate", lambda **kw: fake_metrics)
    out_path = tmp_path / "realworld_eval.json"

    import argparse

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            checkpoint="c.pth",
            data_dir="data/asl_crossval",
            device="cpu",
            hand_crop=True,
            output=str(out_path),
            thresholds_json=None,
            margin=None,
            tta=False,
        ),
    )

    rc = erw.main()
    assert rc == 0
    written = json.loads(out_path.read_text())
    assert written == fake_metrics


def test_main_output_arg_routes_to_distinct_file(monkeypatch, tmp_path):
    """--output must route metrics to a candidate file, not the baseline path."""
    fake_metrics = {
        "source": "cross-dataset:x",
        "num_samples": 0,
        "hand_crop_used": True,
        "num_no_hand_fallback": 0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "accuracy_ay": 0.0,
        "macro_f1_ay": 0.0,
        "num_samples_ay": 0,
        "most_confused_pairs": [],
    }
    monkeypatch.setattr(erw, "evaluate", lambda **kw: fake_metrics)
    # Point the baseline path at a sentinel that must NOT be written.
    baseline = tmp_path / "realworld_eval.json"
    candidate = tmp_path / "realworld_eval_cropped.json"
    monkeypatch.setattr(erw, "OUTPUT_PATH", baseline)

    import argparse

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            checkpoint="c.pth",
            data_dir="data/asl_crossval",
            device="cpu",
            hand_crop=True,
            output=str(candidate),
            thresholds_json=None,
            margin=None,
            tta=False,
        ),
    )

    rc = erw.main()
    assert rc == 0
    assert candidate.exists()
    assert not baseline.exists()  # baseline preserved


def test_parse_args_output_defaults_to_baseline(monkeypatch):
    """Without --output, the default is the deployed baseline path."""
    monkeypatch.setattr(sys, "argv", ["eval_realworld", "--checkpoint", "c.pth"])
    args = erw.parse_args()
    assert args.output == str(erw.OUTPUT_PATH)


# --------------------------------------------------------------------------- #
# Decision policy (per-class thresholds + margin) and A-Y subset metric
# --------------------------------------------------------------------------- #


def test_decision_policy_default_is_argmax():
    """No thresholds + no margin == plain argmax (byte-identical base path)."""
    probs = np.array([0.1, 0.7, 0.2])
    assert erw.apply_decision_policy(probs) == 1


def test_decision_policy_threshold_redirects_off_sink_class():
    """A sink class below its threshold falls through to the next that clears."""
    names = ["S", "T", "U"]
    probs = np.array([0.55, 0.40, 0.05])  # argmax = S
    # S needs 0.6 to be accepted; it's at 0.55 → fall through to T (no floor).
    pred = erw.apply_decision_policy(
        probs, class_thresholds={"S": 0.6}, class_names=names
    )
    assert names[pred] == "T"


def test_decision_policy_threshold_kept_when_cleared():
    names = ["S", "T", "U"]
    probs = np.array([0.75, 0.20, 0.05])  # S clears its 0.6 floor
    pred = erw.apply_decision_policy(
        probs, class_thresholds={"S": 0.6}, class_names=names
    )
    assert names[pred] == "S"


def test_decision_policy_margin_only_does_not_reorder():
    """Margin gates acceptance but apply_decision_policy still returns a class."""
    probs = np.array([0.52, 0.48])
    # Small margin: argmax still returned (policy never invents abstention here;
    # the web layer maps low-margin to 'unsure'). Index 0 remains the pick.
    assert erw.apply_decision_policy(probs, margin=0.1) == 0


def test_subset_metrics_excludes_motion_letters():
    names = ["A", "J", "Z", "B"]
    # 4 samples, all correct except the J sample. Excluding J/Z should drop them.
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 0, 0, 3])  # A ok, J wrong, Z wrong, B ok
    full = erw._subset_metrics(y_true, y_pred, names, exclude=())
    ay = erw._subset_metrics(y_true, y_pred, names, exclude=("J", "Z"))
    assert full["num_samples"] == 4
    assert ay["num_samples"] == 2  # only A, B remain
    assert ay["accuracy"] == 1.0  # both remaining are correct


def test_build_metrics_includes_ay_fields():
    names = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    y_true = np.arange(26)
    y_pred = np.arange(26)
    m = erw._build_metrics(
        y_true,
        y_pred,
        names,
        checkpoint="c",
        data_dir="d",
        use_hand_crop=True,
        num_no_hand=0,
    )
    assert "accuracy_ay" in m and "macro_f1_ay" in m and "num_samples_ay" in m
    assert m["num_samples_ay"] == 24  # J and Z excluded
    assert m["accuracy_ay"] == 1.0


def test_tta_views_are_scale_only_no_flip():
    """TTA must produce centre crops only — never a horizontal mirror."""
    img = Image.fromarray(
        np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (64, 1, 3))
    )  # horizontal gradient: a flip would be detectable
    views = erw._tta_views(img)
    assert len(views) >= 2
    base = np.asarray(views[0])
    # No view may equal the horizontal mirror of the original.
    mirror = base[:, ::-1, :]
    for v in views:
        arr = np.asarray(v.resize(base.shape[1::-1]))
        assert not np.array_equal(arr, mirror)
