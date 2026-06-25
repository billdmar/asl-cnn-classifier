"""Cross-language preprocessing parity — Python side of the CI-blocking gate.

The browser/TS path is verified against the committed golden fixtures by the
web test suite (`web/lib/__tests__/parity.strict.test.ts`). This module guards
the *Python* end of the same contract using only committed assets (the web ONNX
model + the golden manifest + the pre-resized fixture PNGs) — so it runs in CI
without needing the multi-hundred-MB real dataset.

If preprocessing, the model, or the ONNX export drifts, the committed manifest
will no longer reproduce and this test fails, forcing a fixture regeneration
(`make export-onnx-web`) and a conscious review of the change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from src.dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "web" / "test-fixtures" / "golden"
WEB_ONNX = REPO_ROOT / "web" / "public" / "model" / "model.onnx"
MANIFEST = GOLDEN_DIR / "manifest.json"

# Strict tolerance: feeding the SAME pre-resized pixels through Python's
# ToTensor+Normalize and the ONNX model must reproduce the manifest essentially
# exactly (measured agreement ~5e-7). This isolates the normalize/layout/model
# arithmetic from the resize kernel (which legitimately differs cross-language).
STRICT_ATOL = 1e-3

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists() or not WEB_ONNX.exists(),
    reason="web parity fixtures or web ONNX model not present",
)


def _load_manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


def test_manifest_matches_canonical_constants() -> None:
    """The fixtures were generated with the same constants the code uses today."""
    manifest = _load_manifest()
    assert manifest["image_size"] == IMAGE_SIZE
    assert manifest["imagenet_mean"] == IMAGENET_MEAN
    assert manifest["imagenet_std"] == IMAGENET_STD
    assert len(manifest["class_names"]) == 26
    assert manifest["class_names"][0] == "A"
    assert manifest["class_names"][-1] == "Z"


def _preprocess_resized(path: Path) -> np.ndarray:
    """Apply ToTensor+Normalize to an already-128x128 RGB image (no resize)."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    # HWC -> CHW, normalize per channel.
    chw = np.transpose(arr, (2, 0, 1))
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    normalized = (chw - mean) / std
    return normalized[np.newaxis, ...].astype(np.float32)


def test_golden_fixtures_reproduce_from_committed_assets() -> None:
    """Re-running the committed ONNX model over the pre-resized fixtures must
    reproduce the manifest's predicted class exactly and probabilities within
    the strict tolerance."""
    manifest = _load_manifest()
    session = ort.InferenceSession(str(WEB_ONNX))
    input_name = session.get_inputs()[0].name
    class_names = manifest["class_names"]

    for fixture in manifest["fixtures"]:
        cls = fixture["true_class"]
        resized_path = GOLDEN_DIR / f"{cls}_resized.png"
        assert resized_path.exists(), f"missing pre-resized fixture {resized_path}"

        tensor = _preprocess_resized(resized_path)
        logits = session.run(None, {input_name: tensor})[0][0]
        probs = torch.softmax(torch.tensor(logits), 0).numpy()

        pred_index = int(probs.argmax())
        assert class_names[pred_index] == fixture["pred_class"], (
            f"{cls}: predicted {class_names[pred_index]} != "
            f"manifest {fixture['pred_class']}"
        )
        expected = np.array(fixture["probs"], dtype=np.float32)
        max_diff = float(np.max(np.abs(probs - expected)))
        assert max_diff < STRICT_ATOL, (
            f"{cls}: max prob diff {max_diff:.2e} exceeds {STRICT_ATOL} — "
            "regenerate fixtures with `make export-onnx-web`."
        )
