"""Drift guard: the DEPLOYED checkpoint must match the committed web ONNX model.

`web/public/model/model.onnx` is the model the live site runs; it's exported from
`artifacts/checkpoints/best_model.pth` via `make export-onnx-web`. If someone
retrains the checkpoint but forgets to re-export (or vice versa), the live site
would silently serve a stale/mismatched model. This test re-runs the PyTorch
checkpoint and the committed ONNX over the same golden fixtures and asserts their
logits agree — failing loudly on drift.

Skips gracefully when the checkpoint isn't present (CI doesn't train a real model;
this guard runs locally / wherever the checkpoint exists). The committed-ONNX
parity against the *fixtures* is covered separately by test_parity_fixtures.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from src.dataset import IMAGENET_MEAN, IMAGENET_STD

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "web" / "test-fixtures" / "golden"
WEB_ONNX = REPO_ROOT / "web" / "public" / "model" / "model.onnx"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "best_model.pth"

# Logit agreement between the eager PyTorch checkpoint and the exported ONNX
# graph on identical inputs. The ONNX export is a faithful trace, so this is
# tight (the existing fixture parity runs at ~5e-7); 1e-3 is a safe ceiling.
SYNC_ATOL = 1e-3

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists() or not WEB_ONNX.exists() or not GOLDEN_DIR.exists(),
    reason="deployed checkpoint, web ONNX, or golden fixtures not present",
)


def _preprocess_resized(path: Path) -> np.ndarray:
    """ToTensor+Normalize on an already-128x128 RGB image (no resize)."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    return ((chw - mean) / std)[np.newaxis, ...].astype(np.float32)


def test_deployed_checkpoint_matches_committed_onnx() -> None:
    """The .pth the repo trains and the .onnx the site serves must agree."""
    from src.infer_camera import load_checkpoint

    device = torch.device("cpu")
    model, class_names = load_checkpoint(str(CHECKPOINT), device)
    model.eval()

    session = ort.InferenceSession(str(WEB_ONNX))
    input_name = session.get_inputs()[0].name
    onnx_classes = session.get_outputs()[0].shape[-1]

    # Only meaningful when the on-disk checkpoint IS the deployed model. CI's
    # "sample train" step writes a throwaway 29-class custom_cnn checkpoint to
    # this path that has nothing to do with the committed 26-class ONNX — skip
    # those rather than false-fail. The drift guard runs wherever the real
    # deployed checkpoint exists (locally, release machines).
    if len(class_names) != onnx_classes:
        pytest.skip(
            f"checkpoint has {len(class_names)} classes but the web ONNX has "
            f"{onnx_classes} — not the deployed model (e.g. CI's sample train); "
            "drift guard only applies to the real deployed checkpoint."
        )

    fixtures = sorted(GOLDEN_DIR.glob("*_resized.png"))
    assert fixtures, "no golden fixture images found"

    max_diff = 0.0
    for path in fixtures:
        tensor = _preprocess_resized(path)
        with torch.no_grad():
            pth_logits = model(torch.from_numpy(tensor)).numpy()[0]
        onnx_logits = session.run(None, {input_name: tensor})[0][0]
        # Same argmax (deploy-critical) and tight logit agreement.
        assert int(pth_logits.argmax()) == int(onnx_logits.argmax()), (
            f"{path.name}: checkpoint and ONNX disagree on the predicted class — "
            "the web model.onnx is out of sync with the deployed checkpoint. "
            "Re-run `make export-onnx-web`."
        )
        max_diff = max(max_diff, float(np.max(np.abs(pth_logits - onnx_logits))))

    assert max_diff < SYNC_ATOL, (
        f"checkpoint↔ONNX logit drift {max_diff:.2e} exceeds {SYNC_ATOL}; "
        "re-run `make export-onnx-web` to resync the deployed model."
    )
