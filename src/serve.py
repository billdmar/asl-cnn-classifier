"""FastAPI inference service for the ASL classifier.

Exposes two endpoints:

* ``GET /health`` — liveness probe returning ``{"status": "ok", ...}``.
* ``POST /predict`` — multipart image upload → JSON with ``predicted_class``,
  ``confidence`` (in ``[0, 1]``), and a ``top5`` list of
  ``{"class", "confidence"}`` entries.

The model is loaded once at import time via
:func:`src.infer_camera.load_checkpoint` (random-init fallback when no
checkpoint exists), and every request reuses
:func:`src.dataset.get_eval_transforms` for preprocessing — there is no second
copy of the resize/normalize logic. The module-level ``app`` object is importable
for ``fastapi.testclient.TestClient`` and for ``uvicorn src.serve:app``.

Run::

    uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torch import nn

from src.dataset import get_eval_transforms
from src.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint

# Checkpoint path is overridable via env var so the same app object can serve a
# real checkpoint in production without code changes.
CHECKPOINT_PATH = os.environ.get("ASL_CHECKPOINT", DEFAULT_CHECKPOINT)

# Loaded once at import; CPU keeps the service portable (no GPU assumed).
_DEVICE = torch.device("cpu")
_MODEL, _CLASS_NAMES = load_checkpoint(CHECKPOINT_PATH, _DEVICE)
_TRANSFORM = get_eval_transforms()

app = FastAPI(title="ASL CNN Classifier", version="1.0.0")


@torch.no_grad()
def predict_image(
    image: Image.Image,
    model: nn.Module,
    transform,
    device: torch.device,
    class_names: list[str],
    top_k: int = 5,
) -> dict[str, object]:
    """Classify a single PIL image and return the prediction payload.

    Applies the canonical eval transform, runs a forward pass, softmaxes the
    logits, and returns the argmax label plus the ``top_k`` ranked classes.

    Returns:
        ``{"predicted_class", "confidence", "top5"}`` where ``top5`` is a list
        of ``{"class", "confidence"}`` dicts (length ``min(top_k, n_classes)``).
    """
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0)
    k = min(top_k, probs.numel())
    top_conf, top_idx = torch.topk(probs, k)

    top5 = [
        {"class": class_names[int(i)], "confidence": float(c)}
        for c, i in zip(top_conf, top_idx)
    ]
    return {
        "predicted_class": top5[0]["class"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
    }


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe: confirms the model is loaded and reports class count."""
    return {
        "status": "ok",
        "num_classes": len(_CLASS_NAMES),
        "checkpoint": CHECKPOINT_PATH,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    """Classify an uploaded image file.

    Raises:
        HTTPException: 400 if the upload is empty or not a decodable image.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode image: {exc}"
        ) from exc

    return predict_image(image, _MODEL, _TRANSFORM, _DEVICE, _CLASS_NAMES)
