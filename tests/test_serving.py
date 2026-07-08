"""Tests for the serving & performance workstream.

Covers, all on CPU with the random-init checkpoint fallback (no trained model
required):

* **ONNX parity** — PyTorch logits vs. ONNX Runtime logits agree within
  ``atol=1e-4`` on the same input. This is the headline correctness check: the
  exported graph must compute the same function as the eager model.
* **INT8 quantization** — the quantized model produces same-shape output and is
  no larger on disk than FP32 (measured, not asserted as a magic number).
* **FastAPI service** — ``/health`` returns 200; ``/predict`` on a real sample
  image returns a valid class and a confidence in ``[0, 1]``.
* **Multi-backend benchmark** — a tiny-frame smoke run produces stats for every
  backend and writes the JSON artifact.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnxruntime as ort
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src import benchmark_backends as bb
from src import export_onnx, quantize, serve
from src.dataset import get_eval_transforms
from src.checkpoint import load_checkpoint

DATA_DIR = "data/sample"


def _repo_path(rel: str) -> str:
    repo_root = serve.__file__.rsplit("/src/", 1)[0]
    return os.path.join(repo_root, rel)


def _sample_image() -> str:
    return _repo_path(os.path.join(DATA_DIR, "A", "0.png"))


# --------------------------------------------------------------------------- #
# ONNX export + parity (the headline correctness check)
# --------------------------------------------------------------------------- #
def test_onnx_parity(tmp_path):
    """PyTorch and ONNX Runtime must agree on logits within atol=1e-4."""
    model, _ = load_checkpoint("missing.pth", torch.device("cpu"))
    model.eval()
    onnx_path = tmp_path / "model.onnx"
    export_onnx.export_to_onnx(model, onnx_path, torch.device("cpu"))
    assert onnx_path.exists() and onnx_path.stat().st_size > 0

    # Batch of 2 exercises the dynamic batch axis as well.
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        torch_logits = model(x).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]

    assert onnx_logits.shape == torch_logits.shape == (2, 29)
    np.testing.assert_allclose(onnx_logits, torch_logits, atol=1e-4)


def test_onnx_dynamic_batch_axis(tmp_path):
    """The exported graph accepts a batch size different from the trace (1)."""
    model, _ = load_checkpoint("missing.pth", torch.device("cpu"))
    onnx_path = tmp_path / "model.onnx"
    export_onnx.export_to_onnx(model, onnx_path, torch.device("cpu"))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = session.run(
        None,
        {
            session.get_inputs()[0]
            .name: np.random.randn(4, 3, 128, 128)
            .astype(np.float32)
        },
    )[0]
    assert out.shape == (4, 29)


def test_export_onnx_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            checkpoint="missing.pth",
            output="artifacts/model.onnx",
            device="cpu",
        ),
    )
    assert export_onnx.main() == 0
    assert (tmp_path / "artifacts" / "model.onnx").stat().st_size > 0


# --------------------------------------------------------------------------- #
# INT8 quantization
# --------------------------------------------------------------------------- #
def test_quantized_output_shape():
    model, _ = load_checkpoint("missing.pth", torch.device("cpu"))
    quantized = quantize.quantize_dynamic_int8(model)
    with torch.no_grad():
        out = quantized(torch.randn(1, 3, 128, 128))
    assert out.shape == (1, 29)


def test_quantization_smaller_on_disk(tmp_path):
    model, _ = load_checkpoint("missing.pth", torch.device("cpu"))
    results = quantize.measure_quantization(model, tmp_path)
    assert results["int8_bytes"] > 0
    assert results["fp32_bytes"] > 0
    # Dynamic quantization of the Linear layers must not grow the model;
    # CustomCNN is conv-heavy so the gain is modest but real.
    assert results["int8_bytes"] <= results["fp32_bytes"]
    assert results["size_reduction_bytes"] >= 0
    assert "Linear" in results["quantized_layer_types"]


def test_quantize_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            checkpoint="missing.pth",
            output="artifacts/quantization.json",
            device="cpu",
        ),
    )
    assert quantize.main() == 0
    assert (tmp_path / "artifacts" / "quantization.json").exists()


# --------------------------------------------------------------------------- #
# FastAPI service
# --------------------------------------------------------------------------- #
def test_health_endpoint():
    client = TestClient(serve.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # Class count reflects the loaded checkpoint (29 for the random-init/sample
    # fallback, 26 for the real A–Z model), not a hardcoded constant.
    assert body["num_classes"] == len(serve._CLASS_NAMES)


def test_predict_endpoint():
    client = TestClient(serve.app)
    with open(_sample_image(), "rb") as fh:
        resp = client.post("/predict", files={"file": ("0.png", fh, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in serve._CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top5"]) == 5
    assert all(0.0 <= e["confidence"] <= 1.0 for e in body["top5"])
    # top5 is ranked descending; the first entry is the prediction.
    assert body["top5"][0]["class"] == body["predicted_class"]


def test_predict_empty_upload():
    client = TestClient(serve.app)
    resp = client.post("/predict", files={"file": ("e.png", b"", "image/png")})
    assert resp.status_code == 400


def test_predict_bad_image():
    client = TestClient(serve.app)
    resp = client.post(
        "/predict", files={"file": ("bad.png", b"not-an-image", "image/png")}
    )
    assert resp.status_code == 400


def test_predict_image_helper():
    model, class_names = load_checkpoint("missing.pth", torch.device("cpu"))
    img = Image.new("RGB", (200, 200), (120, 130, 140))
    out = serve.predict_image(
        img, model, get_eval_transforms(), torch.device("cpu"), class_names
    )
    assert out["predicted_class"] in class_names
    assert 0.0 <= out["confidence"] <= 1.0
    assert len(out["top5"]) == 5


# --------------------------------------------------------------------------- #
# Multi-backend benchmark (smoke)
# --------------------------------------------------------------------------- #
def test_run_backends_smoke(tmp_path):
    model, _ = load_checkpoint("missing.pth", torch.device("cpu"))
    frames = bb._load_frames(None, 6)
    results = bb.run_backends(model, frames, get_eval_transforms(), tmp_path)
    # The three CPU backends are always present.
    for backend in ("pytorch_fp32_cpu", "onnxruntime_cpu", "int8_dynamic_cpu"):
        assert backend in results
        assert set(results[backend]) == {
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "fps",
        }
        assert results[backend]["mean_ms"] >= 0.0
    # MPS key appears iff MPS is actually available (graceful skip otherwise).
    mps_ok = bool(getattr(torch.backends, "mps", None)) and (
        torch.backends.mps.is_available()
    )
    assert ("pytorch_fp32_mps" in results) == mps_ok


def test_format_markdown_table():
    table = bb.format_markdown_table(
        {
            "pytorch_fp32_cpu": {
                "mean_ms": 1.0,
                "p50_ms": 1.0,
                "p95_ms": 1.0,
                "p99_ms": 1.0,
                "fps": 1000.0,
            }
        }
    )
    assert "| Backend |" in table
    assert "pytorch_fp32_cpu" in table


def test_benchmark_backends_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = _repo_path(DATA_DIR)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            checkpoint="missing.pth",
            num_frames=6,
            source=data,
            output="artifacts/backend_benchmark.json",
        ),
    )
    assert bb.main() == 0
    out = tmp_path / "artifacts" / "backend_benchmark.json"
    assert out.exists()


def test_summarize_helper():
    stats = bb._summarize([1.0, 2.0, 3.0, 4.0])
    assert stats["mean_ms"] == pytest.approx(2.5)
    assert stats["fps"] == pytest.approx(400.0)
