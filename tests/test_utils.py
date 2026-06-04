"""Tests for the shared utilities: seeding, device selection, and JSON I/O.

All tests are CPU-only and deterministic. ``get_device`` is exercised across
its preference branches; the CUDA/MPS hardware paths are simulated via
monkeypatch so they run on any host.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils import get_device, load_json, save_json, set_seed


def test_set_seed_is_reproducible():
    set_seed(42)
    a_py = random.random()
    a_np = np.random.rand(3)
    a_torch = torch.rand(3)

    set_seed(42)
    b_py = random.random()
    b_np = np.random.rand(3)
    b_torch = torch.rand(3)

    assert a_py == b_py
    assert np.array_equal(a_np, b_np)
    assert torch.equal(a_torch, b_torch)


def test_get_device_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_auto_returns_valid_device():
    dev = get_device("auto")
    assert dev.type in ("cpu", "cuda", "mps")


def test_get_device_cuda_preference(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device("cuda") == torch.device("cuda")


def test_get_device_cuda_preference_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device("cuda") == torch.device("cpu")


def test_get_device_mps_preference_falls_back_to_cpu(monkeypatch):
    # Force MPS unavailable regardless of host.
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device("mps") == torch.device("cpu")


def test_get_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device("auto") == torch.device("cuda")


def test_save_and_load_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "data.json"
    payload = {"a": 1, "b": [1, 2, 3], "c": {"d": True}}
    save_json(path, payload)
    assert path.exists()
    assert load_json(path) == payload
