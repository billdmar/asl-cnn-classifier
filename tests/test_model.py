"""Tests for the model factory, forward shapes, and freeze/unfreeze behavior.

All tests are CPU-only and run with random-init weights (``pretrained=False``)
so they stay fast and need no network access.
"""

from __future__ import annotations

import pytest
import torch

from src.model import CustomCNN, TransferModel, build_model

BATCH = 2
NUM_CLASSES = 29
INPUT = (BATCH, 3, 128, 128)


def test_custom_cnn_forward_shape():
    model = build_model("custom_cnn")
    out = model(torch.randn(*INPUT))
    assert tuple(out.shape) == (BATCH, NUM_CLASSES)


def test_mobilenet_v2_forward_shape():
    model = build_model("mobilenet_v2", pretrained=False)
    out = model(torch.randn(*INPUT))
    assert tuple(out.shape) == (BATCH, NUM_CLASSES)


def test_resnet18_forward_shape():
    model = build_model("resnet18", pretrained=False)
    out = model(torch.randn(*INPUT))
    assert tuple(out.shape) == (BATCH, NUM_CLASSES)


@pytest.mark.parametrize("arch", ["mobilenet_v3_small", "efficientnet_b0"])
def test_new_transfer_arches_forward_and_freeze(arch):
    model = build_model(arch, pretrained=False)
    assert isinstance(model, TransferModel)
    out = model(torch.randn(*INPUT))
    assert tuple(out.shape) == (BATCH, NUM_CLASSES)
    # Freeze isolates the head: non-head frozen, head trainable.
    model.freeze_backbone()
    head_ids = model._head_param_ids
    non_head = [p for p in model.backbone.parameters() if id(p) not in head_ids]
    head = [p for p in model.backbone.parameters() if id(p) in head_ids]
    assert non_head and head
    assert all(not p.requires_grad for p in non_head)
    assert all(p.requires_grad for p in head)


def test_custom_cnn_param_count_in_band():
    # Guards the README's ~656,829-parameter figure from silent drift.
    model = build_model("custom_cnn")
    n_params = sum(p.numel() for p in model.parameters())
    assert 550_000 < n_params < 750_000, n_params


def test_build_model_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_model("nonsense")


def test_transfer_freeze_then_unfreeze_toggles_requires_grad():
    model = build_model("mobilenet_v2", pretrained=False)
    assert isinstance(model, TransferModel)

    model.freeze_backbone()
    # Non-head backbone params are frozen; the classifier head stays trainable.
    head_ids = model._head_param_ids
    non_head = [p for p in model.backbone.parameters() if id(p) not in head_ids]
    head = [p for p in model.backbone.parameters() if id(p) in head_ids]
    assert non_head, "expected backbone to have non-head parameters"
    assert all(not p.requires_grad for p in non_head)
    assert all(p.requires_grad for p in head)

    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_custom_cnn_is_custom_cnn_instance():
    model = build_model("custom_cnn")
    assert isinstance(model, CustomCNN)
