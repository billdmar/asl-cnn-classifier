"""Tests for the training entry point and its helpers.

Training runs on the committed ``data/sample`` fixture for a single epoch on CPU
with ``num_workers=0`` so the loop, checkpointing, and history writing are all
covered quickly and deterministically.
"""

from __future__ import annotations

import argparse
import sys

import pytest
import torch
import yaml
from torch import nn

from src import train
from src.model import build_model

DATA_DIR = "data/sample"


def _write_config(tmp_path, **overrides):
    """Write a minimal YAML config pointed at the sample fixture + tmp outputs."""
    config = {
        "data_dir": DATA_DIR,
        "image_size": 32,
        "arch": "custom_cnn",
        "num_classes": 29,
        "pretrained": False,
        "batch_size": 64,
        "num_epochs": 1,
        "learning_rate": 0.001,
        "optimizer": "adamw",
        "lr_scheduler": "cosine",
        "warmup_epochs": 0,
        "seed": 42,
        "num_workers": 0,
        "amp": False,
        "early_stopping_patience": 10,
        "device": "cpu",
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "tensorboard_dir": str(tmp_path / "runs"),
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


# --------------------------------------------------------------------------- #
# Helper-level tests
# --------------------------------------------------------------------------- #
def test_build_optimizer_adamw_and_sgd():
    model = build_model("custom_cnn")
    adamw = train.build_optimizer(
        model, {"optimizer": "adamw", "weight_decay": 1e-4}, 1e-3
    )
    assert isinstance(adamw, torch.optim.AdamW)
    sgd = train.build_optimizer(
        model, {"optimizer": "sgd", "weight_decay": 1e-4, "momentum": 0.9}, 1e-3
    )
    assert isinstance(sgd, torch.optim.SGD)


def test_build_optimizer_unknown_raises():
    model = build_model("custom_cnn")
    with pytest.raises(ValueError):
        train.build_optimizer(
            model, {"optimizer": "rmsprop", "weight_decay": 0.0}, 1e-3
        )


def test_build_scheduler_cosine_and_plateau():
    model = build_model("custom_cnn")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cosine = train.build_scheduler(opt, {"lr_scheduler": "cosine"}, t_max=5)
    assert isinstance(cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
    plateau = train.build_scheduler(opt, {"lr_scheduler": "plateau"}, t_max=5)
    assert isinstance(plateau, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_build_scheduler_unknown_raises():
    model = build_model("custom_cnn")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        train.build_scheduler(opt, {"lr_scheduler": "step"}, t_max=5)


def test_run_epoch_train_and_eval():
    model = build_model("custom_cnn")
    device = torch.device("cpu")
    # 2 tiny synthetic batches: (inputs, targets, paths).
    batches = [
        (torch.randn(2, 3, 32, 32), torch.tensor([0, 1]), ["a", "b"]),
        (torch.randn(2, 3, 32, 32), torch.tensor([2, 3]), ["c", "d"]),
    ]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=False)

    train_loss, train_acc = train.run_epoch(
        model, batches, criterion, device, optimizer, scaler, False, False
    )
    assert train_loss >= 0.0
    assert 0.0 <= train_acc <= 1.0

    eval_loss, eval_acc = train.run_epoch(
        model, batches, criterion, device, None, scaler, False, False
    )
    assert eval_loss >= 0.0
    assert 0.0 <= eval_acc <= 1.0


def test_load_config_applies_overrides_and_defaults(tmp_path):
    cfg_path = _write_config(tmp_path, optimizer="adamw")
    args = argparse.Namespace(
        config=str(cfg_path),
        data_dir=None,
        arch="resnet18",
        num_epochs=3,
        batch_size=None,
        learning_rate=None,
        seed=None,
        amp=None,
        device=None,
        resume_checkpoint=None,
    )
    config = train.load_config(args)
    assert config["arch"] == "resnet18"  # CLI override applied
    assert config["num_epochs"] == 3
    assert config["weight_decay"] == 1e-4  # default filled in


# --------------------------------------------------------------------------- #
# Full main() runs
# --------------------------------------------------------------------------- #
def test_main_custom_cnn_end_to_end(tmp_path, monkeypatch, capsys):
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["train", "--config", str(cfg_path), "--device", "cpu"]
    )
    monkeypatch.chdir(tmp_path)
    # data_dir is absolute-ish relative to repo; point at repo's sample copy.
    import os

    repo_root = train.__file__.rsplit("/src/", 1)[0]
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            config=str(cfg_path),
            data_dir=os.path.join(repo_root, DATA_DIR),
            arch=None,
            num_epochs=None,
            batch_size=None,
            learning_rate=None,
            seed=None,
            amp=None,
            device="cpu",
            resume_checkpoint=None,
        ),
    )
    train.main()
    out = capsys.readouterr().out
    assert "Best validation accuracy" in out
    assert (tmp_path / "checkpoints" / "best_model.pth").exists()


def test_main_transfer_with_warmup_and_resume(tmp_path, monkeypatch, capsys):
    """Covers the freeze->unfreeze warmup path and resume_checkpoint loading."""
    import os

    repo_root = train.__file__.rsplit("/src/", 1)[0]
    abs_data = os.path.join(repo_root, DATA_DIR)

    # First, produce a checkpoint to resume from.
    ckpt_dir = tmp_path / "ck"
    cfg1 = _write_config(
        tmp_path,
        arch="mobilenet_v2",
        checkpoint_dir=str(ckpt_dir),
        data_dir=abs_data,
    )
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            config=str(cfg1),
            data_dir=abs_data,
            arch=None,
            num_epochs=None,
            batch_size=None,
            learning_rate=None,
            seed=None,
            amp=None,
            device="cpu",
            resume_checkpoint=None,
        ),
    )
    monkeypatch.chdir(tmp_path)
    train.main()
    ckpt = ckpt_dir / "best_model.pth"
    assert ckpt.exists()

    # Now run again with warmup_epochs=1, num_epochs=2 (triggers unfreeze) and
    # resume from the checkpoint, plateau scheduler, SGD.
    cfg2 = _write_config(
        tmp_path,
        arch="mobilenet_v2",
        warmup_epochs=1,
        num_epochs=2,
        lr_scheduler="plateau",
        optimizer="sgd",
        checkpoint_dir=str(tmp_path / "ck2"),
        data_dir=abs_data,
    )
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            config=str(cfg2),
            data_dir=abs_data,
            arch=None,
            num_epochs=None,
            batch_size=None,
            learning_rate=None,
            seed=None,
            amp=None,
            device="cpu",
            resume_checkpoint=str(ckpt),
        ),
    )
    train.main()
    out = capsys.readouterr().out
    assert "unfroze backbone" in out
    assert "Resumed weights" in out
