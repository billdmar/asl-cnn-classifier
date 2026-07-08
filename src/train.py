"""Training entry point for the ASL classifier.

Reads a YAML config (see ``configs/``) that any command-line flag can override,
builds the requested model via :func:`src.model.build_model`, and runs a
training loop with:

* stratified file-level train/val/test splits (no augmentation leakage),
* optional freeze-then-fine-tune schedule for transfer-learning backbones,
* AdamW or SGD + cosine / plateau LR scheduling,
* device-safe automatic mixed precision (a true no-op off CUDA/MPS),
* TensorBoard logging and early stopping on validation loss.

The best checkpoint (by validation accuracy) is written to
``{checkpoint_dir}/best_model.pth`` with a fixed schema that the evaluation,
inference, and benchmark scripts depend on::

    {"model_state_dict", "arch", "class_names", "config", "val_accuracy"}

Run, e.g.::

    python -m src.train --config configs/train_custom_cnn.yaml --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as tv_utils

from src.dataset import (
    ASLDataset,
    _list_samples,
    get_class_names,
    get_eval_transforms,
    get_train_transforms,
    get_union_class_names,
    make_stratified_splits,
    normalize_data_dirs,
)
from src.model import TransferModel, build_model
from src.utils import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Unset overrides stay ``None``."""
    parser = argparse.ArgumentParser(description="Train the ASL classifier.")
    parser.add_argument(
        "--config", required=True, help="Path to a YAML training config."
    )
    parser.add_argument("--data_dir", default=None, help="Override data directory.")
    parser.add_argument("--arch", default=None, help="Override model architecture.")
    parser.add_argument(
        "--num_epochs", type=int, default=None, help="Override epoch count."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Override LR."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override AMP flag (auto-disabled off CUDA/MPS).",
    )
    parser.add_argument(
        "--device", default=None, help="Override device (auto|cpu|cuda|mps)."
    )
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Path to a checkpoint to resume weights from.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load YAML config and apply any provided CLI overrides.

    CLI values take precedence over YAML whenever they are explicitly set
    (i.e. not ``None``).
    """
    with open(args.config, encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    overrides = {
        "data_dir": args.data_dir,
        "arch": args.arch,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "amp": args.amp,
        "device": args.device,
        "resume_checkpoint": args.resume_checkpoint,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    # Sensible defaults for keys a minimal config may omit.
    config.setdefault("image_size", 128)
    config.setdefault("train_frac", 0.70)
    config.setdefault("val_frac", 0.15)
    config.setdefault("test_frac", 0.15)
    config.setdefault("num_classes", 29)
    config.setdefault("pretrained", False)
    config.setdefault("weight_decay", 1e-4)
    config.setdefault("optimizer", "adamw")
    config.setdefault("lr_scheduler", "cosine")
    config.setdefault("warmup_epochs", 0)
    config.setdefault("momentum", 0.9)
    config.setdefault("num_workers", 4)
    config.setdefault("amp", False)
    config.setdefault("early_stopping_patience", 10)
    config.setdefault("device", "auto")
    config.setdefault("checkpoint_dir", "artifacts/checkpoints")
    config.setdefault("tensorboard_dir", "artifacts/runs")
    config.setdefault("resume_checkpoint", None)
    return config


def build_optimizer(
    model: nn.Module, config: dict[str, Any], lr: float
) -> torch.optim.Optimizer:
    """Create the optimizer over the currently trainable parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    name = str(config["optimizer"]).lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=config["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
    raise ValueError(
        f"Unknown optimizer '{config['optimizer']}'. Expected 'adamw' or 'sgd'."
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any], t_max: int
) -> Any:
    """Create the LR scheduler (cosine annealing or reduce-on-plateau)."""
    name = str(config["lr_scheduler"]).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, t_max)
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5
        )
    raise ValueError(
        f"Unknown lr_scheduler '{config['lr_scheduler']}'. Expected 'cosine' or 'plateau'."
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    autocast_enabled: bool,
) -> tuple[float, float]:
    """Run one train (``optimizer`` set) or eval (``optimizer=None``) epoch.

    Returns:
        ``(mean_loss, accuracy)`` over the loader.
    """
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        # Dataset yields a 3-tuple; the filepath is unused during train/eval.
        for inputs, targets, _paths in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_train:
                assert optimizer is not None  # implied by is_train
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if is_train:
                assert optimizer is not None  # implied by is_train
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    mean_loss = running_loss / max(1, total)
    accuracy = correct / max(1, total)
    return mean_loss, accuracy


def main() -> None:
    """Parse config, train, log, checkpoint, and write training history."""
    args = parse_args()
    config = load_config(args)

    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))
    print(f"Using device: {device}")

    # --- AMP setup (device-safe). On CPU/MPS the GradScaler is disabled. ---
    use_amp = bool(config["amp"]) and device.type == "cuda"
    autocast_enabled = bool(config["amp"]) and device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # --- Data ---
    # `data_dir` may be a single dir (str) or several (list, or comma-separated
    # str via CLI override). Multiple dirs train on the UNION of class-folder
    # datasets for diversity; class_names is the sorted union so the label↔index
    # map covers every class present in any source. The single-dir path is
    # unchanged (byte-identical split).
    data_dirs = normalize_data_dirs(config["data_dir"])
    if len(data_dirs) == 1:
        class_names = get_class_names(data_dirs[0])
        train_samples, val_samples, _test_samples = make_stratified_splits(
            data_dirs[0],
            train_frac=config["train_frac"],
            val_frac=config["val_frac"],
            test_frac=config["test_frac"],
            seed=int(config["seed"]),
            class_names=class_names,
        )
    else:
        class_names = get_union_class_names(data_dirs)
        merged_samples: list[tuple[str, int]] = []
        for d in data_dirs:
            merged_samples.extend(_list_samples(d, class_names))
        print(
            f"Multi-source training on {len(data_dirs)} dirs "
            f"({len(merged_samples)} images, {len(class_names)} classes): "
            f"{', '.join(str(d) for d in data_dirs)}"
        )
        train_samples, val_samples, _test_samples = make_stratified_splits(
            samples=merged_samples,
            train_frac=config["train_frac"],
            val_frac=config["val_frac"],
            test_frac=config["test_frac"],
            seed=int(config["seed"]),
            class_names=class_names,
        )

    image_size = int(config["image_size"])
    # Augmentation regime: prefer the explicit `augmentation` config key; fall
    # back to the legacy boolean `heavy_augmentation` so existing configs behave
    # identically. `get_train_transforms` resolves None → standard/heavy.
    aug_regime = config.get("augmentation")
    if aug_regime is None and bool(config.get("heavy_augmentation", False)):
        aug_regime = "heavy"
    train_ds = ASLDataset(
        samples=train_samples,
        transform=get_train_transforms(image_size, regime=aug_regime),
        class_names=class_names,
    )
    val_ds = ASLDataset(
        samples=val_samples,
        transform=get_eval_transforms(image_size),
        class_names=class_names,
    )

    pin_memory = device.type == "cuda"
    # Don't spawn more worker processes than there are batches to feed them;
    # for tiny datasets the fork/IPC overhead dwarfs any speedup, so on small
    # splits we fall back to in-process loading (num_workers=0).
    batch_size = int(config["batch_size"])
    n_train_batches = (len(train_samples) + batch_size - 1) // batch_size
    num_workers = min(int(config["num_workers"]), n_train_batches)
    if n_train_batches <= 4:
        num_workers = 0
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # --- Model ---
    model = build_model(
        arch=str(config["arch"]),
        num_classes=int(config["num_classes"]),
        pretrained=bool(config["pretrained"]),
    ).to(device)

    if config["resume_checkpoint"]:
        ckpt = torch.load(
            config["resume_checkpoint"], map_location=device, weights_only=False
        )
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"Resumed weights from {config['resume_checkpoint']}")

    # Freeze-then-fine-tune schedule applies only to transfer backbones.
    warmup_epochs = int(config["warmup_epochs"])
    is_transfer = isinstance(model, TransferModel)
    use_warmup = is_transfer and warmup_epochs > 0
    if use_warmup:
        assert isinstance(model, TransferModel)  # implied by is_transfer
        model.freeze_backbone()
        print(f"Backbone frozen for {warmup_epochs} warm-up epoch(s).")

    base_lr = float(config["learning_rate"])
    num_epochs = int(config["num_epochs"])
    # Optional inverse-frequency class weighting to counter source/class
    # imbalance (e.g. an over-predicted "sink" class). Off by default →
    # plain CrossEntropyLoss, byte-identical to before.
    class_weight = None
    if str(config.get("class_weights", "")).lower() == "auto":
        counts = np.bincount(
            [lbl for _f, lbl in train_samples], minlength=len(class_names)
        ).astype(np.float64)
        # inverse frequency, normalized to mean 1 so the loss scale is unchanged
        inv = np.where(counts > 0, counts.sum() / (counts * len(counts)), 0.0)
        inv = inv / inv[inv > 0].mean() if (inv > 0).any() else inv
        class_weight = torch.tensor(inv, dtype=torch.float32, device=device)
        print(
            f"Class-weighted loss (inverse-frequency) enabled: {class_weight.tolist()}"
        )
    # Optional label smoothing softens the one-hot targets, which can improve
    # generalization + calibration. Off by default (0.0) → standard CE,
    # byte-identical to before.
    label_smoothing = float(config.get("label_smoothing", 0.0))
    if label_smoothing > 0.0:
        print(f"Label smoothing enabled: {label_smoothing:g}")
    criterion = nn.CrossEntropyLoss(
        weight=class_weight, label_smoothing=label_smoothing
    )
    optimizer = build_optimizer(model, config, base_lr)
    scheduler = build_scheduler(optimizer, config, t_max=num_epochs)
    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    # --- Optional Stochastic Weight Averaging (SWA) ---
    # When enabled, the final ~tail of training switches to a constant SWA LR and
    # the weights from each tail epoch are averaged; BN stats are recomputed over
    # the train loader at the end. The averaged model is saved as the checkpoint.
    # Off by default → no AveragedModel, no SWALR, byte-identical to before.
    use_swa = bool(config.get("use_swa", False))
    swa_start_epoch = int(config.get("swa_start_epoch", num_epochs))
    swa_lr = float(config.get("swa_lr", base_lr / 10.0))
    swa_model: AveragedModel | None = None
    swa_scheduler: SWALR | None = None
    swa_n_updates = 0
    if use_swa:
        swa_model = AveragedModel(model)
        print(
            f"SWA enabled: averaging from epoch {swa_start_epoch} "
            f"at constant LR {swa_lr:g}."
        )

    writer = SummaryWriter(log_dir=config["tensorboard_dir"])

    # --- Training loop ---
    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = int(config["early_stopping_patience"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pth"

    history: list[dict[str, float]] = []
    fine_tune_started = False

    for epoch in range(1, num_epochs + 1):
        # Transition from frozen warm-up to full fine-tuning at 10x-lower LR.
        if use_warmup and not fine_tune_started and epoch > warmup_epochs:
            assert isinstance(model, TransferModel)  # implied by use_warmup
            model.unfreeze_backbone()
            fine_tune_started = True
            ft_lr = base_lr / 10.0
            optimizer = build_optimizer(model, config, ft_lr)
            scheduler = build_scheduler(
                optimizer, config, t_max=num_epochs - warmup_epochs
            )
            is_plateau = isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            )
            print(f"Epoch {epoch}: unfroze backbone, fine-tuning at LR {ft_lr:g}.")

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            autocast_enabled=autocast_enabled,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            autocast_enabled=autocast_enabled,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # Log the first augmented training batch once, at epoch 1.
        if epoch == 1:
            sample_batch, _, _ = next(iter(train_loader))
            grid = tv_utils.make_grid(
                sample_batch[:32], nrow=8, normalize=True, scale_each=True
            )
            writer.add_image("train/augmented_batch", grid, global_step=epoch)

        # Step the scheduler (plateau needs the monitored metric). Once the SWA
        # phase begins, switch to the constant SWA LR and average the weights
        # from each tail epoch instead of following the base schedule.
        in_swa_phase = use_swa and epoch >= swa_start_epoch
        if in_swa_phase:
            assert swa_model is not None  # implied by use_swa
            if swa_scheduler is None:
                # Bind SWALR to the current (post-warmup) optimizer the first
                # time we enter the SWA phase.
                swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
            swa_model.update_parameters(model)
            swa_n_updates += 1
            swa_scheduler.step()
        elif is_plateau:
            scheduler.step(val_loss)
        else:
            scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} | lr {current_lr:.2e}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
        )

        # Save the best checkpoint by validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arch": str(config["arch"]),
                    "class_names": class_names,
                    "config": config,
                    "val_accuracy": val_acc,
                },
                best_path,
            )

        # Early stopping is tracked on validation loss.
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"val loss has not improved for {patience} epoch(s)."
                )
                break

    # --- SWA finalization ---
    # The averaged weights need their BatchNorm running stats recomputed over the
    # train data (SWA never updated them). We then evaluate the averaged model and
    # save IT as the checkpoint — the averaged model is the SWA deliverable, so it
    # overwrites the per-epoch best regardless of val accuracy.
    if use_swa and swa_model is not None and swa_n_updates > 0:
        print(
            f"SWA: recomputing BN stats over the train loader "
            f"({swa_n_updates} averaged epoch(s))."
        )
        update_bn(train_loader, swa_model, device=device)
        swa_val_loss, swa_val_acc = run_epoch(
            swa_model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            autocast_enabled=autocast_enabled,
        )
        print(f"SWA averaged model | val_loss {swa_val_loss:.4f} acc {swa_val_acc:.4f}")
        # AveragedModel wraps the net in `.module`; save the underlying state dict
        # so it loads into a plain build_model() net like every other checkpoint.
        torch.save(
            {
                "model_state_dict": swa_model.module.state_dict(),
                "arch": str(config["arch"]),
                "class_names": class_names,
                "config": config,
                "val_accuracy": swa_val_acc,
            },
            best_path,
        )
        best_val_acc = swa_val_acc

    writer.close()

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {best_path}")

    # Write history alongside the checkpoint so separate runs (e.g. a robustness
    # retrain to a different checkpoint_dir) don't clobber each other's history.
    # Also mirror to the canonical artifacts path for the default run.
    history_path = checkpoint_dir / "training_history.json"
    save_json(str(history_path), history)
    print(f"Training history written to {history_path}")
    if checkpoint_dir == Path("artifacts/checkpoints"):
        save_json("artifacts/training_history.json", history)


if __name__ == "__main__":
    main()
