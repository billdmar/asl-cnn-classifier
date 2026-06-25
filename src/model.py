"""Model definitions: a compact from-scratch CNN and a transfer-learning wrapper.

Two architectures are exposed through the :func:`build_model` factory:

* :class:`CustomCNN` — a ~657K-parameter convolutional network trained from
  scratch on 3×128×128 inputs. Global average pooling (rather than a large
  flattened dense layer) keeps the parameter count low while preserving the
  four convolutional stages.
* :class:`TransferModel` — wraps a torchvision backbone (``mobilenet_v2`` or
  ``resnet18``) with a fresh classifier head and exposes
  :meth:`TransferModel.freeze_backbone` / :meth:`TransferModel.unfreeze_backbone`
  for the freeze-then-fine-tune training schedule.

Both ``train.py`` and the evaluation/benchmark scripts construct models only via
:func:`build_model`, so the architecture↔checkpoint mapping stays consistent.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
)


class CustomCNN(nn.Module):
    """From-scratch CNN for 3×128×128 ASL images.

    Four convolutional blocks progressively halve the spatial resolution while
    growing channel depth (32 → 64 → 128 → 256). A global average pooling layer
    collapses the final 256×8×8 feature map to a 256-vector before a small
    fully-connected head, which keeps the model around 657K parameters.

    Args:
        num_classes: Number of output logits (29 ASL classes by default).
    """

    def __init__(self, num_classes: int = 29) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32, /2
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            # Block 2: 32 -> 64, /2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            # Block 3: 64 -> 128, /2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            # Block 4: 128 -> 256, /2 (single conv)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        # Global average pooling collapses spatial dims -> (N, 256, 1, 1).
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return ``(N, num_classes)`` logits."""
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


class TransferModel(nn.Module):
    """Transfer-learning wrapper around a torchvision backbone.

    Replaces the backbone's classifier head with one sized to ``num_classes``
    and provides helpers to (un)freeze the feature extractor for a
    freeze-then-fine-tune schedule.

    Args:
        arch: ``"mobilenet_v2"`` or ``"resnet18"``.
        num_classes: Number of output logits.
        pretrained: If ``True``, load ImageNet weights via the modern
            ``weights=`` API; otherwise initialize randomly.
    """

    def __init__(
        self,
        arch: str = "mobilenet_v2",
        num_classes: int = 29,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.arch = arch

        if arch == "mobilenet_v2":
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
            # The final classifier Linear is the head we keep trainable.
            self._head_param_ids = {
                id(p) for p in self.backbone.classifier[1].parameters()
            }
        elif arch == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            self._head_param_ids = {id(p) for p in self.backbone.fc.parameters()}
        elif arch == "mobilenet_v3_small":
            weights = (
                MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.backbone = models.mobilenet_v3_small(weights=weights)
            # Final classifier Linear is classifier[-1] (a small head MLP precedes it).
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            self._head_param_ids = {
                id(p) for p in self.backbone.classifier[-1].parameters()
            }
        elif arch == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            self._head_param_ids = {
                id(p) for p in self.backbone.classifier[-1].parameters()
            }
        else:
            raise ValueError(
                f"Unsupported transfer arch '{arch}'. Expected one of: "
                "mobilenet_v2, resnet18, mobilenet_v3_small, efficientnet_b0."
            )

    def freeze_backbone(self) -> None:
        """Freeze every parameter except the newly added classifier head."""
        for param in self.backbone.parameters():
            param.requires_grad = id(param) in self._head_param_ids

    def unfreeze_backbone(self) -> None:
        """Make all parameters trainable (for the fine-tuning phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return ``(N, num_classes)`` logits."""
        return self.backbone(x)


def build_model(
    arch: str, num_classes: int = 29, pretrained: bool = False
) -> nn.Module:
    """Construct a model by architecture name.

    Args:
        arch: One of ``"custom_cnn"``, ``"mobilenet_v2"``, ``"resnet18"``,
            ``"mobilenet_v3_small"``, ``"efficientnet_b0"``.
        num_classes: Number of output logits.
        pretrained: Whether transfer backbones load ImageNet weights (ignored
            for ``custom_cnn``).

    Returns:
        An instantiated :class:`torch.nn.Module`.

    Raises:
        ValueError: If ``arch`` is not recognized.
    """
    if arch == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    if arch in (
        "mobilenet_v2",
        "resnet18",
        "mobilenet_v3_small",
        "efficientnet_b0",
    ):
        return TransferModel(arch=arch, num_classes=num_classes, pretrained=pretrained)
    raise ValueError(
        f"Unknown arch '{arch}'. Expected 'custom_cnn', 'mobilenet_v2', "
        "'resnet18', 'mobilenet_v3_small', or 'efficientnet_b0'."
    )
