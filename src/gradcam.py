"""Grad-CAM explainability for the ASL classifier (no extra dependencies).

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of
an input image that most influenced a class prediction. We implement it directly
with forward/backward hooks on the last convolutional block — no
``pytorch-grad-cam`` dependency is pulled in.

The algorithm:

1. Register a forward hook to capture the target conv layer's activations
   ``A`` (shape ``(1, C, h, w)``) and a backward hook to capture the gradients
   ``dY_c/dA`` of the target class score.
2. Run a forward pass, pick the target class (argmax by default), and
   backpropagate that single logit.
3. Global-average-pool the gradients over the spatial dims to get per-channel
   weights ``alpha_c``; the CAM is ``ReLU(sum_c alpha_c * A_c)``.
4. Upsample the CAM to the input resolution and min-max normalize it to
   ``[0, 1]``.

The target layer is chosen automatically: the last ``nn.Conv2d`` inside the
``features`` sub-module for both :class:`~src.model.CustomCNN` and the
MobileNetV2 backbone. Any other module that exposes a ``features`` Sequential
with at least one ``Conv2d`` degrades gracefully through the same path.

Preprocessing reuses :func:`src.dataset.get_eval_transforms` and checkpoint
loading reuses :func:`src.infer_camera.load_checkpoint`, so there is no second
copy of either piece of logic.

CRITICAL HONESTY NOTE: with no trained checkpoint, ``load_checkpoint`` falls
back to a RANDOM-init ``custom_cnn`` and the only available imagery is the tiny
synthetic ``data/sample`` fixture of colored glyphs (NOT real ASL hands). The
saved overlay is therefore a *wiring / demonstration* artifact that proves the
Grad-CAM plumbing works end to end — it is NOT a meaningful saliency map of a
real model. Train on the full ASL Alphabet dataset for interpretable heatmaps.

Run, e.g.::

    python -m src.gradcam --checkpoint artifacts/checkpoints/best_model.pth \
        --source data/sample/A/0.png --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — no display required.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402

from src.dataset import IMAGE_SIZE, get_eval_transforms  # noqa: E402
from src.infer_camera import load_checkpoint  # noqa: E402
from src.utils import get_device  # noqa: E402

DEFAULT_CHECKPOINT = "artifacts/checkpoints/best_model.pth"
GRADCAM_DIR = Path("artifacts/gradcam")

SAMPLE_DATA_NOTE = (
    "Grad-CAM overlay on the synthetic data/sample fixture with an untrained "
    "(random-init) model is a wiring demonstration, not a meaningful saliency "
    "map. Train on the full ASL Alphabet dataset for interpretable heatmaps."
)


def find_target_layer(model: nn.Module) -> nn.Conv2d:
    """Return the last ``nn.Conv2d`` inside the model's ``features`` block.

    Handles :class:`~src.model.CustomCNN` (``model.features``) and the
    :class:`~src.model.TransferModel` MobileNetV2 backbone
    (``model.backbone.features``). Falls back to the last ``Conv2d`` anywhere in
    the model if no ``features`` Sequential is found, so callers degrade
    gracefully rather than crashing.

    Raises:
        ValueError: If the model contains no ``nn.Conv2d`` at all.
    """
    features = getattr(model, "features", None)
    if features is None:
        backbone = getattr(model, "backbone", None)
        features = getattr(backbone, "features", None)

    search_space = features if features is not None else model
    last_conv: nn.Conv2d | None = None
    for module in search_space.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No nn.Conv2d layer found; cannot run Grad-CAM.")
    return last_conv


class GradCAM:
    """Grad-CAM via forward/backward hooks on a target convolutional layer.

    Args:
        model: A model in eval mode.
        target_layer: The conv layer whose activations/gradients drive the CAM.
            Defaults to :func:`find_target_layer`'s choice.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Conv2d | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or find_target_layer(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(
        self, _module: nn.Module, _inp: tuple, output: torch.Tensor
    ) -> None:
        self._activations = output.detach()

    def _save_gradient(
        self, _module: nn.Module, _grad_in: tuple, grad_out: tuple
    ) -> None:
        # grad_out is a tuple; element 0 is dLoss/dActivation.
        self._gradients = grad_out[0].detach()

    def remove(self) -> None:
        """Detach all hooks (call once finished to avoid leaks)."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()

    def __call__(
        self, input_tensor: torch.Tensor, class_idx: int | None = None
    ) -> tuple[np.ndarray, int]:
        """Compute the Grad-CAM heatmap for a single image.

        Args:
            input_tensor: A ``(1, 3, H, W)`` preprocessed image tensor.
            class_idx: Target class. If ``None``, the model's argmax is used.

        Returns:
            ``(cam, class_idx)`` where ``cam`` is an ``(H, W)`` float32 numpy
            array normalized to ``[0, 1]`` and upsampled to the input size.
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                "Grad-CAM expects a single image tensor of shape (1,3,H,W)."
            )

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        # alpha_c = global-average-pooled gradients (per channel).
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy().astype(np.float32)

        # Min-max normalize to [0, 1]; a flat map (e.g. all-zero ReLU) stays 0.
        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np, class_idx


def overlay_cam_on_image(
    image: Image.Image, cam: np.ndarray, alpha: float = 0.5
) -> Image.Image:
    """Blend a ``[0, 1]`` CAM heatmap (jet colormap) over an RGB image.

    Args:
        image: The original RGB PIL image (any size; resized to the CAM size).
        cam: An ``(H, W)`` array in ``[0, 1]``.
        alpha: Heatmap opacity in ``[0, 1]``.

    Returns:
        A blended RGB PIL image at the CAM's resolution.
    """
    h, w = cam.shape
    base = image.convert("RGB").resize((w, h))
    base_arr = np.asarray(base).astype(np.float32) / 255.0

    heatmap = matplotlib.colormaps["jet"](cam)[..., :3].astype(np.float32)  # (H,W,3)
    blended = (1.0 - alpha) * base_arr + alpha * heatmap
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def run_gradcam(
    source: str | Path,
    model: nn.Module,
    device: torch.device,
    class_names: list[str],
    out_dir: Path = GRADCAM_DIR,
) -> tuple[Path, str]:
    """Compute and save a Grad-CAM overlay for a single image.

    Args:
        source: Path to an input image.
        model: Model in eval mode on ``device``.
        device: Compute device.
        class_names: Index→label mapping.
        out_dir: Directory for the saved overlay.

    Returns:
        ``(output_path, predicted_class)``.
    """
    transform = get_eval_transforms()
    image = Image.open(source).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with GradCAM(model) as cam_engine:
        cam, class_idx = cam_engine(input_tensor)

    label = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
    overlay = overlay_cam_on_image(image.resize((IMAGE_SIZE, IMAGE_SIZE)), cam)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.png"
    overlay.save(out_path)
    return out_path, label


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Grad-CAM explainability for the ASL classifier."
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--source",
        default="data/sample/A/0.png",
        help="Path to the input image to explain.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Compute device.",
    )
    return parser.parse_args()


def main() -> int:
    """Load the model and write a Grad-CAM overlay for ``--source``."""
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    model, class_names = load_checkpoint(args.checkpoint, device)
    out_path, label = run_gradcam(args.source, model, device, class_names)

    print(f"predicted_class: {label}")
    print(f"Saved Grad-CAM overlay to {out_path}")
    print(f"\nNOTE: {SAMPLE_DATA_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
