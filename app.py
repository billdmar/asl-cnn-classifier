"""Gradio demo for the ASL Sign-Language CNN (Hugging Face Spaces entry point).

Upload (or click a built-in example of) a cropped hand-sign image and the app
returns the predicted ASL class, its confidence, and the top-5 class
probabilities.

The model and preprocessing are reused verbatim from the training pipeline:

* :func:`src.infer_camera.load_checkpoint` loads a trained checkpoint (or, if
  none is present, falls back to an **untrained random-init** ``custom_cnn`` so
  the app still runs — predictions are then meaningless, and the UI says so).
* :func:`src.dataset.get_eval_transforms` provides the exact eval-time
  resize/normalize pipeline, so there is no second copy of preprocessing here.

The model, class names, transform, and device are loaded **once** into a
module-level :data:`ModelBundle` singleton at import time, so a deployed Space
does not reload the weights on every request. Loading is cheap and needs no
checkpoint (the random-init fallback is fine), which keeps importing ``app`` in
tests fast and CI-safe.

The :func:`predict` function is deliberately factored to be importable and
callable **without launching the server** (see ``tests/test_app.py``); likewise
:func:`build_demo` constructs the UI without launching. Only the
``demo.launch()`` call is guarded behind ``if __name__ == "__main__"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.dataset import get_eval_transforms
from src.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
from src.utils import get_device

# Number of class probabilities to surface in the UI / return value.
TOP_K = 5

# Link back to the source repository, surfaced in the UI description.
REPO_URL = "https://github.com/billdmar/asl-cnn-classifier"

# Classes used to wire up click-to-try examples. Chosen to be visually
# distinct hand shapes (open palm, fist, fingers extended, etc.) so a visitor
# can sample a spread of signs without uploading anything.
EXAMPLE_CLASSES = ("A", "C", "L", "W", "Y", "space")

# Directory of committed, *real* ASL hand-photo examples (``<CLASS>.png``),
# preferred over the synthetic ``data/sample`` fixtures when present. Populate it
# with ``scripts/extract_examples.py`` from a real class-folder dataset.
REAL_EXAMPLES_DIR = Path("docs/examples")

# Synthetic fallback fixtures (colored squares) committed for CI; used only when
# no real examples are available so the demo still has clickable examples.
SAMPLE_DIR = Path("data/sample")


@dataclass(frozen=True)
class ModelBundle:
    """Everything needed to run a prediction, loaded once and shared.

    Attributes:
        model: The CNN in eval mode on :attr:`device`.
        class_names: Index→label mapping for the model outputs.
        transform: The canonical eval-time preprocessing pipeline.
        device: The compute device (CPU on the HF Spaces free tier).
        val_accuracy: The checkpoint's recorded validation accuracy in
            ``[0, 1]`` if a trained checkpoint was loaded, else ``None``.
        trained: ``True`` if a real checkpoint was loaded, ``False`` if the
            untrained random-init fallback is in use.
    """

    model: torch.nn.Module
    class_names: list[str]
    transform: object
    device: torch.device
    val_accuracy: float | None
    trained: bool


def _read_val_accuracy(path: str | Path) -> float | None:
    """Best-effort read of ``val_accuracy`` from a checkpoint, else ``None``.

    Reads the checkpoint metadata without disturbing
    :func:`src.infer_camera.load_checkpoint` (which owns model construction).
    Any failure (missing file, unreadable archive, absent key, wrong type)
    yields ``None`` so the UI degrades gracefully.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - any load failure → unknown accuracy.
        return None
    val_acc = checkpoint.get("val_accuracy") if isinstance(checkpoint, dict) else None
    return float(val_acc) if isinstance(val_acc, (int, float)) else None


def _build_bundle() -> ModelBundle:
    """Load the model, class names, transform, and device exactly once.

    The demo runs CPU-only on Hugging Face Spaces' free tier. When no checkpoint
    is present, :func:`load_checkpoint` returns an untrained random-init model;
    we record that in :attr:`ModelBundle.trained` so the UI can be honest.
    """
    device = get_device("cpu")
    model, class_names = load_checkpoint(DEFAULT_CHECKPOINT, device)
    transform = get_eval_transforms()
    trained = Path(DEFAULT_CHECKPOINT).exists()
    val_accuracy = _read_val_accuracy(DEFAULT_CHECKPOINT) if trained else None
    return ModelBundle(
        model=model,
        class_names=class_names,
        transform=transform,
        device=device,
        val_accuracy=val_accuracy,
        trained=trained,
    )


# Module-level singleton: built once at import, reused for every prediction.
MODEL: ModelBundle = _build_bundle()


def get_model() -> ModelBundle:
    """Return the shared :data:`ModelBundle` singleton (loaded once at import)."""
    return MODEL


def is_using_trained_checkpoint() -> bool:
    """Return True if a real checkpoint was loaded (not the random-init fallback).

    Drives the honesty banner in the UI: when no checkpoint exists,
    :func:`load_checkpoint` falls back to untrained random weights and
    predictions are meaningless.
    """
    return get_model().trained


@torch.no_grad()
def predict(image: Image.Image) -> tuple[dict[str, float], str]:
    """Classify a single PIL image into ASL classes.

    Applies the canonical eval transform, runs a forward pass, and softmaxes the
    logits. This is the headless prediction entry point — it performs no Gradio
    or server calls, so it can be unit-tested directly. It uses the shared
    module-level model singleton (no per-call weight loading).

    Args:
        image: An RGB (or convertible) ``PIL.Image``.

    Returns:
        A tuple ``(top_probs, summary)`` where ``top_probs`` maps the top-``K``
        class labels to their probabilities in ``[0, 1]`` (suitable for a
        ``gr.Label``), and ``summary`` is a short human-readable string with the
        best class and its confidence.

    Raises:
        ValueError: If ``image`` is ``None`` or is not a usable image.
    """
    if image is None:
        raise ValueError("No image provided. Upload or pick an example image.")
    if not isinstance(image, Image.Image):
        raise ValueError(
            "Expected an image; received "
            f"{type(image).__name__}. Upload or pick an example image."
        )

    bundle = get_model()
    try:
        pil = image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read the provided image: {exc}") from exc

    tensor = bundle.transform(pil).unsqueeze(0).to(bundle.device)
    logits = bundle.model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(TOP_K, probs.numel())
    top_vals, top_idx = torch.topk(probs, k)
    top_probs = {
        bundle.class_names[int(i)]: float(v) for v, i in zip(top_vals, top_idx)
    }

    best_label = bundle.class_names[int(top_idx[0])]
    best_conf = float(top_vals[0])
    summary = f"Predicted: {best_label}  ({best_conf * 100:.1f}% confidence)"
    return top_probs, summary


def _banner_markdown(bundle: ModelBundle) -> str:
    """Build the title + honest status banner for the given model bundle."""
    header = (
        "# ASL Sign-Language CNN — demo\n"
        "Classify a cropped image of a single static American Sign Language hand "
        "sign (A–Z, plus *space* / *del* / *nothing*). Upload your own crop or "
        "click an example below; the model returns the predicted class and the "
        f"top-{TOP_K} probabilities. "
        f"Source, training code, and model card: [{REPO_URL}]({REPO_URL}).\n\n"
    )
    if bundle.trained:
        if bundle.val_accuracy is not None:
            status = (
                "> A **trained checkpoint is loaded** "
                f"(reported validation accuracy: **{bundle.val_accuracy * 100:.2f}%**). "
                "Predictions reflect that trained model."
            )
        else:
            status = (
                "> A **trained checkpoint is loaded**; predictions reflect that "
                "trained model. (No validation accuracy was recorded in the "
                "checkpoint.)"
            )
    else:
        status = (
            "> **No trained checkpoint is loaded.** This Space is running an "
            "**untrained, random-init** model — its predictions are "
            "**meaningless** and demonstrate the wiring only. Train a model "
            "(`make train`) and add `best_model.pth` to produce real results. "
            "See the README's accuracy note and `MODEL_CARD.md`."
        )
    return header + status


def _real_example_paths() -> list[list[str]]:
    """Return ``Examples`` rows for the real hand-photo examples on disk.

    Reads every ``*.png`` in :data:`REAL_EXAMPLES_DIR` (sorted for a stable
    order). Returns an empty list when the directory is absent or holds no PNGs,
    so the caller can fall back to the synthetic fixtures.
    """
    if not REAL_EXAMPLES_DIR.is_dir():
        return []
    return [[str(p)] for p in sorted(REAL_EXAMPLES_DIR.glob("*.png"))]


def _sample_example_paths() -> list[list[str]]:
    """Return ``Examples`` rows for the synthetic ``data/sample`` fixtures.

    One row per class in :data:`EXAMPLE_CLASSES` whose ``0.png`` fixture exists;
    missing files are skipped so the UI never references a path that would 404.
    """
    rows: list[list[str]] = []
    for cls in EXAMPLE_CLASSES:
        path = SAMPLE_DIR / cls / "0.png"
        if path.exists():
            rows.append([str(path)])
    return rows


def _example_paths() -> list[list[str]]:
    """Return Gradio ``Examples`` rows, preferring real images over fixtures.

    Each row is a single-element list ``[path]`` matching the single image
    input. Real ASL hand photos committed under :data:`REAL_EXAMPLES_DIR` are
    used when present; otherwise the demo falls back to the synthetic
    ``data/sample`` colored-square fixtures so it always has clickable examples.
    """
    real = _real_example_paths()
    if real:
        return real
    return _sample_example_paths()


def build_demo():
    """Construct the Gradio Blocks UI (no server launch).

    Imported lazily so the heavy Gradio import is not paid by headless tests
    that only exercise :func:`predict`.
    """
    import gradio as gr

    bundle = get_model()
    with gr.Blocks(title="ASL Sign-Language CNN") as demo:
        gr.Markdown(_banner_markdown(bundle))
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Hand-sign image")
                submit = gr.Button("Classify", variant="primary")
                examples = _example_paths()
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=image_in,
                        label="Examples (click to try)",
                    )
            with gr.Column():
                summary_out = gr.Textbox(label="Prediction", interactive=False)
                label_out = gr.Label(
                    num_top_classes=TOP_K, label=f"Top-{TOP_K} probabilities"
                )

        def _ui_predict(img):
            """Gradio callback: surface prediction errors as a UI message."""
            try:
                top_probs, summary = predict(img)
            except ValueError as exc:
                return str(exc), {}
            return summary, top_probs

        submit.click(_ui_predict, inputs=image_in, outputs=[summary_out, label_out])
        image_in.upload(_ui_predict, inputs=image_in, outputs=[summary_out, label_out])

    return demo


if __name__ == "__main__":
    import os

    # Bind to 0.0.0.0 so the app is reachable inside a container / on Hugging
    # Face Spaces (the default 127.0.0.1 is not accessible there, which makes
    # Gradio raise "a shareable link must be created"). Spaces sets the port via
    # the PORT env var; fall back to Gradio's default 7860 locally.
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    build_demo().launch(server_name="0.0.0.0", server_port=port)
