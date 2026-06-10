"""Gradio demo for the ASL Sign-Language CNN (Hugging Face Spaces entry point).

Upload a cropped hand-sign image and the app returns the predicted ASL class,
its confidence, and a bar chart of the top-5 class probabilities.

The model and preprocessing are reused verbatim from the training pipeline:

* :func:`src.infer_camera.load_checkpoint` loads a trained checkpoint (or, if
  none is present, falls back to an **untrained random-init** ``custom_cnn`` so
  the app still runs — predictions are then meaningless, and the UI says so).
* :func:`src.dataset.get_eval_transforms` provides the exact eval-time
  resize/normalize pipeline, so there is no second copy of preprocessing here.

The :func:`predict` function is deliberately factored to be importable and
callable **without launching the server** (see ``tests/test_app.py``); only the
``demo.launch()`` call is guarded behind ``if __name__ == "__main__"``.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image

from src.dataset import get_eval_transforms
from src.infer_camera import DEFAULT_CHECKPOINT, load_checkpoint
from src.utils import get_device

# Number of class probabilities to surface in the UI / return value.
TOP_K = 5


@lru_cache(maxsize=1)
def _load() -> tuple[torch.nn.Module, list[str], object, torch.device]:
    """Load (and cache) the model, class names, transform, and device.

    Cached so the checkpoint is loaded once per process rather than on every
    prediction. The demo runs CPU-only on Hugging Face Spaces' free tier.
    """
    device = get_device("cpu")
    model, class_names = load_checkpoint(DEFAULT_CHECKPOINT, device)
    transform = get_eval_transforms()
    return model, class_names, transform, device


def is_using_trained_checkpoint() -> bool:
    """Return True if a real checkpoint was found (not the random-init fallback).

    Used to drive the honesty banner in the UI: when no checkpoint exists,
    ``load_checkpoint`` falls back to untrained random weights and predictions
    are meaningless.
    """
    from pathlib import Path

    return Path(DEFAULT_CHECKPOINT).exists()


@torch.no_grad()
def predict(image: Image.Image) -> tuple[dict[str, float], str]:
    """Classify a single PIL image into ASL classes.

    Applies the canonical eval transform, runs a forward pass, and softmaxes the
    logits. This is the headless prediction entry point — it performs no Gradio
    or server calls, so it can be unit-tested directly.

    Args:
        image: An RGB (or convertible) ``PIL.Image``.

    Returns:
        A tuple ``(top_probs, summary)`` where ``top_probs`` maps the top-``K``
        class labels to their probabilities in ``[0, 1]`` (suitable for a
        ``gr.Label``), and ``summary`` is a short human-readable string with the
        best class and its confidence.

    Raises:
        ValueError: If ``image`` is ``None``.
    """
    if image is None:
        raise ValueError("No image provided.")

    model, class_names, transform, device = _load()
    pil = image.convert("RGB")
    tensor = transform(pil).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(TOP_K, probs.numel())
    top_vals, top_idx = torch.topk(probs, k)
    top_probs = {class_names[int(i)]: float(v) for v, i in zip(top_vals, top_idx)}

    best_label = class_names[int(top_idx[0])]
    best_conf = float(top_vals[0])
    summary = f"Predicted: {best_label}  ({best_conf * 100:.1f}% confidence)"
    return top_probs, summary


_TRAINED = is_using_trained_checkpoint()
_BANNER = (
    "### ASL Sign-Language CNN — demo\n"
    "Upload a cropped image of a single static ASL hand sign (A–Z, plus "
    "*space* / *del* / *nothing*). The model predicts the class and shows the "
    "top-5 probabilities.\n\n"
    + (
        "> Predictions reflect the **loaded trained checkpoint**."
        if _TRAINED
        else "> **No trained checkpoint is loaded**, so this Space is running an "
        "**untrained, random-init** model and its predictions are "
        "**meaningless** — they only demonstrate the wiring. Train a model "
        "(`make train`) and add `best_model.pth` to produce real results. See "
        "the README's accuracy note and `MODEL_CARD.md`."
    )
)


def build_demo():
    """Construct the Gradio Blocks UI. Imported lazily to keep tests headless."""
    import gradio as gr

    with gr.Blocks(title="ASL Sign-Language CNN") as demo:
        gr.Markdown(_BANNER)
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Hand-sign image")
                submit = gr.Button("Classify", variant="primary")
                gr.Examples(
                    examples=[
                        ["data/sample/A/0.png"],
                        ["data/sample/B/0.png"],
                        ["data/sample/C/0.png"],
                    ],
                    inputs=image_in,
                )
            with gr.Column():
                summary_out = gr.Textbox(label="Prediction", interactive=False)
                label_out = gr.Label(num_top_classes=TOP_K, label="Top-5 probabilities")

        def _ui_predict(img):
            top_probs, summary = predict(img)
            return summary, top_probs

        submit.click(_ui_predict, inputs=image_in, outputs=[summary_out, label_out])
        image_in.upload(_ui_predict, inputs=image_in, outputs=[summary_out, label_out])

    return demo


if __name__ == "__main__":
    build_demo().launch()
