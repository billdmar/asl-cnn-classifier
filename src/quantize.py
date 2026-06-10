"""Dynamic INT8 quantization and on-disk size measurement.

Applies :func:`torch.ao.quantization.quantize_dynamic` to the model, quantizing
``nn.Linear`` (and ``nn.LSTM``/``nn.GRU`` where present) to INT8 with dynamic
activation quantization. Dynamic quantization targets the weight-heavy Linear
layers; convolutions are left in FP32 because eager-mode *dynamic* quantization
does not support ``nn.Conv2d`` (that requires static/QAT flows), so we quantize
"Linear at minimum" exactly as the deployment story requires.

The script then measures the **real on-disk size** (bytes) of the FP32 vs. INT8
``state_dict`` serializations and writes them to ``artifacts/quantization.json``.
The model is loaded via :func:`src.infer_camera.load_checkpoint`, so it runs with
the random-init fallback when no checkpoint exists.

Example::

    python -m src.quantize --checkpoint artifacts/checkpoints/best_model.pth \
        --output artifacts/quantization.json --device cpu
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch import nn

from src.infer_camera import DEFAULT_CHECKPOINT, load_checkpoint
from src.utils import get_device, save_json

DEFAULT_OUTPUT = "artifacts/quantization.json"

# Layer types we ask dynamic quantization to convert. Linear is the minimum
# required; LSTM/GRU are included for completeness in case a future arch adds
# recurrent heads. Conv2d is intentionally excluded (unsupported by eager-mode
# dynamic quantization).
QUANTIZE_TYPES = {nn.Linear, nn.LSTM, nn.GRU}


def _ensure_quantized_engine() -> None:
    """Select an available quantized backend engine if one isn't set.

    The quantized engine defaults to ``"none"`` on a fresh process, which makes
    ``quantized::linear_prepack`` (used when serializing INT8 weights) fail.
    We pick the first supported engine — ``qnnpack`` on ARM/Apple-Silicon,
    ``fbgemm`` on x86 — so quantization works portably.
    """
    if torch.backends.quantized.engine != "none":
        return
    supported = list(torch.backends.quantized.supported_engines)
    for candidate in ("qnnpack", "fbgemm"):
        if candidate in supported:
            torch.backends.quantized.engine = candidate
            return
    # Fall back to whatever is supported (skipping the "none" sentinel).
    real = [e for e in supported if e != "none"]
    if real:
        torch.backends.quantized.engine = real[0]


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """Return an INT8 dynamically-quantized copy of ``model`` (CPU, eval).

    Dynamic quantization converts the weights of supported layers
    (:data:`QUANTIZE_TYPES`) to INT8 and quantizes activations on the fly at
    inference. The returned module runs on CPU only.
    """
    _ensure_quantized_engine()
    model.to("cpu").eval()
    return torch.ao.quantization.quantize_dynamic(
        model, QUANTIZE_TYPES, dtype=torch.qint8
    )


def _state_dict_size_bytes(model: nn.Module, tmp_path: Path) -> int:
    """Serialize ``model``'s state_dict to disk and return its size in bytes."""
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), tmp_path)
    return tmp_path.stat().st_size


def measure_quantization(
    model: nn.Module, workdir: str | Path | None = None
) -> dict[str, object]:
    """Quantize ``model`` and measure FP32 vs INT8 on-disk size.

    Both state_dicts are serialized to disk and their byte sizes are read back
    — these are REAL measured numbers, not estimates. Serialization uses a
    private temporary directory (cleaned up automatically); ``workdir`` is
    accepted for backward compatibility but ignored.

    Returns:
        A dict with ``fp32_bytes``, ``int8_bytes``, ``size_reduction_bytes``,
        ``compression_ratio``, and ``quantized_layer_types``.
    """
    quantized = quantize_dynamic_int8(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fp32_bytes = _state_dict_size_bytes(model, tmp / "fp32_state.pt")
        int8_bytes = _state_dict_size_bytes(quantized, tmp / "int8_state.pt")

    return {
        "fp32_bytes": fp32_bytes,
        "int8_bytes": int8_bytes,
        "size_reduction_bytes": fp32_bytes - int8_bytes,
        "compression_ratio": (fp32_bytes / int8_bytes) if int8_bytes > 0 else 0.0,
        "quantized_layer_types": sorted(t.__name__ for t in QUANTIZE_TYPES),
    }


def main() -> int:
    """Parse args, quantize the model, and persist size measurements."""
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization.")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Destination JSON path."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for loading (quantization itself always runs on CPU).",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    model, _ = load_checkpoint(args.checkpoint, device)
    out_path = Path(args.output)
    results = measure_quantization(model, out_path.parent)

    save_json(out_path, results)
    print(
        f"FP32 state_dict: {results['fp32_bytes']} bytes | "
        f"INT8 state_dict: {results['int8_bytes']} bytes | "
        f"reduction: {results['size_reduction_bytes']} bytes "
        f"({results['compression_ratio']:.3f}x)."
    )
    print(f"Saved quantization results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
