"""Per-class F1 bar chart from ``artifacts/metrics.json``.

``src.eval`` writes a ``metrics.json`` containing a ``per_class`` mapping of
``{class_name: {"precision", "recall", "f1", "support"}}``. This module reads
that file and renders a horizontal bar chart of per-class F1, sorted ascending
so the weakest classes are immediately visible.

CRITICAL HONESTY NOTE: when ``metrics.json`` was produced by ``src.eval`` on the
tiny synthetic ``data/sample`` fixture with an untrained model, the F1 values
are a wiring sanity check, not a meaningful measure of model quality (the chart
faithfully mirrors that caveat). Train on the full ASL Alphabet dataset for real
per-class F1.

Run, e.g.::

    python -m src.plot_per_class --metrics artifacts/metrics.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — no display required.
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import load_json  # noqa: E402

ARTIFACTS = Path("artifacts")
DEFAULT_METRICS = ARTIFACTS / "metrics.json"


def plot_per_class_f1(metrics: dict, path: Path) -> Path:
    """Render a per-class F1 horizontal bar chart and save it to ``path``.

    Args:
        metrics: A parsed ``metrics.json`` dict with a ``per_class`` mapping.
        path: Destination PNG path.

    Returns:
        The path written.

    Raises:
        ValueError: If ``metrics`` has no ``per_class`` entry.
    """
    per_class = metrics.get("per_class")
    if not per_class:
        raise ValueError("metrics.json has no 'per_class' section to plot.")

    # Sort ascending by F1 so the weakest classes surface at the top.
    items = sorted(per_class.items(), key=lambda kv: kv[1].get("f1", 0.0))
    names = [name for name, _ in items]
    f1s = [float(stats.get("f1", 0.0)) for _, stats in items]

    fig_height = max(4.0, 0.3 * len(names))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(names, f1s, color="#3b78c2", edgecolor="black")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_ylabel("Class")
    ax.set_title("Per-class F1 (held-out test split)")
    macro_f1 = metrics.get("macro_f1")
    if isinstance(macro_f1, (int, float)):
        ax.axvline(
            macro_f1,
            color="darkorange",
            linestyle="--",
            label=f"Macro F1 = {macro_f1:.3f}",
        )
        ax.legend(loc="lower right")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot per-class F1 from a metrics.json file."
    )
    parser.add_argument(
        "--metrics",
        default=str(DEFAULT_METRICS),
        help="Path to the metrics.json written by src.eval.",
    )
    parser.add_argument(
        "--out",
        default=str(ARTIFACTS / "per_class_f1.png"),
        help="Destination PNG path.",
    )
    return parser.parse_args()


def main() -> int:
    """Read ``--metrics`` and write the per-class F1 chart."""
    args = parse_args()
    metrics = load_json(args.metrics)
    out_path = plot_per_class_f1(metrics, Path(args.out))
    note = metrics.get("note")
    print(f"Saved per-class F1 chart to {out_path}")
    if note:
        print(f"\nNOTE: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
