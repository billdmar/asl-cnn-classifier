"""Split a class-folder dataset into a TRAIN partition and a held-out GATE.

Hemg serves double duty: most of it joins the training union (adding J/Z static
frames + a plain-background regime), while a disjoint held-out slice becomes a
SECOND independent honest eval gate (complementing EitanG98's cluttered-background
gate). This script makes the two partitions from one download, with NO shared
files, deterministically (seed 42) so the split is reproducible and auditable.

It copies image files into ``<out_prefix>_train/<CLASS>/`` and
``<out_prefix>_gate/<CLASS>/`` and writes a manifest. Class-stratified at the
file level (reuses :func:`src.dataset.make_stratified_splits`), so each class is
represented in both partitions in the same proportion.

CRITICAL follow-up: run ``scripts/check_eval_overlap.py`` between the two
partitions — Hemg is single-signer, so near-duplicate frames could straddle the
train/gate boundary and silently contaminate the gate. Inspect the histogram,
not just the rate.

Usage:
    python scripts/split_hemg.py --in_dir data/asl_hemg --out_prefix data/asl_hemg \
        --gate_frac 0.2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from src.dataset import _list_samples, get_class_names  # noqa: E402


def split_dataset(
    in_dir: str | Path,
    out_prefix: str | Path,
    gate_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Partition ``in_dir`` into ``<out_prefix>_train`` and ``<out_prefix>_gate``.

    A single class-stratified 2-way split (train vs held-out gate). Deterministic
    given ``seed`` so the partition is reproducible and auditable.
    """
    in_root = Path(in_dir)
    class_names = get_class_names(in_root)
    samples = _list_samples(in_root, class_names)
    if not samples:
        raise RuntimeError(f"No images found under {in_root}")

    labels = [s[1] for s in samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=gate_frac, random_state=seed)
    train_idx, gate_idx = next(sss.split(samples, labels))
    train_samples = [samples[i] for i in train_idx]
    gate = [samples[i] for i in gate_idx]

    train_root = Path(f"{out_prefix}_train")
    gate_root = Path(f"{out_prefix}_gate")

    def materialize(samples: list[tuple[str, int]], root: Path) -> int:
        for src, label in samples:
            cls = class_names[label]
            dst_dir = root / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / Path(src).name)
        return len(samples)

    n_train = materialize(train_samples, train_root)
    n_gate = materialize(gate, gate_root)

    manifest = {
        "in_dir": str(in_root),
        "train_dir": str(train_root),
        "gate_dir": str(gate_root),
        "gate_frac": gate_frac,
        "seed": seed,
        "num_train": n_train,
        "num_gate": n_gate,
        "num_classes": len(class_names),
    }
    (train_root / "_split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (gate_root / "_split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_dir", default="data/asl_hemg")
    parser.add_argument("--out_prefix", default="data/asl_hemg")
    parser.add_argument("--gate_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    m = split_dataset(args.in_dir, args.out_prefix, args.gate_frac, args.seed)
    print(f"Split {args.in_dir} ({m['num_classes']} classes):")
    print(f"  train: {m['num_train']} -> {m['train_dir']}")
    print(f"  gate : {m['num_gate']} -> {m['gate_dir']}")
    print("Next: run scripts/check_eval_overlap.py between the two partitions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
