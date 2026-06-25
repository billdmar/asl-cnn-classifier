"""Guard the honest eval gate: detect near-duplicate overlap between a candidate
TRAINING dataset and the held-out cross-dataset EVAL set (``data/asl_crossval``).

Why this exists: we are adding a diverse training source (`aliciiavs/...`) that
shares a "real-room webcam" style with the EitanG98 eval set. If any training
image is a near-duplicate of an eval image, a retrained model could "beat" the
33.4% gate by memorizing eval images rather than generalizing — the exact failure
mode the whole diversity effort is guarding against. This script must report ~0
overlap BEFORE training is trusted.

Mechanism (reuses the codebase's perceptual-hash machinery): for each class
present in BOTH the train dir and the eval dir, compute ``imagehash.phash`` for
every image, then for each training image find its minimum Hamming distance to
any **same-class** eval image. A distance ``<= threshold`` (default
``CROSS_DATASET_PHASH_THRESHOLD = 10``) is flagged as a near-duplicate.
Class-scoped comparison keeps it tractable. NOTE: 10 is much tighter than the
sequential-video dedup radius (22) — across two photo datasets, distinct photos
of the same static sign collide at ~16-26 from shared coarse structure, so 22
floods with false positives; a true re-encode/resize/crop of the same image
lands at <=~8. See the constant's comment for the empirical justification.

Read-only by default — prints a per-class report, a min-distance histogram, the
closest pairs, and writes ``<train_dir>/_overlap_report.json``. Opt-in mutation:
``--remove`` deletes flagged training files; ``--exclude-manifest PATH`` writes
the flagged paths to a file instead of deleting.

Usage:
    python scripts/check_eval_overlap.py --train_dir data/asl_diverse
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_class_names  # noqa: E402

# Cross-dataset near-duplicate threshold (pHash Hamming distance). This is much
# tighter than the sequential-video DEDUP_PHASH_THRESHOLD (22): that radius is
# tuned for consecutive frames of ONE recording, but across two different photo
# datasets, distinct photos of the SAME static sign (same letter, centered hand,
# plain background) collide at distance ~16-26 from shared coarse structure
# alone. Empirically (aliciiavs vs EitanG98) the min-distance distribution is a
# clean bell centered at 24 with ZERO images at d<=10 — so a true re-encode /
# resize / crop of the same image (which lands at <=~8) is well separated from
# same-sign-different-photo collisions. 10 catches real dups without the
# false-positive flood that threshold 22 produces (~36% of legit images).
CROSS_DATASET_PHASH_THRESHOLD = 10

# Above this fraction of training images flagged, the candidate set is considered
# contaminated and must be cleaned (--remove/--exclude-manifest) before training.
CONTAMINATION_FAIL_RATE = 0.005  # 0.5%

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _phash_dir(class_dir: Path) -> list[tuple[Path, Any]]:
    """Return ``(path, phash)`` for every image in a class folder."""
    import imagehash

    out: list[tuple[Path, Any]] = []
    for p in sorted(class_dir.iterdir()):
        if p.suffix.lower() in _IMG_EXTS:
            out.append((p, imagehash.phash(Image.open(p).convert("RGB"))))
    return out


def check_overlap(
    train_dir: str | Path,
    eval_dir: str | Path = "data/asl_crossval",
    threshold: int = CROSS_DATASET_PHASH_THRESHOLD,
) -> dict:
    """Scan ``train_dir`` for near-duplicates of ``eval_dir`` (same-class pHash).

    Returns a report dict with per-class flagged counts, the global flag rate,
    a min-distance histogram, and the closest cross-set pairs.
    """
    train_root = Path(train_dir)
    eval_root = Path(eval_dir)
    classes = sorted(set(get_class_names(train_root)) & set(get_class_names(eval_root)))

    per_class: dict[str, dict[str, int]] = {}
    flagged_pairs: list[dict] = []
    dist_hist: Counter[int] = Counter()
    total_train = 0
    total_flagged = 0

    for name in classes:
        train_imgs = _phash_dir(train_root / name)
        eval_imgs = _phash_dir(eval_root / name)
        if not train_imgs or not eval_imgs:
            continue
        n_flagged = 0
        for tpath, th in train_imgs:
            # Min Hamming distance to any same-class eval image.
            best = min((th - eh, epath) for epath, eh in eval_imgs)
            min_dist, epath = int(best[0]), best[1]
            dist_hist[min_dist] += 1
            if min_dist <= threshold:
                n_flagged += 1
                flagged_pairs.append(
                    {
                        "class": name,
                        "train": str(tpath),
                        "eval": str(epath),
                        "distance": min_dist,
                    }
                )
        per_class[name] = {"train": len(train_imgs), "flagged": n_flagged}
        total_train += len(train_imgs)
        total_flagged += n_flagged
        print(
            f"  {name}: {len(train_imgs)} train vs {len(eval_imgs)} eval "
            f"-> {n_flagged} flagged"
        )

    flagged_pairs.sort(key=lambda d: d["distance"])
    rate = (total_flagged / total_train) if total_train else 0.0
    report = {
        "train_dir": str(train_root),
        "eval_dir": str(eval_root),
        "threshold": threshold,
        "classes_compared": len(per_class),
        "total_train_images": total_train,
        "total_flagged": total_flagged,
        "flag_rate": rate,
        "contaminated": rate > CONTAMINATION_FAIL_RATE,
        "distance_histogram": {str(k): dist_hist[k] for k in sorted(dist_hist)},
        "closest_pairs": flagged_pairs[:20],
        "per_class": per_class,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_dir", required=True, help="Candidate training dataset to scan."
    )
    parser.add_argument(
        "--eval_dir", default="data/asl_crossval", help="Held-out eval set."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=CROSS_DATASET_PHASH_THRESHOLD,
        help=(
            "pHash Hamming distance for a cross-dataset near-dup "
            f"(default {CROSS_DATASET_PHASH_THRESHOLD}; tighter than the "
            "sequential-video dedup radius of 22 — see module docstring)."
        ),
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="DELETE flagged training files (opt-in mutation; default report-only).",
    )
    parser.add_argument(
        "--exclude-manifest",
        dest="exclude_manifest",
        default=None,
        help="Write flagged training paths to this file instead of deleting them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        f"Checking {args.train_dir} for near-duplicates of {args.eval_dir} "
        f"(threshold={args.threshold})"
    )
    report = check_overlap(
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        threshold=args.threshold,
    )

    out_path = Path(args.train_dir) / "_overlap_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print("\n=== Eval-overlap summary ===")
    print(f"Classes compared : {report['classes_compared']}")
    print(f"Train images     : {report['total_train_images']}")
    print(
        f"Flagged near-dups: {report['total_flagged']} " f"({report['flag_rate']:.2%})"
    )
    print(
        f"Distance histogram (min-dist -> count): "
        f"{dict(list(report['distance_histogram'].items())[:12])}"
    )
    if report["closest_pairs"]:
        print("Closest cross-set pairs (class, dist):")
        for p in report["closest_pairs"][:10]:
            print(f"  {p['class']} d={p['distance']}: {p['train']} ~ {p['eval']}")

    # closest_pairs is capped at 20 for display; re-scan for the full flagged
    # list only when we actually need to mutate.
    if report["total_flagged"] > 0 and (args.remove or args.exclude_manifest):
        full = _all_flagged_paths(args.train_dir, args.eval_dir, args.threshold)
        if args.exclude_manifest:
            Path(args.exclude_manifest).write_text("\n".join(full) + "\n")
            print(f"\nWrote {len(full)} flagged paths to {args.exclude_manifest}")
        elif args.remove:
            for fp in full:
                Path(fp).unlink(missing_ok=True)
            print(f"\nDeleted {len(full)} flagged training files.")

    print(f"\nReport: {out_path}")
    if report["contaminated"]:
        print(
            f"\n*** CONTAMINATED: flag rate {report['flag_rate']:.2%} exceeds "
            f"{CONTAMINATION_FAIL_RATE:.1%}. Clean (--remove / --exclude-manifest) "
            "before training, or the eval gate is poisoned. ***"
        )
        return 1
    print("\nOK: overlap below the contamination threshold — safe to train.")
    return 0


def _all_flagged_paths(
    train_dir: str | Path, eval_dir: str | Path, threshold: int
) -> list[str]:
    """Re-scan and return ALL flagged training paths (not just the top-20 shown)."""
    train_root = Path(train_dir)
    eval_root = Path(eval_dir)
    classes = sorted(set(get_class_names(train_root)) & set(get_class_names(eval_root)))
    flagged: list[str] = []
    for name in classes:
        train_imgs = _phash_dir(train_root / name)
        eval_imgs = _phash_dir(eval_root / name)
        if not train_imgs or not eval_imgs:
            continue
        for tpath, th in train_imgs:
            if min(th - eh for _e, eh in eval_imgs) <= threshold:
                flagged.append(str(tpath))
    return flagged


if __name__ == "__main__":
    raise SystemExit(main())
