"""Validate the dedup-split clustering constants against the REAL dataset.

The dedup split (make_stratified_splits(dedup=True)) assumes DEDUP_PHASH_THRESHOLD
and DEDUP_WINDOW cluster sequential video near-duplicates correctly. The unit
tests only exercise synthetic data, so this script measures what actually happens
on data/asl_real (~11k frames):

  1. Cluster-size distribution at the shipped threshold/window.
  2. Threshold sweep — how cluster count / largest cluster move with the radius,
     so we can see whether 22 sits on a plateau or a cliff.
  3. Ordering sensitivity — lexical (0,1,10,100,...) vs numeric (0,1,2,...) file
     order, since the windowed scan depends on frames being in sequence and
     _list_samples uses lexical sort.

Read-only: loads images, computes hashes, prints stats. Writes nothing.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import dataset  # noqa: E402
from src.dataset import (  # noqa: E402
    DEDUP_PHASH_THRESHOLD,
    DEDUP_WINDOW,
    _phash_groups,
)

DATA_DIR = "data/asl_real"


def _numeric_key(path: str) -> tuple:
    """Sort key that orders 0,1,2,...,10 numerically (true frame sequence)."""
    nums = re.findall(r"\d+", Path(path).name)
    return (int(nums[0]) if nums else -1, path)


def _cluster_stats(group_ids: list[int]) -> dict:
    sizes = Counter(group_ids)
    size_hist = Counter(sizes.values())
    n = len(group_ids)
    n_groups = len(sizes)
    singletons = size_hist.get(1, 0)
    largest = max(sizes.values())
    return {
        "n_frames": n,
        "n_groups": n_groups,
        "compression": n / n_groups,
        "singletons": singletons,
        "singleton_frac": singletons / n_groups,
        "largest_cluster": largest,
        "size_hist": dict(sorted(size_hist.items())),
    }


def _print_stats(label: str, stats: dict) -> None:
    print(f"\n[{label}]")
    print(f"  frames            : {stats['n_frames']}")
    print(f"  clusters          : {stats['n_groups']}")
    print(f"  frames/cluster    : {stats['compression']:.2f}")
    print(
        f"  singletons        : {stats['singletons']} "
        f"({stats['singleton_frac']:.1%} of clusters)"
    )
    print(f"  largest cluster   : {stats['largest_cluster']} frames")
    hist = stats["size_hist"]
    shown = {k: hist[k] for k in list(hist)[:12]}
    print(f"  size histogram    : {shown}{' …' if len(hist) > 12 else ''}")


def main() -> int:
    class_names = dataset.get_class_names(DATA_DIR)
    samples = dataset._list_samples(DATA_DIR, class_names)
    print(
        f"Loaded {len(samples)} samples across {len(class_names)} classes "
        f"from {DATA_DIR}"
    )
    print(
        f"Shipped constants: THRESHOLD={DEDUP_PHASH_THRESHOLD}, "
        f"WINDOW={DEDUP_WINDOW}"
    )

    # 1 + 3: lexical (as _list_samples returns) vs numeric frame order.
    lex_ids = _phash_groups(samples)
    _print_stats("lexical order (shipped path)", _cluster_stats(lex_ids))

    numeric_samples = sorted(samples, key=lambda s: (s[1], _numeric_key(s[0])))
    num_ids = _phash_groups(numeric_samples)
    _print_stats("numeric frame order", _cluster_stats(num_ids))

    # 2: threshold sweep at the shipped window, numeric order (true sequence).
    print("\n[threshold sweep — numeric order, window=" f"{DEDUP_WINDOW}]")
    print(
        f"  {'thresh':>6}  {'clusters':>9}  {'frames/clu':>10}  "
        f"{'largest':>8}  {'singletons':>10}"
    )
    for thr in (8, 12, 16, 18, 20, 22, 24, 28, 32):
        ids = _phash_groups(numeric_samples, threshold=thr)
        s = _cluster_stats(ids)
        print(
            f"  {thr:>6}  {s['n_groups']:>9}  {s['compression']:>10.2f}  "
            f"{s['largest_cluster']:>8}  {s['singletons']:>10}"
        )

    # Window sweep at shipped threshold — does a wider window chain more?
    print(f"\n[window sweep — numeric order, threshold={DEDUP_PHASH_THRESHOLD}]")
    print(f"  {'window':>6}  {'clusters':>9}  {'frames/clu':>10}  {'largest':>8}")
    for win in (2, 4, 8, 16, 32):
        ids = _phash_groups(numeric_samples, window=win)
        s = _cluster_stats(ids)
        print(
            f"  {win:>6}  {s['n_groups']:>9}  {s['compression']:>10.2f}  "
            f"{s['largest_cluster']:>8}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
