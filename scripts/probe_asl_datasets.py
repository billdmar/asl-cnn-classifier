"""Probe candidate HF ASL-alphabet datasets for a 3rd, training-diverse source.

Tries a list of public A-Z ASL alphabet datasets and reports, for each, whether
it loads anonymously, its split sizes, label feature, and a normalized A-Z class
count. We need a dataset DISTINCT from both Marxulia (asl_real, training) and
EitanG98/asl_letters (asl_crossval, our held-out real-world test). Read-only.
"""

from __future__ import annotations

import sys

CANDIDATES = [
    ("aliciiavs/sign-language-image-dataset", None),
    ("Akash190104/american_sign_language", None),
    ("mariosasko/asl-alphabet", None),
    ("NeuML/asl-alphabet", None),
    ("dewa/american-sign-language", None),
    ("pitssm/asl_alphabet", None),
    ("Voxel51/ASL-Alphabet", None),
    ("Marxulia/asl_sign_languages_alphabets_v01", None),
    ("Marxulia/asl_sign_languages_alphabets_v02", None),
]


def probe(hf_id: str, split: str | None) -> None:
    from datasets import load_dataset

    try:
        ds = load_dataset(hf_id, split=split or "train")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {hf_id}: {type(exc).__name__}: {str(exc)[:160]}")
        return
    feats = getattr(ds, "features", {})
    label = feats.get("label") if feats else None
    names = getattr(label, "names", None)
    n = len(ds)
    cols = list(feats.keys()) if feats else []
    print(f"OK    {hf_id}: n={n} cols={cols} label_names={names}")


def main() -> int:
    for hf_id, split in CANDIDATES:
        probe(hf_id, split)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
