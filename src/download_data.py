"""Download the ASL Alphabet dataset from Kaggle (or print manual instructions).

The real training data is the public `grassknoted/asl-alphabet` Kaggle dataset
(~87k images, ~1 GB), far too large to commit. This helper automates the
download when Kaggle credentials are present, and otherwise prints clear,
copy-pasteable manual instructions — it never crashes when ``kaggle`` is
missing or unauthenticated.

Run ``python -m src.download_data --data_dir data`` to fetch the dataset.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
from pathlib import Path

KAGGLE_DATASET = "grassknoted/asl-alphabet"
KAGGLE_URL = "https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
CREDENTIALS_PATH = Path.home() / ".kaggle" / "kaggle.json"

# Layout produced after unzipping the Kaggle archive (the dataset nests the
# train folder twice). train.py / eval.py point --data_dir at the innermost dir.
EXPECTED_LAYOUT = (
    "data/asl_alphabet_train/asl_alphabet_train/<class folders>\n"
    "    e.g. data/asl_alphabet_train/asl_alphabet_train/A/A1.jpg\n"
    "         data/asl_alphabet_train/asl_alphabet_train/space/space1.jpg"
)


def _kaggle_available() -> bool:
    """True if the kaggle CLI/package is installed.

    Detection is import-free on purpose: importing the ``kaggle`` package
    eagerly authenticates and calls ``sys.exit`` when no credentials are
    present, which would kill this process. We therefore probe the CLI on PATH
    and the importable module spec without executing it.
    """
    if shutil.which("kaggle") is not None:
        return True
    return importlib.util.find_spec("kaggle") is not None


def _print_manual_instructions() -> None:
    """Print step-by-step manual download instructions (no creds / no kaggle)."""
    print("=" * 70)
    print("Kaggle download unavailable — follow these MANUAL steps:")
    print("=" * 70)
    print("")
    print("1. Get a Kaggle API token:")
    print("   - Sign in at https://www.kaggle.com")
    print("   - Go to Account -> Create New API Token")
    print("   - This downloads a 'kaggle.json' file.")
    print("")
    print("2. Install the credentials:")
    print(f"   mkdir -p {CREDENTIALS_PATH.parent}")
    print(f"   mv ~/Downloads/kaggle.json {CREDENTIALS_PATH}")
    print(f"   chmod 600 {CREDENTIALS_PATH}")
    print("")
    print("3. Re-run this script (it will auto-download), OR download manually:")
    print(f"   {KAGGLE_URL}")
    print("   then unzip the archive into your data dir.")
    print("")
    print("Expected extracted layout:")
    print(f"    {EXPECTED_LAYOUT}")
    print("=" * 70)


def download(data_dir: str = "data") -> None:
    """Download + unzip the ASL Alphabet dataset, or print manual instructions.

    If the ``kaggle`` CLI/package is available AND ``~/.kaggle/kaggle.json``
    exists, runs ``kaggle datasets download`` with ``--unzip``. Otherwise prints
    manual instructions and returns. Never raises on a missing/failed Kaggle
    setup.
    """
    if not _kaggle_available() or not CREDENTIALS_PATH.exists():
        if not _kaggle_available():
            print("The 'kaggle' package/CLI was not found.")
        if not CREDENTIALS_PATH.exists():
            print(f"No Kaggle credentials found at {CREDENTIALS_PATH}.")
        print("")
        _print_manual_instructions()
        return

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        data_dir,
        "--unzip",
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        print(f"Kaggle download failed: {exc}")
        print("")
        _print_manual_instructions()
        return

    print(f"Download complete. Data extracted under {data_dir}/")
    print("Expected layout:")
    print(f"    {EXPECTED_LAYOUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the ASL Alphabet dataset from Kaggle."
    )
    parser.add_argument(
        "--data_dir", default="data", help="Directory to download/extract into."
    )
    args = parser.parse_args()
    download(args.data_dir)
