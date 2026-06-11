"""Idempotent one-command deploy of the Gradio demo to a Hugging Face Space.

This script packages the repository's Gradio app (``app.py`` + the ``src/``
inference path) and uploads it to a Hugging Face Space using the ``HfApi`` from
``huggingface_hub``. It is designed to be run repeatedly: ``create_repo`` is
called with ``exist_ok=True`` and ``upload_folder`` overwrites changed files, so
re-running simply syncs the latest state.

Authentication is read from the ``HF_TOKEN`` environment variable (or
``--token``); the token is **never** hardcoded or logged. If no token is
available, the script refuses to perform any network call and exits cleanly —
use ``--dry-run`` to preview exactly what would be uploaded without a token.

Typical use::

    export HF_TOKEN=hf_...                       # write-scoped token
    python scripts/deploy_hf.py --space-id you/asl-cnn-classifier

Preview only (no token, no network)::

    python scripts/deploy_hf.py --space-id you/asl --dry-run

The Space-side ``README.md`` (with the required HF YAML frontmatter) and
``requirements.txt`` are sourced from the ``space/`` directory and uploaded to
the Space root under their canonical names; see ``docs/DEPLOY.md``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repository root (this file lives at <root>/scripts/deploy_hf.py).
REPO_ROOT = Path(__file__).resolve().parent.parent

# Files/dirs uploaded to the Space, expressed as allow patterns for
# ``upload_folder``. These are the runtime essentials: the Gradio entry point,
# the inference source, configs, and the small committed sample images that back
# the in-app examples. A trained checkpoint is included only if present.
ALLOW_PATTERNS: list[str] = [
    "app.py",
    "src/**",
    "configs/**",
    "data/sample/**",
]

# Patterns never uploaded, even if matched above or added later. Keeps the Space
# lean and avoids shipping the venv, git internals, caches, tests, or large
# regenerated artifacts.
#
# NOTE: ``artifacts/`` is excluded, but the trained checkpoint lives under it and
# IS shipped when present. In ``huggingface_hub.upload_folder`` an
# ``ignore_patterns`` match always wins over ``allow_patterns``, so a blanket
# ``artifacts/**`` ignore would silently drop the checkpoint even though it is in
# ``allow_patterns``. We therefore exclude the artifacts subdirectories
# individually and leave ``artifacts/checkpoints/`` shippable.
IGNORE_PATTERNS: list[str] = [
    ".venv/**",
    ".git/**",
    "artifacts/runs/**",
    "artifacts/camera_snapshots/**",
    "artifacts/gradcam/**",
    "artifacts/*.png",
    "artifacts/*.json",
    "artifacts/*.txt",
    "artifacts/*.onnx",
    "**/__pycache__/**",
    "*.pyc",
    "tests/**",
    ".github/**",
    "runs/**",
]

# A trained checkpoint, if it exists, is uploaded so predictions are real.
# Otherwise the app falls back to the random-init demo (documented in the UI).
CHECKPOINT_REL = "artifacts/checkpoints/best_model.pth"

# Space-side files staged in ``space/`` and uploaded to the Space root under
# their canonical names (so they don't collide with the repo's own files).
SPACE_README = "space/README.md"
SPACE_REQUIREMENTS = "space/requirements.txt"


def space_url(space_id: str) -> str:
    """Return the public Space URL for ``user/space``."""
    return f"https://huggingface.co/spaces/{space_id}"


def validate_space_id(space_id: str) -> str:
    """Validate and normalize a ``user/space`` identifier.

    Args:
        space_id: The Space id, e.g. ``"alice/asl-cnn-classifier"``.

    Returns:
        The stripped ``space_id``.

    Raises:
        ValueError: If it is not exactly one ``user/space`` pair.
    """
    cleaned = space_id.strip()
    parts = cleaned.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"--space-id must be of the form '<user>/<space>', got {space_id!r}."
        )
    return cleaned


def resolve_token(cli_token: str | None) -> str | None:
    """Resolve the HF token from ``--token`` then the ``HF_TOKEN`` env var.

    The CLI value takes precedence over the environment. Returns ``None`` if
    neither is set; callers decide whether that is fatal (real run) or fine
    (dry run).
    """
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or None


def planned_uploads(repo_root: Path = REPO_ROOT) -> list[str]:
    """Compute the concrete list of repo-relative files that would be uploaded.

    This expands :data:`ALLOW_PATTERNS` against the actual filesystem, applies
    :data:`IGNORE_PATTERNS`, and appends the checkpoint and the two Space-side
    files when present. It performs no network access, so it is safe for
    ``--dry-run`` and for unit tests.

    Args:
        repo_root: Repository root to resolve patterns against (overridable for
            tests).

    Returns:
        Sorted, de-duplicated list of POSIX-style relative paths.
    """
    selected: set[str] = set()

    for pattern in ALLOW_PATTERNS:
        # ``upload_folder`` treats ``dir/**`` (fnmatch) as "everything under
        # dir", but pathlib's ``**`` only matches directories — it needs
        # ``dir/**/*`` to reach files. Translate so the preview matches the
        # real upload's file set exactly.
        glob_pattern = pattern + "/*" if pattern.endswith("**") else pattern
        for path in repo_root.glob(glob_pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_root).as_posix()
            if not _is_ignored(rel):
                selected.add(rel)

    # Space-side files uploaded under their canonical Space names.
    if (repo_root / SPACE_README).is_file():
        selected.add(SPACE_README)
    if (repo_root / SPACE_REQUIREMENTS).is_file():
        selected.add(SPACE_REQUIREMENTS)

    # Trained checkpoint is optional; include it only if it exists on disk.
    if (repo_root / CHECKPOINT_REL).is_file():
        selected.add(CHECKPOINT_REL)

    return sorted(selected)


def _is_ignored(rel_path: str) -> bool:
    """Return True if ``rel_path`` matches any pattern in :data:`IGNORE_PATTERNS`.

    Mirrors the ``**``-spanning glob semantics that
    ``huggingface_hub.upload_folder`` applies, so the dry-run preview matches the
    real upload's ignore behavior.
    """
    from fnmatch import fnmatch

    for pattern in IGNORE_PATTERNS:
        if fnmatch(rel_path, pattern):
            return True
        # ``dir/**`` should also match files directly under ``dir``.
        if pattern.endswith("/**"):
            base = pattern[:-3]
            if rel_path == base or rel_path.startswith(base + "/"):
                return True
        # ``**/<seg>/**`` (e.g. __pycache__) should match the segment anywhere.
        if "/**/" in pattern:
            segment = pattern.split("/**/")[-1].rstrip("/*")
            if segment and (
                f"/{segment}/" in f"/{rel_path}/" or rel_path.startswith(segment + "/")
            ):
                return True
    return False


def _print_plan(space_id: str, files: list[str], *, has_token: bool) -> None:
    """Print the dry-run / pre-flight plan to stdout."""
    url = space_url(space_id)
    has_checkpoint = CHECKPOINT_REL in files
    print(f"Space:            {space_id}")
    print(f"Space URL:        {url}")
    print("SDK:              gradio")
    print(f"Token present:    {'yes' if has_token else 'no'}")
    print(
        "Checkpoint:       "
        + (
            f"{CHECKPOINT_REL} (real predictions)"
            if has_checkpoint
            else "none found — Space will run the untrained random-init demo"
        )
    )
    print(f"Files to upload ({len(files)}):")
    for rel in files:
        if rel == SPACE_README:
            print(f"  {rel}  ->  README.md")
        elif rel == SPACE_REQUIREMENTS:
            print(f"  {rel}  ->  requirements.txt")
        else:
            print(f"  {rel}")


def deploy(
    space_id: str,
    *,
    token: str | None,
    sdk: str = "gradio",
    dry_run: bool = False,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Create (if needed) and sync the Space, or preview the plan on dry-run.

    Args:
        space_id: Validated ``user/space`` identifier.
        token: HF token, or ``None``.
        sdk: Space SDK (only ``"gradio"`` is supported/tested here).
        dry_run: If True, print the plan and return without any network call.
        repo_root: Repository root (overridable for tests).

    Returns:
        The resolved Space URL.

    Raises:
        SystemExit: If a real (non-dry-run) deploy is requested without a token.
    """
    space_id = validate_space_id(space_id)
    files = planned_uploads(repo_root)
    url = space_url(space_id)

    if dry_run:
        print("DRY RUN — no network calls will be made.\n")
        _print_plan(space_id, files, has_token=bool(token))
        print(f"\nWould deploy to: {url}")
        return url

    if not token:
        raise SystemExit(
            "ERROR: no HF token. Set the HF_TOKEN environment variable or pass "
            "--token (write scope). Get one at "
            "https://huggingface.co/settings/tokens. Re-run with --dry-run to "
            "preview without a token."
        )

    _print_plan(space_id, files, has_token=True)

    # Imported lazily so --dry-run and tests don't require the heavy dependency.
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk=sdk,
        exist_ok=True,
    )

    # Upload the Space-side README and requirements under their canonical names
    # first, then the application folder. upload_folder is idempotent.
    api.upload_file(
        path_or_fileobj=str(repo_root / SPACE_README),
        path_in_repo="README.md",
        repo_id=space_id,
        repo_type="space",
    )
    api.upload_file(
        path_or_fileobj=str(repo_root / SPACE_REQUIREMENTS),
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space",
    )
    api.upload_folder(
        folder_path=str(repo_root),
        repo_id=space_id,
        repo_type="space",
        allow_patterns=ALLOW_PATTERNS
        + ([CHECKPOINT_REL] if (repo_root / CHECKPOINT_REL).is_file() else []),
        ignore_patterns=IGNORE_PATTERNS,
    )

    print(f"\nDeployed. Space URL: {url}")
    return url


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Deploy the Gradio demo to a Hugging Face Space (idempotent).",
    )
    parser.add_argument(
        "--space-id",
        required=True,
        help="Target Space as '<user>/<space>', e.g. 'alice/asl-cnn-classifier'.",
    )
    parser.add_argument(
        "--sdk",
        default="gradio",
        choices=["gradio"],
        help="Space SDK (only 'gradio' is supported).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF write token. Defaults to the HF_TOKEN env var. Never logged.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exactly what would be uploaded and the Space URL; no network.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    token = resolve_token(args.token)

    try:
        deploy(
            args.space_id,
            token=token,
            sdk=args.sdk,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
