"""Tests for the Hugging Face Space deploy script (``scripts/deploy_hf.py``).

These tests exercise the script's pure logic only — they never touch the
network. ``huggingface_hub`` is monkeypatched with a fake ``HfApi`` so the
"real" deploy path can be asserted on without authenticating or uploading
anything. The dry-run and missing-token paths are covered directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load scripts/deploy_hf.py as a module (the scripts/ dir isn't a package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "deploy_hf", _REPO_ROOT / "scripts" / "deploy_hf.py"
)
assert _SPEC and _SPEC.loader
deploy_hf = importlib.util.module_from_spec(_SPEC)
sys.modules["deploy_hf"] = deploy_hf
_SPEC.loader.exec_module(deploy_hf)


# --------------------------------------------------------------------------- #
# space_url / validate_space_id
# --------------------------------------------------------------------------- #
def test_space_url_resolves_correctly():
    assert (
        deploy_hf.space_url("alice/asl-cnn-classifier")
        == "https://huggingface.co/spaces/alice/asl-cnn-classifier"
    )


@pytest.mark.parametrize("bad", ["", "noslash", "a/b/c", "/b", "a/", "  /  "])
def test_validate_space_id_rejects_malformed(bad):
    with pytest.raises(ValueError):
        deploy_hf.validate_space_id(bad)


def test_validate_space_id_strips_and_passes():
    assert deploy_hf.validate_space_id("  alice/asl  ") == "alice/asl"


# --------------------------------------------------------------------------- #
# resolve_token
# --------------------------------------------------------------------------- #
def test_resolve_token_cli_takes_precedence(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert deploy_hf.resolve_token("from-cli") == "from-cli"


def test_resolve_token_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert deploy_hf.resolve_token(None) == "from-env"


def test_resolve_token_none_when_absent(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert deploy_hf.resolve_token(None) is None


# --------------------------------------------------------------------------- #
# planned_uploads / ignore patterns
# --------------------------------------------------------------------------- #
def test_planned_uploads_lists_expected_core_files():
    files = deploy_hf.planned_uploads(_REPO_ROOT)
    assert "app.py" in files
    assert "space/README.md" in files
    assert "space/requirements.txt" in files
    # src/ inference path and configs are bundled.
    assert any(f.startswith("src/") and f.endswith(".py") for f in files)
    assert any(f.startswith("configs/") for f in files)
    # Sample images back the in-app examples.
    assert any(f.startswith("data/sample/") for f in files)


def test_planned_uploads_excludes_noise():
    files = deploy_hf.planned_uploads(_REPO_ROOT)
    assert not any("__pycache__" in f for f in files)
    assert not any(f.startswith(".venv") for f in files)
    assert not any(f.startswith(".git") for f in files)
    assert not any(f.startswith("tests/") for f in files)
    # Regenerated artifacts (confusion matrix, metrics.json, ONNX exports, …) are
    # never shipped. The ONE deliberate exception is the trained checkpoint,
    # which is uploaded when present so the Space gives real predictions.
    artifact_files = [f for f in files if f.startswith("artifacts/")]
    assert all(
        f == deploy_hf.CHECKPOINT_REL for f in artifact_files
    ), f"only the checkpoint may be uploaded from artifacts/, got: {artifact_files}"


def test_planned_uploads_is_sorted_and_unique():
    files = deploy_hf.planned_uploads(_REPO_ROOT)
    assert files == sorted(files)
    assert len(files) == len(set(files))


@pytest.mark.parametrize(
    "path",
    [
        ".venv/lib/python3.12/site-packages/foo.py",
        ".git/config",
        "artifacts/checkpoints/best_model.pth",
        "src/__pycache__/model.cpython-312.pyc",
        "tests/test_app.py",
        ".github/workflows/ci.yml",
    ],
)
def test_is_ignored_matches_noise(path):
    assert deploy_hf._is_ignored(path) is True


@pytest.mark.parametrize("path", ["app.py", "src/model.py", "configs/x.yaml"])
def test_is_ignored_keeps_real_files(path):
    assert deploy_hf._is_ignored(path) is False


def test_checkpoint_included_when_present(tmp_path):
    # Build a minimal fake repo with a checkpoint and assert it's picked up.
    (tmp_path / "app.py").write_text("# app\n")
    (tmp_path / "space").mkdir()
    (tmp_path / "space" / "README.md").write_text("readme\n")
    (tmp_path / "space" / "requirements.txt").write_text("gradio\n")
    ckpt = tmp_path / "artifacts" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "best_model.pth").write_bytes(b"weights")

    files = deploy_hf.planned_uploads(tmp_path)
    assert deploy_hf.CHECKPOINT_REL in files


def test_checkpoint_absent_when_missing(tmp_path):
    (tmp_path / "app.py").write_text("# app\n")
    files = deploy_hf.planned_uploads(tmp_path)
    assert deploy_hf.CHECKPOINT_REL not in files


# --------------------------------------------------------------------------- #
# deploy() — dry-run and token handling (no network)
# --------------------------------------------------------------------------- #
def test_dry_run_prints_plan_and_url(capsys):
    url = deploy_hf.deploy("demo/asl", token=None, dry_run=True)
    out = capsys.readouterr().out
    assert url == "https://huggingface.co/spaces/demo/asl"
    assert "DRY RUN" in out
    assert "https://huggingface.co/spaces/demo/asl" in out
    assert "app.py" in out
    assert "README.md" in out
    assert "requirements.txt" in out


def test_dry_run_does_not_import_hfapi(monkeypatch):
    # If deploy() touched HfApi on a dry run, this poisoned import would blow up.
    import builtins

    real_import = builtins.__import__

    def _guard(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise AssertionError("dry-run must not import huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guard)
    deploy_hf.deploy("demo/asl", token=None, dry_run=True)


def test_real_deploy_without_token_exits_cleanly(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        deploy_hf.deploy("demo/asl", token=None, dry_run=False)
    assert "HF_TOKEN" in str(exc.value)


def test_main_missing_token_real_run_exits(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        deploy_hf.main(["--space-id", "demo/asl"])


def test_main_bad_space_id_returns_error_code(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    rc = deploy_hf.main(["--space-id", "noslash"])
    assert rc == 2


def test_main_dry_run_returns_zero(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc = deploy_hf.main(["--space-id", "demo/asl", "--dry-run"])
    assert rc == 0
    assert "demo/asl" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# deploy() — real path with a fake HfApi (still no network)
# --------------------------------------------------------------------------- #
class _FakeHfApi:
    """Records calls instead of hitting the network."""

    instances: list["_FakeHfApi"] = []

    def __init__(self, token=None):
        self.token = token
        self.created = []
        self.uploaded_files = []
        self.uploaded_folders = []
        _FakeHfApi.instances.append(self)

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_file(self, **kwargs):
        self.uploaded_files.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploaded_folders.append(kwargs)


def test_real_deploy_uses_hfapi_idempotently(monkeypatch, capsys):
    import types

    _FakeHfApi.instances.clear()
    fake_module = types.ModuleType("huggingface_hub")
    fake_module.HfApi = _FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    url = deploy_hf.deploy("alice/asl", token="hf_secret", dry_run=False)

    assert url == "https://huggingface.co/spaces/alice/asl"
    api = _FakeHfApi.instances[-1]
    # Token threaded into the client, never printed.
    assert api.token == "hf_secret"
    assert "hf_secret" not in capsys.readouterr().out
    # create_repo is idempotent (exist_ok) and targets a gradio space.
    assert api.created[0]["exist_ok"] is True
    assert api.created[0]["repo_type"] == "space"
    assert api.created[0]["space_sdk"] == "gradio"
    # Space README + requirements uploaded under canonical names.
    dests = {f["path_in_repo"] for f in api.uploaded_files}
    assert {"README.md", "requirements.txt"} <= dests
    # Application folder uploaded once with the allow/ignore patterns.
    assert len(api.uploaded_folders) == 1
    folder = api.uploaded_folders[0]
    assert "app.py" in folder["allow_patterns"]
    assert ".venv/**" in folder["ignore_patterns"]
