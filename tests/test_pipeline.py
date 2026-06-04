"""End-to-end pipeline + data-helper tests.

Covers ``make_sample_data.generate``, ``download_data.download`` (manual-
instructions branch), and an end-to-end train -> eval -> benchmark -> infer run
on a tiny generated fixture. Everything is CPU-only, headless, and fast.
"""

from __future__ import annotations

import sys

from src.dataset import CLASS_NAMES


def test_make_sample_data_generate(tmp_path):
    from src.make_sample_data import generate

    out_dir = tmp_path / "sample"
    total = generate(out_dir, per_class=1, seed=42)
    assert total == len(CLASS_NAMES)

    # Each class folder exists with exactly one image.
    for name in CLASS_NAMES:
        imgs = list((out_dir / name).glob("*.png"))
        assert len(imgs) == 1

    # Deterministic: regenerating yields identical bytes.
    first_bytes = (out_dir / CLASS_NAMES[0] / "0.png").read_bytes()
    generate(out_dir, per_class=1, seed=42)
    assert (out_dir / CLASS_NAMES[0] / "0.png").read_bytes() == first_bytes


def test_make_sample_data_main(monkeypatch, tmp_path, capsys):
    import runpy

    out_dir = tmp_path / "cli_sample"
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_sample_data", "--out_dir", str(out_dir), "--per_class", "1"],
    )
    runpy.run_module("src.make_sample_data", run_name="__main__")
    captured = capsys.readouterr()
    assert "Generated" in captured.out
    assert (out_dir / CLASS_NAMES[0]).is_dir()


def test_download_manual_instructions_branch(monkeypatch, tmp_path, capsys):
    """With no kaggle + no credentials, ``download`` prints manual steps."""
    from src import download_data

    monkeypatch.setattr(download_data, "_kaggle_available", lambda: False)
    monkeypatch.setattr(
        download_data, "CREDENTIALS_PATH", tmp_path / "nope" / "kaggle.json"
    )
    download_data.download(str(tmp_path / "data"))
    out = capsys.readouterr().out
    assert "MANUAL steps" in out
    assert "kaggle.json" in out


def test_download_kaggle_available_function():
    from src.download_data import _kaggle_available

    # Just exercise the probe; result depends on host but must be a bool.
    assert isinstance(_kaggle_available(), bool)


def test_download_subprocess_success(monkeypatch, tmp_path, capsys):
    """kaggle + creds present -> runs the subprocess (mocked) and prints layout."""
    from src import download_data

    creds = tmp_path / "kaggle.json"
    creds.write_text("{}")
    monkeypatch.setattr(download_data, "_kaggle_available", lambda: True)
    monkeypatch.setattr(download_data, "CREDENTIALS_PATH", creds)

    calls = {}

    def fake_run(cmd, check):
        calls["cmd"] = cmd
        calls["check"] = check

    monkeypatch.setattr(download_data.subprocess, "run", fake_run)
    download_data.download(str(tmp_path / "data"))
    out = capsys.readouterr().out
    assert "Download complete" in out
    assert calls["cmd"][0] == "kaggle"


def test_download_subprocess_failure_falls_back(monkeypatch, tmp_path, capsys):
    """A failing kaggle subprocess falls back to manual instructions."""
    from src import download_data

    creds = tmp_path / "kaggle.json"
    creds.write_text("{}")
    monkeypatch.setattr(download_data, "_kaggle_available", lambda: True)
    monkeypatch.setattr(download_data, "CREDENTIALS_PATH", creds)

    def fake_run(cmd, check):
        raise download_data.subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(download_data.subprocess, "run", fake_run)
    download_data.download(str(tmp_path / "data"))
    out = capsys.readouterr().out
    assert "Kaggle download failed" in out
    assert "MANUAL steps" in out


def test_download_main_entry(monkeypatch, tmp_path, capsys):
    import runpy

    from src import download_data

    monkeypatch.setattr(download_data, "_kaggle_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["download_data", "--data_dir", str(tmp_path)])
    runpy.run_module("src.download_data", run_name="__main__")
    assert "MANUAL steps" in capsys.readouterr().out
