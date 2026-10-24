"""Shared test fixtures. The tests use generated images, so they run without the real data."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def git(folder: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(folder), *args], check=True, capture_output=True)


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """Return a temporary project folder: a Git repository with the committed settings and DATA_DIR=fake."""
    folder = tmp_path / "project"
    folder.mkdir()
    for name in ("experiment.yaml", "experiment_multi.yaml", ".gitignore"):
        shutil.copy(REPO / name, folder / name)
    for name in ("configs", "prompts", "slurm"):
        shutil.copytree(REPO / name, folder / name)
    git(folder, "init", "-q")
    git(folder, "-c", "user.name=test", "-c", "user.email=test@example.com", "add", "-A")
    git(folder, "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-q", "-m", "test project")
    monkeypatch.chdir(folder)
    monkeypatch.setenv("PROJECT_DIR", str(folder))
    monkeypatch.setenv("RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", "fake")
    for name in ("STATE_DIR", "EXPERIMENT_FILE", "SLURM_JOB_ID", "SLURM_RESTART_COUNT"):
        monkeypatch.delenv(name, raising=False)
    return folder
