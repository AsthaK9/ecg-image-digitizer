"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point env-driven config paths at a temp dir so tests never touch
    real project data/artifacts directories.
    """
    monkeypatch.setenv("ECG_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ECG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    os.environ.setdefault("PYTHONHASHSEED", "42")
