"""General-purpose I/O helpers shared across data, training, and inference code.

Deliberately dependency-light and free of any ML framework imports so it can
be unit tested in isolation and reused anywhere in the codebase (including
lightweight preprocessing scripts that shouldn't need to import torch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (and parents) if it doesn't exist; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def list_record_ids(directory: str | Path, suffix: str = ".hea") -> list[str]:
    """Return sorted, de-duplicated record IDs found in `directory`.

    A "record ID" is the filename stem shared by a WFDB header/signal pair
    and its associated image(s), e.g. `00001_lr` for `00001_lr.hea`.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted({p.stem for p in directory.glob(f"*{suffix}")})
