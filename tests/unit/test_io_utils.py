from __future__ import annotations

from pathlib import Path

import pytest

from ecg_digitization.utils.io_utils import (
    ensure_dir,
    list_record_ids,
    read_json,
    read_yaml,
    write_json,
    write_yaml,
)

pytestmark = pytest.mark.unit


def test_ensure_dir_creates_nested_path(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.exists()
    assert result.is_dir()


def test_yaml_roundtrip(tmp_path: Path) -> None:
    data = {"a": 1, "b": {"c": [1, 2, 3]}}
    path = tmp_path / "config.yaml"
    write_yaml(data, path)
    loaded = read_yaml(path)
    assert loaded == data


def test_json_roundtrip(tmp_path: Path) -> None:
    data = {"a": 1, "b": ["x", "y"]}
    path = tmp_path / "nested" / "data.json"
    write_json(data, path)
    loaded = read_json(path)
    assert loaded == data


def test_list_record_ids_deduplicates_and_sorts(tmp_path: Path) -> None:
    for name in ["00002.hea", "00001.hea", "00001.dat", "00003.hea"]:
        (tmp_path / name).touch()

    record_ids = list_record_ids(tmp_path, suffix=".hea")
    assert record_ids == ["00001", "00002", "00003"]


def test_list_record_ids_missing_dir_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert list_record_ids(missing) == []
