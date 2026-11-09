"""Unit tests for Kaggle orchestration manager."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ecg_digitization.training.kaggle_manager import KaggleManager


def test_kaggle_manager_auth_check() -> None:
    manager = KaggleManager()
    status = manager.check_auth()

    assert isinstance(status.cli_available, bool)
    assert isinstance(status.auth_configured, bool)
    # Check that secrets are not exposed in masked_user
    if status.masked_user:
        assert "***" in status.masked_user or len(status.masked_user) <= 15


def test_kaggle_manager_prepare_kernel_dry_run() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = KaggleManager(workspace_dir=tmp_dir)
        card = manager.prepare_kernel(
            output_dir=tmp_dir,
            kernel_slug="test-ecg-training",
            title="Test ECG Training",
            datasets=["erhmrai/ecg-image-data"],
            enable_gpu=True,
        )

        assert card.kernel_slug == "test-ecg-training"
        assert card.enable_gpu is True
        assert "kaggle kernels push" in card.dry_run_command

        # Verify kernel-metadata.json
        meta_file = Path(tmp_dir) / "kernel-metadata.json"
        assert meta_file.is_file()
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["id"] == "test-ecg-training"
        assert meta["enable_gpu"] == "true"
        assert "erhmrai/ecg-image-data" in meta["dataset_sources"]

        # Verify train_kaggle.py script
        script_file = Path(tmp_dir) / "train_kaggle.py"
        assert script_file.is_file()
        content = script_file.read_text(encoding="utf-8")
        assert "ECGResNet1D" in content
