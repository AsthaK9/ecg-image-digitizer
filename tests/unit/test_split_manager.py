"""Unit tests for LeakageSafeSplitManager."""

from __future__ import annotations

import pandas as pd
import pytest

from ecg_digitization.data.split_manager import LeakageSafeSplitManager, SplitResult


def test_extract_base_record_id():
    """Extracts root parent record ID from augmented and synthetic variant identifiers."""
    manager = LeakageSafeSplitManager()

    # Standard PTB-XL record format
    assert manager.extract_base_record_id("00001_lr") == "00001_lr"
    assert manager.extract_base_record_id("00001_lr_deg01") == "00001_lr"
    assert manager.extract_base_record_id("00001_lr_rot2") == "00001_lr"
    assert manager.extract_base_record_id("00001_lr_seed42") == "00001_lr"
    assert manager.extract_base_record_id("12345-crop") == "12345"
    assert manager.extract_base_record_id("00042") == "00042"


def test_assign_ptbxl_splits_with_metadata():
    """Assigns folds 1-8 to train, 9 to val, 10 to test without leakage."""
    # Dummy PTB-XL metadata table with 5 patients across folds
    df = pd.DataFrame(
        {
            "strat_fold": [1, 2, 8, 9, 10],
            "patient_id": [101, 102, 103, 104, 105],
        },
        index=[1, 2, 3, 4, 5],
    )

    sample_ids = [
        "00001_lr",
        "00001_lr_deg1",
        "00002_lr",
        "00003_lr_noise",
        "00004_lr",
        "00005_lr_rot",
    ]

    result = LeakageSafeSplitManager.assign_ptbxl_splits(sample_ids, df)

    assert isinstance(result, SplitResult)
    assert not result.leakage_detected

    # Folds 1, 2, 8 are in train -> 00001_lr, 00001_lr_deg1, 00002_lr, 00003_lr_noise
    assert "00001_lr" in result.train_ids
    assert "00001_lr_deg1" in result.train_ids
    assert "00002_lr" in result.train_ids
    assert "00003_lr_noise" in result.train_ids

    # Fold 9 is in val -> 00004_lr
    assert "00004_lr" in result.val_ids

    # Fold 10 is in test -> 00005_lr_rot
    assert "00005_lr_rot" in result.test_ids


def test_verify_no_leakage_raises_on_overlap():
    """Asserts that leakage detection raises AssertionError if records cross splits."""
    train = ["00001_lr", "00002_lr"]
    val = ["00001_lr_deg1", "00003_lr"]  # 00001_lr leaks into val!
    test = ["00004_lr"]

    with pytest.raises(AssertionError, match="Leakage detected"):
        LeakageSafeSplitManager.verify_no_leakage(train, val, test)


def test_verify_no_leakage_passes_on_disjoint():
    """Passes cleanly when base records are strictly disjoint."""
    train = ["00001_lr", "00001_lr_deg1", "00002_lr"]
    val = ["00003_lr", "00003_lr_aug"]
    test = ["00004_lr"]

    assert LeakageSafeSplitManager.verify_no_leakage(train, val, test) is True
