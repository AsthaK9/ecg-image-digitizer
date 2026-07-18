"""Integration test for ECGDataModule against a real, composed Hydra config
and a real (synthetic) on-disk record collection.

This is the test that proves the data layer actually integrates with the
existing project scaffold end-to-end: real config composition ->
ECGDataModule -> real PyTorch DataLoader -> real batches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import wfdb
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from ecg_digitization.data.datamodule import ECGDataModule
from ecg_digitization.data.types import EmptyDatasetError, UnsupportedDatasetSourceError

pytestmark = pytest.mark.integration

CONFIG_DIR = str((Path(__file__).parents[2] / "configs").resolve())

LEAD_NAMES = ["I", "II", "III"]
SIGNAL_LENGTH = 200
FS = 500


def _write_record(directory: Path, record_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, SIGNAL_LENGTH / FS, SIGNAL_LENGTH, endpoint=False)
    p_signal = np.stack([np.sin(2 * np.pi * (i + 1) * t) for i in range(len(LEAD_NAMES))], axis=1)
    wfdb.wrsamp(
        record_name=record_id,
        fs=FS,
        units=["mV"] * len(LEAD_NAMES),
        sig_name=LEAD_NAMES,
        p_signal=p_signal.astype(np.float64),
        fmt=["16"] * len(LEAD_NAMES),
        write_dir=str(directory),
    )
    # A matching synthetic "image" (arbitrary pixel content; loader only
    # needs a decodable file).
    import cv2

    image = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(directory / f"{record_id}-0.png"), image)


def _build_cfg(data_root: Path, num_workers: int = 0):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"env.data_root={data_root}",
                f"data.dataloader.num_workers={num_workers}",
                "data.dataloader.batch_size=2",
                "data.image.target_size=[32,32]",
            ],
        )
    return cfg


class TestECGDataModuleSinglePoolSplit:
    def test_setup_and_dataloaders_end_to_end(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        for i in range(10):
            _write_record(raw_dir, f"rec{i:03d}")

        cfg = _build_cfg(tmp_path)
        datamodule = ECGDataModule(cfg)
        datamodule.setup()

        datasets = datamodule.datasets
        assert set(datasets.keys()) == {"train", "val", "test"}
        total = sum(len(ds) for ds in datasets.values())
        assert total == 10

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        assert batch["image"].shape[1:] == (3, 32, 32)
        assert "signal" in batch  # train split always has ground truth

    def test_split_is_deterministic_given_same_seed(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        for i in range(10):
            _write_record(raw_dir, f"rec{i:03d}")

        cfg_a = _build_cfg(tmp_path)
        cfg_b = _build_cfg(tmp_path)

        dm_a = ECGDataModule(cfg_a)
        dm_a.setup()
        dm_b = ECGDataModule(cfg_b)
        dm_b.setup()

        assert dm_a.datasets["train"].record_ids == dm_b.datasets["train"].record_ids
        assert dm_a.datasets["val"].record_ids == dm_b.datasets["val"].record_ids
        assert dm_a.datasets["test"].record_ids == dm_b.datasets["test"].record_ids

    def test_no_leakage_of_record_ids_across_splits(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        for i in range(20):
            _write_record(raw_dir, f"rec{i:03d}")

        cfg = _build_cfg(tmp_path)
        datamodule = ECGDataModule(cfg)
        datamodule.setup()

        train_ids = set(datamodule.datasets["train"].record_ids)
        val_ids = set(datamodule.datasets["val"].record_ids)
        test_ids = set(datamodule.datasets["test"].record_ids)

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert train_ids | val_ids | test_ids == {f"rec{i:03d}" for i in range(20)}

    def test_empty_dataset_raises_clear_error(self, tmp_path: Path) -> None:
        cfg = _build_cfg(tmp_path)  # no records written at all
        datamodule = ECGDataModule(cfg)
        with pytest.raises(EmptyDatasetError):
            datamodule.setup()

    def test_dataloader_before_setup_raises_runtime_error(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        _write_record(raw_dir, "rec000")
        cfg = _build_cfg(tmp_path)
        datamodule = ECGDataModule(cfg)
        with pytest.raises(RuntimeError, match="not been built yet"):
            datamodule.train_dataloader()


class TestECGDataModulePreSplitLayout:
    def test_uses_pre_split_directories_when_present(self, tmp_path: Path) -> None:
        for i in range(4):
            _write_record(tmp_path / "processed" / "train", f"train{i}")
        for i in range(2):
            _write_record(tmp_path / "processed" / "val", f"val{i}")
        for i in range(2):
            _write_record(tmp_path / "processed" / "test", f"test{i}")

        cfg = _build_cfg(tmp_path)
        datamodule = ECGDataModule(cfg)
        datamodule.setup()

        assert len(datamodule.datasets["train"]) == 4
        assert len(datamodule.datasets["val"]) == 2
        assert len(datamodule.datasets["test"]) == 2
        assert set(datamodule.datasets["train"].record_ids) == {
            "train0",
            "train1",
            "train2",
            "train3",
        }


class TestECGDataModuleSourceRegistry:
    def test_unregistered_source_raises_clear_error(self, tmp_path: Path) -> None:
        cfg = _build_cfg(tmp_path)
        with open_dict(cfg):
            cfg.data.source = "not_a_real_dataset_source"
        with pytest.raises(UnsupportedDatasetSourceError):
            ECGDataModule(cfg)

    def test_default_source_is_physionet_when_unset(self, tmp_path: Path) -> None:
        cfg = _build_cfg(tmp_path)
        datamodule = ECGDataModule(cfg)
        assert datamodule._source_name == "physionet"  # noqa: SLF001 (white-box check by design)
