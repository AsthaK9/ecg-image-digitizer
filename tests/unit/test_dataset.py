"""Unit tests for ecg_digitization.data.dataset and ecg_digitization.data.io.

Split into three groups:
  * TestECGImageDataset / TestCollateEcgBatch — exercise ECGImageDataset and
    collate_ecg_batch against lightweight in-memory fakes of the
    RecordLocator/RecordParser/ImageLoader protocols, so these tests are
    fast and fully isolated from real file formats.
  * TestPhysioNetRecordLocator / TestOpenCVImageLoader — exercise the real
    io.py implementations against real files on a temp filesystem, since
    their entire job is filesystem/codec interaction that fakes cannot
    meaningfully verify.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ecg_digitization.data.dataset import ECGImageDataset, collate_ecg_batch
from ecg_digitization.data.io import OpenCVImageLoader, PhysioNetRecordLocator
from ecg_digitization.data.types import (
    ECGRecordMetadata,
    ECGRecordPaths,
    ECGSignal,
    ImageLoadError,
    RecordNotFoundError,
    RecordParsingError,
)

pytestmark = pytest.mark.unit

IMAGE_SIZE = (32, 32)  # (H, W)
NUM_LEADS = 3
SIGNAL_LENGTH = 100


# ============================================================================
# In-memory test doubles implementing the data-layer protocols
# ============================================================================


class FakeLocator:
    """Minimal RecordLocator: every record has a real image path and a
    header/signal path only if `with_ground_truth` includes it.
    """

    def __init__(
        self,
        record_ids: list[str],
        image_dir: Path,
        with_ground_truth: set[str] | None = None,
        missing: set[str] | None = None,
    ) -> None:
        self.record_ids = record_ids
        self.image_dir = image_dir
        self.with_ground_truth = (
            with_ground_truth if with_ground_truth is not None else set(record_ids)
        )
        self.missing = missing or set()
        self.get_paths_calls: list[str] = []

    def list_record_ids(self) -> list[str]:
        return list(self.record_ids)

    def get_paths(self, record_id: str) -> ECGRecordPaths:
        self.get_paths_calls.append(record_id)
        if record_id in self.missing:
            raise RecordNotFoundError(f"No files for '{record_id}'.")

        has_gt = record_id in self.with_ground_truth
        return ECGRecordPaths(
            record_id=record_id,
            header_path=(self.image_dir / f"{record_id}.hea") if has_gt else None,
            signal_path=(self.image_dir / f"{record_id}.dat") if has_gt else None,
            image_paths=[self.image_dir / f"{record_id}.png"],
        )


class FakeParser:
    """Minimal RecordParser returning a deterministic synthetic signal per record_id."""

    def __init__(self, num_leads: int = NUM_LEADS, signal_length: int = SIGNAL_LENGTH) -> None:
        self.num_leads = num_leads
        self.signal_length = signal_length
        self.parse_calls: list[str] = []

    def parse(self, paths: ECGRecordPaths, source_dataset: str) -> ECGSignal:
        self.parse_calls.append(paths.record_id)
        values = np.full((self.num_leads, self.signal_length), fill_value=1.0, dtype=np.float32)
        metadata = ECGRecordMetadata(
            record_id=paths.record_id,
            source_dataset=source_dataset,
            sampling_rate_hz=500.0,
            num_leads=self.num_leads,
            lead_names=[f"L{i}" for i in range(self.num_leads)],
            signal_length=self.signal_length,
        )
        return ECGSignal(values=values, metadata=metadata)


class FailingParser:
    """RecordParser that always raises, to test error propagation."""

    def parse(self, paths: ECGRecordPaths, source_dataset: str) -> ECGSignal:
        raise RecordParsingError(f"Simulated parse failure for '{paths.record_id}'.")


class FakeImageLoader:
    """Minimal ImageLoader returning a fixed-size synthetic RGB array."""

    def __init__(self, size: tuple[int, int] = IMAGE_SIZE) -> None:
        self.size = size
        self.load_calls: list[Path] = []

    def load(self, path: Path) -> np.ndarray:
        self.load_calls.append(path)
        height, width = self.size
        return np.full((height, width, 3), fill_value=128, dtype=np.uint8)


def _make_dataset(
    record_ids: list[str],
    tmp_path: Path,
    with_ground_truth: set[str] | None = None,
    require_signal: bool = True,
    parser: object | None = None,
) -> tuple[ECGImageDataset, FakeLocator, object, FakeImageLoader]:
    locator = FakeLocator(record_ids, tmp_path, with_ground_truth=with_ground_truth)
    used_parser = parser if parser is not None else FakeParser()
    image_loader = FakeImageLoader()
    dataset = ECGImageDataset(
        record_ids=record_ids,
        locator=locator,
        parser=used_parser,
        image_loader=image_loader,
        target_image_size=IMAGE_SIZE,
        require_signal=require_signal,
        source_dataset_name="fake",
    )
    return dataset, locator, used_parser, image_loader


# ============================================================================
# ECGImageDataset
# ============================================================================


class TestECGImageDataset:
    def test_len_matches_record_ids(self, tmp_path: Path) -> None:
        dataset, *_ = _make_dataset(["a", "b", "c"], tmp_path)
        assert len(dataset) == 3

    def test_empty_record_ids_raises_value_error(self, tmp_path: Path) -> None:
        locator = FakeLocator([], tmp_path)
        with pytest.raises(ValueError, match="at least one record_id"):
            ECGImageDataset(
                record_ids=[],
                locator=locator,
                parser=FakeParser(),
                image_loader=FakeImageLoader(),
            )

    def test_construction_does_not_touch_locator_or_parser(self, tmp_path: Path) -> None:
        """Core lazy-loading contract: nothing is read from disk at construction time."""
        dataset, locator, parser, image_loader = _make_dataset(["a", "b"], tmp_path)
        assert locator.get_paths_calls == []
        assert parser.parse_calls == []
        assert image_loader.load_calls == []
        assert len(dataset) == 2  # __len__ itself must not trigger loading

    def test_getitem_triggers_lazy_load_of_exactly_one_record(self, tmp_path: Path) -> None:
        dataset, locator, parser, image_loader = _make_dataset(["a", "b", "c"], tmp_path)
        _ = dataset[1]
        assert locator.get_paths_calls == ["b"]
        assert parser.parse_calls == ["b"]
        assert len(image_loader.load_calls) == 1

    def test_getitem_returns_expected_keys_and_shapes(self, tmp_path: Path) -> None:
        dataset, *_ = _make_dataset(["a"], tmp_path)
        sample = dataset[0]

        assert set(sample.keys()) == {
            "record_id",
            "image",
            "signal",
            "sampling_rate_hz",
            "lead_names",
        }
        assert sample["record_id"] == "a"
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape == (3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        assert sample["image"].dtype == torch.float32
        assert sample["signal"].shape == (NUM_LEADS, SIGNAL_LENGTH)
        assert sample["sampling_rate_hz"] == 500.0
        assert sample["lead_names"] == ["L0", "L1", "L2"]

    def test_image_values_normalized_to_unit_range(self, tmp_path: Path) -> None:
        dataset, *_ = _make_dataset(["a"], tmp_path)
        sample = dataset[0]
        assert torch.all(sample["image"] >= 0.0)
        assert torch.all(sample["image"] <= 1.0)
        # FakeImageLoader returns uint8 value 128 everywhere -> 128/255.
        assert torch.allclose(sample["image"], torch.full_like(sample["image"], 128 / 255.0))

    def test_require_signal_false_returns_none_for_missing_ground_truth(
        self, tmp_path: Path
    ) -> None:
        dataset, *_ = _make_dataset(["a"], tmp_path, with_ground_truth=set(), require_signal=False)
        sample = dataset[0]
        assert sample["signal"] is None
        assert sample["sampling_rate_hz"] is None
        assert sample["lead_names"] is None

    def test_require_signal_true_raises_for_missing_ground_truth(self, tmp_path: Path) -> None:
        dataset, *_ = _make_dataset(["a"], tmp_path, with_ground_truth=set(), require_signal=True)
        with pytest.raises(RecordParsingError, match="no ground-truth signal"):
            dataset[0]

    def test_out_of_range_index_raises_index_error(self, tmp_path: Path) -> None:
        dataset, *_ = _make_dataset(["a", "b"], tmp_path)
        with pytest.raises(IndexError):
            dataset[5]
        with pytest.raises(IndexError):
            dataset[-1]

    def test_locator_not_found_propagates_as_record_not_found_error(self, tmp_path: Path) -> None:
        locator = FakeLocator(["a"], tmp_path, missing={"a"})
        dataset = ECGImageDataset(
            record_ids=["a"],
            locator=locator,
            parser=FakeParser(),
            image_loader=FakeImageLoader(),
        )
        with pytest.raises(RecordNotFoundError):
            dataset[0]

    def test_parser_failure_propagates_as_record_parsing_error(self, tmp_path: Path) -> None:
        dataset, *_ = _make_dataset(["a"], tmp_path, parser=FailingParser())
        with pytest.raises(RecordParsingError, match="Simulated parse failure"):
            dataset[0]

    def test_transform_is_applied_to_image(self, tmp_path: Path) -> None:
        locator = FakeLocator(["a"], tmp_path)
        calls: list[np.ndarray] = []

        def transform(image: np.ndarray) -> np.ndarray:
            calls.append(image)
            return np.zeros_like(image)

        dataset = ECGImageDataset(
            record_ids=["a"],
            locator=locator,
            parser=FakeParser(),
            image_loader=FakeImageLoader(),
            target_image_size=IMAGE_SIZE,
            transform=transform,
        )
        sample = dataset[0]
        assert len(calls) == 1
        assert torch.all(sample["image"] == 0.0)

    def test_record_ids_property_is_a_copy(self, tmp_path: Path) -> None:
        dataset, *_ = _make_dataset(["a", "b"], tmp_path)
        ids = dataset.record_ids
        ids.append("mutated")
        assert dataset.record_ids == ["a", "b"]

    def test_validate_returns_failures_without_raising(self, tmp_path: Path) -> None:
        locator = FakeLocator(["good", "bad"], tmp_path, missing={"bad"})
        dataset = ECGImageDataset(
            record_ids=["good", "bad"],
            locator=locator,
            parser=FakeParser(),
            image_loader=FakeImageLoader(),
        )
        failures = dataset.validate()
        assert len(failures) == 1
        assert failures[0][0] == "bad"

    def test_validate_returns_empty_when_all_records_load(self, tmp_path: Path) -> None:
        dataset, *_ = _make_dataset(["a", "b"], tmp_path)
        assert dataset.validate() == []


# ============================================================================
# collate_ecg_batch
# ============================================================================


class TestCollateEcgBatch:
    def _sample(self, record_id: str, signal_len: int | None) -> dict:
        return {
            "record_id": record_id,
            "image": torch.zeros(3, 8, 8),
            "signal": torch.ones(NUM_LEADS, signal_len) if signal_len is not None else None,
            "sampling_rate_hz": 500.0 if signal_len is not None else None,
            "lead_names": ["L0", "L1", "L2"] if signal_len is not None else None,
        }

    def test_stacks_images_and_pads_equal_length_signals(self) -> None:
        batch = [self._sample("a", 10), self._sample("b", 10)]
        collated = collate_ecg_batch(batch)
        assert collated["image"].shape == (2, 3, 8, 8)
        assert collated["signal"].shape == (2, NUM_LEADS, 10)
        assert torch.equal(collated["signal_length"], torch.tensor([10, 10]))

    def test_pads_variable_length_signals_with_zeros(self) -> None:
        batch = [self._sample("a", 5), self._sample("b", 10)]
        collated = collate_ecg_batch(batch)
        assert collated["signal"].shape == (2, NUM_LEADS, 10)
        # Shorter sample's tail must be zero-padded.
        assert torch.all(collated["signal"][0, :, 5:] == 0.0)
        assert torch.all(collated["signal"][0, :, :5] == 1.0)
        assert torch.equal(collated["signal_length"], torch.tensor([5, 10]))

    def test_omits_signal_key_when_all_samples_lack_signal(self) -> None:
        batch = [self._sample("a", None), self._sample("b", None)]
        collated = collate_ecg_batch(batch)
        assert "signal" not in collated
        assert "signal_length" not in collated
        assert collated["record_id"] == ["a", "b"]

    def test_raises_on_mixed_present_and_absent_signals(self) -> None:
        batch = [self._sample("a", 10), self._sample("b", None)]
        with pytest.raises(ValueError, match="mixes records with and without"):
            collate_ecg_batch(batch)

    def test_raises_on_empty_batch(self) -> None:
        with pytest.raises(ValueError, match="empty batch"):
            collate_ecg_batch([])

    def test_integrates_with_real_dataset_via_dataloader(self, tmp_path: Path) -> None:
        dataset, *_ = _make_dataset(["a", "b", "c"], tmp_path)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False, collate_fn=collate_ecg_batch
        )
        batches = list(loader)
        assert len(batches) == 2  # 3 records, batch_size=2 -> batches of 2 and 1
        assert batches[0]["image"].shape == (2, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        assert batches[1]["image"].shape == (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])


# ============================================================================
# PhysioNetRecordLocator (real filesystem)
# ============================================================================


class TestPhysioNetRecordLocator:
    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    def test_lists_record_with_header_and_image(self, tmp_path: Path) -> None:
        self._touch(tmp_path / "00001.hea")
        self._touch(tmp_path / "00001.dat")
        self._touch(tmp_path / "00001-0.png")

        locator = PhysioNetRecordLocator(root_dir=tmp_path)
        assert locator.list_record_ids() == ["00001"]

    def test_lists_image_only_record(self, tmp_path: Path) -> None:
        self._touch(tmp_path / "00002-0.png")
        locator = PhysioNetRecordLocator(root_dir=tmp_path)
        assert locator.list_record_ids() == ["00002"]

    def test_deduplicates_multiple_images_per_record(self, tmp_path: Path) -> None:
        self._touch(tmp_path / "00003-0.png")
        self._touch(tmp_path / "00003-1.png")
        locator = PhysioNetRecordLocator(root_dir=tmp_path)
        assert locator.list_record_ids() == ["00003"]

    def test_list_record_ids_sorted_and_unique_across_multiple_records(
        self, tmp_path: Path
    ) -> None:
        for rid in ["00003", "00001", "00002"]:
            self._touch(tmp_path / f"{rid}.hea")
            self._touch(tmp_path / f"{rid}-0.png")
        locator = PhysioNetRecordLocator(root_dir=tmp_path)
        assert locator.list_record_ids() == ["00001", "00002", "00003"]

    def test_missing_root_dir_returns_empty_list(self, tmp_path: Path) -> None:
        locator = PhysioNetRecordLocator(root_dir=tmp_path / "does_not_exist")
        assert locator.list_record_ids() == []

    def test_get_paths_resolves_header_signal_and_images(self, tmp_path: Path) -> None:
        self._touch(tmp_path / "00001.hea")
        self._touch(tmp_path / "00001.dat")
        self._touch(tmp_path / "00001-0.png")
        self._touch(tmp_path / "00001-1.png")

        locator = PhysioNetRecordLocator(root_dir=tmp_path)
        paths = locator.get_paths("00001")

        assert paths.record_id == "00001"
        assert paths.header_path == tmp_path / "00001.hea"
        assert paths.signal_path == tmp_path / "00001.dat"
        assert paths.has_ground_truth is True
        assert paths.image_paths == [tmp_path / "00001-0.png", tmp_path / "00001-1.png"]

    def test_get_paths_image_only_record_has_no_ground_truth(self, tmp_path: Path) -> None:
        self._touch(tmp_path / "00002-0.png")
        locator = PhysioNetRecordLocator(root_dir=tmp_path)
        paths = locator.get_paths("00002")
        assert paths.has_ground_truth is False
        assert paths.header_path is None
        assert paths.signal_path is None
        assert len(paths.image_paths) == 1

    def test_get_paths_header_without_dat_has_no_ground_truth(self, tmp_path: Path) -> None:
        # A .hea with no matching .dat should not be reported as having
        # ground truth — has_ground_truth requires both files.
        self._touch(tmp_path / "00003.hea")
        self._touch(tmp_path / "00003-0.png")
        locator = PhysioNetRecordLocator(root_dir=tmp_path)
        paths = locator.get_paths("00003")
        assert paths.has_ground_truth is False

    def test_get_paths_unknown_record_raises_not_found(self, tmp_path: Path) -> None:
        locator = PhysioNetRecordLocator(root_dir=tmp_path)
        with pytest.raises(RecordNotFoundError):
            locator.get_paths("nonexistent")

    def test_custom_image_extensions(self, tmp_path: Path) -> None:
        self._touch(tmp_path / "00004-0.bmp")
        locator = PhysioNetRecordLocator(root_dir=tmp_path, image_extensions=[".bmp"])
        assert locator.list_record_ids() == ["00004"]


# ============================================================================
# OpenCVImageLoader (real filesystem)
# ============================================================================


class TestOpenCVImageLoader:
    def test_loads_valid_image_as_rgb(self, tmp_path: Path) -> None:
        image_path = tmp_path / "sample.png"
        # Write a BGR image where channels are clearly distinguishable.
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[..., 0] = 10  # B
        bgr[..., 1] = 20  # G
        bgr[..., 2] = 30  # R
        cv2.imwrite(str(image_path), bgr)

        loader = OpenCVImageLoader()
        rgb = loader.load(image_path)

        assert rgb.shape == (4, 4, 3)
        assert rgb.dtype == np.uint8
        # After BGR->RGB conversion, channel order must be flipped.
        assert np.all(rgb[..., 0] == 30)  # R
        assert np.all(rgb[..., 1] == 20)  # G
        assert np.all(rgb[..., 2] == 10)  # B

    def test_missing_file_raises_image_load_error(self, tmp_path: Path) -> None:
        loader = OpenCVImageLoader()
        with pytest.raises(ImageLoadError, match="does not exist"):
            loader.load(tmp_path / "missing.png")

    def test_corrupt_file_raises_image_load_error(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "corrupt.png"
        bad_path.write_bytes(b"not a real png file, just garbage bytes")

        loader = OpenCVImageLoader()
        with pytest.raises(ImageLoadError, match="failed to decode"):
            loader.load(bad_path)
