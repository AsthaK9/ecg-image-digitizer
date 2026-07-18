"""PyTorch ``Dataset`` for lazily-loaded ECG image/signal pairs.

:class:`ECGImageDataset` is deliberately decoupled from any specific ECG
dataset's on-disk layout or ground-truth format: it depends only on the
:class:`~ecg_digitization.data.types.RecordLocator`,
:class:`~ecg_digitization.data.types.RecordParser`, and
:class:`~ecg_digitization.data.types.ImageLoader` protocols from
:mod:`ecg_digitization.data.types`, supplied via constructor injection. This
means the dataset class itself never needs to change to support a new ECG
data source — only a new locator/parser implementation is required (see
:mod:`ecg_digitization.data.io`, :mod:`ecg_digitization.data.parser`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_digitization.data.types import (
    ECGDataError,
    ECGRecordMetadata,
    ECGRecordPaths,
    ImageLoader,
    RecordLocator,
    RecordNotFoundError,
    RecordParser,
    RecordParsingError,
)

logger = logging.getLogger(__name__)

ImageTransform = Callable[[np.ndarray], np.ndarray]


class ECGImageDataset(Dataset):
    """Lazily-loaded PyTorch ``Dataset`` over ECG image/signal record pairs.

    Nothing is read from disk until :meth:`__getitem__` is called for a
    given index — construction only stores the (cheap) list of record IDs
    and the injected collaborators, so datasets over very large record
    collections can be created instantly and are safe to fork across
    ``DataLoader`` worker processes.

    Each item returned by :meth:`__getitem__` is a dict::

        {
            "record_id": str,
            "image": FloatTensor [C, H, W] in [0, 1],
            "signal": FloatTensor [num_leads, signal_length] or None,
            "sampling_rate_hz": float or None,
            "lead_names": list[str] or None,
        }

    Use :func:`collate_ecg_batch` as a DataLoader's ``collate_fn`` — the
    default PyTorch collate cannot batch samples whose ``signal`` may be
    ``None`` or of varying length.

    Args:
        record_ids: Ordered record IDs this dataset instance exposes. Index
            ``i`` of the dataset always maps to ``record_ids[i]``.
        locator: Resolves file paths for a given ``record_id``.
        parser: Parses the ground-truth signal for a record.
        image_loader: Decodes an ECG image file into a pixel array.
        target_image_size: Optional ``(height, width)`` every image is
            resized to, so images can be batched without a custom resize
            step downstream. If ``None``, images are returned at native
            resolution — the caller must then ensure a consistent size
            (e.g. via ``transform``) or use ``batch_size=1``.
        transform: Optional callable applied to the raw ``(H, W, C)``
            ``uint8`` image array after resizing (e.g. an Albumentations
            pipeline). Must return an ``(H, W, C)`` array. Kept as a generic
            ``Callable`` rather than a concrete Albumentations type so this
            module has no hard dependency on any specific augmentation
            library.
        require_signal: If ``True`` (default), a record with no
            ground-truth signal raises :class:`RecordParsingError` when
            accessed. Set to ``False`` for image-only inference/test
            splits, in which case ``sample["signal"]`` will be ``None``.
        source_dataset_name: Recorded into each sample's metadata for
            provenance; has no effect on loading behavior.

    Raises:
        ValueError: if ``record_ids`` is empty.
    """

    def __init__(
        self,
        record_ids: Sequence[str],
        locator: RecordLocator,
        parser: RecordParser,
        image_loader: ImageLoader,
        target_image_size: tuple[int, int] | None = None,
        transform: ImageTransform | None = None,
        require_signal: bool = True,
        source_dataset_name: str = "unknown",
    ) -> None:
        if len(record_ids) == 0:
            raise ValueError("ECGImageDataset requires at least one record_id.")

        self._record_ids: list[str] = list(record_ids)
        self._locator = locator
        self._parser = parser
        self._image_loader = image_loader
        self._target_image_size = target_image_size
        self._transform = transform
        self._require_signal = require_signal
        self._source_dataset_name = source_dataset_name

        logger.info(
            "Initialized ECGImageDataset: %d records (source=%s, require_signal=%s)",
            len(self._record_ids),
            source_dataset_name,
            require_signal,
        )

    def __len__(self) -> int:
        return len(self._record_ids)

    @property
    def record_ids(self) -> list[str]:
        """Read-only view of the record IDs this dataset exposes."""
        return list(self._record_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Lazily load and return the sample at ``index``.

        Raises:
            IndexError: if ``index`` is out of range.
            RecordNotFoundError: if the record's files cannot be located.
            RecordParsingError: if the image or ground-truth signal cannot
                be loaded/parsed, or ground truth is required but absent.
        """
        if not 0 <= index < len(self._record_ids):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self._record_ids)}."
            )

        record_id = self._record_ids[index]

        try:
            paths = self._locator.get_paths(record_id)
            image = self._load_image(paths)
            signal, metadata = self._load_signal(paths)
        except ECGDataError:
            raise
        except Exception as exc:  # unexpected error: wrap with context, never swallow silently
            raise RecordParsingError(
                f"Unexpected error loading record '{record_id}' at index {index}: {exc}"
            ) from exc

        return self._to_tensors(record_id, image, signal, metadata)

    def validate(self) -> list[tuple[str, str]]:
        """Eagerly attempt to load every record without raising.

        Useful as an offline data-quality check (e.g. in CI, or before a
        long training run) to catch corrupt/missing files early rather than
        mid-epoch.

        Returns:
            A list of ``(record_id, error_message)`` tuples for every
            record that failed to load. Empty if every record loaded
            successfully.
        """
        failures: list[tuple[str, str]] = []
        for index, record_id in enumerate(self._record_ids):
            try:
                self[index]
            except ECGDataError as exc:
                failures.append((record_id, str(exc)))
                logger.warning("Validation failed for record '%s': %s", record_id, exc)
        return failures

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, paths: ECGRecordPaths) -> np.ndarray:
        if not paths.image_paths:
            raise RecordNotFoundError(f"Record '{paths.record_id}' has no associated image files.")
        image = self._image_loader.load(paths.image_paths[0])

        if self._target_image_size is not None:
            height, width = self._target_image_size
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        if self._transform is not None:
            image = self._transform(image)

        return image

    def _load_signal(
        self, paths: ECGRecordPaths
    ) -> tuple[np.ndarray | None, ECGRecordMetadata | None]:
        if not paths.has_ground_truth:
            if self._require_signal:
                raise RecordParsingError(
                    f"Record '{paths.record_id}' has no ground-truth signal "
                    f"available, but this dataset was built with require_signal=True."
                )
            return None, None

        ecg_signal = self._parser.parse(paths, source_dataset=self._source_dataset_name)
        return ecg_signal.values, ecg_signal.metadata

    @staticmethod
    def _to_tensors(
        record_id: str,
        image: np.ndarray,
        signal: np.ndarray | None,
        metadata: ECGRecordMetadata | None,
    ) -> dict[str, Any]:
        image_chw = np.ascontiguousarray(image.transpose(2, 0, 1))
        image_tensor = torch.from_numpy(image_chw).float() / 255.0

        signal_tensor = (
            torch.from_numpy(np.ascontiguousarray(signal)).float() if signal is not None else None
        )

        return {
            "record_id": record_id,
            "image": image_tensor,
            "signal": signal_tensor,
            "sampling_rate_hz": metadata.sampling_rate_hz if metadata is not None else None,
            "lead_names": metadata.lead_names if metadata is not None else None,
        }


def collate_ecg_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """``collate_fn`` tolerant of variable-length signals and missing ground truth.

    ``torch.utils.data.default_collate`` cannot batch this dataset's output
    whenever signal lengths differ across a batch, or when some samples have
    ``signal=None`` (image-only records). This function instead:

    * Stacks images directly — they are assumed to be a fixed size (see
      ``ECGImageDataset(target_image_size=...)``).
    * If every sample in the batch has a signal, zero-pads them to the
      batch's longest signal length along the time axis and returns an
      additional ``signal_length`` tensor so padding can be masked out
      downstream.
    * If no sample in the batch has a signal, omits the ``signal`` and
      ``signal_length`` keys entirely.
    * Raises ``ValueError`` if the batch mixes samples with and without a
      signal — silently dropping ground truth for part of a batch is a
      likely-latent bug, not a case to paper over.

    Args:
        batch: A list of per-sample dicts as produced by
            ``ECGImageDataset.__getitem__``.

    Returns:
        A batched dict with ``"record_id"`` (list[str]), ``"image"``
        (stacked FloatTensor), and, when applicable, ``"signal"`` (padded
        FloatTensor) and ``"signal_length"`` (LongTensor).

    Raises:
        ValueError: if the batch is empty, or mixes present/absent signals.
    """
    if not batch:
        raise ValueError("collate_ecg_batch received an empty batch.")

    record_ids = [item["record_id"] for item in batch]
    images = torch.stack([item["image"] for item in batch], dim=0)

    signals_present = [item["signal"] is not None for item in batch]

    output: dict[str, Any] = {
        "record_id": record_ids,
        "image": images,
        "sampling_rate_hz": [item["sampling_rate_hz"] for item in batch],
        "lead_names": [item["lead_names"] for item in batch],
    }

    if all(signals_present):
        signal_batch, length_tensor = _pad_signals([item["signal"] for item in batch])
        output["signal"] = signal_batch
        output["signal_length"] = length_tensor
    elif any(signals_present):
        raise ValueError(
            "Batch mixes records with and without ground-truth signals; "
            "collate_ecg_batch does not support this. Split image-only and "
            "labeled records into separate datasets/dataloaders."
        )

    return output


def _pad_signals(signals: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad a list of ``(num_leads, length)`` tensors to a common length."""
    lengths = [signal.shape[-1] for signal in signals]
    max_length = max(lengths)
    num_leads = signals[0].shape[0]

    for signal, _length in zip(signals, lengths, strict=True):
        if signal.shape[0] != num_leads:
            raise ValueError(
                f"Cannot batch signals with differing lead counts: expected "
                f"{num_leads}, got {signal.shape[0]}."
            )

    padded = torch.zeros(len(signals), num_leads, max_length, dtype=torch.float32)
    for i, (signal, length) in enumerate(zip(signals, lengths, strict=True)):
        padded[i, :, :length] = signal

    return padded, torch.tensor(lengths, dtype=torch.long)
