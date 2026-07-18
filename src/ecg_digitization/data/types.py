"""Core data types, protocols, and exceptions for the ECG data layer.

This module is the single shared foundation that every other module in
``ecg_digitization.data`` depends on. It intentionally contains no I/O, no
framework-specific imports (no ``torch``, no ``wfdb``, no ``cv2``), and no
concrete implementations — only:

* Plain data containers (frozen dataclasses) describing an ECG record, its
  metadata, and a dataset sample.
* Structural-typing ``Protocol`` s that define the *interfaces* concrete
  components (record locators, parsers, image loaders) must satisfy.
* A shared exception hierarchy used consistently across the data layer.

Depending only on this module — rather than on concrete classes in
``parser.py`` / ``io.py`` — is what lets ``dataset.py`` and ``datamodule.py``
stay agnostic to which ECG dataset backend is in use. Support for a new ECG
dataset is added by implementing these protocols and registering the
implementation (see ``io.register_locator`` / ``parser.register_parser``),
never by modifying the consumers of them. This is the Dependency Inversion
and Open/Closed principles applied directly to the data layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "DatasetSplit",
    "ECGDataError",
    "RecordNotFoundError",
    "RecordParsingError",
    "ImageLoadError",
    "EmptyDatasetError",
    "UnsupportedDatasetSourceError",
    "ECGRecordPaths",
    "ECGRecordMetadata",
    "ECGSignal",
    "ECGSample",
    "RecordLocator",
    "RecordParser",
    "ImageLoader",
]


# ============================================================================
# Enums
# ============================================================================


class DatasetSplit(StrEnum):
    """Canonical split names used throughout the data layer and configs.

    Inherits from ``str`` so instances compare equal to plain strings
    (``DatasetSplit.TRAIN == "train"``), which keeps interop with Hydra
    configs (which store split names as plain strings) friction-free.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ============================================================================
# Exceptions
# ============================================================================


class ECGDataError(Exception):
    """Base class for all ECG data-layer errors.

    Catching this (rather than bare ``Exception``) anywhere in the codebase
    signals "a data-layer problem, not a bug" — useful both for logging
    policy and for deciding whether an error is safe to skip/retry.
    """


class RecordNotFoundError(ECGDataError):
    """Raised when a requested ``record_id`` has no discoverable files."""


class RecordParsingError(ECGDataError):
    """Raised when a record's header/signal files exist but cannot be
    parsed into a valid :class:`ECGSignal`.
    """


class ImageLoadError(ECGDataError):
    """Raised when an ECG image file cannot be loaded or decoded."""


class EmptyDatasetError(ECGDataError):
    """Raised when a configured dataset split resolves to zero records."""


class UnsupportedDatasetSourceError(ECGDataError):
    """Raised when a ``data.source`` name has no registered locator/parser."""


# ============================================================================
# Data containers
# ============================================================================


@dataclass(frozen=True)
class ECGRecordPaths:
    """Filesystem locations of the files that make up a single ECG record.

    Attributes:
        record_id: Stable identifier shared by every file belonging to this
            record.
        header_path: Path to the WFDB ``.hea`` header, or ``None`` if this
            record has no available ground-truth signal (e.g. an image-only
            held-out test split).
        signal_path: Path to the WFDB ``.dat`` signal file, or ``None``
            under the same conditions as ``header_path``.
        image_paths: Paths to the record's rendered/scanned ECG image(s).
            A locator should never return an :class:`ECGRecordPaths` with
            an empty ``image_paths`` list for a record it considers to
            exist — an empty list is treated as "not found" by consumers.
    """

    record_id: str
    header_path: Path | None
    signal_path: Path | None
    image_paths: list[Path] = field(default_factory=list)

    @property
    def has_ground_truth(self) -> bool:
        """Whether both a header and a signal file are available."""
        return self.header_path is not None and self.signal_path is not None


@dataclass(frozen=True)
class ECGRecordMetadata:
    """Metadata describing a parsed ECG record, independent of pixel data.

    Attributes:
        record_id: Identifier matching the originating :class:`ECGRecordPaths`.
        source_dataset: Name of the dataset source this record came from
            (e.g. ``"physionet"``), for provenance and debugging.
        sampling_rate_hz: Sampling frequency of the ground-truth signal.
        num_leads: Number of leads/channels in the signal.
        lead_names: Ordered lead names (e.g. ``["I", "II", ..., "V6"]``),
            matching the first axis of the corresponding signal array.
        signal_length: Number of samples per lead.
        units: Physical unit of the signal values (conventionally ``"mV"``).
        extra: Free-form bag for source-specific metadata (header comments,
            demographics, diagnosis codes, ...) that does not need a typed
            field. New datasets can populate this without changing the
            dataclass shape.
    """

    record_id: str
    source_dataset: str
    sampling_rate_hz: float
    num_leads: int
    lead_names: list[str]
    signal_length: int
    units: str = "mV"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ECGSignal:
    """A parsed, physically-calibrated multi-lead ECG signal.

    Attributes:
        values: Array of shape ``(num_leads, signal_length)`` expressed in
            the physical unit declared by ``metadata.units``.
        metadata: Descriptive metadata consistent with ``values``' shape.

    Raises:
        ValueError: if ``values`` is not 2-D, or its lead count does not
            match ``metadata.num_leads``. Validation happens at
            construction time so an invalid ``ECGSignal`` can never exist —
            callers do not need to re-check its shape.
    """

    values: np.ndarray
    metadata: ECGRecordMetadata

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                f"ECGSignal.values must be 2D (num_leads, signal_length); "
                f"got shape {self.values.shape} for record "
                f"'{self.metadata.record_id}'."
            )
        if self.values.shape[0] != self.metadata.num_leads:
            raise ValueError(
                f"ECGSignal lead-count mismatch for record "
                f"'{self.metadata.record_id}': values has "
                f"{self.values.shape[0]} leads but metadata declares "
                f"{self.metadata.num_leads}."
            )


@dataclass(frozen=True)
class ECGSample:
    """A single conceptual item produced by the data layer.

    This mirrors the dict returned by ``ECGImageDataset.__getitem__`` but as
    a typed, immutable object — useful in tests and in any non-PyTorch code
    path that wants a strongly-typed sample rather than a loosely-typed
    dict of tensors.

    Attributes:
        record_id: Identifier of the underlying record.
        image: Pixel array, shape ``(H, W, C)``, dtype ``uint8`` or as
            produced by an applied transform.
        signal: Ground-truth signal values, or ``None`` for image-only
            records.
        metadata: Metadata paired with ``signal``, or ``None`` if ``signal``
            is ``None``.
    """

    record_id: str
    image: np.ndarray
    signal: np.ndarray | None
    metadata: ECGRecordMetadata | None


# ============================================================================
# Protocols (structural interfaces)
# ============================================================================


@runtime_checkable
class RecordLocator(Protocol):
    """Discovers which records exist and where their files live on disk.

    Implementations own dataset-specific naming/layout conventions (for
    example, the PhysioNet/Kaggle ``<record_id>.hea`` / ``<record_id>-N.png``
    convention implemented by :class:`ecg_digitization.data.io.PhysioNetRecordLocator`).
    Supporting a new ECG dataset means implementing this protocol, not
    modifying :class:`ecg_digitization.data.dataset.ECGImageDataset` or
    :class:`ecg_digitization.data.datamodule.ECGDataModule`.
    """

    def list_record_ids(self) -> list[str]:
        """Return every record ID discoverable by this locator, sorted."""
        ...

    def get_paths(self, record_id: str) -> ECGRecordPaths:
        """Resolve the files belonging to ``record_id``.

        Raises:
            RecordNotFoundError: if no files exist for ``record_id``.
        """
        ...


@runtime_checkable
class RecordParser(Protocol):
    """Parses a record's header/signal files into a calibrated :class:`ECGSignal`."""

    def parse(self, paths: ECGRecordPaths, source_dataset: str) -> ECGSignal:
        """Parse the ground-truth signal referenced by ``paths``.

        Args:
            paths: Must satisfy ``paths.has_ground_truth``.
            source_dataset: Name recorded into the resulting metadata.

        Raises:
            RecordParsingError: if the files cannot be parsed or fail
                validation.
        """
        ...


@runtime_checkable
class ImageLoader(Protocol):
    """Loads a single ECG image file into an ``(H, W, C)`` ``uint8`` array."""

    def load(self, path: Path) -> np.ndarray:
        """Decode the image at ``path``.

        Raises:
            ImageLoadError: if the file is missing or cannot be decoded.
        """
        ...
