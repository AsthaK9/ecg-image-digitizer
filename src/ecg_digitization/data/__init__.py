"""ECG data layer: dataset-agnostic record discovery, parsing, and loading.

Public API surface for this package. Internal modules
(:mod:`ecg_digitization.data.io`, :mod:`ecg_digitization.data.parser`)
expose additional registration functions (``register_locator``,
``register_parser``) used to add support for new ECG dataset sources; import
them directly from those modules rather than from here.
"""

from ecg_digitization.data.datamodule import ECGDataModule
from ecg_digitization.data.dataset import ECGImageDataset, collate_ecg_batch
from ecg_digitization.data.io import OpenCVImageLoader, PhysioNetRecordLocator
from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import (
    DatasetSplit,
    ECGDataError,
    ECGRecordMetadata,
    ECGRecordPaths,
    ECGSample,
    ECGSignal,
    EmptyDatasetError,
    ImageLoader,
    ImageLoadError,
    RecordLocator,
    RecordNotFoundError,
    RecordParser,
    RecordParsingError,
    UnsupportedDatasetSourceError,
)

__all__ = [
    "ECGDataModule",
    "ECGImageDataset",
    "collate_ecg_batch",
    "OpenCVImageLoader",
    "PhysioNetRecordLocator",
    "WFDBRecordParser",
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
