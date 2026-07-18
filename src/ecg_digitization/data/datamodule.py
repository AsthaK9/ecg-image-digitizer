"""Hydra-config-driven orchestration of ECG datasets and dataloaders.

:class:`ECGDataModule` is the single integration point between the
project's Hydra ``data`` config group (``configs/data/base.yaml``) and the
dataset primitives in this package. It:

1. Resolves which :class:`~ecg_digitization.data.types.RecordLocator` /
   :class:`~ecg_digitization.data.types.RecordParser` implementation to use
   for the configured ``data.source`` (defaulting to ``"physionet"`` for
   backward compatibility with the existing config, which has no
   ``source`` key), via the registries in :mod:`ecg_digitization.data.io`
   and :mod:`ecg_digitization.data.parser`.
2. Discovers train/val/test record IDs — either from pre-split directories
   (used directly if ``data.train_dir`` / ``data.val_dir`` / ``data.test_dir``
   already contain records) or by splitting a single pool of records by
   ``record_id`` according to ``data.split.*`` when no pre-split layout is
   found. Splitting is always done by ``record_id``, never by image, so
   multiple rendered images of the same ground-truth signal cannot leak
   across splits.
3. Builds :class:`~ecg_digitization.data.dataset.ECGImageDataset` instances
   and ``DataLoader`` s from ``data.dataloader.*``.

Adding a new ECG dataset source requires implementing and registering a
``RecordLocator`` / ``RecordParser`` pair (see
:func:`ecg_digitization.data.io.register_locator` and
:func:`ecg_digitization.data.parser.register_parser`) — this module does not
need to change, satisfying the project's Open/Closed extensibility
requirement for the data layer.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from pathlib import Path

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ecg_digitization.data.dataset import ECGImageDataset, collate_ecg_batch
from ecg_digitization.data.io import LOCATOR_REGISTRY, OpenCVImageLoader
from ecg_digitization.data.parser import PARSER_REGISTRY
from ecg_digitization.data.types import (
    DatasetSplit,
    EmptyDatasetError,
    RecordLocator,
    RecordParser,
    UnsupportedDatasetSourceError,
)

logger = logging.getLogger(__name__)

_DEFAULT_SOURCE = "physionet"
_SPLITS: tuple[str, str, str] = (
    DatasetSplit.TRAIN.value,
    DatasetSplit.VAL.value,
    DatasetSplit.TEST.value,
)

# Maps split name -> whether ECGImageDataset should require ground truth.
_REQUIRE_SIGNAL_BY_SPLIT: dict[str, bool] = {
    DatasetSplit.TRAIN.value: True,
    DatasetSplit.VAL.value: True,
    DatasetSplit.TEST.value: False,
}


class ECGDataModule:
    """Builds train/val/test ``ECGImageDataset`` s and ``DataLoader`` s from a Hydra config.

    Args:
        cfg: The fully-resolved root Hydra config (i.e. ``cfg``, not
            ``cfg.data`` — this module reads both ``cfg.data`` and
            ``cfg.seed`` for split reproducibility, matching how
            ``configs/config.yaml`` composes the ``data`` group).

    Raises:
        UnsupportedDatasetSourceError: if ``cfg.data.source`` (or the
            default, ``"physionet"``) has no registered locator/parser.

    Example:
        >>> datamodule = ECGDataModule(cfg)
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._source_name: str = self._data_cfg.get("source", _DEFAULT_SOURCE)

        locator_factory = LOCATOR_REGISTRY.get(self._source_name)
        parser_factory = PARSER_REGISTRY.get(self._source_name)
        if locator_factory is None or parser_factory is None:
            raise UnsupportedDatasetSourceError(
                f"No locator/parser registered for data.source='{self._source_name}'. "
                f"Available locators: {sorted(LOCATOR_REGISTRY)}; "
                f"available parsers: {sorted(PARSER_REGISTRY)}."
            )
        self._locator_factory: Callable[[Path], RecordLocator] = locator_factory
        self._parser_factory: Callable[[], RecordParser] = parser_factory

        self._parser: RecordParser = self._parser_factory()
        self._image_loader = OpenCVImageLoader()
        self._datasets: dict[str, ECGImageDataset] = {}

        logger.info("ECGDataModule initialized with source='%s'", self._source_name)

    def setup(self) -> None:
        """Discover records and build the train/val/test datasets.

        Idempotent: calling this more than once rebuilds the datasets from
        the current config, discarding any previously built ones.

        Raises:
            EmptyDatasetError: if any split resolves to zero records.
        """
        split_record_ids = self._resolve_split_record_ids()

        for split, (record_ids, root_dir) in split_record_ids.items():
            if not record_ids:
                raise EmptyDatasetError(
                    f"Split '{split}' resolved to zero records under '{root_dir}'; "
                    f"check data.{split}_dir / data.split settings."
                )
            locator = self._locator_factory(root_dir)
            self._datasets[split] = ECGImageDataset(
                record_ids=record_ids,
                locator=locator,
                parser=self._parser,
                image_loader=self._image_loader,
                target_image_size=tuple(self._data_cfg.image.target_size),
                require_signal=_REQUIRE_SIGNAL_BY_SPLIT[split],
                source_dataset_name=self._source_name,
            )
            logger.info(
                "Built '%s' dataset: %d records (root=%s, source=%s)",
                split,
                len(record_ids),
                root_dir,
                self._source_name,
            )

    @property
    def datasets(self) -> dict[str, ECGImageDataset]:
        """The datasets built by :meth:`setup`, keyed by split name."""
        return dict(self._datasets)

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        """Build the training ``DataLoader`` (shuffled, per ``data.dataloader.shuffle_train``)."""
        return self._build_dataloader(
            DatasetSplit.TRAIN.value,
            shuffle=bool(self._data_cfg.dataloader.shuffle_train),
            drop_last=bool(self._data_cfg.dataloader.drop_last_train),
        )

    def val_dataloader(self) -> DataLoader:
        """Build the validation ``DataLoader`` (never shuffled)."""
        return self._build_dataloader(DatasetSplit.VAL.value, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Build the test ``DataLoader`` (never shuffled)."""
        return self._build_dataloader(DatasetSplit.TEST.value, shuffle=False, drop_last=False)

    def _build_dataloader(self, split: str, shuffle: bool, drop_last: bool) -> DataLoader:
        dataset = self._datasets.get(split)
        if dataset is None:
            raise RuntimeError(
                f"Dataset for split '{split}' has not been built yet; call "
                f"`ECGDataModule.setup()` before requesting a dataloader."
            )
        dl_cfg = self._data_cfg.dataloader
        num_workers = int(dl_cfg.num_workers)
        return DataLoader(
            dataset,
            batch_size=int(dl_cfg.batch_size),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(dl_cfg.pin_memory),
            persistent_workers=bool(dl_cfg.persistent_workers) and num_workers > 0,
            drop_last=drop_last,
            collate_fn=collate_ecg_batch,
        )

    # ------------------------------------------------------------------
    # Record discovery / splitting
    # ------------------------------------------------------------------

    def _resolve_split_record_ids(self) -> dict[str, tuple[list[str], Path]]:
        """Resolve each split's record IDs and the root directory they live under."""
        pre_split = self._discover_pre_split_records()
        if pre_split is not None:
            logger.info("Using pre-split train/val/test directories from config.")
            return pre_split

        logger.info(
            "No pre-split directories found with existing records; splitting a "
            "single record pool by record_id using data.split settings."
        )
        return self._split_single_pool()

    def _discover_pre_split_records(self) -> dict[str, tuple[list[str], Path]] | None:
        """Attempt to use ``data.{split}_dir`` directly, if each already has records.

        Returns ``None`` (triggering the single-pool fallback) unless *all
        three* split directories independently resolve to at least one
        record — a partially pre-split layout is treated as "not pre-split"
        rather than silently producing an empty split.
        """
        result: dict[str, tuple[list[str], Path]] = {}
        for split in _SPLITS:
            root_dir = Path(str(self._data_cfg[f"{split}_dir"]))
            locator = self._locator_factory(root_dir)
            record_ids = locator.list_record_ids()
            if not record_ids:
                return None
            result[split] = (record_ids, root_dir)
        return result

    def _split_single_pool(self) -> dict[str, tuple[list[str], Path]]:
        """Discover all records under ``data.raw_dir`` and split by ``record_id``."""
        root_dir = Path(str(self._data_cfg.get("raw_dir", self._data_cfg.root_dir)))
        locator = self._locator_factory(root_dir)
        all_record_ids = locator.list_record_ids()

        if not all_record_ids:
            raise EmptyDatasetError(
                f"No records discovered under '{root_dir}' for source '{self._source_name}'. "
                f"Populate data.raw_dir or pre-split data.{{train,val,test}}_dir."
            )

        split_cfg = self._data_cfg.split
        seed = int(split_cfg.get("random_seed", self._cfg.get("seed", 42)))
        rng = random.Random(seed)
        shuffled = list(all_record_ids)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * float(split_cfg.train_ratio)))
        n_val = int(round(n * float(split_cfg.val_ratio)))

        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train : n_train + n_val]
        test_ids = shuffled[n_train + n_val :]

        logger.info(
            "Split %d records by record_id (seed=%d): train=%d, val=%d, test=%d",
            n,
            seed,
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )

        return {
            DatasetSplit.TRAIN.value: (train_ids, root_dir),
            DatasetSplit.VAL.value: (val_ids, root_dir),
            DatasetSplit.TEST.value: (test_ids, root_dir),
        }
