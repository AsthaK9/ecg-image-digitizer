"""Filesystem discovery and image loading for the ECG data layer.

Owns all direct interaction with the filesystem: enumerating which records
exist, resolving each record's associated files, and decoding image bytes
into pixel arrays. Parsing WFDB *content* is deliberately out of scope here
— see :mod:`ecg_digitization.data.parser` — this module only knows about
paths and pixels.

New ECG dataset sources with a different on-disk naming convention add
support by implementing :class:`~ecg_digitization.data.types.RecordLocator`
and registering it with :func:`register_locator` — no changes to
:mod:`dataset` or :mod:`datamodule` are required.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from ecg_digitization.data.types import (
    ECGRecordPaths,
    ImageLoadError,
    RecordLocator,
    RecordNotFoundError,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")

# Matches "<record_id>-<index>" so a record's images (e.g. "00001-0.png",
# "00001-1.png") can be grouped back to their shared record_id.
_IMAGE_INDEX_SUFFIX_RE = re.compile(r"^(?P<record_id>.+)-(?P<index>\d+)$")


class PhysioNetRecordLocator:
    """Locates records laid out using the PhysioNet/Kaggle ECG-digitization
    convention::

        <root>/<record_id>.hea
        <root>/<record_id>.dat
        <root>/<record_id>-0.<ext>   (one or more images per record)

    Records with no ``.hea`` / ``.dat`` (e.g. an image-only held-out test
    split) are still discoverable purely from their image filenames.

    Implements the :class:`~ecg_digitization.data.types.RecordLocator` protocol.

    Args:
        root_dir: Directory to scan for record files. Does not need to
            exist at construction time (a missing directory simply yields
            zero records) — this keeps datamodule setup robust to
            not-yet-downloaded data during config validation.
        image_extensions: Case-insensitive file extensions considered valid
            ECG images.
    """

    def __init__(
        self,
        root_dir: Path,
        image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._image_extensions: tuple[str, ...] = tuple(ext.lower() for ext in image_extensions)
        if not self._root_dir.exists():
            logger.warning(
                "PhysioNetRecordLocator root directory does not exist yet: %s",
                self._root_dir,
            )

    @property
    def root_dir(self) -> Path:
        """The directory this locator scans."""
        return self._root_dir

    def list_record_ids(self) -> list[str]:
        """Return sorted, de-duplicated record IDs discoverable under ``root_dir``.

        A record is considered to exist if it has a ``.hea`` file OR at
        least one matching image file — this lets image-only test splits
        share the same locator implementation used for fully-labeled
        train/val splits.
        """
        if not self._root_dir.exists():
            return []

        record_ids: set[str] = set()

        for hea_path in self._root_dir.glob("*.hea"):
            record_ids.add(hea_path.stem)

        for ext in self._image_extensions:
            for image_path in self._root_dir.glob(f"*{ext}"):
                record_ids.add(self._record_id_from_image_stem(image_path.stem))

        return sorted(record_ids)

    def get_paths(self, record_id: str) -> ECGRecordPaths:
        """Resolve all files belonging to ``record_id``.

        Raises:
            RecordNotFoundError: if neither a header file nor any image
                file exists for ``record_id``.
        """
        header_path = self._root_dir / f"{record_id}.hea"
        signal_path = self._root_dir / f"{record_id}.dat"
        image_paths = self._discover_image_paths(record_id)

        resolved_header = header_path if header_path.exists() else None
        resolved_signal = (
            signal_path if (resolved_header is not None and signal_path.exists()) else None
        )

        if resolved_header is None and not image_paths:
            raise RecordNotFoundError(
                f"No header or image files found for record_id='{record_id}' "
                f"under '{self._root_dir}'."
            )

        return ECGRecordPaths(
            record_id=record_id,
            header_path=resolved_header,
            signal_path=resolved_signal,
            image_paths=image_paths,
        )

    def _discover_image_paths(self, record_id: str) -> list[Path]:
        matches: set[Path] = set()
        for ext in self._image_extensions:
            # Indexed images: "<record_id>-0.png", "<record_id>-1.png", ...
            matches.update(self._root_dir.glob(f"{record_id}-*{ext}"))
            # Unindexed single image: "<record_id>.png"
            single = self._root_dir / f"{record_id}{ext}"
            if single.exists():
                matches.add(single)
        return sorted(matches)

    @staticmethod
    def _record_id_from_image_stem(stem: str) -> str:
        match = _IMAGE_INDEX_SUFFIX_RE.match(stem)
        return match.group("record_id") if match else stem


class OpenCVImageLoader:
    """Loads ECG images from disk using OpenCV.

    Implements the :class:`~ecg_digitization.data.types.ImageLoader`
    protocol. Always returns images as RGB ``(H, W, C)`` ``uint8`` arrays
    regardless of the source file's on-disk channel order, so downstream
    code never has to reason about OpenCV's BGR convention.
    """

    def load(self, path: Path) -> np.ndarray:
        """Decode the image at ``path``.

        Raises:
            ImageLoadError: if the file does not exist or cannot be decoded
                (corrupt, empty, or an unsupported format).
        """
        if not path.exists():
            raise ImageLoadError(f"Image file does not exist: '{path}'.")

        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ImageLoadError(
                f"OpenCV failed to decode image '{path}' — the file may be "
                f"corrupt, empty, or an unsupported format."
            )

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# ============================================================================
# Registry
#
# Maps a `data.source` config value to the RecordLocator implementation that
# understands that dataset's on-disk layout. New datasets register here (or
# via `register_locator` from anywhere, e.g. a plugin module) rather than
# requiring changes to `datamodule.py`.
# ============================================================================

LOCATOR_REGISTRY: dict[str, Callable[[Path], RecordLocator]] = {}


def register_locator(name: str, locator_factory: Callable[[Path], RecordLocator]) -> None:
    """Register a :class:`~ecg_digitization.data.types.RecordLocator` factory.

    Args:
        name: The ``data.source`` config value this locator should handle
            (e.g. ``"physionet"``).
        locator_factory: A callable constructible as ``locator_factory(root_dir)``
            and returning a ``RecordLocator`` — typically the locator class
            itself (its constructor), e.g. ``PhysioNetRecordLocator``.

    Note:
        Registering the same ``name`` twice overwrites the previous
        registration; a warning is logged so accidental shadowing is
        visible rather than silent.
    """
    existing = LOCATOR_REGISTRY.get(name)
    if existing is not None and existing is not locator_factory:
        logger.warning(
            "Overwriting locator registration for source '%s': %s -> %s",
            name,
            getattr(existing, "__name__", existing),
            getattr(locator_factory, "__name__", locator_factory),
        )
    LOCATOR_REGISTRY[name] = locator_factory
    logger.debug(
        "Registered RecordLocator '%s' for source '%s'",
        getattr(locator_factory, "__name__", locator_factory),
        name,
    )


register_locator("physionet", PhysioNetRecordLocator)
