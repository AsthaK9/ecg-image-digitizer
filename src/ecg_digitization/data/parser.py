"""WFDB-based parsing of ECG ground-truth records.

Wraps the third-party ``wfdb`` library behind the :class:`~ecg_digitization.data.types.RecordParser`
protocol, translating its exceptions and quirks into the data layer's own
exception hierarchy with structured logging, so callers never need to know
``wfdb`` is involved at all.

New ECG dataset sources that ship ground truth in a different format (not
WFDB) add support by implementing :class:`~ecg_digitization.data.types.RecordParser`
themselves and registering it with :func:`register_parser` — no changes to
this module, :mod:`dataset`, or :mod:`datamodule` are required.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import wfdb

from ecg_digitization.data.types import (
    ECGRecordMetadata,
    ECGRecordPaths,
    ECGSignal,
    RecordParser,
    RecordParsingError,
)

logger = logging.getLogger(__name__)


class WFDBRecordParser:
    """Parses WFDB (``.hea`` / ``.dat``) records into calibrated :class:`ECGSignal` objects.

    Implements the :class:`~ecg_digitization.data.types.RecordParser` protocol.
    Stateless and thread-safe: every call to :meth:`parse` is independent and
    touches only the files referenced by the given
    :class:`~ecg_digitization.data.types.ECGRecordPaths`, so a single
    instance can safely be shared across ``DataLoader`` worker processes.
    """

    def parse(self, paths: ECGRecordPaths, source_dataset: str) -> ECGSignal:
        """Parse a single record's WFDB files into a validated :class:`ECGSignal`.

        Args:
            paths: Resolved file locations for the record. Both
                ``header_path`` and ``signal_path`` must be set — check
                ``paths.has_ground_truth`` before calling if that is not
                already guaranteed by the caller.
            source_dataset: Name of the originating dataset source, stored
                in the resulting metadata for provenance/debugging.

        Returns:
            A fully populated, shape-validated :class:`ECGSignal`.

        Raises:
            RecordParsingError: if the files are missing, unreadable, or the
                parsed record fails basic sanity checks (missing physical
                signal, zero leads, non-finite values, empty signal).
        """
        if not paths.has_ground_truth:
            raise RecordParsingError(
                f"Record '{paths.record_id}' has no header/signal path to "
                f"parse (header_path={paths.header_path}, "
                f"signal_path={paths.signal_path})."
            )

        record_base = self._record_base_path(paths)
        logger.debug("Parsing WFDB record '%s' from '%s'", paths.record_id, record_base)

        record = self._read_wfdb_record(record_base, paths.record_id)

        values = self._extract_physical_signal(record, paths.record_id)
        lead_names = self._extract_lead_names(record, paths.record_id)
        units = self._extract_units(record)

        metadata = ECGRecordMetadata(
            record_id=paths.record_id,
            source_dataset=source_dataset,
            sampling_rate_hz=float(record.fs),
            num_leads=values.shape[0],
            lead_names=lead_names,
            signal_length=values.shape[1],
            units=units,
            extra={"comments": list(getattr(record, "comments", []) or [])},
        )

        try:
            signal = ECGSignal(values=values, metadata=metadata)
        except ValueError as exc:
            raise RecordParsingError(str(exc)) from exc

        self._validate_signal(signal)

        logger.debug(
            "Parsed record '%s': %d leads x %d samples @ %.1f Hz",
            paths.record_id,
            signal.values.shape[0],
            signal.values.shape[1],
            metadata.sampling_rate_hz,
        )
        return signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_base_path(paths: ECGRecordPaths) -> Path:
        assert paths.header_path is not None  # narrowed by has_ground_truth check above
        return paths.header_path.with_suffix("")

    @staticmethod
    def _read_wfdb_record(record_base: Path, record_id: str) -> wfdb.Record:
        try:
            return wfdb.rdrecord(str(record_base))
        except FileNotFoundError as exc:
            raise RecordParsingError(
                f"WFDB files for record '{record_id}' not found at '{record_base}'."
            ) from exc
        except Exception as exc:  # wfdb raises plain Exception/ValueError on malformed headers
            raise RecordParsingError(
                f"Failed to parse WFDB record '{record_id}' at '{record_base}': {exc}"
            ) from exc

    @staticmethod
    def _extract_physical_signal(record: wfdb.Record, record_id: str) -> np.ndarray:
        p_signal = getattr(record, "p_signal", None)
        if p_signal is None:
            raise RecordParsingError(
                f"WFDB record '{record_id}' has no physical signal "
                f"('p_signal'); its header may declare a digital-only "
                f"format this parser does not support."
            )
        # wfdb returns (samples, leads); this data layer's convention is
        # (leads, samples) throughout — transpose once, here, at the source,
        # so every downstream consumer can rely on a single, documented axis
        # order.
        values = np.asarray(p_signal, dtype=np.float32).T
        if values.size == 0:
            raise RecordParsingError(f"WFDB record '{record_id}' has an empty signal.")
        return values

    @staticmethod
    def _extract_lead_names(record: wfdb.Record, record_id: str) -> list[str]:
        lead_names = list(getattr(record, "sig_name", []) or [])
        if not lead_names:
            raise RecordParsingError(
                f"WFDB record '{record_id}' does not declare any lead/signal names."
            )
        return lead_names

    @staticmethod
    def _extract_units(record: wfdb.Record) -> str:
        units = getattr(record, "units", None)
        if units:
            # WFDB stores one unit string per lead; every standard ECG
            # record is homogeneous across leads, so the first is authoritative.
            return str(units[0])
        return "mV"

    @staticmethod
    def _validate_signal(signal: ECGSignal) -> None:
        if not np.isfinite(signal.values).all():
            raise RecordParsingError(
                f"Record '{signal.metadata.record_id}' contains non-finite "
                f"values (NaN/Inf) in its parsed signal."
            )


# ============================================================================
# Registry
#
# Maps a `data.source` config value to the RecordParser implementation that
# understands that dataset's ground-truth format. New datasets register
# here (or via `register_parser` from anywhere, e.g. a plugin module) rather
# than requiring changes to `datamodule.py`.
# ============================================================================

PARSER_REGISTRY: dict[str, Callable[[], RecordParser]] = {}


def register_parser(name: str, parser_factory: Callable[[], RecordParser]) -> None:
    """Register a :class:`~ecg_digitization.data.types.RecordParser` factory.

    Args:
        name: The ``data.source`` config value this parser should handle
            (e.g. ``"physionet"``).
        parser_factory: A callable constructible as ``parser_factory()`` and
            returning a ``RecordParser`` — typically the parser class itself
            (its constructor), e.g. ``WFDBRecordParser``.

    Note:
        Registering the same ``name`` twice overwrites the previous
        registration; a warning is logged so accidental shadowing is visible
        rather than silent.
    """
    existing = PARSER_REGISTRY.get(name)
    if existing is not None and existing is not parser_factory:
        logger.warning(
            "Overwriting parser registration for source '%s': %s -> %s",
            name,
            getattr(existing, "__name__", existing),
            getattr(parser_factory, "__name__", parser_factory),
        )
    PARSER_REGISTRY[name] = parser_factory
    logger.debug(
        "Registered RecordParser '%s' for source '%s'",
        getattr(parser_factory, "__name__", parser_factory),
        name,
    )


register_parser("physionet", WFDBRecordParser)
