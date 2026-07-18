"""Unit tests for ecg_digitization.data.parser.

Uses real WFDB files written to disk via `wfdb.wrsamp` rather than mocks —
the parser's entire job is to correctly interface with the `wfdb` library,
so exercising the real library against real files is the only way to
actually verify that behavior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import wfdb

from ecg_digitization.data.parser import PARSER_REGISTRY, WFDBRecordParser, register_parser
from ecg_digitization.data.types import (
    ECGRecordPaths,
    RecordParser,
    RecordParsingError,
)

pytestmark = pytest.mark.unit

SAMPLING_RATE_HZ = 500
NUM_LEADS = 12
SIGNAL_LENGTH = 1000
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _write_synthetic_record(
    directory: Path,
    record_name: str = "00001",
    num_leads: int = NUM_LEADS,
    signal_length: int = SIGNAL_LENGTH,
    fs: int = SAMPLING_RATE_HZ,
    lead_names: list[str] | None = None,
) -> np.ndarray:
    """Write a real, valid WFDB record to `directory` and return its physical signal."""
    lead_names = lead_names if lead_names is not None else LEAD_NAMES[:num_leads]
    rng = np.random.default_rng(seed=42)
    t = np.linspace(0, signal_length / fs, signal_length, endpoint=False)
    # A simple deterministic multi-lead waveform, distinct per lead, with
    # small noise so it is not perfectly flat/degenerate.
    p_signal = np.stack(
        [
            0.5 * np.sin(2 * np.pi * (1.0 + 0.1 * i) * t)
            + 0.01 * rng.standard_normal(signal_length)
            for i in range(num_leads)
        ],
        axis=1,
    ).astype(np.float64)

    wfdb.wrsamp(
        record_name=record_name,
        fs=fs,
        units=["mV"] * num_leads,
        sig_name=lead_names,
        p_signal=p_signal,
        fmt=["16"] * num_leads,
        write_dir=str(directory),
        comments=["age: 45", "sex: F"],
    )
    return p_signal.T  # (num_leads, signal_length), matching this parser's convention


@pytest.fixture
def synthetic_record_dir(tmp_path: Path) -> Path:
    _write_synthetic_record(tmp_path)
    return tmp_path


class TestWFDBRecordParser:
    def test_parses_signal_shape_and_values(self, synthetic_record_dir: Path) -> None:
        expected_signal = _write_synthetic_record(synthetic_record_dir, record_name="00002")
        paths = ECGRecordPaths(
            record_id="00002",
            header_path=synthetic_record_dir / "00002.hea",
            signal_path=synthetic_record_dir / "00002.dat",
            image_paths=[],
        )

        parser = WFDBRecordParser()
        signal = parser.parse(paths, source_dataset="physionet")

        assert signal.values.shape == (NUM_LEADS, SIGNAL_LENGTH)
        # WFDB round-trips through 16-bit ADC quantization; allow small tolerance.
        np.testing.assert_allclose(signal.values, expected_signal, atol=1e-2)

    def test_parses_metadata_correctly(self, synthetic_record_dir: Path) -> None:
        paths = ECGRecordPaths(
            record_id="00001",
            header_path=synthetic_record_dir / "00001.hea",
            signal_path=synthetic_record_dir / "00001.dat",
            image_paths=[],
        )

        parser = WFDBRecordParser()
        signal = parser.parse(paths, source_dataset="physionet")
        metadata = signal.metadata

        assert metadata.record_id == "00001"
        assert metadata.source_dataset == "physionet"
        assert metadata.sampling_rate_hz == pytest.approx(SAMPLING_RATE_HZ)
        assert metadata.num_leads == NUM_LEADS
        assert metadata.lead_names == LEAD_NAMES
        assert metadata.signal_length == SIGNAL_LENGTH
        assert metadata.units == "mV"
        assert "comments" in metadata.extra
        assert any("age" in c for c in metadata.extra["comments"])

    def test_values_and_metadata_are_internally_consistent(
        self, synthetic_record_dir: Path
    ) -> None:
        paths = ECGRecordPaths(
            record_id="00001",
            header_path=synthetic_record_dir / "00001.hea",
            signal_path=synthetic_record_dir / "00001.dat",
            image_paths=[],
        )
        signal = WFDBRecordParser().parse(paths, source_dataset="physionet")
        assert signal.values.shape[0] == signal.metadata.num_leads
        assert signal.values.shape[1] == signal.metadata.signal_length

    def test_single_lead_record(self, tmp_path: Path) -> None:
        _write_synthetic_record(tmp_path, record_name="single", num_leads=1, lead_names=["II"])
        paths = ECGRecordPaths(
            record_id="single",
            header_path=tmp_path / "single.hea",
            signal_path=tmp_path / "single.dat",
            image_paths=[],
        )
        signal = WFDBRecordParser().parse(paths, source_dataset="physionet")
        assert signal.values.shape[0] == 1
        assert signal.metadata.lead_names == ["II"]

    def test_missing_ground_truth_paths_raises(self) -> None:
        paths = ECGRecordPaths(
            record_id="missing",
            header_path=None,
            signal_path=None,
            image_paths=[],
        )
        with pytest.raises(RecordParsingError, match="no header/signal path"):
            WFDBRecordParser().parse(paths, source_dataset="physionet")

    def test_nonexistent_files_raises_parsing_error(self, tmp_path: Path) -> None:
        paths = ECGRecordPaths(
            record_id="ghost",
            header_path=tmp_path / "ghost.hea",
            signal_path=tmp_path / "ghost.dat",
            image_paths=[],
        )
        with pytest.raises(RecordParsingError):
            WFDBRecordParser().parse(paths, source_dataset="physionet")

    def test_corrupt_header_raises_parsing_error(self, tmp_path: Path) -> None:
        _write_synthetic_record(tmp_path, record_name="corrupt")
        # Overwrite the valid header with garbage.
        (tmp_path / "corrupt.hea").write_text("this is not a valid wfdb header\n")

        paths = ECGRecordPaths(
            record_id="corrupt",
            header_path=tmp_path / "corrupt.hea",
            signal_path=tmp_path / "corrupt.dat",
            image_paths=[],
        )
        with pytest.raises(RecordParsingError):
            WFDBRecordParser().parse(paths, source_dataset="physionet")

    def test_non_finite_values_are_rejected(self, synthetic_record_dir: Path) -> None:
        paths = ECGRecordPaths(
            record_id="00001",
            header_path=synthetic_record_dir / "00001.hea",
            signal_path=synthetic_record_dir / "00001.dat",
            image_paths=[],
        )
        signal = WFDBRecordParser().parse(paths, source_dataset="physionet")

        # Directly exercise the validation guard with a corrupted copy of an
        # otherwise-valid signal, since WFDB's own ADC round-trip cannot
        # itself produce NaN/Inf from a finite p_signal.
        bad_values = signal.values.copy()
        bad_values[0, 0] = np.nan
        from dataclasses import replace

        bad_signal = replace(signal, values=bad_values)
        with pytest.raises(RecordParsingError):
            WFDBRecordParser._validate_signal(bad_signal)

    def test_parser_is_stateless_and_reusable_across_records(
        self, synthetic_record_dir: Path
    ) -> None:
        _write_synthetic_record(synthetic_record_dir, record_name="second")
        parser = WFDBRecordParser()

        paths_a = ECGRecordPaths(
            record_id="00001",
            header_path=synthetic_record_dir / "00001.hea",
            signal_path=synthetic_record_dir / "00001.dat",
            image_paths=[],
        )
        paths_b = ECGRecordPaths(
            record_id="second",
            header_path=synthetic_record_dir / "second.hea",
            signal_path=synthetic_record_dir / "second.dat",
            image_paths=[],
        )

        signal_a = parser.parse(paths_a, source_dataset="physionet")
        signal_b = parser.parse(paths_b, source_dataset="physionet")

        assert signal_a.metadata.record_id == "00001"
        assert signal_b.metadata.record_id == "second"


class TestParserRegistry:
    def test_physionet_parser_is_pre_registered(self) -> None:
        assert PARSER_REGISTRY["physionet"] is WFDBRecordParser

    def test_register_parser_adds_new_source(self) -> None:
        class DummyParser:
            def parse(self, paths: ECGRecordPaths, source_dataset: str):  # pragma: no cover
                raise NotImplementedError

        register_parser("dummy_source_for_test", DummyParser)
        try:
            assert PARSER_REGISTRY["dummy_source_for_test"] is DummyParser
            assert isinstance(DummyParser(), RecordParser)
        finally:
            del PARSER_REGISTRY["dummy_source_for_test"]

    def test_register_parser_overwrite_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        class ParserOne:
            def parse(self, paths, source_dataset):  # pragma: no cover
                raise NotImplementedError

        class ParserTwo:
            def parse(self, paths, source_dataset):  # pragma: no cover
                raise NotImplementedError

        register_parser("overwrite_test", ParserOne)
        try:
            with caplog.at_level("WARNING"):
                register_parser("overwrite_test", ParserTwo)
            assert PARSER_REGISTRY["overwrite_test"] is ParserTwo
            assert any("Overwriting parser registration" in msg for msg in caplog.messages)
        finally:
            del PARSER_REGISTRY["overwrite_test"]
