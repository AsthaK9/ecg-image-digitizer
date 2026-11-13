"""Unit and integration tests for End-to-End 12-Lead ECG Sheet Digitization.

Verifies:
1. Lead layout detection and bounding box cropping for 3x4+1 and 12x1 sheets.
2. Routing through deep digitizer with 1:1 physical vertical scaling and rhythm chunking.
3. Formally validated vs experimental lead categorization.
4. Calibration preservation and nominal voltage handling.
5. Unified multi-lead CSV generation and schema conformance.
6. Diagnostic sheet overlay generation with bounding boxes and reconstructed traces.
7. FastAPI service integration for digitisation and CSV export.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import ECGRecordPaths
from ecg_digitization.pipeline.digitizer import DigitizationPipelineResult, ECGDigitizerPipeline
from ecg_digitization.preprocessing.lead_detector import LeadLayoutDetector
from ecg_digitization.reconstruction.calibrator import CalibrationConfig, ReconstructedECG
from ecg_digitization.service.app import (
    DigitizeRequest,
    digitize_image,
    export_csv,
)
from ecg_digitization.synthetic.generator import SyntheticECGGenerator


@pytest.fixture(scope="module")
def sample_synthetic_sheet():
    """Generates a reproducible synthetic 12-lead ECG sheet from Record 00001_lr."""
    parser = WFDBRecordParser()
    sig = parser.parse(
        ECGRecordPaths(
            "00001_lr",
            Path("data/interim/ptbxl_sample/00001_lr.hea"),
            Path("data/interim/ptbxl_sample/00001_lr.dat"),
        ),
        source_dataset="ptbxl",
    )
    gen = SyntheticECGGenerator()
    synth = gen.generate(sig, seed=42)
    return synth, sig


@pytest.fixture(scope="module")
def shared_digitized_result(sample_synthetic_sheet):
    """Executes single deep digitisation pass shared across unit tests to avoid redundant CPU inference."""
    synth, _ = sample_synthetic_sheet
    pipeline = ECGDigitizerPipeline(extractor_mode="deep")
    result = pipeline.digitize(
        synth.image,
        layout_hint="3x4_1",
        extractor_mode="deep",
        is_calibration_reliable=True,
        calibration_note="Verified synthetic 25 mm/s, 10 mm/mV",
    )
    return result


class TestEndToEnd12LeadDigitization:
    """Tests the full end-to-end 12-lead digitisation pipeline."""

    def test_lead_layout_detector_3x4_1(self):
        detector = LeadLayoutDetector()
        res = detector.detect_layout((1240, 1654, 3), layout_hint="3x4_1")
        assert res.is_supported is True
        assert len(res.leads) == 13  # 12 grid leads + 1 rhythm strip
        assert "II" in res.leads
        assert "V1" in res.leads
        assert "II_RHYTHM" in res.leads
        assert res.leads["II_RHYTHM"].is_rhythm is True
        assert res.leads["II"].is_rhythm is False

    def test_lead_layout_detector_12x1(self):
        detector = LeadLayoutDetector()
        res = detector.detect_layout((1200, 1600, 3), layout_hint="12x1")
        assert res.is_supported is True
        assert len(res.leads) == 12
        assert "I" in res.leads
        assert "V6" in res.leads

    def test_end_to_end_digitize_deep_mode(self, shared_digitized_result, tmp_path):
        result = shared_digitized_result

        assert isinstance(result, DigitizationPipelineResult)
        assert result.success is True
        assert result.extractor_mode_used in ("deep", "classical")
        assert len(result.extracted_traces) >= 12

        # Check lead categorization: Formally Validated vs Experimental
        trace_II = result.extracted_traces["II"]
        assert trace_II.is_trained_lead is True
        assert trace_II.validation_status == "validated_lead"

        trace_V1 = result.extracted_traces["V1"]
        assert trace_V1.is_trained_lead is True
        assert trace_V1.validation_status == "validated_lead"

        trace_I = result.extracted_traces["I"]
        assert trace_I.validation_status in ("validated_lead", "experimental_unvalidated")

        # Check reconstructed ECG
        rec = result.reconstructed_ecg
        assert rec.is_calibration_reliable is True
        assert "II" in rec.leads
        assert "V1" in rec.leads
        assert "II_RHYTHM" in rec.leads

        # Check rhythm strip duration (~10s vs 2.5s for grid leads)
        assert rec.leads["II_RHYTHM"].duration_seconds == pytest.approx(10.0, rel=1e-2)
        assert rec.leads["II"].duration_seconds == pytest.approx(2.5, rel=1e-2)

        # Check unified CSV export
        csv_file = tmp_path / "unified_test_signal.csv"
        rec.save_csv(csv_file)
        assert csv_file.is_file()

        # Check CSV content with pandas (ignoring # comment headers)
        df = pd.read_csv(csv_file, comment="#")
        assert "time_s" in df.columns
        assert "II" in df.columns
        assert "V1" in df.columns
        assert "II_RHYTHM" in df.columns
        assert len(df) == 1000  # 10.0 seconds at 100 Hz

    def test_generate_diagnostic_sheet_overlay(self, sample_synthetic_sheet, shared_digitized_result, tmp_path):
        synth, _ = sample_synthetic_sheet
        result = shared_digitized_result
        overlay_path = tmp_path / "sheet_overlay_test.png"
        result.generate_sheet_overlay(overlay_path)

        assert overlay_path.is_file()
        overlay_img = cv2.imread(str(overlay_path))
        assert overlay_img is not None
        # Should include top banner (+32 px)
        assert overlay_img.shape[0] == synth.image.shape[0] + 32
        assert overlay_img.shape[1] == synth.image.shape[1]

    def test_uncalibrated_nominal_voltage_flagging(self, sample_synthetic_sheet, tmp_path):
        synth, _ = sample_synthetic_sheet
        pipeline = ECGDigitizerPipeline(extractor_mode="classical")

        result = pipeline.digitize(
            synth.image,
            layout_hint="3x4_1",
            extractor_mode="classical",
            is_calibration_reliable=False,
            calibration_note="Uncalibrated scan: voltages are nominal estimates",
        )

        assert result.reconstructed_ecg.is_calibration_reliable is False
        assert "nominal estimates" in result.reconstructed_ecg.calibration_note
        assert result.reconstructed_ecg.leads["II"].is_calibration_reliable is False

        # Verify comment header in CSV
        csv_file = tmp_path / "nominal_test.csv"
        result.reconstructed_ecg.save_csv(csv_file)
        content = csv_file.read_text(encoding="utf-8")
        assert "Calibration Reliable: False" in content


class TestFastAPISheetIntegration:
    """Tests FastAPI backend endpoint handlers for 12-lead sheet digitisation and CSV export."""

    def test_digitize_api_returns_sheet_overlay_and_lead_provenance(self):
        req = DigitizeRequest(
            record_id="00001_lr",
            layout_hint="3x4_1",
            extractor_mode="classical",  # Fast mode for endpoint handler test
        )
        data = digitize_image(req)

        assert data["success"] is True
        assert "sheet_overlay_b64" in data
        assert len(data["sheet_overlay_b64"]) > 100
        assert "leads" in data
        assert "II" in data["leads"]
        assert data["leads"]["II"]["is_trained_lead"] is True
        assert data["leads"]["II"]["validation_status"] in ("validated_lead", "calibrated_lead")
        assert data["leads"]["I"]["is_trained_lead"] is False
        assert data["leads"]["I"]["validation_status"] in ("experimental_unvalidated", "calibrated_lead", "uncalibrated_lead", "quality_warning")

    def test_export_csv_endpoint(self):
        resp = export_csv("00001_lr")
        assert resp.status_code == 200
        assert "text/csv" in resp.media_type
        content = resp.body.decode("utf-8")
        assert "time_s" in content
        assert "II" in content
        assert "Validated Leads" in content


class TestECG1AttachedImageRegression:
    """Verifies that the user-attached ecg1.jpeg test image reconstructs accurately without sawtooth noise."""

    def test_ecg1_end_to_end_digitization(self, tmp_path):
        ecg1_file = Path("tests/fixtures/ecg1.jpeg")
        assert ecg1_file.is_file(), "Fixture tests/fixtures/ecg1.jpeg not found"

        img = cv2.imread(str(ecg1_file))
        assert img is not None
        assert img.shape == (540, 720, 3)

        pipeline = ECGDigitizerPipeline(extractor_mode="deep")
        res = pipeline.digitize(img, layout_hint="auto", is_calibration_reliable=False)

        assert res.success is True
        assert res.layout.layout_name == "3x4_3"
        assert len(res.reconstructed_ecg.leads) == 15
        assert "II" in res.reconstructed_ecg.leads
        assert "II_RHYTHM" in res.reconstructed_ecg.leads
        assert "V1_RHYTHM" in res.reconstructed_ecg.leads
        assert "V5_RHYTHM" in res.reconstructed_ecg.leads

        # Verify no sawtooth vertical jumping
        for l_name in ["I", "II", "II_RHYTHM"]:
            trace = res.extracted_traces[l_name]
            traj = trace.pixel_trajectory
            jumps = np.abs(np.diff(traj))
            assert np.mean(jumps) < 2.5, f"Lead {l_name} has high mean jump ({np.mean(jumps):.2f} px), indicating noise"

        # Verify diagnostic sheet overlay creation
        overlay_p = res.generate_sheet_overlay(tmp_path / "ecg1_test_overlay.png")
        assert overlay_p.is_file()
        assert overlay_p.stat().st_size > 10000


class TestECG2AttachedImageRegression:
    """Verifies that user-attached ecg2.jpeg (3x4+1) reconstructs cleanly without Row 2/3 oscillations."""

    def test_ecg2_end_to_end_digitization(self, tmp_path):
        ecg2_file = Path("tests/fixtures/ecg2.jpeg")
        assert ecg2_file.is_file(), "Fixture tests/fixtures/ecg2.jpeg not found"

        img = cv2.imread(str(ecg2_file))
        assert img is not None
        assert img.shape == (388, 791, 3)

        pipeline = ECGDigitizerPipeline()
        res = pipeline.digitize(img, layout_hint="auto", is_calibration_reliable=True)

        assert res.success is True
        assert res.layout.layout_name == "3x4_1"
        assert len(res.reconstructed_ecg.leads) == 13
        assert "III" in res.reconstructed_ecg.leads
        assert "AVF" in res.reconstructed_ecg.leads
        assert "II_RHYTHM" in res.reconstructed_ecg.leads

        # Verify Lead III, aVF and rhythm strip do not have square-wave oscillations
        for l_name in ["III", "AVF", "II_RHYTHM"]:
            trace = res.extracted_traces[l_name]
            traj = trace.pixel_trajectory
            jumps = np.abs(np.diff(traj))
            assert np.mean(jumps) < 2.0, f"Lead {l_name} has high mean jump ({np.mean(jumps):.2f} px)"
            assert np.sum(jumps > 15.0) <= 8, f"Lead {l_name} has excessive large jumps"

        # Overlay verification
        overlay_p = res.generate_sheet_overlay(tmp_path / "ecg2_test_overlay.png")
        assert overlay_p.is_file()
        assert overlay_p.stat().st_size > 10000


class TestSingleLeadImageRegression:
    """Verifies that single-lead crops (N10007, Q1) reconstruct with sub-pixel ink fidelity and trim zero-tails."""

    def test_n10007_single_lead_digitization(self, tmp_path):
        n10007_file = Path("data/real_ecg_eval/samples/N10007.png")
        assert n10007_file.is_file()

        img = cv2.imread(str(n10007_file))
        pipeline = ECGDigitizerPipeline()
        res = pipeline.digitize(img, layout_hint="auto", is_calibration_reliable=False)

        assert res.success is True
        assert res.layout.layout_name == "single_lead"
        assert len(res.reconstructed_ecg.leads) == 1
        assert "II" in res.reconstructed_ecg.leads

        tr = res.extracted_traces["II"]
        # In N10007, 81 samples of flat zero-pad tail should be cleanly trimmed
        assert tr.tail_trimmed_samples >= 70, f"Expected >= 70 trimmed tail samples, got {tr.tail_trimmed_samples}"
        # Waveform should be smooth and faithful to cyan trace
        assert tr.mean_jump_px < 3.5

        # Verify distance to true ink
        from scipy.ndimage import distance_transform_edt
        cyan_mask = (img[:, :, 0] > 150) & (img[:, :, 1] > 150) & (img[:, :, 2] < 100)
        cyan_mask[:15, :] = False
        cyan_mask[-15:, :] = False
        cyan_mask[:, :15] = False
        cyan_mask[:, -15:] = False
        dt = distance_transform_edt(~cyan_mask)

        overlay_p = res.generate_sheet_overlay(tmp_path / "n10007_overlay.png")
        ov = cv2.imread(str(overlay_p))
        red_pts = (ov[32:, :, 2] > 180) & (ov[32:, :, 1] < 60) & (ov[32:, :, 0] < 60)
        mean_dist = float(np.mean(dt[red_pts]))
        assert mean_dist < 0.5, f"Expected sub-pixel mean distance to cyan ink < 0.5px, got {mean_dist:.3f}px"

    def test_q1_single_lead_digitization(self, tmp_path):
        q1_file = Path("data/real_ecg_eval/samples/Q1.png")
        assert q1_file.is_file()

        img = cv2.imread(str(q1_file))
        pipeline = ECGDigitizerPipeline()
        res = pipeline.digitize(img, layout_hint="auto", is_calibration_reliable=False)

        assert res.success is True
        assert len(res.reconstructed_ecg.leads) == 1
        tr = res.extracted_traces["II"]
        assert tr.tail_trimmed_samples >= 90
