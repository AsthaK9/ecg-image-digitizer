"""Comprehensive Automated Verification Tests for Signal Fidelity, Calibration,
CSV Round-Trip, and Ground-Truth Discipline (Tests A through F).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import (
    CanonicalECGSignal,
    ECGRecordPaths,
    LeadSignal,
)
from ecg_digitization.evaluation.metrics import evaluate_digitization_fidelity
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from ecg_digitization.reconstruction.calibrator import (
    CalibrationConfig,
    ReconstructedECG,
    ReconstructedLead,
    SignalReconstructor,
    export_to_csv,
    import_from_csv,
)
from ecg_digitization.service.app import app
from ecg_digitization.synthetic.generator import (
    DegradationConfig,
    SyntheticECGGenerator,
)


# ============================================================================
# Test A: CSV Round-Trip Consistency
# ============================================================================
class TestCSVRoundTrip:
    """Verifies that exporting an ECG signal to CSV and importing it back is lossless."""

    def test_round_trip_canonical_signal(self) -> None:
        fs = 100.0
        n_samples = 250
        time_axis = np.arange(n_samples, dtype=np.float32) / fs

        leads = {
            "I": LeadSignal(
                lead_name="I",
                voltage=np.sin(2 * np.pi * 1.5 * time_axis).astype(np.float32),
                time_seconds=time_axis,
                sampling_rate_hz=fs,
                units="mV",
                is_calibrated=True,
                completeness=1.0,
                is_trained_lead=True,
            ),
            "II": LeadSignal(
                lead_name="II",
                voltage=(np.cos(2 * np.pi * 1.5 * time_axis) * 1.2).astype(np.float32),
                time_seconds=time_axis,
                sampling_rate_hz=fs,
                units="mV",
                is_calibrated=True,
                completeness=1.0,
                is_trained_lead=True,
            ),
            "V1": LeadSignal(
                lead_name="V1",
                voltage=(np.sin(2 * np.pi * 3.0 * time_axis) * 0.5).astype(np.float32),
                time_seconds=time_axis,
                sampling_rate_hz=fs,
                units="mV",
                is_calibrated=True,
                completeness=1.0,
                is_trained_lead=True,
            ),
        }

        orig_signal = CanonicalECGSignal(
            record_id="test_roundtrip_01",
            leads=leads,
            sampling_rate_hz=fs,
            layout_name="3x4_1",
            units="mV",
            is_calibration_reliable=True,
            calibration_standard="Calibrated Physical Voltage (mV)",
            calibration_note="Test round-trip execution",
            source_type="synthetic_paired",
        )

        csv_text = export_to_csv(orig_signal)
        imported_signal = import_from_csv(csv_text, record_id="test_roundtrip_01")

        # 1. Metadata match
        assert imported_signal.record_id == "test_roundtrip_01"
        assert imported_signal.sampling_rate_hz == fs
        assert imported_signal.layout_name == "3x4_1"
        assert imported_signal.is_calibration_reliable is True

        # 2. Leads match
        assert set(imported_signal.lead_names) == {"I", "II", "V1"}

        # 3. Sample arrays match within 4-decimal CSV precision (atol=1e-4)
        for l_name in ("I", "II", "V1"):
            orig_lead = orig_signal.leads[l_name]
            imp_lead = imported_signal.leads[l_name]

            assert len(imp_lead.voltage) == len(orig_lead.voltage)
            assert np.allclose(imp_lead.voltage, orig_lead.voltage, atol=1e-4)
            assert np.allclose(imp_lead.time_seconds, orig_lead.time_seconds, atol=1e-4)

    def test_round_trip_uncalibrated_single_lead(self) -> None:
        fs = 100.0
        n_samples = 153
        time_axis = np.arange(n_samples, dtype=np.float32) / fs
        voltages = np.linspace(-1.5, 1.5, n_samples, dtype=np.float32)

        leads = {
            "II": LeadSignal(
                lead_name="II",
                voltage=voltages,
                time_seconds=time_axis,
                sampling_rate_hz=fs,
                units="relative_units",
                is_calibrated=False,
                completeness=1.0,
            )
        }

        orig = CanonicalECGSignal(
            record_id="uncal_crop",
            leads=leads,
            sampling_rate_hz=fs,
            layout_name="single_lead",
            units="relative_units",
            is_calibration_reliable=False,
            calibration_standard="Uncalibrated / Normalized Amplitude",
            calibration_note="Nominal estimates",
            source_type="real_unpaired",
        )

        csv_text = export_to_csv(orig)
        imported = import_from_csv(csv_text, record_id="uncal_crop")

        assert imported.is_calibration_reliable is False
        assert imported.layout_name == "single_lead"
        assert len(imported.leads["II"].voltage) == n_samples
        assert np.allclose(imported.leads["II"].voltage, voltages, atol=1e-4)


# ============================================================================
# Test B: Synthetic Image Digitisation Against Genuine Digital Ground Truth
# ============================================================================
class TestSyntheticGroundTruthDigitization:
    """Tests digitising a rendered synthetic ECG sheet against genuine digital GT."""

    def test_synthetic_sheet_fidelity(self) -> None:
        # Load sample PTB-XL record
        hea = Path("data/interim/ptbxl_sample/00001_lr.hea")
        dat = Path("data/interim/ptbxl_sample/00001_lr.dat")
        if not (hea.is_file() and dat.is_file()):
            pytest.skip("PTB-XL sample 00001_lr not available.")

        parser = WFDBRecordParser()
        rec_signal = parser.parse(ECGRecordPaths("00001_lr", hea, dat), source_dataset="ptbxl")

        # Generate clean synthetic 3x4_1 ECG sheet
        generator = SyntheticECGGenerator()
        deg = DegradationConfig(enabled=False)
        synth = generator.generate(rec_signal, degradation=deg, seed=42)

        # Digitize synthetic image
        pipeline = ECGDigitizerPipeline()
        res = pipeline.digitize(synth.image, layout_hint="3x4_1")

        assert res.success is True
        assert "II" in res.reconstructed_ecg.leads
        assert "V1" in res.reconstructed_ecg.leads
        assert "II_RHYTHM" in res.reconstructed_ecg.leads

        # Map ground-truth leads to columns
        lead_map = {name.upper(): i for i, name in enumerate(rec_signal.metadata.lead_names)}
        gt_leads = {}
        rec_leads = {}
        col_leads = generator.LAYOUT_3X4_COLUMNS

        for col_idx, cols in enumerate(col_leads):
            t_start = int(col_idx * 2.5 * 100)
            t_end = int((col_idx + 1) * 2.5 * 100)
            for l_name in cols:
                if l_name in lead_map and l_name in res.reconstructed_ecg.leads:
                    gt_leads[l_name] = rec_signal.values[lead_map[l_name], t_start:t_end]
                    rec_leads[l_name] = res.reconstructed_ecg.leads[l_name].voltage_mv

        if "II" in lead_map and "II_RHYTHM" in res.reconstructed_ecg.leads:
            gt_leads["II_RHYTHM"] = rec_signal.values[lead_map["II"], :1000]
            rec_leads["II_RHYTHM"] = res.reconstructed_ecg.leads["II_RHYTHM"].voltage_mv

        report = evaluate_digitization_fidelity(gt_leads, rec_leads, record_id="00001_lr")

        # Verify high correlation on key validated leads without arbitrary curve fabrication
        assert report.per_lead["II"].pearson_r > 0.85
        assert report.per_lead["II_RHYTHM"].pearson_r > 0.85
        assert report.per_lead["II"].rmse_mv < 0.20


# ============================================================================
# Test C: Mathematical Calibration & Grid Scaling
# ============================================================================
class TestCalibrationMathematics:
    """Verifies that pixels-to-mV and pixels-to-seconds conversions match physical standards."""

    def test_physical_calibration_constants(self) -> None:
        cal = CalibrationConfig(speed_mm_s=25.0, gain_mm_mv=10.0, dpi=150, target_fs_hz=100.0)

        # 1 inch = 25.4 mm -> pixels_per_mm = 150 / 25.4
        expected_px_mm = 150.0 / 25.4
        assert np.isclose(cal.pixels_per_mm, expected_px_mm, atol=1e-3)

        # 25 mm/s paper speed
        expected_px_sec = 25.0 * expected_px_mm
        assert np.isclose(cal.pixels_per_sec, expected_px_sec, atol=1e-3)

        # 10 mm/mV voltage gain
        expected_px_mv = 10.0 * expected_px_mm
        assert np.isclose(cal.pixels_per_mv, expected_px_mv, atol=1e-3)

    def test_displacement_to_voltage_mapping(self) -> None:
        cal = CalibrationConfig(speed_mm_s=25.0, gain_mm_mv=10.0, dpi=150, target_fs_hz=100.0)
        reconstructor = SignalReconstructor(calibration=cal)

        # Construct synthetic extracted trace with known pixel deflections (5 points across 0.04s at 100 Hz)
        # 59.055 px = 1.0 mV deflection
        px_1mv = cal.pixels_per_mv
        relative_y = np.array([0.0, px_1mv, -px_1mv, 2.0 * px_1mv, 0.0], dtype=np.float32)
        pixel_trajectory = 100.0 - relative_y

        from ecg_digitization.reconstruction.extractor import ExtractedTrace
        trace = ExtractedTrace(
            lead_name="II",
            pixel_trajectory=pixel_trajectory,
            baseline_y=100.0,
            relative_trajectory=relative_y,
            completeness_ratio=1.0,
            active_start_col=0,
            active_end_col=4,
            crop_width_px=5,
            crop_height_px=200,
        )

        lead = reconstructor.reconstruct_lead(trace, lead_duration_s=0.04)

        # At index 0: 0.0 mV; at index 1: 1.0 mV; at index 2: -1.0 mV; at index 3: 2.0 mV
        assert np.isclose(lead.voltage_mv[0], 0.0, atol=1e-2)
        assert np.isclose(lead.voltage_mv[1], 1.0, atol=1e-2)
        assert np.isclose(lead.voltage_mv[2], -1.0, atol=1e-2)
        assert np.isclose(lead.voltage_mv[3], 2.0, atol=1e-2)

        # Verify time spacing at 100 Hz
        assert np.isclose(lead.time_seconds[1] - lead.time_seconds[0], 0.01, atol=1e-4)


# ============================================================================
# Test D: Reference CSV Import & API Consistency
# ============================================================================
class TestReferenceCSVImport:
    """Tests loading the user's reference N10007_signal.csv and API round-trip."""

    def test_import_reference_csv_file(self) -> None:
        csv_path = Path("data/interim/session1_demo_output/N10007_signal.csv")
        if not csv_path.is_file():
            # Check user upload location
            cand = list(Path("/Users/akashraj/.gemini/antigravity/brain").glob("*/.user_uploaded/*d5126b83.csv"))
            if cand and cand[0].is_file():
                csv_path = cand[0]
            else:
                pytest.skip("N10007_signal.csv not found.")

        sig = import_from_csv(csv_path)

        assert sig.layout_name == "single_lead"
        assert sig.is_calibration_reliable is False
        assert sig.sampling_rate_hz == 100.0
        assert "II" in sig.leads

        lead_ii = sig.leads["II"]
        assert len(lead_ii.voltage) == 209
        assert np.isclose(lead_ii.voltage[0], 1.7018, atol=1e-3)
        assert np.isclose(lead_ii.voltage[-1], -1.6510, atol=1e-3)

    def test_api_import_csv_endpoint(self) -> None:
        client = TestClient(app)
        sample_csv = (
            "# ECG DigiAI — Unified Reconstructed ECG Waveform Record\n"
            "# Layout: 3x4_1 | Sampling Rate: 100.0 Hz | Calibration Reliable: True\n"
            "# Calibration Standard: Calibrated Physical Voltage (mV)\n"
            "# Validated Leads: I, II\n"
            "# Leads Present (2): I, II\n"
            "time_s,I,II\n"
            "0.0000,0.1000,0.2000\n"
            "0.0100,0.1200,0.2500\n"
            "0.0200,0.1400,0.3000\n"
        )

        response = client.post(
            "/api/import-csv",
            json={"csv_text": sample_csv, "record_id": "test_api_import"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["record_id"] == "test_api_import"
        assert data["sampling_rate_hz"] == 100.0
        assert "I" in data["leads"]
        assert "II" in data["leads"]
        assert len(data["leads"]["I"]["voltage_mv"]) == 3
        assert np.isclose(data["leads"]["I"]["voltage_mv"][1], 0.12, atol=1e-4)


# ============================================================================
# Test E: Ground-Truth Evaluation Discipline (Unpaired Real Produces N/A)
# ============================================================================
class TestEvaluationDiscipline:
    """Verifies that unpaired real inputs produce N/A metrics and no mock numbers."""

    def test_unpaired_real_record_ground_truth(self) -> None:
        client = TestClient(app)

        # Request ground truth for an unpaired real beat sample
        res = client.get("/api/ground-truth/N10007")
        assert res.status_code == 200
        data = res.json()

        assert data["available"] is False
        assert "unpaired" in data["note"].lower()
        assert len(data["leads"]) == 0

    def test_unpaired_digitize_has_no_mock_fidelity_metrics(self) -> None:
        client = TestClient(app)
        img_path = Path("data/real_ecg_eval/samples/N10007.png")
        if not img_path.is_file():
            pytest.skip("N10007.png not available.")

        import base64
        b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

        res = client.post(
            "/api/digitize",
            json={
                "record_id": "N10007",
                "image_base64": b64,
                "layout_hint": "single_lead",
            },
        )
        assert res.status_code == 200
        data = res.json()

        # Fidelity metrics must be strictly None (omitted for unpaired real inputs)
        assert data["fidelity_metrics"] is None


# ============================================================================
# Test F: Multi-Lead Consistency and Standard Ordering
# ============================================================================
class TestMultiLeadConsistency:
    """Verifies standard 12-lead clinical sequence, DataFrame structure, and chunk alignment."""

    def test_standard_lead_sequence_and_alignment(self) -> None:
        expected_order = [
            "I", "II", "III",
            "AVR", "AVL", "AVF",
            "V1", "V2", "V3",
            "V4", "V5", "V6",
            "II_RHYTHM",
        ]

        fs = 100.0
        leads = {}
        for idx, name in enumerate(expected_order):
            duration = 10.0 if "RHYTHM" in name else 2.5
            n_samples = int(duration * fs)
            leads[name] = LeadSignal(
                lead_name=name,
                voltage=np.zeros(n_samples, dtype=np.float32) + idx * 0.1,
                time_seconds=np.arange(n_samples, dtype=np.float32) / fs,
                sampling_rate_hz=fs,
                units="mV",
                is_calibrated=True,
            )

        signal = CanonicalECGSignal(
            record_id="standard_12lead",
            leads=leads,
            sampling_rate_hz=fs,
            layout_name="3x4_1",
        )

        assert signal.lead_names == expected_order

        df = signal.to_dataframe()
        assert df.columns.tolist() == ["time_s"] + expected_order
        assert len(df) == 1000  # 10.0 seconds at 100 Hz

        # Check that 2.5s leads have NaNs after sample 250 while rhythm strip is full 1000
        assert not np.isnan(df["II_RHYTHM"].iloc[999])
        assert np.isnan(df["I"].iloc[500])
        assert not np.isnan(df["I"].iloc[249])


# ============================================================================
# Test G: Morphological Fidelity, Apex Preservation & No Flat Tail Truncation
# ============================================================================
class TestFineMorphologyPreservation:
    """Verifies that waveform extraction and calibration preserve extrema and flat segments."""

    def test_no_flat_tail_truncation(self) -> None:
        from ecg_digitization.reconstruction.extractor import WaveformExtractor

        # Create binary mask with signal that ends in a 100-pixel flat line
        h, w = 100, 300
        mask = np.zeros((h, w), dtype=np.uint8)

        # Sine wave for first 150 px
        for col in range(150):
            row = int(50 + 30 * np.sin(2 * np.pi * col / 75))
            mask[row, col] = 255

        # Flat baseline at row 50 for the remaining 150 px
        for col in range(150, 300):
            mask[50, col] = 255

        extractor = WaveformExtractor()
        extracted = extractor.extract_from_crop(mask, lead_name="II")

        # Extraction must cover all 300 columns and not truncate the flat tail
        assert len(extracted.pixel_trajectory) == 300
        # The last 50 samples must be exactly at row 50
        assert np.allclose(extracted.pixel_trajectory[-50:], 50.0, atol=0.5)

    def test_apex_preservation_on_steep_peaks(self) -> None:
        from ecg_digitization.reconstruction.extractor import WaveformExtractor

        # In an image coordinate system, row 0 is top (highest physical voltage), row 120 is bottom
        # A sharp R-peak is a vertical stroke reaching up to row 10 (apex = 10)
        h, w = 120, 50
        mask = np.zeros((h, w), dtype=np.uint8)
        for col in range(w):
            mask[60, col] = 255  # baseline at row 60

        # Draw a sharp vertical R-stroke at col 25 from row 10 to row 60
        mask[10:61, 25] = 255

        extractor = WaveformExtractor()
        extracted = extractor.extract_from_crop(mask, lead_name="II")

        # The peak apex must be preserved near row 10, NOT the midpoint (35)
        # Because in ECG upward deflection = min row in image coordinates
        assert extracted.pixel_trajectory[25] <= 20.0, f"Expected apex near row 10, got {extracted.pixel_trajectory[25]}"

    def test_pchip_interpolation_preserves_extrema(self) -> None:
        from ecg_digitization.reconstruction.calibrator import SignalReconstructor
        from ecg_digitization.reconstruction.extractor import ExtractedTrace

        reconstructor = SignalReconstructor()
        # Original discrete column pixels with a sharp peak at col 50 out of 101 cols
        col_x = np.arange(101)
        pixel_traj = 50.0 - 40.0 * np.exp(-((col_x - 50) / 3.0) ** 2)
        trace = ExtractedTrace(
            lead_name="II",
            pixel_trajectory=pixel_traj,
            baseline_y=50.0,
            relative_trajectory=50.0 - pixel_traj,
            completeness_ratio=1.0,
            active_start_col=0,
            active_end_col=100,
            crop_width_px=101,
            crop_height_px=100,
            is_trained_lead=True,
            validation_status="validated",
            confidence=1.0,
            extraction_method="test",
            crop_y1=0,
            crop_y2=100,
            is_clipped_top=False,
            is_clipped_bottom=False,
            clipping_margin_px=10,
            cross_talk_risk=False,
            is_overlapping=False,
            mean_jump_px=0.0,
            max_jump_px=0.0,
            is_quality_passed=True,
            quality_note="test",
        )
        reconstructed = reconstructor.reconstruct_lead(trace, lead_duration_s=1.0)

        # Check peak value is strictly preserved without overshoot/undershoot
        peak_idx = np.argmax(reconstructed.voltage_mv)
        assert reconstructed.voltage_mv[peak_idx] > 0.60
        assert np.isclose(reconstructed.time_seconds[peak_idx], 0.50, atol=0.03)


# ============================================================================
# Test H: QRS Complex Detection and Inverted Polarity Handling
# ============================================================================
class TestQRSDetectionAndPolarity:
    """Verifies QRS complex detection, Q-onset/S-offset boundaries, and inverted lead support."""

    def test_upright_lead_qrs_detection(self) -> None:
        from ecg_digitization.inference.analyzer import ECGRhythmAnalyzer

        analyzer = ECGRhythmAnalyzer(target_fs_hz=100.0)
        # Synthesize 3 regular beats at 60 bpm (1 beat/sec)
        t = np.arange(300) / 100.0
        sig = np.zeros_like(t)

        for beat_center in [0.5, 1.5, 2.5]:
            idx_q = int((beat_center - 0.04) * 100)
            idx_r = int(beat_center * 100)
            idx_s = int((beat_center + 0.05) * 100)
            sig[idx_q] = -0.25
            sig[idx_r] = 1.20
            sig[idx_s] = -0.35

        complexes = analyzer.detect_qrs_complexes(sig, lead_name="II")
        assert len(complexes) >= 2, f"Expected at least 2 QRS complexes, got {len(complexes)}"

        for qrs in complexes:
            assert qrs.is_inverted is False
            assert qrs.qrs_duration_ms is not None
            assert 35.0 <= qrs.qrs_duration_ms <= 220.0
            assert qrs.confidence > 0.60

    def test_inverted_lead_qrs_detection(self) -> None:
        from ecg_digitization.inference.analyzer import ECGRhythmAnalyzer

        analyzer = ECGRhythmAnalyzer(target_fs_hz=100.0)
        # Synthesize inverted aVR beats with negative QS deflection
        t = np.arange(300) / 100.0
        sig = np.zeros_like(t)

        for beat_center in [0.5, 1.5, 2.5]:
            idx_r = int(beat_center * 100)
            sig[idx_r] = -1.10  # Dominant negative deflection

        complexes = analyzer.detect_qrs_complexes(sig, lead_name="aVR")
        assert len(complexes) >= 2, f"Expected at least 2 inverted QRS complexes, got {len(complexes)}"
        for qrs in complexes:
            assert qrs.is_inverted is True
            assert qrs.r_voltage_mv < 0


# ============================================================================
# Test I: QRS Overlay Generation & Pipeline Integration
# ============================================================================
class TestQRSOverlayIntegration:
    """Verifies that QRS detection overlay image is generated and returned via API."""

    def test_generate_qrs_overlay_file(self, tmp_path: Path) -> None:
        from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline

        img_path = Path("data/real_ecg_eval/samples/N10007.png")
        if not img_path.is_file():
            pytest.skip("N10007.png not found")

        pipeline = ECGDigitizerPipeline()
        res = pipeline.digitize(img_path)

        out_qrs = tmp_path / "test_qrs_overlay.png"
        generated_path = res.generate_qrs_overlay(out_qrs)

        assert generated_path.is_file()
        assert generated_path.stat().st_size > 1000  # Non-empty valid image


# ============================================================================
# Test J: Morphological Fidelity & Non-Clipping Regression
# ============================================================================
class TestMorphologicalFidelityAndNonClipping:
    """Verifies that narrow, broad, inverted, and boundary-touching peaks are preserved."""

    def test_narrow_qrs_peak_preservation(self) -> None:
        """Verifies that narrow needle-like QRS complexes are not smoothed into plateaus."""
        from ecg_digitization.inference.deep_digitizer import DeepCropDigitizer

        h, w = 128, 256
        crop_bin = np.zeros((h, w), dtype=np.uint8)
        crop_color = np.full((h, w, 3), 252, dtype=np.uint8)

        base_y = 64
        crop_bin[base_y, :] = 255

        # Draw narrow QRS at col 100: width 3px, ascending to y=20
        col = 100
        crop_bin[20:base_y, col - 1 : col + 2] = 255
        crop_color[crop_bin > 0] = [30, 30, 30]

        digitizer = DeepCropDigitizer()
        trace = digitizer.extract_trace_from_lead_crop(
            lead_crop=crop_color,
            binary_crop=crop_bin,
            lead_name="II",
            expected_baseline_y=float(base_y),
            is_rhythm=False,
        )

        traj = trace.pixel_trajectory
        # Peak must reach near y=20 (within 3px)
        peak_y = np.min(traj)
        assert peak_y <= 24.0, f"Narrow peak flattened: min_y={peak_y}, expected <= 24"

        # Peak must be narrow (within 2px of peak value should span <= 4 columns)
        near_top = np.sum(np.abs(traj - peak_y) < 2.0)
        assert near_top <= 4, f"Narrow peak converted to wide plateau ({near_top} cols)"

    def test_v4_like_peak_preservation_at_ceiling(self) -> None:
        """Verifies that tall QRS peaks near the upper crop boundary are not clipped."""
        from ecg_digitization.inference.deep_digitizer import DeepCropDigitizer

        h, w = 92, 180
        crop_bin = np.zeros((h, w), dtype=np.uint8)
        crop_color = np.full((h, w, 3), 252, dtype=np.uint8)

        base_y = 47
        crop_bin[base_y, :] = 255

        # Tall peak extending up to y=6
        crop_bin[6:base_y, 88:92] = 255
        crop_color[crop_bin > 0] = [30, 30, 30]

        digitizer = DeepCropDigitizer()
        trace = digitizer.extract_trace_from_lead_crop(
            lead_crop=crop_color,
            binary_crop=crop_bin,
            lead_name="V4",
            expected_baseline_y=float(base_y),
            is_rhythm=False,
        )

        peak_y = np.min(trace.pixel_trajectory)
        assert peak_y <= 10.0, f"Ceiling peak clipped: min_y={peak_y}, expected <= 10"

    def test_inverted_qrs_trough_preservation(self) -> None:
        """Verifies that dominant negative deflections (S-waves / inverted leads) are preserved."""
        from ecg_digitization.inference.deep_digitizer import DeepCropDigitizer

        h, w = 128, 256
        crop_bin = np.zeros((h, w), dtype=np.uint8)
        crop_color = np.full((h, w, 3), 252, dtype=np.uint8)

        base_y = 64
        crop_bin[base_y, :] = 255

        # Inverted QRS: downward stroke descending to y=108
        crop_bin[base_y:109, 120:124] = 255
        crop_color[crop_bin > 0] = [30, 30, 30]

        digitizer = DeepCropDigitizer()
        trace = digitizer.extract_trace_from_lead_crop(
            lead_crop=crop_color,
            binary_crop=crop_bin,
            lead_name="AVR",
            expected_baseline_y=float(base_y),
            is_rhythm=False,
        )

        trough_y = np.max(trace.pixel_trajectory)
        assert trough_y >= 104.0, f"Inverted trough attenuated: max_y={trough_y}, expected >= 104"

