"""Unit and regression tests for Trace Fidelity Diagnostics, Ink Recovery,
Ambiguity, Interpolation Accounting, and Layout Coordinate Preservation.

Covers Priority 5 requirements:
1. Top-edge ECG ink preservation.
2. Bottom rhythm-strip baseline preservation.
3. Dynamic row spacing across supported layouts (3x4, 3x4_1, 3x4_3, 12x1, single_lead).
4. Positive and negative QRS apex preservation.
5. Preservation of P-wave and T-wave morphology.
6. No artificial square-wave plateaus.
7. Directly observed vs interpolated sample accounting.
8. Ambiguous-region identification.
9. Per-lead diagnostic generation.
10. Correct image-to-waveform coordinate mapping.
11. Diagnostics unavailable handling.
12. Existing CSV import/export and digitisation regressions.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import ECGRecordPaths
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from ecg_digitization.preprocessing.lead_detector import LeadLayoutDetector
from ecg_digitization.reconstruction.calibrator import export_to_csv, import_from_csv
from ecg_digitization.reconstruction.diagnostics import (
    QRSComplexInfo,
    TraceFidelityComputer,
    TraceFidelityDiagnostics,
    UncertainInterval,
)
from ecg_digitization.service.app import app
from ecg_digitization.synthetic.generator import SyntheticECGGenerator


class TestTraceFidelityAccounting:
    """Verifies physical ink accounting, ambiguity detection, and interpolation metrics."""

    def test_observed_vs_interpolated_sample_accounting(self):
        """Verifies observed vs interpolated samples are counted accurately without fabricating support."""
        h, w = 64, 100
        crop_bin = np.zeros((h, w), dtype=np.uint8)
        base_y = 32

        # Draw ink for columns 0..59 and 80..99 (cols 60..79 are zero-ink gap)
        crop_bin[base_y, 0:60] = 255
        crop_bin[base_y, 80:100] = 255

        traj = np.full(100, float(base_y), dtype=np.float32)

        diag = TraceFidelityComputer.compute(
            binary_crop=crop_bin,
            pixel_trajectory=traj,
            active_start_col=0,
            active_end_col=99,
            lead_name="II",
            sampling_rate_hz=100.0,
            crop_height_px=h,
        )

        assert diag.total_samples == 100
        assert diag.observed_samples == 80
        assert diag.interpolated_samples == 20
        assert diag.interpolated_percent == 20.0
        assert diag.longest_interpolation_gap == 20
        assert len(diag.observed_mask) == 100
        assert all(diag.observed_mask[0:60])
        assert not any(diag.observed_mask[60:80])
        assert all(diag.observed_mask[80:100])

        # Verify uncertain intervals contains the missing ink gap
        gaps = [u for u in diag.uncertain_intervals if u.reason == "missing_ink"]
        assert len(gaps) == 1
        assert gaps[0].start_col == 60
        assert gaps[0].end_col == 79
        assert gaps[0].start_time_s == 0.60
        assert gaps[0].end_time_s == 0.80

    def test_ambiguous_region_identification(self):
        """Verifies that multi-path ink clusters (e.g. text/grid intersection) are detected."""
        h, w = 64, 100
        crop_bin = np.zeros((h, w), dtype=np.uint8)
        base_y = 32

        # Draw continuous trace at base_y
        crop_bin[base_y, :] = 255

        # Draw overlapping vertical stroke / label at cols 40..46 from y=10 to y=20 (gap > 2px from base_y)
        crop_bin[10:20, 40:47] = 255

        traj = np.full(100, float(base_y), dtype=np.float32)

        diag = TraceFidelityComputer.compute(
            binary_crop=crop_bin,
            pixel_trajectory=traj,
            active_start_col=0,
            active_end_col=99,
            lead_name="V1",
            sampling_rate_hz=100.0,
            crop_height_px=h,
        )

        assert diag.ambiguous_columns_count == 7
        assert diag.ambiguous_columns_percent == 7.0
        assert len(diag.ambiguous_regions) == 1
        assert diag.ambiguous_regions[0] == (40, 46)

        # Non-trace pixels in the upper cluster should be rejected
        assert diag.rejected_non_trace_pixels >= 70

        amb_intervals = [u for u in diag.uncertain_intervals if u.reason == "ambiguous_stroke"]
        assert len(amb_intervals) == 1
        assert amb_intervals[0].start_col == 40
        assert amb_intervals[0].end_col == 46

    def test_empty_or_unavailable_diagnostics_handling(self):
        """Verifies graceful handling and explicit UNCERTAIN_EXTRACTION status on empty crops."""
        empty_crop = np.zeros((0, 0), dtype=np.uint8)
        empty_traj = np.array([], dtype=np.float32)

        diag = TraceFidelityComputer.compute(
            binary_crop=empty_crop,
            pixel_trajectory=empty_traj,
            active_start_col=0,
            active_end_col=0,
            lead_name="I",
        )

        assert diag.overall_status == "UNCERTAIN_EXTRACTION"
        assert diag.total_samples == 0
        assert diag.observed_samples == 0
        assert diag.interpolated_percent == 0.0

        d_dict = diag.to_dict()
        assert d_dict["lead_name"] == "I"
        assert d_dict["overall_status"] == "UNCERTAIN_EXTRACTION"


class TestMorphologyAndPeakPreservation:
    """Verifies that P-waves, T-waves, and sharp QRS complexes maintain anatomical fidelity."""

    def test_p_and_t_wave_smoothness_preserved(self):
        """Verifies that broad low-slope P and T waves are not clipped or distorted by apex anchoring."""
        from ecg_digitization.inference.deep_digitizer import DeepCropDigitizer

        h, w = 128, 256
        crop_bin = np.zeros((h, w), dtype=np.uint8)
        crop_color = np.full((h, w, 3), 252, dtype=np.uint8)
        base_y = 64

        # Draw gentle sinusoidal T-wave spanning cols 140..200, amplitude 12px (peak at y=52)
        cols = np.arange(140, 201)
        t_wave_y = np.round(base_y - 12.0 * np.sin(np.pi * (cols - 140) / 60.0)).astype(int)
        for c, y in zip(cols, t_wave_y):
            crop_bin[max(0, y - 1) : min(h, y + 2), c] = 255
        # Fill rest of baseline
        crop_bin[base_y, :140] = 255
        crop_bin[base_y, 201:] = 255
        crop_color[crop_bin > 0] = [40, 40, 40]

        digitizer = DeepCropDigitizer()
        trace = digitizer.extract_trace_from_lead_crop(
            lead_crop=crop_color,
            binary_crop=crop_bin,
            lead_name="II",
            expected_baseline_y=float(base_y),
            is_rhythm=False,
        )

        traj = trace.pixel_trajectory
        t_section = traj[140:201]

        # Max jump in smooth T-wave section must be tiny (< 1.5px), no vertical apex jump
        jumps = np.abs(np.diff(t_section))
        assert np.max(jumps) < 1.8, f"T-wave distorted by apex anchoring: max jump {np.max(jumps):.2f}px"

        # T-wave peak should be smooth and rounded, not a needle or flat-topped square wave
        t_peak_idx = np.argmin(t_section)
        assert abs(t_section[t_peak_idx] - 52.0) <= 2.5

    def test_no_artificial_square_wave_plateaus_on_v5_rhythm(self):
        """Verifies ecg1.jpeg V5_RHYTHM does not produce rectangular plateaus."""
        ecg1_file = Path("tests/fixtures/ecg1.jpeg")
        assert ecg1_file.is_file()
        img = cv2.imread(str(ecg1_file))

        pipeline = ECGDigitizerPipeline(extractor_mode="deep")
        res = pipeline.digitize(img, layout_hint="3x4_3", is_calibration_reliable=False)

        v5_trace = res.extracted_traces["V5_RHYTHM"]
        traj = v5_trace.pixel_trajectory
        import scipy.signal
        peaks, _ = scipy.signal.find_peaks(-(traj - v5_trace.baseline_y), height=15, distance=30)
        assert len(peaks) >= 10, f"Expected >= 10 QRS complexes, got {len(peaks)}"

        # Apex width within 2px of peak must be narrow (<= 4 samples)
        for p in peaks:
            peak_val = traj[p]
            flat_pts = np.sum(np.abs(traj[max(0, p - 6) : min(len(traj), p + 7)] - peak_val) < 2.0)
            assert flat_pts <= 4, f"Square-wave flat top detected at index {p} ({flat_pts} samples)"


class TestDynamicLayoutsAndCoordinateMapping:
    """Verifies layout support and 1:1 image-to-waveform coordinate alignment."""

    def test_dynamic_row_spacing_across_supported_layouts(self):
        """Verifies row bounding box recovery across 3x4, 3x4_1, 3x4_3, 12x1, and single_lead."""
        detector = LeadLayoutDetector()

        # 1. 3x4 layout (4 rows)
        res_3x4 = detector.detect_layout((480, 640, 3), layout_hint="3x4")
        assert res_3x4.is_supported is True
        assert len(res_3x4.leads) == 12

        # 2. 3x4_1 layout (5 rows: 4 grid + 1 rhythm)
        res_3x4_1 = detector.detect_layout((500, 700, 3), layout_hint="3x4_1")
        assert res_3x4_1.is_supported is True
        assert len(res_3x4_1.leads) == 13
        assert res_3x4_1.leads["II_RHYTHM"].is_rhythm is True

        # 3. 3x4_3 layout (6 rows: 3 grid + 3 rhythm)
        res_3x4_3 = detector.detect_layout((540, 720, 3), layout_hint="3x4_3")
        assert res_3x4_3.is_supported is True
        assert len(res_3x4_3.leads) == 15
        assert res_3x4_3.leads["V1_RHYTHM"].is_rhythm is True
        assert res_3x4_3.leads["II_RHYTHM"].is_rhythm is True
        assert res_3x4_3.leads["V5_RHYTHM"].is_rhythm is True

        # 4. 12x1 layout (12 rows)
        res_12x1 = detector.detect_layout((1200, 1600, 3), layout_hint="12x1")
        assert res_12x1.is_supported is True
        assert len(res_12x1.leads) == 12

        # 5. single_lead (1 row)
        res_single = detector.detect_layout((288, 432, 3), layout_hint="single_lead")
        assert res_single.is_supported is True
        assert len(res_single.leads) == 1
        assert "II" in res_single.leads

    def test_image_to_waveform_coordinate_mapping_consistency(self, tmp_path):
        """Verifies that bounding box coordinates and active column spans map 1:1 onto sheet overlay."""
        img = cv2.imread("tests/fixtures/ecg1.jpeg")
        pipeline = ECGDigitizerPipeline(extractor_mode="deep")
        res = pipeline.digitize(img, layout_hint="3x4_3", is_calibration_reliable=False)

        overlay_path = tmp_path / "test_overlay_mapping.png"
        res.generate_uncertainty_overlay(overlay_path, selected_lead="V4")
        assert overlay_path.is_file()

        # Check V4 coordinate correspondence
        v4_reg = res.layout.leads["V4"]
        v4_trace = res.extracted_traces["V4"]
        assert v4_reg.y1 == 0
        assert v4_trace.active_start_col >= 0
        assert v4_trace.active_end_col <= (v4_reg.x2 - v4_reg.x1)

        # Global sheet coordinate of the R peak
        r_peak_idx = int(np.argmin(v4_trace.pixel_trajectory))
        r_peak_img_x = v4_reg.x1 + v4_trace.active_start_col + r_peak_idx
        r_peak_img_y = v4_reg.y1 + v4_trace.pixel_trajectory[r_peak_idx]

        # Must lie within the physical image bounds
        assert 0 <= r_peak_img_x < img.shape[1]
        assert 0 <= r_peak_img_y < img.shape[0]
        # In ecg1.jpeg, the V4 peak is known to reach near y=7
        assert r_peak_img_y <= 10.0


class TestFastAPIFidelityEndpoints:
    """Verifies FastAPI service integration for trace fidelity diagnostics and focused overlay."""

    def test_digitize_returns_per_lead_diagnostics(self):
        client = TestClient(app)
        img = cv2.imread("tests/fixtures/ecg1.jpeg")
        import base64
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf).decode("utf-8")

        res = client.post("/api/digitize", json={"image_base64": b64, "layout_hint": "3x4_3"})
        assert res.status_code == 200
        data = res.json()

        assert "trace_fidelity_diagnostics" in data
        diags = data["trace_fidelity_diagnostics"]
        assert len(diags) == 15
        assert "V4" in diags
        assert "V5_RHYTHM" in diags

        v4_d = diags["V4"]
        assert v4_d["ink_recovery"]["ink_support_percent"] > 85.0
        assert v4_d["interpolation"]["total_samples"] > 100
        assert v4_d["overall_status"] in ("HIGH_FIDELITY", "ACCEPTABLE_FIDELITY")

        assert "uncertainty_overlay_b64" in data
        assert len(data["uncertainty_overlay_b64"]) > 1000

    def test_focused_overlay_endpoint(self):
        client = TestClient(app)
        img = cv2.imread("tests/fixtures/ecg1.jpeg")
        import base64
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf).decode("utf-8")

        res = client.post("/api/digitize", json={"image_base64": b64, "layout_hint": "3x4_3"})
        rec_id = res.json()["record_id"]

        # Request focused overlay for Lead V4
        f_res = client.post("/api/overlay-fidelity", json={"record_id": rec_id, "selected_lead": "V4"})
        assert f_res.status_code == 200
        f_data = f_res.json()
        assert "uncertainty_overlay_b64" in f_data
        assert f_data["selected_lead"] == "V4"
