"""Unit tests for digitizer pipeline and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_digitization.evaluation.metrics import (
    compute_lead_metrics,
    evaluate_digitization_fidelity,
)
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from ecg_digitization.reconstruction.calibrator import CalibrationConfig
from ecg_digitization.reconstruction.extractor import WaveformExtractor
from ecg_digitization.synthetic.generator import SyntheticECGGenerator


def test_waveform_extractor() -> None:
    # 50 rows x 100 cols binary mask with horizontal line at row 25
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[25, 10:90] = 255

    extractor = WaveformExtractor(method="median")
    trace = extractor.extract_from_crop(mask, lead_name="I", baseline_y=25.0)

    assert trace.lead_name == "I"
    assert trace.active_start_col == 10
    assert trace.active_end_col == 89
    assert trace.completeness_ratio == 1.0
    # Line is exactly on baseline 25, so relative trajectory should be ~0
    np.testing.assert_allclose(trace.relative_trajectory, 0.0, atol=1e-3)


def test_compute_lead_metrics() -> None:
    t = np.linspace(0, 1, 100)
    sig = np.sin(2 * np.pi * 5 * t)

    metrics = compute_lead_metrics(ground_truth=sig, reconstructed=sig, lead_name="I")
    assert pytest.approx(metrics.pearson_r, rel=1e-3) == 1.0
    assert pytest.approx(metrics.rmse_mv, abs=1e-4) == 0.0
    assert metrics.snr_db > 50.0


def test_end_to_end_digitizer_pipeline() -> None:
    # Generate clean synthetic ECG
    dummy_signal = np.sin(np.linspace(0, 20 * np.pi, 1000)).reshape(1, -1)
    dummy_signal = np.repeat(dummy_signal, 12, axis=0) * 0.5

    gen = SyntheticECGGenerator()
    synth_res = gen.generate(dummy_signal, sampling_rate_hz=100.0, seed=42)

    pipeline = ECGDigitizerPipeline()
    res = pipeline.digitize(synth_res.image, layout_hint="3x4_1")

    assert res.success is True
    assert len(res.extracted_traces) == 13
    assert len(res.reconstructed_ecg.leads) == 13
    assert "I" in res.reconstructed_ecg.leads
    assert len(res.reconstructed_ecg.leads["I"].voltage_mv) == 250
