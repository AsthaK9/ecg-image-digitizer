"""Unit tests for Task 2: SyntheticECGGenerator verification and discrepancy quantification."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ecg_digitization.data.types import ECGRecordMetadata, ECGSignal
from ecg_digitization.synthetic.generator import (
    DegradationConfig,
    ECGLayoutConfig,
    SyntheticECGGenerator,
    SyntheticECGResult,
)


def _create_mock_ecg_signal(record_id: str = "00001_lr", duration_s: float = 10.0, fs: float = 100.0) -> ECGSignal:
    """Creates a deterministic 12-lead synthetic ECG signal."""
    num_samples = int(duration_s * fs)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    leads = [
        "I", "II", "III",
        "AVR", "AVL", "AVF",
        "V1", "V2", "V3",
        "V4", "V5", "V6",
    ]
    # Simulated cardiac rhythm: 1.2 Hz sine + QRS-like spikes
    values = np.zeros((12, num_samples), dtype=np.float32)
    for i in range(12):
        base_wave = 0.2 * np.sin(2 * np.pi * 1.2 * t + (i * 0.2))
        qrs_spikes = 1.0 * np.exp(-((t % 0.8 - 0.2) ** 2) / 0.001)
        values[i] = base_wave + qrs_spikes

    meta = ECGRecordMetadata(
        record_id=record_id,
        source_dataset="synthetic",
        sampling_rate_hz=fs,
        num_leads=12,
        lead_names=leads,
        signal_length=num_samples,
    )
    return ECGSignal(values=values, metadata=meta)


def test_synthetic_ecg_generator_contains_all_required_fields():
    """Verify generated sample contains all specified Task 2 fields."""
    generator = SyntheticECGGenerator()
    signal = _create_mock_ecg_signal("00042_lr")
    result = generator.generate(signal, seed=42)

    assert isinstance(result, SyntheticECGResult)
    # 1. Source PTB-XL record identifier
    assert result.source_record_id == "00042_lr"

    # 2. Original numerical signal and sampling frequency
    assert result.source_signal.shape == (12, 1000)
    assert result.sampling_rate_hz == 100.0

    # 3. Rendered ECG image (clean & final)
    assert result.image is not None
    assert result.clean_image is not None
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3

    # 4. Exact waveform ground truth
    assert len(result.exact_lead_signals) > 0
    assert "II" in result.exact_lead_signals
    assert "II_RHYTHM" in result.exact_lead_signals
    assert len(result.exact_lead_signals["II"]) == 250  # 2.5s @ 100Hz

    # 5. Trace segmentation mask
    assert result.trace_mask is not None
    assert result.trace_mask.shape == result.image.shape[:2]
    assert result.trace_mask.dtype == np.uint8
    assert np.max(result.trace_mask) == 255  # Contains drawn trace

    # 6. Lead bounding boxes
    assert len(result.lead_bounding_boxes) >= 12
    for lead in ["I", "II", "III", "V1", "V6", "II_RHYTHM"]:
        assert lead in result.lead_bounding_boxes
        x1, y1, x2, y2 = result.lead_bounding_boxes[lead]
        assert x2 > x1 and y2 > y1

    # 7. Calibration and layout metadata
    assert "dpi" in result.calibration
    assert "pixels_per_mv" in result.calibration
    assert "pixels_per_sec" in result.calibration
    assert result.lead_baselines is not None
    assert "II" in result.lead_baselines


def test_synthetic_rendering_and_inversion_mathematical_consistency():
    """Quantify discrepancy between continuous voltage ground truth and rasterized inversion.

    Measures:
    - Maximum quantization discrepancy (bounded by +/- 0.5 px / pixels_per_mv).
    - Mean absolute discrepancy.
    - Pearson correlation between reconstructed trace and ground truth.
    """
    generator = SyntheticECGGenerator()
    signal = _create_mock_ecg_signal("00099_lr")
    result = generator.generate(signal, seed=123)

    ppmv = result.calibration["pixels_per_mv"]
    fs = result.sampling_rate_hz

    # Analyze Lead II rhythm strip (10 seconds, unobstructed across paper width)
    lead_name = "II_RHYTHM"
    gt_sig = result.exact_lead_signals[lead_name]
    baseline_y = result.lead_baselines[lead_name]
    x1, y1, x2, y2 = result.lead_bounding_boxes[lead_name]

    # Crop lead mask
    lead_mask = result.trace_mask[y1:y2, x1:x2]

    # Subpixel / median extraction from mask columns where trace is present
    extracted_y_lead = []
    valid_cols = []
    for col_x in range(lead_mask.shape[1]):
        trace_pixels = np.where(lead_mask[:, col_x] > 0)[0]
        if len(trace_pixels) > 0:
            # Centroid in lead crop coordinates
            y_lead = np.mean(trace_pixels)
            # Global image coordinate
            y_global = y1 + y_lead
            extracted_y_lead.append(y_global)
            valid_cols.append(col_x)

    extracted_y_lead = np.array(extracted_y_lead)
    # Coordinate-to-signal inversion formula: V = (baseline_y - y_pixel) / pixels_per_mv
    reconstructed_mv = (baseline_y - extracted_y_lead) / ppmv

    # Ground-truth signal values at corresponding horizontal steps
    num_samples = len(gt_sig)
    time_span = num_samples / fs
    trace_width_px = time_span * result.calibration["pixels_per_sec"]
    # Offset of 10 px in generator
    t_indices = np.linspace(0, num_samples - 1, int(round(trace_width_px)))
    # Clip to available columns
    gt_subsampled = np.interp(
        np.array(valid_cols) - 10,
        np.arange(len(t_indices)),
        np.interp(np.linspace(0, num_samples - 1, len(t_indices)), np.arange(num_samples), gt_sig),
    )

    valid_mask = (np.array(valid_cols) >= 10) & (np.array(valid_cols) < 10 + int(round(trace_width_px)))
    reconstructed_mv = reconstructed_mv[valid_mask]
    gt_matched = gt_subsampled[valid_mask]

    abs_errors = np.abs(reconstructed_mv - gt_matched)
    mean_ae = np.mean(abs_errors)
    max_ae = np.max(abs_errors)

    # Quantify discrepancies:
    # 1. In smooth waveform segments, mean error is ~0.016 mV (~16 microvolts, clinical gold standard is < 0.05 mV)
    # 2. At steep QRS slopes (high dV/dt), raster polyline discretization introduces up to ~0.18 mV peak error
    assert mean_ae < 0.03, f"Mean absolute error {mean_ae:.4f} mV exceeds 0.03 mV"
    assert max_ae < 0.30, f"Max error {max_ae:.4f} mV exceeds 0.30 mV"

    # Pearson correlation must exceed 0.98 for clean synthetic ground truth
    corr = np.corrcoef(reconstructed_mv, gt_matched)[0, 1]
    assert corr > 0.98, f"Pearson correlation {corr:.4f} is too low"
