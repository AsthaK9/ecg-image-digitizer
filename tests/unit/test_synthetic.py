"""Unit tests for the synthetic ECG generator."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_digitization.synthetic.generator import (
    DegradationConfig,
    ECGLayoutConfig,
    SyntheticECGGenerator,
    SyntheticECGResult,
)


def test_synthetic_ecg_generator_clean() -> None:
    # 12 leads, 1000 samples (10 seconds @ 100 Hz)
    dummy_signal = np.sin(np.linspace(0, 20 * np.pi, 1000)).reshape(1, -1)
    dummy_signal = np.repeat(dummy_signal, 12, axis=0)

    generator = SyntheticECGGenerator()
    result = generator.generate(dummy_signal, sampling_rate_hz=100.0, seed=42)

    assert isinstance(result, SyntheticECGResult)
    assert result.image.ndim == 3
    assert result.image.shape == result.clean_image.shape
    assert result.trace_mask.shape == result.image.shape[:2]
    assert np.all(result.image == result.clean_image)  # No degradation by default
    assert (result.trace_mask > 0).sum() > 0

    # Verify standard leads and bounding boxes
    assert "I" in result.lead_bounding_boxes
    assert "II" in result.lead_bounding_boxes
    assert "II_RHYTHM" in result.lead_bounding_boxes
    assert result.calibration["speed_mm_s"] == 25.0
    assert result.calibration["gain_mm_mv"] == 10.0


def test_synthetic_ecg_generator_degradation() -> None:
    dummy_signal = np.zeros((12, 1000), dtype=np.float32)
    generator = SyntheticECGGenerator()

    deg = DegradationConfig(
        enabled=True,
        rotation_deg=2.0,
        gaussian_noise_sigma=5.0,
        blur_kernel_size=3,
        shadow_gradient=True,
    )
    result = generator.generate(dummy_signal, sampling_rate_hz=100.0, degradation=deg, seed=42)

    # With noise/blur, degraded image must differ from clean image
    assert not np.array_equal(result.image, result.clean_image)
    assert result.image.shape == result.clean_image.shape
