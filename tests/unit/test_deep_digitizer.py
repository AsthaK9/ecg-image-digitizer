"""Unit tests for DeepCropDigitizer inference engine."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from ecg_digitization.inference.deep_digitizer import (
    DeepCropDigitizer,
    RealCropDigitizationResult,
    RealCropPreprocessingResult,
)


def test_deep_crop_digitizer_initialization():
    """DeepCropDigitizer correctly loads weights and config from checkpoint."""
    digitizer = DeepCropDigitizer(checkpoint_path="models/ecg_digitizer_unet.pt")
    assert digitizer.model is not None
    assert digitizer.epoch >= 1
    assert digitizer.best_val_r > 0.90


def test_deep_crop_preprocessing_synthetic_image():
    """Preprocessing detects content bbox and outputs 128x256 float tensor."""
    digitizer = DeepCropDigitizer(checkpoint_path="models/ecg_digitizer_unet.pt")

    # Create dummy image with white margin around a center trace
    img = np.full((200, 300, 3), 255, dtype=np.uint8)
    # Draw dark line in center [50:150, 50:250]
    img[95:105, 50:250] = (20, 20, 30)

    prep = digitizer.preprocess_image(img)
    assert isinstance(prep, RealCropPreprocessingResult)
    assert prep.input_tensor.shape == (1, 3, 128, 256)
    assert prep.input_tensor.dtype == torch.float32
    assert prep.content_bbox[2] > prep.content_bbox[0]
    assert prep.content_bbox[3] > prep.content_bbox[1]


def test_deep_crop_digitize_and_export():
    """Runs digitization and exports CSV and overlay visualization."""
    digitizer = DeepCropDigitizer(checkpoint_path="models/ecg_digitizer_unet.pt")

    # Test image
    img = np.full((128, 256, 3), 252, dtype=np.uint8)
    img[60:68, :] = (20, 20, 30)

    result = digitizer.digitize_image(img, sample_name="test_lead")
    assert isinstance(result, RealCropDigitizationResult)
    assert len(result.pixel_coordinates_crop) == 256
    assert result.pixel_coordinates_original.shape == (256, 2)
    assert len(result.column_confidence) == 256
    assert len(result.normalized_amplitude) == 256
    assert len(result.calibrated_voltage_mv) == 256

    # Test CSV export
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "waveform.csv"
        saved_csv = result.save_csv(csv_path)
        assert saved_csv.is_file()
        content = saved_csv.read_text(encoding="utf-8")
        assert "sample_index,pixel_x_orig" in content
        assert "normalized_amplitude" in content

        plot_path = Path(tmp_dir) / "overlay.png"
        saved_plot = result.save_overlay_plot(plot_path)
        assert saved_plot.is_file()
        assert saved_plot.stat().st_size > 1000
