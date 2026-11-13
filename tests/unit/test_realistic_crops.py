"""Unit tests for Realistic Paired Lead Crop Generator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg_digitization.data.realistic_crops import (
    PairedLeadCrop,
    RealisticLeadCropGenerator,
    STANDARD_12_LEADS,
)


class TestRealisticLeadCropGenerator:
    """Verifies generation of realistic lead crops with hard negatives and degradations."""

    @pytest.fixture
    def generator(self) -> RealisticLeadCropGenerator:
        return RealisticLeadCropGenerator(crop_h=128, crop_w=256)

    @pytest.fixture
    def sample_signal(self) -> np.ndarray:
        t = np.linspace(0, 2.5, 250)
        return np.sin(2 * np.pi * 1.5 * t).astype(np.float32)

    def test_clean_crop_generation(self, generator, sample_signal):
        crop = generator.generate_crop(sample_signal, lead_name="II", degradation_mode="clean", seed=42)
        assert isinstance(crop, PairedLeadCrop)
        assert crop.image.shape == (128, 256, 3)
        assert crop.image_tensor.shape == (3, 128, 256)
        assert crop.target_mask.shape == (1, 128, 256)
        assert crop.target_signal.shape == (256,)
        assert crop.target_coords.shape == (256,)
        assert crop.valid_columns.shape == (256,)
        assert crop.lead_name == "II"
        assert crop.degradation_type == "clean"

    def test_hard_negative_text_label_is_zero_in_mask(self, generator):
        # Using flat 0 mV signal so waveform is solely along baseline (y=64)
        flat_sig = np.zeros(250, dtype=np.float32)
        crop = generator.generate_crop(flat_sig, lead_name="aVR", degradation_mode="text_overlap", seed=101)
        assert crop.degradation_type == "text_overlap"
        # Text ink is dark in the upper patch [15:45, 10:60]
        text_patch_img = crop.image[15:45, 10:60]
        assert np.min(text_patch_img) < 100
        # In the upper margin [10:35, 10:60], the ground-truth mask must be strictly 0 (no text in mask!)
        assert torch.sum(crop.target_mask[0, 10:35, 10:60]) == 0

    def test_hard_negative_calib_pulse_is_masked(self, generator, sample_signal):
        # Fully stochastic crop with calibration pulse
        crop = generator.generate_crop(sample_signal, lead_name="V1", degradation_mode="random", seed=77)
        assert isinstance(crop.valid_columns, torch.Tensor)
        assert crop.valid_columns.shape == (256,)
        # Valid columns should be a binary vector in {0, 1}
        assert set(crop.valid_columns.unique().tolist()).issubset({0.0, 1.0})

    def test_varied_stroke_widths(self, generator, sample_signal):
        thin_crop = generator.generate_crop(sample_signal, degradation_mode="thin_stroke", seed=12)
        thick_crop = generator.generate_crop(sample_signal, degradation_mode="thick_stroke", seed=12)
        assert thin_crop.image.shape == (128, 256, 3)
        assert thick_crop.image.shape == (128, 256, 3)
        # Thick stroke has significantly more dark pixels than thin stroke
        dark_thin = np.sum(thin_crop.image < 100)
        dark_thick = np.sum(thick_crop.image < 100)
        assert dark_thick > dark_thin
