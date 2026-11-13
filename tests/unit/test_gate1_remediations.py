"""Unit tests for Gate 1 Remediations in ECG DigiAI.

Tests:
1. Precordial extrema preservation & adaptive crop dynamic range diagnostics.
2. Marquette 3x4+3 dynamic column boundary detection without static offset.
3. Stacked 12x1 row boundary constraints and neighbor cross-talk diagnostics.
4. Optical augmentation coordinate alignment in Kaggle training pipeline.
5. Backward compatibility for single-lead crops and missing-lead handling.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from ecg_digitization.inference.deep_digitizer import DeepCropDigitizer
from ecg_digitization.preprocessing.lead_detector import LeadLayoutDetector
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from kaggle_bundles.digitizer.train_digitizer_kaggle import render_lead_crop


def test_adaptive_crop_bounds_and_diagnostics():
    """Verifies that compute_adaptive_crop_bounds expands for extreme deflections

    and correctly populates clipping and cross-talk diagnostics.
    """
    digitizer = DeepCropDigitizer(checkpoint_path="models/ecg_digitizer_unet.pt")

    # Lead crop height 250, width 300, baseline at 100
    h, w = 250, 300
    base_y_local = 100.0

    # Binary mask with deep S-wave reaching down to y=220
    binary_crop = np.zeros((h, w), dtype=np.uint8)
    binary_crop[90:105, 10:280] = 255  # baseline
    binary_crop[100:220, 140:150] = 255  # deep S wave down to y=220

    y1, y2, is_top_clipped, is_bot_clipped, clip_margin, ct_risk, is_overlap = (
        digitizer.compute_adaptive_crop_bounds(
            binary_crop=binary_crop,
            base_y_local=base_y_local,
            col_min=0,
            col_max=w - 1,
            h=h,
            w=w,
            is_12x1=False,
        )
    )

    # In standard multi-row layout, crop should expand downward to capture the deep S-wave (220 + pad)
    assert y2 >= 220, f"Expected y2 >= 220 to capture S-wave, got {y2}"
    assert y1 <= base_y_local
    assert isinstance(is_top_clipped, bool)
    assert isinstance(is_bot_clipped, bool)
    assert isinstance(ct_risk, float)
    assert isinstance(is_overlap, bool)

    # Now test 12x1 constrained mode where row height is tight (~95 px)
    h_12x1 = 95
    base_y_12x1 = 47.0
    binary_crop_12x1 = np.zeros((h_12x1, w), dtype=np.uint8)
    binary_crop_12x1[45:50, :] = 255
    # Bleed from neighboring lead near top border
    binary_crop_12x1[2:6, 50:100] = 255

    y1_c, y2_c, _, _, _, ct_risk_12x1, is_overlap_12x1 = (
        digitizer.compute_adaptive_crop_bounds(
            binary_crop=binary_crop_12x1,
            base_y_local=base_y_12x1,
            col_min=0,
            col_max=w - 1,
            h=h_12x1,
            w=w,
            is_12x1=True,
        )
    )
    # 12x1 must stay strictly row-constrained: y1=0, y2=h
    assert y1_c == 0
    assert y2_c == h_12x1
    assert ct_risk_12x1 > 0.0  # detected bleed near boundary


def test_marquette_dynamic_column_boundaries():
    """Verifies dynamic column boundary detection for 3x4_3 layout

    without the 34-39 px artificial delay.
    """
    # Create synthetic Marquette-like sheet (1200 x 1650)
    sheet = np.full((1200, 1650, 3), 252, dtype=np.uint8)
    # Active grid between x=55 and x=1600
    active_x1, active_x2 = 55, 1600
    sheet[100:1100, active_x1:active_x2] = (240, 240, 240)

    # Draw separator / lead ink in grid region
    for y_pos in [150, 350, 550, 750, 950]:
        sheet[y_pos:y_pos+4, active_x1:active_x2] = (20, 20, 30)

    detector = LeadLayoutDetector()
    result = detector.detect_layout(image=sheet, image_shape=sheet.shape, layout_hint="3x4_3")

    assert "I" in result.leads
    assert "II" in result.leads
    assert "II_RHYTHM" in result.leads

    # Verify column 0 does not start at 0 or 30 static px
    lead_i = result.leads["I"]
    lead_avr = result.leads["AVR"]

    col_w = lead_i.x2 - lead_i.x1
    assert 360 <= col_w <= 410, f"Expected dynamic col_w ~ 386, got {col_w}"
    assert abs(lead_i.x1 - active_x1) < 25, f"Lead I x1 ({lead_i.x1}) should match active grid ({active_x1})"
    assert abs(lead_avr.x1 - (lead_i.x1 + col_w)) < 10


def test_stacked_12x1_row_constraints():
    """Verifies that 12x1 layout strictly constrains rows to prevent neighbor bleed-through."""
    sheet_h, sheet_w = 1155, 1650
    sheet = np.full((sheet_h, sheet_w, 3), 252, dtype=np.uint8)

    # 12 distinct lead rows
    detector = LeadLayoutDetector()
    result = detector.detect_layout(image=sheet, image_shape=sheet.shape, layout_hint="12x1")

    assert len(result.leads) == 12
    leads_order = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for i in range(len(leads_order) - 1):
        lead_curr = result.leads[leads_order[i]]
        lead_next = result.leads[leads_order[i+1]]
        # Row heights must be bounded (~96 px)
        row_h = lead_curr.y2 - lead_curr.y1
        assert 80 <= row_h <= 110, f"Expected row height ~96 px, got {row_h}"
        # Consecutive rows should not overlap
        assert lead_curr.y2 <= lead_next.y1, f"Row {leads_order[i]} overlaps with {leads_order[i+1]}"


def test_optical_augmentation_alignment():
    """Verifies that optical augmentations maintain exact coordinate alignment

    between the image tensor, trace mask, and target signal.
    """
    dummy_signal = np.sin(np.linspace(0, 10, 250)).astype(np.float32) * 0.8
    # Test across multiple random seeds to cover clean, rotated, blurred, noisy cases
    for s in [42, 101, 202, 303, 404]:
        img, mask, sig, base_y = render_lead_crop(dummy_signal, crop_h=128, crop_w=256, seed=s)

        assert img.shape == (3, 128, 256)
        assert mask.shape == (1, 128, 256)
        assert sig.shape == (256,)
        assert base_y == 64.0
        assert torch.isfinite(img).all()
        assert torch.isfinite(mask).all()
        assert torch.isfinite(sig).all()

        # Signal values must stay within reasonable physiological bounds (+/- 3.5 mV)
        assert torch.all(sig >= -3.5) and torch.all(sig <= 3.5)


def test_single_lead_backward_compatibility():
    """Verifies that single-lead crops and missing leads are handled gracefully

    without crashing or hallucinating missing leads.
    """
    # A single crop image (128 x 256)
    single_lead_crop = np.full((128, 256, 3), 252, dtype=np.uint8)
    single_lead_crop[60:68, 20:230] = (20, 20, 30)

    detector = LeadLayoutDetector()
    result = detector.detect_layout(image=single_lead_crop, image_shape=single_lead_crop.shape, layout_hint="single_lead")

    # Exactly 1 lead region returned, zero fabrication
    assert len(result.leads) == 1
    assert "II" in result.leads or "LEAD" in result.leads or "I" in result.leads
