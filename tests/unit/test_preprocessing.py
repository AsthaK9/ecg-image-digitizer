"""Unit tests for the image preprocessing and lead detection modules."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_digitization.preprocessing.enhancement import (
    binarize_trace,
    deskew_image,
    enhance_contrast,
    estimate_skew_angle,
    load_and_validate_image,
    suppress_grid,
    to_grayscale,
)
from ecg_digitization.preprocessing.lead_detector import (
    LeadDetectionResult,
    LeadLayoutDetector,
)


def test_load_and_validate_image() -> None:
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    validated = load_and_validate_image(img)
    assert validated.shape == (100, 100, 3)

    with pytest.raises(ValueError):
        load_and_validate_image(np.ones((20, 20, 3), dtype=np.uint8))


def test_grayscale_and_contrast() -> None:
    img = np.ones((100, 100, 3), dtype=np.uint8) * 150
    gray = to_grayscale(img)
    assert gray.shape == (100, 100)
    enhanced = enhance_contrast(gray)
    assert enhanced.shape == (100, 100)


def test_grid_suppression() -> None:
    # White background with pink grid line (BGR: 200, 200, 255)
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    img[25, :, :] = [200, 200, 255]  # pink line
    img[:, 25, :] = [20, 20, 20]     # black trace line

    suppressed = suppress_grid(img)
    assert suppressed.shape == (50, 50)
    # The black trace line should remain dark
    assert suppressed[10, 25] < 50


def test_lead_layout_detector() -> None:
    detector = LeadLayoutDetector()
    shape = (1200, 1600, 3)

    res = detector.detect_layout(shape, layout_hint="3x4_1")
    assert res.is_supported is True
    assert len(res.leads) == 13
    assert "I" in res.leads
    assert "II_RHYTHM" in res.leads

    # Test unsupported layout
    unsupported_res = detector.detect_layout(shape, layout_hint="unsupported_custom")
    assert unsupported_res.is_supported is False
    assert len(unsupported_res.warnings) > 0
