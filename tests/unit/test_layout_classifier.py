"""Unit tests for ECG Sheet Layout Classifier and LayoutDetectorService."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg_digitization.models.layout_classifier import (
    ECGLayoutCNN,
    LAYOUT_CLASSES,
    LayoutDetectorService,
)


class TestECGLayoutClassifier:
    """Verifies layout CNN forward pass and hybrid layout detector service."""

    def test_cnn_architecture_forward_pass(self):
        model = ECGLayoutCNN(num_classes=len(LAYOUT_CLASSES))
        x = torch.randn(2, 1, 128, 192)
        logits = model(x)
        assert logits.shape == (2, len(LAYOUT_CLASSES))

    def test_layout_service_single_lead_detection(self):
        svc = LayoutDetectorService(model_path="models/ecg_layout_classifier.pt")
        # Single lead crop shape
        img = np.full((288, 432, 3), 255, dtype=np.uint8)
        layout, conf, probs = svc.detect_layout(img)
        assert layout == "single_lead"
        assert conf >= 0.90
        assert probs["single_lead"] >= 0.90

    def test_layout_service_real_fixtures(self):
        svc = LayoutDetectorService(model_path="models/ecg_layout_classifier.pt")

        # Test on ecg1.jpeg fixture if available
        import cv2
        from pathlib import Path

        p_ecg1 = Path("tests/fixtures/ecg1.jpeg")
        if p_ecg1.is_file():
            img1 = cv2.imread(str(p_ecg1))
            l1, c1, _ = svc.detect_layout(img1)
            assert l1 in ("3x4_3", "3x4_1")

        p_n10007 = Path("data/real_ecg_eval/samples/N10007.png")
        if p_n10007.is_file():
            img_n = cv2.imread(str(p_n10007))
            ln, cn, _ = svc.detect_layout(img_n)
            assert ln == "single_lead"
