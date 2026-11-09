"""ECG Neural Network Models."""

from __future__ import annotations

from ecg_digitization.models.classifier_1d import ECGResNet1D, ResBlock1D

__all__ = [
    "ECGResNet1D",
    "ResBlock1D",
]
