"""Synthetic ECG paper image generator module.

Generates realistic paper ECG images from digital 12-lead signals with standard
clinical calibration (25 mm/s, 10 mm/mV), realistic grid patterns, lead labeling,
and configurable degradation (blur, noise, rotation, perspective, lighting).
"""

from __future__ import annotations

from ecg_digitization.synthetic.generator import (
    DegradationConfig,
    ECGLayoutConfig,
    SyntheticECGGenerator,
    SyntheticECGResult,
)

__all__ = [
    "DegradationConfig",
    "ECGLayoutConfig",
    "SyntheticECGGenerator",
    "SyntheticECGResult",
]
