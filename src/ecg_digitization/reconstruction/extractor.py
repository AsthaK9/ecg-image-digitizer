"""Classical Computer Vision Waveform Extractor.

Extracts 1D ECG lead trajectories from binary or grayscale lead region crops
using column-wise centroid/median tracking, active trace span bounding,
robust interpolation of missing columns, and baseline estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTrace:
    """Extracted 1D waveform from a single lead region."""

    lead_name: str
    pixel_trajectory: np.ndarray      # y(x) along active trace span (0 at crop top)
    baseline_y: float                 # Estimated or provided isoelectric baseline y
    relative_trajectory: np.ndarray   # y_baseline - y(x) (positive upwards in mV)
    completeness_ratio: float         # Fraction of columns with direct foreground pixels
    active_start_col: int             # First column with trace pixels
    active_end_col: int               # Last column with trace pixels
    crop_width_px: int
    crop_height_px: int


class WaveformExtractor:
    """Extracts 1D ECG waveforms from 2D segmented lead region masks."""

    def __init__(self, method: str = "median") -> None:
        if method not in ("median", "centroid"):
            raise ValueError(f"Method must be 'median' or 'centroid', got {method}")
        self.method = method

    def extract_from_crop(
        self,
        lead_mask: np.ndarray,
        lead_name: str,
        baseline_y: Optional[float] = None,
    ) -> ExtractedTrace:
        """Extracts 1D waveform trace from a 2D lead mask crop.

        Trims unprinted outer box margins to the active trace span [col_min, col_max]
        to ensure exact temporal alignment without boundary dilation.
        """
        h, w = lead_mask.shape[:2]
        if w == 0 or h == 0:
            raise ValueError(f"Empty lead mask provided: shape {lead_mask.shape}")

        # Scan active columns
        active_cols = np.where(lead_mask.sum(axis=0) > 0)[0]

        if len(active_cols) == 0:
            logger.warning("Lead '%s' contains no detected trace pixels; defaulting to center.", lead_name)
            col_min, col_max = 0, w - 1
            interpolated_y = np.full(w, h / 2.0, dtype=np.float32)
            completeness = 0.0
        else:
            col_min = int(active_cols[0])
            col_max = int(active_cols[-1])
            active_w = col_max - col_min + 1

            raw_y = np.full(active_w, np.nan, dtype=np.float32)

            for offset, c in enumerate(range(col_min, col_max + 1)):
                fg_indices = np.where(lead_mask[:, c] > 0)[0]
                if len(fg_indices) > 0:
                    if self.method == "median":
                        raw_y[offset] = float(np.median(fg_indices))
                    else:  # centroid
                        weights = lead_mask[fg_indices, c].astype(np.float32)
                        raw_y[offset] = float(np.sum(fg_indices * weights) / (np.sum(weights) + 1e-6))

            valid_mask = ~np.isnan(raw_y)
            completeness = float(np.sum(valid_mask) / float(active_w))

            if np.sum(valid_mask) == active_w:
                interpolated_y = raw_y
            else:
                x_valid = np.where(valid_mask)[0]
                y_valid = raw_y[valid_mask]
                all_x = np.arange(active_w)
                interpolated_y = np.interp(all_x, x_valid, y_valid).astype(np.float32)

        if baseline_y is None:
            baseline = float(np.median(interpolated_y))
        else:
            baseline = float(baseline_y)

        # In image coordinates, y increases downwards.
        # Positive voltage (R peak) goes upwards -> relative trajectory = baseline - y
        relative_y = baseline - interpolated_y

        return ExtractedTrace(
            lead_name=lead_name,
            pixel_trajectory=interpolated_y,
            baseline_y=baseline,
            relative_trajectory=relative_y,
            completeness_ratio=completeness,
            active_start_col=col_min,
            active_end_col=col_max,
            crop_width_px=w,
            crop_height_px=h,
        )
