"""ECG Lead Region Detection and Layout Analysis.

Identifies bounding boxes and ordering for standard ECG lead layouts:
- Standard 3x4 + 1 Rhythm Strip (12 leads in 4 columns x 3 rows + full-width bottom Lead II)
- Stacked 12x1 layout (12 continuous rows)

Emits clear warnings for unsupported or ambiguous layouts instead of guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LeadRegion:
    """Bounding box and metadata for a single lead window in an ECG image."""

    lead_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    row: int
    col: int
    is_rhythm: bool = False

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class LeadDetectionResult:
    """Result of lead layout detection."""

    layout_name: str
    is_supported: bool
    leads: Dict[str, LeadRegion]
    warnings: List[str]


class LeadLayoutDetector:
    """Detects lead regions for standard clinical ECG sheet layouts."""

    SUPPORTED_LAYOUTS = ("3x4_1", "3x4", "12x1")

    # Standard 3x4 lead ordering:
    # Col 0: I, II, III
    # Col 1: aVR, aVL, aVF
    # Col 2: V1, V2, V3
    # Col 3: V4, V5, V6
    LAYOUT_3X4_COLUMNS = [
        ["I", "II", "III"],
        ["AVR", "AVL", "AVF"],
        ["V1", "V2", "V3"],
        ["V4", "V5", "V6"],
    ]

    LAYOUT_12X1_ROWS = [
        "I", "II", "III", "AVR", "AVL", "AVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    def detect_layout(
        self,
        image_shape: Tuple[int, ...],
        layout_hint: str = "3x4_1",
        margin_ratio: float = 0.035,
    ) -> LeadDetectionResult:
        """Determines lead region bounding boxes based on image geometry and layout convention.

        Args:
            image_shape: (H, W) or (H, W, C) of the preprocessed image.
            layout_hint: One of '3x4_1' (default clinical sheet), '3x4', or '12x1'.
            margin_ratio: Outer paper margin fraction.

        Returns:
            LeadDetectionResult containing lead bounding boxes or unsupported warning.
        """
        h = image_shape[0]
        w = image_shape[1]
        warnings: List[str] = []

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.8:
            warnings.append(
                f"Image aspect ratio {aspect_ratio:.2f} is unusually tall for a standard ECG sheet."
            )

        if layout_hint not in self.SUPPORTED_LAYOUTS:
            return LeadDetectionResult(
                layout_name=layout_hint,
                is_supported=False,
                leads={},
                warnings=[f"Unsupported ECG layout '{layout_hint}'. Supported: {self.SUPPORTED_LAYOUTS}"],
            )

        margin_x = int(round(w * margin_ratio))
        margin_y = int(round(h * margin_ratio))
        usable_w = w - 2 * margin_x
        usable_h = h - 2 * margin_y

        leads: Dict[str, LeadRegion] = {}

        if layout_hint == "3x4_1":
            # 4 horizontal rows: 3 for 3x4 grid + 1 for bottom rhythm strip
            row_h = usable_h // 4
            col_w = usable_w // 4

            for col_idx, col_leads in enumerate(self.LAYOUT_3X4_COLUMNS):
                x1 = margin_x + col_idx * col_w
                x2 = margin_x + (col_idx + 1) * col_w

                for row_idx, lead_name in enumerate(col_leads):
                    y1 = margin_y + row_idx * row_h
                    y2 = margin_y + (row_idx + 1) * row_h
                    leads[lead_name] = LeadRegion(
                        lead_name=lead_name,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        row=row_idx,
                        col=col_idx,
                        is_rhythm=False,
                    )

            # Row 3: rhythm strip (typically Lead II)
            rhythm_y1 = margin_y + 3 * row_h
            rhythm_y2 = margin_y + 4 * row_h
            leads["II_RHYTHM"] = LeadRegion(
                lead_name="II_RHYTHM",
                x1=margin_x,
                y1=rhythm_y1,
                x2=margin_x + usable_w,
                y2=rhythm_y2,
                row=3,
                col=0,
                is_rhythm=True,
            )

        elif layout_hint == "12x1":
            # 12 horizontal stacked rows
            row_h = usable_h // 12
            for row_idx, lead_name in enumerate(self.LAYOUT_12X1_ROWS):
                y1 = margin_y + row_idx * row_h
                y2 = margin_y + (row_idx + 1) * row_h
                leads[lead_name] = LeadRegion(
                    lead_name=lead_name,
                    x1=margin_x,
                    y1=y1,
                    x2=margin_x + usable_w,
                    y2=y2,
                    row=row_idx,
                    col=0,
                    is_rhythm=False,
                )

        return LeadDetectionResult(
            layout_name=layout_hint,
            is_supported=True,
            leads=leads,
            warnings=warnings,
        )
