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

import cv2
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
    baseline_y_hint: Optional[float] = None

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

    SUPPORTED_LAYOUTS = ("3x4_1", "3x4_3", "3x4", "12x1", "single_lead")

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

    _cached_layout_service = None

    @classmethod
    def get_layout_service(cls):
        if cls._cached_layout_service is None:
            try:
                from ecg_digitization.models.layout_classifier import LayoutDetectorService
                cls._cached_layout_service = LayoutDetectorService(model_path="models/ecg_layout_classifier.pt")
            except Exception as exc:
                logger.warning("Could not instantiate LayoutDetectorService singleton: %s", exc)
        return cls._cached_layout_service

    def detect_active_canvas(self, image: np.ndarray) -> Tuple[int, int, int, int, str]:
        """Detects the active waveform grid area, suppressing clinical report headers and footers."""
        try:
            svc = self.get_layout_service()
            if svc is not None:
                return svc.detect_active_canvas(image)
            from ecg_digitization.models.layout_classifier import LayoutDetectorService
            return LayoutDetectorService.detect_active_canvas(image)
        except Exception:
            h, w = image.shape[:2]
            return int(0.02 * h), int(0.98 * h), int(0.02 * w), int(0.98 * w), "fallback"

    def infer_layout_detailed(self, image: np.ndarray) -> Dict[str, Any]:
        """Infers the ECG sheet layout with confidence, ambiguity detection,
        class probabilities, and active canvas metadata."""
        try:
            svc = self.get_layout_service()
            if svc is not None:
                return svc.detect_layout_detailed(image)
            fallback = self.infer_layout(image)
            return {
                "predicted_layout": fallback,
                "confidence": 0.70,
                "is_ambiguous": True,
                "class_probabilities": {fallback: 0.70},
                "n_rows": 4,
                "notes": "LayoutDetectorService unavailable",
            }
        except Exception as exc:
            logger.warning("LayoutDetectorService failed: %s; using heuristic fallback.", exc)
            fallback = self.infer_layout(image)
            return {
                "predicted_layout": fallback,
                "confidence": 0.70,
                "is_ambiguous": True,
                "class_probabilities": {fallback: 0.70},
                "n_rows": 4 if fallback == "3x4_1" else 1,
                "active_canvas": (0, image.shape[0], 0, image.shape[1]),
                "notes": f"Fallback heuristic layout detection: {exc}",
            }

    def infer_layout(self, image: np.ndarray) -> str:
        """Infers the most probable ECG sheet layout from image geometry, row projections,
        and trained layout classification service."""
        detailed = self.infer_layout_detailed(image)
        return detailed["predicted_layout"]

    def detect_layout(
        self,
        image_shape: Union[Tuple[int, ...], np.ndarray],
        layout_hint: str = "3x4_1",
        margin_ratio: float = 0.035,
        image: Optional[np.ndarray] = None,
    ) -> LeadDetectionResult:
        """Determines lead region bounding boxes based on image geometry and layout convention.

        Args:
            image_shape: (H, W) or (H, W, C) of the preprocessed image, or numpy array.
            layout_hint: One of '3x4_1' (default clinical sheet), '3x4', '3x4_3', '12x1', 'single_lead'.
            margin_ratio: Outer paper margin fraction.
            image: Optional preprocessed image for dynamic peak-anchored baseline detection.

        Returns:
            LeadDetectionResult containing lead bounding boxes or unsupported warning.
        """
        if isinstance(image_shape, np.ndarray):
            if image is None:
                image = image_shape
            image_shape = image_shape.shape

        h = int(image_shape[0])
        w = int(image_shape[1])
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

        if image is not None:
            canv_y1, canv_y2, canv_x1, canv_x2, _ = self.detect_active_canvas(image)
            margin_y = canv_y1
            margin_x = canv_x1
            usable_w = max(50, canv_x2 - canv_x1)
            usable_h = max(50, canv_y2 - canv_y1)
        else:
            margin_x = int(round(w * margin_ratio))
            margin_y = int(round(h * margin_ratio))
            usable_w = w - 2 * margin_x
            usable_h = h - 2 * margin_y

        leads: Dict[str, LeadRegion] = {}

        if layout_hint == "3x4_1":
            # 4 horizontal rows: 3 for 3x4 grid + 1 for bottom rhythm strip
            col_w = usable_w / 4.0

            # Dynamic row baseline and boundary detection when image is provided
            peaks = []
            if image is not None:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                    bin_inv = (gray < 150).astype(np.uint8) * 255
                    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.20)), 1))
                    solid_hlines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, horiz_kernel)
                    waveform_ink = cv2.subtract(bin_inv, solid_hlines)

                    canvas_ink = waveform_ink[margin_y : margin_y + usable_h, margin_x : margin_x + usable_w]
                    proj = np.sum(canvas_ink > 0, axis=1).astype(np.float32)
                    win_len = max(5, int(usable_h // 35))
                    smooth = np.convolve(proj, np.ones(win_len) / float(win_len), mode="same")
                    from scipy.signal import find_peaks

                    min_dist = max(15, int(usable_h / 7))
                    det_peaks, _ = find_peaks(
                        smooth,
                        distance=min_dist,
                        prominence=max(2.0, float(np.max(smooth)) * 0.08),
                    )
                    abs_det = [int(p + margin_y) for p in det_peaks]
                    valid = [p for p in abs_det if margin_y <= p <= margin_y + usable_h]
                    if len(valid) == 4:
                        peaks = valid
                except Exception:
                    pass

            if len(peaks) == 4:
                div01 = (peaks[0] + peaks[1]) // 2
                div12 = (peaks[1] + peaks[2]) // 2
                div23 = (peaks[2] + peaks[3]) // 2
                row_bounds = [
                    (margin_y, div01),
                    (div01, div12),
                    (div12, div23),
                    (div23, margin_y + usable_h),
                ]
                row_baselines = [float(p) for p in peaks]
            else:
                row_h = usable_h // 4
                row_bounds = [
                    (margin_y + i * row_h, margin_y + (i + 1) * row_h)
                    for i in range(4)
                ]
                row_baselines = [float(margin_y + (i + 0.5) * row_h) for i in range(4)]

            for col_idx, col_leads in enumerate(self.LAYOUT_3X4_COLUMNS):
                x1 = int(round(margin_x + col_idx * col_w))
                x2 = int(round(margin_x + (col_idx + 1) * col_w))

                for row_idx, lead_name in enumerate(col_leads):
                    y1, y2 = row_bounds[row_idx]
                    local_base = row_baselines[row_idx] - float(y1)
                    leads[lead_name] = LeadRegion(
                        lead_name=lead_name,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        row=row_idx,
                        col=col_idx,
                        is_rhythm=False,
                        baseline_y_hint=local_base,
                    )

            # Row 3: rhythm strip (typically Lead II)
            rhythm_y1, rhythm_y2 = row_bounds[3]
            rhythm_base = row_baselines[3] - float(rhythm_y1)
            leads["II_RHYTHM"] = LeadRegion(
                lead_name="II_RHYTHM",
                x1=margin_x,
                y1=rhythm_y1,
                x2=margin_x + usable_w,
                y2=rhythm_y2,
                row=3,
                col=0,
                is_rhythm=True,
                baseline_y_hint=rhythm_base,
            )

        elif layout_hint == "3x4_3":
            # 6 horizontal rows: 3 for 3x4 grid + 3 bottom rhythm strips (e.g. Marquette MAC format: V1, II, V5)
            # Uses dynamic horizontal and vertical structural geometry when image is provided.
            m_x = int(round(w * 0.018)) if margin_ratio == 0.035 else margin_x
            x_start = m_x
            x_end = w - m_x

            # 1. Dynamic horizontal column boundary detection
            if image is not None:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                    ink_mask = (gray < 110)
                    col_proj = np.sum(ink_mask[:int(h * 0.5), :], axis=0)
                    win_w = max(5, int(w // 100))
                    smooth_col = np.convolve(col_proj, np.ones(win_w) / float(win_w), mode="same")
                    active_cols = np.where(smooth_col > 1.0)[0]
                    if len(active_cols) > 50:
                        det_x_start = int(active_cols[0])
                        det_x_end = int(active_cols[-1])
                        # Ensure reasonable bounds (between 0% and 10% on left, >90% on right)
                        if 0 <= det_x_start < 0.10 * w and 0.88 * w < det_x_end <= w:
                            x_start = det_x_start
                            x_end = det_x_end
                except Exception:
                    pass

            u_w = max(100, x_end - x_start + 1)
            col_w = u_w / 4.0

            # 2. Dynamic row baseline and boundary detection
            peaks = []
            if image is not None:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                    ink = (gray < 110).astype(np.uint8)
                    proj = np.sum(ink, axis=1)
                    win_len = max(5, int(h // 40))
                    smooth = np.convolve(proj, np.ones(win_len) / float(win_len), mode="same")
                    from scipy.signal import find_peaks
                    min_dist = max(15, int(h / 14))
                    det_peaks, _ = find_peaks(smooth, distance=min_dist, prominence=max(10, float(np.max(smooth)) * 0.12))
                    valid = [int(p) for p in det_peaks if 0.04 * h <= p <= 0.96 * h]
                    if len(valid) == 7 and valid[0] < margin_y + 25:
                        valid = valid[1:]
                    if len(valid) == 6:
                        peaks = valid
                except Exception:
                    pass

            if len(peaks) == 6:
                row_bounds = [
                    (0, (peaks[0] + peaks[1]) // 2),
                    ((peaks[0] + peaks[1]) // 2, (peaks[1] + peaks[2]) // 2),
                    ((peaks[1] + peaks[2]) // 2, (peaks[2] + peaks[3]) // 2),
                    ((peaks[2] + peaks[3]) // 2, (peaks[3] + peaks[4]) // 2),
                    ((peaks[3] + peaks[4]) // 2, (peaks[4] + peaks[5]) // 2),
                    ((peaks[4] + peaks[5]) // 2, min(h - int(round(h * 0.033)), int(peaks[5] + (peaks[5] - peaks[4]) * 0.45))),
                ]
                row_baselines = [float(p) for p in peaks]
            else:
                fracs = [0.085, 0.255, 0.425, 0.595, 0.765, 0.935]
                centers = [int(round(h * f)) for f in fracs]
                row_bounds = [
                    (0, (centers[0] + centers[1]) // 2),
                    ((centers[0] + centers[1]) // 2, (centers[1] + centers[2]) // 2),
                    ((centers[1] + centers[2]) // 2, (centers[2] + centers[3]) // 2),
                    ((centers[2] + centers[3]) // 2, (centers[3] + centers[4]) // 2),
                    ((centers[3] + centers[4]) // 2, (centers[4] + centers[5]) // 2),
                    ((centers[4] + centers[5]) // 2, min(h - int(round(h * 0.033)), int(round(h * 0.967)))),
                ]
                row_baselines = [float(c) for c in centers]

            for col_idx, col_leads in enumerate(self.LAYOUT_3X4_COLUMNS):
                x1 = int(round(x_start + col_idx * col_w))
                x2 = int(round(x_start + (col_idx + 1) * col_w))

                for row_idx, lead_name in enumerate(col_leads):
                    y1, y2 = row_bounds[row_idx]
                    local_base = row_baselines[row_idx] - float(y1)
                    leads[lead_name] = LeadRegion(
                        lead_name=lead_name,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        row=row_idx,
                        col=col_idx,
                        is_rhythm=False,
                        baseline_y_hint=local_base,
                    )

            # Rows 3, 4, 5: 3 continuous rhythm strips (V1, II, V5)
            rhythms = [("V1_RHYTHM", 3), ("II_RHYTHM", 4), ("V5_RHYTHM", 5)]
            for r_name, r_idx in rhythms:
                r_y1, r_y2 = row_bounds[r_idx]
                r_base = row_baselines[r_idx] - float(r_y1)
                leads[r_name] = LeadRegion(
                    lead_name=r_name,
                    x1=x_start,
                    y1=r_y1,
                    x2=x_end,
                    y2=r_y2,
                    row=r_idx,
                    col=0,
                    is_rhythm=True,
                    baseline_y_hint=r_base,
                )

        elif layout_hint == "3x4":
            # 3 horizontal rows for standard 3x4 grid without rhythm strip
            row_h = usable_h // 3
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
                        baseline_y_hint=float(row_h // 2),
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
                    baseline_y_hint=float(row_h // 2),
                )

        elif layout_hint == "single_lead":
            # Single lead or beat crop: entire crop treated as Lead II
            leads["II"] = LeadRegion(
                lead_name="II",
                x1=margin_x,
                y1=margin_y,
                x2=max(margin_x + 1, w - margin_x),
                y2=max(margin_y + 1, h - margin_y),
                row=0,
                col=0,
                is_rhythm=False,
                baseline_y_hint=float(usable_h // 2),
            )

        return LeadDetectionResult(
            layout_name=layout_hint,
            is_supported=True,
            leads=leads,
            warnings=warnings,
        )
