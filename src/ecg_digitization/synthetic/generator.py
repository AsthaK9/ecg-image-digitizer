"""Synthetic ECG Image Generator with Paired Ground Truth.

Simulates clinical paper ECGs (standard 25 mm/s, 10 mm/mV) with:
1. Standard 12-lead layout (3x4 + Lead II rhythm strip) or stacked 12x1 layout.
2. Accurate millimeter grid lines (1 mm minor, 5 mm major).
3. 1 mV calibration pulse.
4. Paired trace segmentation mask and lead bounding box metadata.
5. Configurable photographic/scanning degradations:
   - Gaussian noise
   - Gaussian blur
   - Affine rotation
   - Perspective distortion
   - Illumination/brightness gradients
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ecg_digitization.data.types import ECGSignal

logger = logging.getLogger(__name__)


@dataclass
class DegradationConfig:
    """Parameters for simulating real-world scanning and photographic degradation."""

    enabled: bool = False
    rotation_deg: float = 0.0          # Deskew test angle: e.g. -3.0 to +3.0
    perspective_factor: float = 0.0    # 0.0 = none, 0.05 = slight phone angle
    gaussian_noise_sigma: float = 0.0  # e.g. 5.0 to 15.0
    blur_kernel_size: int = 0          # 0 = none, 3 or 5 for optical blur
    brightness_contrast_jitter: bool = False
    shadow_gradient: bool = False      # Non-uniform paper illumination
    jpeg_quality: Optional[int] = None # JPEG compression quality (e.g. 35-75)
    grid_color_style: str = "standard" # standard, faded, dark, or monochrome
    clinical_header: bool = False      # Realistic clinical hospital report header & table rules
    vertical_dividers: bool = False    # Realistic vertical column separator rules
    footer_annotation: bool = False    # Speed/gain/filter footer text annotations


@dataclass
class ECGLayoutConfig:
    """Geometry and clinical formatting for synthetic ECG sheets."""

    dpi: int = 150                      # Standard scanning resolution
    paper_width_mm: float = 280.0       # Standard A4/Letter landscape ECG paper width
    paper_height_mm: float = 210.0      # Standard ECG paper height
    speed_mm_s: float = 25.0            # Standard ECG paper speed (25 mm/s)
    gain_mm_mv: float = 10.0            # Standard clinical gain (10 mm/mV)
    line_thickness: int = 2             # Waveform pen trace thickness in pixels
    grid_minor_color: Tuple[int, int, int] = (215, 215, 255)  # BGR light pink (R=255, G=215, B=215)
    grid_major_color: Tuple[int, int, int] = (140, 140, 235)  # BGR darker pink/red (R=235, G=140, B=140)
    trace_color: Tuple[int, int, int] = (15, 15, 25)          # Near-black ink
    include_calibration_pulse: bool = True
    include_clinical_header: bool = False
    include_vertical_dividers: bool = False
    include_footer_annotation: bool = False


@dataclass
class SyntheticECGResult:
    """Output from the synthetic generation pipeline."""

    image: np.ndarray                   # Final image (H, W, 3) BGR uint8
    clean_image: np.ndarray             # Clean rendered image before degradation
    trace_mask: np.ndarray              # Ground-truth binary mask (H, W) uint8 (255=trace)
    lead_bounding_boxes: Dict[str, Tuple[int, int, int, int]]  # name -> (x1, y1, x2, y2)
    lead_baselines: Dict[str, int]      # name -> y_pixel coordinate of 0 mV baseline
    calibration: Dict[str, Any]         # Calibration scales: pixels_per_sec, pixels_per_mv, etc.
    source_signal: np.ndarray           # (num_leads, signal_length)
    exact_lead_signals: Dict[str, np.ndarray]  # Segmented ground-truth signal per lead window
    lead_names: List[str]
    sampling_rate_hz: float
    seed: int
    source_record_id: str = "synthetic_sample"


class SyntheticECGGenerator:
    """Generates synthetic ECG paper images with exact pixel-level ground truth."""

    STANDARD_12_LEADS = [
        "I", "II", "III",
        "AVR", "AVL", "AVF",
        "V1", "V2", "V3",
        "V4", "V5", "V6",
    ]

    # Standard 3x4 layout columns (each column shows 2.5 seconds of the lead)
    # Row 0: I,   aVR, V1, V4
    # Row 1: II,  aVL, V2, V5
    # Row 2: III, aVF, V3, V6
    # Row 3 (bottom): II (10 seconds continuous rhythm)
    LAYOUT_3X4_COLUMNS = [
        ["I", "II", "III"],
        ["AVR", "AVL", "AVF"],
        ["V1", "V2", "V3"],
        ["V4", "V5", "V6"],
    ]

    LAYOUT_12X1_ROWS = [
        "I", "II", "III",
        "AVR", "AVL", "AVF",
        "V1", "V2", "V3",
        "V4", "V5", "V6",
    ]

    SUPPORTED_LAYOUTS = ["3x4_1", "3x4_3", "3x4", "12x1"]

    def __init__(self, layout: Optional[ECGLayoutConfig] = None) -> None:
        self.layout = layout or ECGLayoutConfig()
        self.pixels_per_mm = self.layout.dpi / 25.4
        self.pixels_per_sec = self.layout.speed_mm_s * self.pixels_per_mm
        self.pixels_per_mv = self.layout.gain_mm_mv * self.pixels_per_mm

    def generate(
        self,
        signal: Union[ECGSignal, np.ndarray],
        sampling_rate_hz: float = 100.0,
        lead_names: Optional[List[str]] = None,
        degradation: Optional[DegradationConfig] = None,
        seed: int = 42,
        layout_type: str = "3x4_1",
    ) -> SyntheticECGResult:
        """Generate a synthetic ECG paper sheet from 12-lead digital signals.

        Supports layouts:
        - '3x4_1': Standard 3x4 grid + 1 rhythm strip (Lead II) (4 rows total)
        - '3x4_3': Marquette MAC format: 3x4 grid + 3 rhythm strips (V1, II, V5) (6 rows total)
        - '3x4': Standard 3x4 grid without rhythm strip (3 rows total)
        - '12x1': 12 continuous stacked horizontal rows (12 rows total)
        """
        if layout_type not in self.SUPPORTED_LAYOUTS:
            raise ValueError(f"Unsupported layout_type '{layout_type}'. Supported: {self.SUPPORTED_LAYOUTS}")

        rng = np.random.RandomState(seed)

        source_rec_id = "synthetic_sample"
        if isinstance(signal, ECGSignal):
            values = signal.values
            fs = signal.metadata.sampling_rate_hz
            names = [l.upper() for l in signal.metadata.lead_names]
            source_rec_id = signal.metadata.record_id
        else:
            values = np.asarray(signal, dtype=np.float32)
            fs = sampling_rate_hz
            names = [l.upper() for l in (lead_names or self.STANDARD_12_LEADS)]

        # Map lead names to indices in values
        lead_idx_map = {name: i for i, name in enumerate(names) if i < values.shape[0]}

        width_px = int(round(self.layout.paper_width_mm * self.pixels_per_mm))
        height_px = int(round(self.layout.paper_height_mm * self.pixels_per_mm))

        # 1. Draw Paper Grid
        grid_style = degradation.grid_color_style if degradation else "standard"
        clean_img = self._create_grid_paper(width_px, height_px, grid_color_style=grid_style)
        trace_mask = np.zeros((height_px, width_px), dtype=np.uint8)

        # 2. Setup Dimensions and Layout Geometry
        margin_x = int(round(10.0 * self.pixels_per_mm))
        margin_y = int(round(10.0 * self.pixels_per_mm))
        usable_w = width_px - 2 * margin_x
        usable_h = height_px - 2 * margin_y

        if layout_type == "3x4_1":
            num_rows = 4
        elif layout_type == "3x4_3":
            num_rows = 6
        elif layout_type == "3x4":
            num_rows = 3
        elif layout_type == "12x1":
            num_rows = 12

        row_h = usable_h // num_rows
        col_w = usable_w // 4

        lead_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
        lead_baselines: Dict[str, int] = {}
        exact_lead_signals: Dict[str, np.ndarray] = {}

        # Draw Metadata Header at the top margin (or realistic clinical report header)
        has_clinical_hdr = self.layout.include_clinical_header or (degradation and degradation.clinical_header)
        if has_clinical_hdr:
            hdr_text1 = f"ECG REPORT  |  Record ID: {source_rec_id}  |  Age: 56  |  Sex: Male  |  HR: 72 bpm"
            hdr_text2 = "Technician: Clinician A  |  Ref-Phys: Cardiology Dept  |  Status: Confirmed Report"
            cv2.putText(clean_img, hdr_text1, (margin_x, max(18, margin_y - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(clean_img, hdr_text2, (margin_x, max(34, margin_y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 50, 50), 1, cv2.LINE_AA)
            cv2.line(clean_img, (margin_x, margin_y - 4), (margin_x + usable_w, margin_y - 4), (60, 60, 60), 1, cv2.LINE_AA)
        else:
            rec_title = f"PTB-XL Record: {source_rec_id} | Layout: {layout_type.upper()} | 25 mm/s, 10 mm/mV | 150 DPI | {fs:.0f} Hz | Clinical 12-Lead ECG"
            cv2.putText(
                clean_img,
                rec_title,
                (margin_x, max(22, margin_y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

        # Draw vertical column dividers if enabled
        has_v_div = self.layout.include_vertical_dividers or (degradation and degradation.vertical_dividers)
        if has_v_div and layout_type in ("3x4", "3x4_1", "3x4_3"):
            for col_i in range(1, 4):
                vx = margin_x + col_i * col_w
                grid_rows_count = 3
                cv2.line(clean_img, (vx, margin_y), (vx, margin_y + grid_rows_count * row_h), (80, 80, 80), 1, cv2.LINE_AA)

        # Draw footer annotation if enabled
        has_footer = self.layout.include_footer_annotation or (degradation and degradation.footer_annotation)
        if has_footer:
            footer_text = "25.0 mm/s  10.0 mm/mV  0.05-150Hz AC50  4*2.5s+1r  SEMIP v1.2"
            cv2.putText(clean_img, footer_text, (margin_x, min(height_px - 8, margin_y + usable_h + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 50, 50), 1, cv2.LINE_AA)

        # Draw standard 1 mV calibration pulse at start of each row if enabled
        if self.layout.include_calibration_pulse:
            pulse_w = int(round(5.0 * self.pixels_per_mm))   # 5 mm = 0.2s
            pulse_h = int(round(1.0 * self.pixels_per_mv))   # 1 mV = 10 mm
            pulse_in = int(round(1.5 * self.pixels_per_mm))  # 1.5 mm baseline lead-in
            p_end_x = margin_x - 12
            p_start_x = p_end_x - pulse_in - pulse_w - pulse_in
            if p_start_x >= 2:
                for r_idx in range(num_rows):
                    base_y = margin_y + r_idx * row_h + row_h // 2
                    pulse_pts = np.array([
                        [p_start_x, base_y],
                        [p_start_x + pulse_in, base_y],
                        [p_start_x + pulse_in, base_y - pulse_h],
                        [p_start_x + pulse_in + pulse_w, base_y - pulse_h],
                        [p_start_x + pulse_in + pulse_w, base_y],
                        [p_end_x, base_y],
                    ], dtype=np.int32)
                    cv2.polylines(
                        clean_img,
                        [pulse_pts],
                        isClosed=False,
                        color=self.layout.trace_color,
                        thickness=self.layout.line_thickness,
                        lineType=cv2.LINE_AA,
                    )

        # 3. Render Waves according to layout
        if layout_type in ("3x4", "3x4_1", "3x4_3"):
            # 3x4 Grid Columns (2.5s duration per lead)
            for col_idx, col_leads in enumerate(self.LAYOUT_3X4_COLUMNS):
                t_start_s = col_idx * 2.5
                t_end_s = (col_idx + 1) * 2.5
                idx_start = int(round(t_start_s * fs))
                idx_end = int(round(t_end_s * fs))

                x_box_start = margin_x + col_idx * col_w
                x_box_end = margin_x + (col_idx + 1) * col_w

                for row_idx, lead_name in enumerate(col_leads):
                    y_box_start = margin_y + row_idx * row_h
                    y_box_end = margin_y + (row_idx + 1) * row_h
                    baseline_y = y_box_start + row_h // 2

                    lead_bboxes[lead_name] = (x_box_start, y_box_start, x_box_end, y_box_end)
                    lead_baselines[lead_name] = baseline_y

                    # Draw Lead Label
                    lbl_font_scale = 0.55 if layout_type == "3x4_3" else 0.65
                    cv2.putText(
                        clean_img,
                        lead_name,
                        (x_box_start + 6, y_box_start + (18 if layout_type == "3x4_3" else 24)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        lbl_font_scale,
                        (40, 40, 40),
                        2 if layout_type != "3x4_3" else 1,
                        cv2.LINE_AA,
                    )

                    if lead_name not in lead_idx_map:
                        continue

                    sig_seg = values[lead_idx_map[lead_name], idx_start:idx_end]
                    if len(sig_seg) == 0:
                        continue
                    exact_lead_signals[lead_name] = sig_seg

                    num_points = len(sig_seg)
                    seg_time_span = num_points / fs
                    seg_width_px = seg_time_span * self.pixels_per_sec

                    x_coords = np.linspace(
                        x_box_start + 10,
                        min(x_box_start + 10 + seg_width_px, x_box_end - 5),
                        num_points,
                    )
                    y_coords = baseline_y - (sig_seg * self.pixels_per_mv)

                    pts = np.column_stack([x_coords, y_coords]).astype(np.int32)
                    cv2.polylines(
                        clean_img,
                        [pts],
                        isClosed=False,
                        color=self.layout.trace_color,
                        thickness=self.layout.line_thickness,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.polylines(
                        trace_mask,
                        [pts],
                        isClosed=False,
                        color=255,
                        thickness=self.layout.line_thickness,
                        lineType=cv2.LINE_8,
                    )

            # Rhythm Strips (if applicable)
            rhythm_configs: List[Tuple[str, str, int]] = []
            if layout_type == "3x4_1":
                rhythm_configs = [("II", "II_RHYTHM", 3)]
            elif layout_type == "3x4_3":
                rhythm_configs = [("V1", "V1_RHYTHM", 3), ("II", "II_RHYTHM", 4), ("V5", "V5_RHYTHM", 5)]

            for src_lead, r_tag, r_row in rhythm_configs:
                y_rhythm_start = margin_y + r_row * row_h
                y_rhythm_end = margin_y + (r_row + 1) * row_h
                rhythm_baseline_y = y_rhythm_start + row_h // 2

                lead_bboxes[r_tag] = (margin_x, y_rhythm_start, margin_x + usable_w, y_rhythm_end)
                lead_baselines[r_tag] = rhythm_baseline_y

                lbl_text = f"{src_lead} (Rhythm Strip)"
                cv2.putText(
                    clean_img,
                    lbl_text,
                    (margin_x + 6, y_rhythm_start + (18 if layout_type == "3x4_3" else 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 if layout_type == "3x4_3" else 0.65,
                    (40, 40, 40),
                    1 if layout_type == "3x4_3" else 2,
                    cv2.LINE_AA,
                )

                if src_lead in lead_idx_map:
                    rhythm_sig = values[lead_idx_map[src_lead], : int(round(10.0 * fs))]
                    exact_lead_signals[r_tag] = rhythm_sig
                    num_points = len(rhythm_sig)
                    rhythm_width_px = (num_points / fs) * self.pixels_per_sec

                    x_coords = np.linspace(
                        margin_x + 10,
                        min(margin_x + 10 + rhythm_width_px, margin_x + usable_w - 5),
                        num_points,
                    )
                    y_coords = rhythm_baseline_y - (rhythm_sig * self.pixels_per_mv)

                    pts = np.column_stack([x_coords, y_coords]).astype(np.int32)
                    cv2.polylines(
                        clean_img,
                        [pts],
                        isClosed=False,
                        color=self.layout.trace_color,
                        thickness=self.layout.line_thickness,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.polylines(
                        trace_mask,
                        [pts],
                        isClosed=False,
                        color=255,
                        thickness=self.layout.line_thickness,
                        lineType=cv2.LINE_8,
                    )

        elif layout_type == "12x1":
            # 12 continuous stacked rows (each 10 seconds continuous across width)
            for row_idx, lead_name in enumerate(self.LAYOUT_12X1_ROWS):
                y_box_start = margin_y + row_idx * row_h
                y_box_end = margin_y + (row_idx + 1) * row_h
                baseline_y = y_box_start + row_h // 2

                lead_bboxes[lead_name] = (margin_x, y_box_start, margin_x + usable_w, y_box_end)
                lead_baselines[lead_name] = baseline_y

                cv2.putText(
                    clean_img,
                    lead_name,
                    (margin_x + 6, y_box_start + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (40, 40, 40),
                    1,
                    cv2.LINE_AA,
                )

                if lead_name in lead_idx_map:
                    row_sig = values[lead_idx_map[lead_name], : int(round(10.0 * fs))]
                    exact_lead_signals[lead_name] = row_sig
                    num_points = len(row_sig)
                    trace_width_px = (num_points / fs) * self.pixels_per_sec

                    x_coords = np.linspace(
                        margin_x + 10,
                        min(margin_x + 10 + trace_width_px, margin_x + usable_w - 5),
                        num_points,
                    )
                    y_coords = baseline_y - (row_sig * self.pixels_per_mv)

                    pts = np.column_stack([x_coords, y_coords]).astype(np.int32)
                    cv2.polylines(
                        clean_img,
                        [pts],
                        isClosed=False,
                        color=self.layout.trace_color,
                        thickness=self.layout.line_thickness,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.polylines(
                        trace_mask,
                        [pts],
                        isClosed=False,
                        color=255,
                        thickness=self.layout.line_thickness,
                        lineType=cv2.LINE_8,
                    )

        # 4. Apply Realistic Photographic / Scanning Degradation
        degraded_img = clean_img.copy()
        if degradation and degradation.enabled:
            degraded_img, trace_mask = self._apply_degradation(
                degraded_img, trace_mask, degradation, rng
            )

        calibration_info = {
            "dpi": self.layout.dpi,
            "pixels_per_mm": float(self.pixels_per_mm),
            "pixels_per_sec": float(self.pixels_per_sec),
            "pixels_per_mv": float(self.pixels_per_mv),
            "speed_mm_s": self.layout.speed_mm_s,
            "gain_mm_mv": self.layout.gain_mm_mv,
            "width_px": width_px,
            "height_px": height_px,
            "layout": layout_type,
        }

        return SyntheticECGResult(
            image=degraded_img,
            clean_image=clean_img,
            trace_mask=trace_mask,
            lead_bounding_boxes=lead_bboxes,
            lead_baselines=lead_baselines,
            calibration=calibration_info,
            source_signal=values,
            exact_lead_signals=exact_lead_signals,
            lead_names=names,
            sampling_rate_hz=fs,
            seed=seed,
            source_record_id=source_rec_id,
        )

    def _create_grid_paper(self, width_px: int, height_px: int, grid_color_style: str = "standard") -> np.ndarray:
        """Draws standard ECG paper background with 1 mm minor and 5 mm major grid lines."""
        # Cream/white paper base
        canvas = np.full((height_px, width_px, 3), 252, dtype=np.uint8)

        px_1mm = self.pixels_per_mm
        px_5mm = 5.0 * px_1mm

        if grid_color_style == "faded":
            minor_col = (235, 235, 255)
            major_col = (180, 180, 245)
        elif grid_color_style == "dark":
            minor_col = (190, 190, 250)
            major_col = (110, 110, 220)
        elif grid_color_style == "monochrome":
            minor_col = (230, 230, 230)
            major_col = (180, 180, 180)
        else:
            minor_col = self.layout.grid_minor_color
            major_col = self.layout.grid_major_color

        # Draw minor 1 mm grid lines
        num_minor_x = int(width_px // px_1mm)
        for i in range(num_minor_x + 1):
            x = int(round(i * px_1mm))
            if x < width_px:
                cv2.line(canvas, (x, 0), (x, height_px), minor_col, 1)

        num_minor_y = int(height_px // px_1mm)
        for i in range(num_minor_y + 1):
            y = int(round(i * px_1mm))
            if y < height_px:
                cv2.line(canvas, (0, y), (width_px, y), minor_col, 1)

        # Draw major 5 mm grid lines
        num_major_x = int(width_px // px_5mm)
        for i in range(num_major_x + 1):
            x = int(round(i * px_5mm))
            if x < width_px:
                cv2.line(canvas, (x, 0), (x, height_px), major_col, 1)

        num_major_y = int(height_px // px_5mm)
        for i in range(num_major_y + 1):
            y = int(round(i * px_5mm))
            if y < height_px:
                cv2.line(canvas, (0, y), (width_px, y), major_col, 1)

        return canvas

    def _apply_degradation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        cfg: DegradationConfig,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies realistic scanning and camera degradations deterministically."""
        h, w = image.shape[:2]

        # 1. Affine Rotation / Skew
        if abs(cfg.rotation_deg) > 1e-3:
            center = (w / 2.0, h / 2.0)
            rot_mat = cv2.getRotationMatrix2D(center, cfg.rotation_deg, 1.0)
            image = cv2.warpAffine(
                image,
                rot_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(252, 252, 252),
            )
            mask = cv2.warpAffine(
                mask,
                rot_mat,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # 2. Perspective Distortion
        if cfg.perspective_factor > 1e-4:
            p = cfg.perspective_factor
            src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dx = w * p
            dy = h * p
            dst_pts = np.float32([
                [dx * rng.uniform(0.5, 1.0), dy * rng.uniform(0.5, 1.0)],
                [w - dx * rng.uniform(0.5, 1.0), dy * rng.uniform(0.5, 1.0)],
                [w - dx * rng.uniform(0.2, 0.6), h - dy * rng.uniform(0.2, 0.6)],
                [dx * rng.uniform(0.2, 0.6), h - dy * rng.uniform(0.2, 0.6)],
            ])
            persp_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            image = cv2.warpPerspective(
                image,
                persp_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(252, 252, 252),
            )
            mask = cv2.warpPerspective(
                mask,
                persp_mat,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # 3. Non-uniform lighting gradient (shadow)
        if cfg.shadow_gradient:
            grad_x = np.linspace(0.85, 1.05, w).reshape(1, w, 1)
            grad_y = np.linspace(0.90, 1.02, h).reshape(h, 1, 1)
            grad = grad_x * grad_y
            image = np.clip(image.astype(np.float32) * grad, 0, 255).astype(np.uint8)

        # 4. Blur
        if cfg.blur_kernel_size > 1:
            k = cfg.blur_kernel_size if cfg.blur_kernel_size % 2 == 1 else cfg.blur_kernel_size + 1
            image = cv2.GaussianBlur(image, (k, k), 0)

        # 5. Gaussian Noise
        if cfg.gaussian_noise_sigma > 1e-3:
            noise = rng.normal(0, cfg.gaussian_noise_sigma, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 6. JPEG Compression Artifacts
        if cfg.jpeg_quality is not None and 10 <= cfg.jpeg_quality <= 95:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)]
            success, enc = cv2.imencode(".jpg", image, encode_param)
            if success:
                image = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        return image, mask

    @staticmethod
    def save_result(result: SyntheticECGResult, output_dir: Union[str, Path], base_name: str) -> Dict[str, str]:
        """Saves image, mask, clean image, and JSON metadata to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        img_file = out_path / f"{base_name}.png"
        clean_file = out_path / f"{base_name}_clean.png"
        mask_file = out_path / f"{base_name}_mask.png"
        meta_file = out_path / f"{base_name}_meta.json"
        signal_file = out_path / f"{base_name}_signal.npy"

        cv2.imwrite(str(img_file), result.image)
        cv2.imwrite(str(clean_file), result.clean_image)
        cv2.imwrite(str(mask_file), result.trace_mask)
        np.save(str(signal_file), result.source_signal)

        meta = {
            "record_id": base_name,
            "seed": result.seed,
            "sampling_rate_hz": result.sampling_rate_hz,
            "lead_names": result.lead_names,
            "lead_bounding_boxes": result.lead_bounding_boxes,
            "lead_baselines": result.lead_baselines,
            "calibration": result.calibration,
            "files": {
                "image": str(img_file.name),
                "clean_image": str(clean_file.name),
                "mask": str(mask_file.name),
                "signal": str(signal_file.name),
            },
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return {k: str(v) for k, v in meta["files"].items()}
