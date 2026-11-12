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
    ) -> SyntheticECGResult:
        """Generate a synthetic ECG paper sheet from 12-lead digital signals."""
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
        clean_img = self._create_grid_paper(width_px, height_px)
        trace_mask = np.zeros((height_px, width_px), dtype=np.uint8)

        # 2. Setup 3x4 + 1 Rhythm Strip Layout
        margin_x = int(round(10.0 * self.pixels_per_mm))
        margin_y = int(round(10.0 * self.pixels_per_mm))
        usable_w = width_px - 2 * margin_x
        usable_h = height_px - 2 * margin_y

        # 4 rows: 3 standard rows + 1 rhythm strip
        row_h = usable_h // 4
        col_w = usable_w // 4

        lead_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
        lead_baselines: Dict[str, int] = {}
        exact_lead_signals: Dict[str, np.ndarray] = {}

        # Draw Metadata Header at the top margin
        rec_title = f"PTB-XL Record: {source_rec_id} | 25 mm/s, 10 mm/mV | 150 DPI | {fs:.0f} Hz | Clinical 12-Lead Diagnostic ECG"
        cv2.putText(
            clean_img,
            rec_title,
            (margin_x, max(22, margin_y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

        # Draw standard 1 mV calibration pulse at start of each row if enabled
        # Keeps a clean >= 10px separation before margin_x so the pulse does not touch Column 0
        if self.layout.include_calibration_pulse:
            pulse_w = int(round(5.0 * self.pixels_per_mm))   # 5 mm = 0.2s
            pulse_h = int(round(1.0 * self.pixels_per_mv))   # 1 mV = 10 mm
            pulse_in = int(round(1.5 * self.pixels_per_mm))  # 1.5 mm baseline lead-in
            p_end_x = margin_x - 12
            p_start_x = p_end_x - pulse_in - pulse_w - pulse_in
            if p_start_x >= 2:
                for r_idx in range(4):
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

        # 3. Draw 3x4 Main Grid Leads
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

                # Draw Lead Label with high-contrast legibility
                cv2.putText(
                    clean_img,
                    lead_name,
                    (x_box_start + 6, y_box_start + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (40, 40, 40),
                    2,
                    cv2.LINE_AA,
                )

                if lead_name not in lead_idx_map:
                    continue

                sig_seg = values[lead_idx_map[lead_name], idx_start:idx_end]
                if len(sig_seg) == 0:
                    continue
                exact_lead_signals[lead_name] = sig_seg

                # Waveform pixel coordinates
                num_points = len(sig_seg)
                seg_time_span = num_points / fs
                seg_width_px = seg_time_span * self.pixels_per_sec

                x_coords = np.linspace(
                    x_box_start + 10,
                    min(x_box_start + 10 + seg_width_px, x_box_end - 5),
                    num_points,
                )
                # ECG voltage: positive mV goes upwards (lower y in image)
                y_coords = baseline_y - (sig_seg * self.pixels_per_mv)

                # Draw trace & update mask
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

        # 4. Draw Row 3: Rhythm Strip (Lead II, full 10s)
        rhythm_lead = "II"
        rhythm_row = 3
        y_rhythm_start = margin_y + rhythm_row * row_h
        y_rhythm_end = margin_y + (rhythm_row + 1) * row_h
        rhythm_baseline_y = y_rhythm_start + row_h // 2

        lead_bboxes["II_RHYTHM"] = (margin_x, y_rhythm_start, margin_x + usable_w, y_rhythm_end)
        lead_baselines["II_RHYTHM"] = rhythm_baseline_y

        cv2.putText(
            clean_img,
            "II (Rhythm Strip)",
            (margin_x + 6, y_rhythm_start + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )

        if rhythm_lead in lead_idx_map:
            rhythm_sig = values[lead_idx_map[rhythm_lead], : int(round(10.0 * fs))]
            exact_lead_signals["II_RHYTHM"] = rhythm_sig
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

        # 5. Apply Realistic Photographic / Scanning Degradation
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
            "layout": "3x4_1",
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

    def _create_grid_paper(self, width_px: int, height_px: int) -> np.ndarray:
        """Draws standard ECG paper background with 1 mm minor and 5 mm major grid lines."""
        # Cream/white paper base
        canvas = np.full((height_px, width_px, 3), 252, dtype=np.uint8)

        px_1mm = self.pixels_per_mm
        px_5mm = 5.0 * px_1mm

        # Draw minor 1 mm grid lines
        num_minor_x = int(width_px // px_1mm)
        for i in range(num_minor_x + 1):
            x = int(round(i * px_1mm))
            if x < width_px:
                cv2.line(canvas, (x, 0), (x, height_px), self.layout.grid_minor_color, 1)

        num_minor_y = int(height_px // px_1mm)
        for i in range(num_minor_y + 1):
            y = int(round(i * px_1mm))
            if y < height_px:
                cv2.line(canvas, (0, y), (width_px, y), self.layout.grid_minor_color, 1)

        # Draw major 5 mm grid lines
        num_major_x = int(width_px // px_5mm)
        for i in range(num_major_x + 1):
            x = int(round(i * px_5mm))
            if x < width_px:
                cv2.line(canvas, (x, 0), (x, height_px), self.layout.grid_major_color, 1)

        num_major_y = int(height_px // px_5mm)
        for i in range(num_major_y + 1):
            y = int(round(i * px_5mm))
            if y < height_px:
                cv2.line(canvas, (0, y), (width_px, y), self.layout.grid_major_color, 1)

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
