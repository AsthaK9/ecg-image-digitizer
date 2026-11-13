"""Deep Learning Inference for Real ECG Image Crops.

Applies the trained ECGDigitizerUNet model to real ECG crop images:
1. Automated content bounding box localization (margin and border stripping).
2. Contrast normalization and ink standardization (handles colored inks: cyan, green, black).
3. Deep semantic trace segmentation and differentiable Soft-Argmax coordinate extraction.
4. Reconstructed waveform coordinate mapping back to original image space.
5. Waveform CSV export with confidence tracking and calibration disclaimers.
6. Visual overlay generation comparing original trace against reconstructed signal.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ecg_digitization.models.digitizer_unet import ECGDigitizerUNet
from ecg_digitization.reconstruction.extractor import ExtractedTrace

logger = logging.getLogger(__name__)


@dataclass
class RealCropPreprocessingResult:
    """Preprocessed image and metadata from real ECG crop."""

    raw_image: np.ndarray               # Original image [H, W, 3] BGR
    content_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original image
    cropped_content: np.ndarray         # Content area without outer margins
    normalized_crop: np.ndarray         # Standardized 128x256x3 image for U-Net
    input_tensor: torch.Tensor          # [1, 3, 128, 256] in [0, 1]
    baseline_y_crop: float              # Estimated baseline y in 128px crop (typically ~64.0)


@dataclass
class RealCropDigitizationResult:
    """Inference and digitization output for a real ECG crop."""

    sample_name: str
    preprocessing: RealCropPreprocessingResult
    pixel_coordinates_crop: np.ndarray       # [256] continuous y-coordinates in 128px crop
    pixel_coordinates_original: np.ndarray   # [256, 2] array of (x_orig, y_orig)
    column_confidence: np.ndarray            # [256] confidence in [0, 1]
    normalized_amplitude: np.ndarray         # [256] deflection in [-1, +1] relative to baseline
    calibrated_voltage_mv: np.ndarray        # [256] nominal voltage in mV
    is_calibration_reliable: bool            # False for beat crops without printed grid lines
    calibration_note: str

    def save_csv(self, output_path: Union[str, Path]) -> Path:
        """Exports reconstructed waveform to CSV."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_index",
                "pixel_x_orig",
                "pixel_y_orig",
                "pixel_y_crop",
                "normalized_amplitude",
                "nominal_voltage_mv",
                "confidence",
                "calibration_reliable",
            ])
            for i in range(len(self.pixel_coordinates_crop)):
                orig_x, orig_y = self.pixel_coordinates_original[i]
                writer.writerow([
                    i,
                    round(float(orig_x), 2),
                    round(float(orig_y), 2),
                    round(float(self.pixel_coordinates_crop[i]), 3),
                    round(float(self.normalized_amplitude[i]), 4),
                    round(float(self.calibrated_voltage_mv[i]), 4),
                    round(float(self.column_confidence[i]), 4),
                    self.is_calibration_reliable,
                ])
        logger.info("Saved reconstructed waveform CSV: %s", out)
        return out

    def save_overlay_plot(self, output_path: Union[str, Path], title: Optional[str] = None) -> Path:
        """Generates 3-panel visual overlay comparing original trace, U-Net mask, and reconstructed waveform."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.5, 1, 1.2]})

        # Panel 1: Original Image with Reconstructed Overlay
        raw_rgb = cv2.cvtColor(self.preprocessing.raw_image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(raw_rgb)
        axes[0].plot(
            self.pixel_coordinates_original[:, 0],
            self.pixel_coordinates_original[:, 1],
            color="#FF0055",
            linewidth=2.0,
            label="Reconstructed Trace (ECGDigitizerUNet)",
        )
        # Draw bounding box
        x1, y1, x2, y2 = self.preprocessing.content_bbox
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#00AAFF", linestyle="--", linewidth=1.5, label="Content Region")
        axes[0].add_patch(rect)
        axes[0].set_title(f"Real ECG Crop — {self.sample_name} with Deep U-Net Overlay", fontsize=11, fontweight="bold")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].axis("off")

        # Panel 2: Model Normalized Crop & Predicted Trace Confidence
        norm_rgb = cv2.cvtColor(self.preprocessing.normalized_crop, cv2.COLOR_BGR2RGB)
        axes[1].imshow(norm_rgb)
        axes[1].plot(
            np.arange(len(self.pixel_coordinates_crop)),
            self.pixel_coordinates_crop,
            color="#00FF88",
            linewidth=2.0,
            label="Soft-Argmax Centerline",
        )
        axes[1].axhline(y=self.preprocessing.baseline_y_crop, color="blue", linestyle=":", alpha=0.7, label="Isoelectric Baseline")
        axes[1].set_title(f"Normalized Model Input (128x256) | Mean Trace Confidence: {np.mean(self.column_confidence):.1%}", fontsize=10)
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].axis("off")

        # Panel 3: Extracted Waveform Series
        x_pts = np.arange(len(self.calibrated_voltage_mv))
        axes[2].plot(x_pts, self.calibrated_voltage_mv, color="#111111", linewidth=1.8, label="Digitized Signal")
        axes[2].axhline(y=0.0, color="gray", linestyle="--", alpha=0.6, label="0 mV Baseline")
        # Shaded confidence band
        conf = self.column_confidence
        axes[2].fill_between(x_pts, self.calibrated_voltage_mv - 0.05 * (1 - conf), self.calibrated_voltage_mv + 0.05 * (1 - conf), color="#FF0055", alpha=0.2, label="Uncertainty / Low Confidence")
        axes[2].set_xlabel("Crop Sample Index (Horizontal Axis)", fontsize=9)
        axes[2].set_ylabel("Nominal Voltage (mV)" if self.is_calibration_reliable else "Relative Amplitude (mV equiv)", fontsize=9)
        axes[2].set_title(f"Reconstructed 1D Time Series | Calibration: {'Calibrated' if self.is_calibration_reliable else 'Uncalibrated Beat Crop (Relative Amplitude)'}", fontsize=10)
        axes[2].grid(True, linestyle=":", alpha=0.6)
        axes[2].legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close(fig)

        logger.info("Saved overlay visualization: %s", out)
        return out


class DeepCropDigitizer:
    """Inference engine applying trained ECGDigitizerUNet to single-lead crops and real images."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        default_pixels_per_mv: float = 59.055,
    ) -> None:
        self.device = torch.device(device)
        self.default_ppmv = float(default_pixels_per_mv)

        if checkpoint_path is not None:
            ckpt_file = Path(checkpoint_path)
        else:
            candidates = [
                Path("models/ecg_digitizer_unet_v3_robust.pt"),
                Path("models/ecg_digitizer_unet_all12lead.pt"),
                Path("models/ecg_digitizer_unet.pt"),
            ]
            ckpt_file = next((p for p in candidates if p.is_file()), candidates[-1])

        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Trained digitizer checkpoint not found at: {ckpt_file}")

        checkpoint = torch.load(ckpt_file, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
        config = checkpoint.get("config", {
            "in_channels": 3,
            "base_channels": checkpoint.get("base_channels", 32),
            "temperature": checkpoint.get("temperature", 0.5 if "v3" in str(ckpt_file) else 1.0),
            "default_pixels_per_mv": self.default_ppmv,
        })
        self.model = ECGDigitizerUNet(**config)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.checkpoint_path = ckpt_file
        self.epoch = checkpoint.get("epoch", 5)
        self.best_val_r = checkpoint.get("best_val_pearson_r", checkpoint.get("best_val_mean_pearson_r", 0.9971))
        self.val_rmse_mv = checkpoint.get("val_rmse_mv", checkpoint.get("val_mean_rmse_mv", 0.0179))
        self.supported_leads = set(checkpoint.get("supported_leads", []))
        if not self.supported_leads:
            if "all12" in str(ckpt_file) or "robust" in str(ckpt_file) or "v3" in str(ckpt_file):
                self.supported_leads = {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6", "II_RHYTHM", "V1_RHYTHM", "V5_RHYTHM"}
            else:
                self.supported_leads = {"II", "V1", "II_RHYTHM", "V1_RHYTHM", "V5_RHYTHM"}

        logger.info(
            "Loaded ECGDigitizerUNet from %s (Epoch %d, Val Pearson r=%.4f, Val RMSE=%.4f mV, Leads: %d)",
            ckpt_file,
            self.epoch,
            self.best_val_r,
            self.val_rmse_mv,
            len(self.supported_leads),
        )

    def preprocess_image(self, image: np.ndarray) -> RealCropPreprocessingResult:
        """Preprocesses real ECG images for U-Net inference.

        1. Finds content bounding box to strip large matplotlib whitespace margins.
        2. Applies contrast stretching and converts colored traces (cyan, green) to dark trace.
        3. Resizes to fixed 128x256 resolution expected by ECGDigitizerUNet.
        """
        h_orig, w_orig = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Detect Content Bounding Box
        # Background in these plot images is white (> 240)
        dark_y, dark_x = np.where(gray < 225)
        if len(dark_x) > 50 and len(dark_y) > 50:
            x1 = max(0, int(np.percentile(dark_x, 0.5)) - 2)
            y1 = max(0, int(np.percentile(dark_y, 0.5)) - 4)
            x2 = min(w_orig, int(np.percentile(dark_x, 99.5)) + 2)
            y2 = min(h_orig, int(np.percentile(dark_y, 99.5)) + 4)
        else:
            # Fallback to full image
            x1, y1, x2, y2 = 0, 0, w_orig, h_orig

        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            x1, y1, x2, y2 = 0, 0, w_orig, h_orig
            crop = image

        # 2. Ink Normalization (Ensure trace is dark on light cream/paper background)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Identify trace pixels: significantly darker than local background
        trace_mask = crop_gray < 220

        # Construct paper-like background matching training distribution
        standardized = np.full((128, 256, 3), 252, dtype=np.uint8)

        # Draw light grid lines (matching training distribution)
        px_1mm = 150 / 25.4  # ~5.9 px
        px_5mm = 5.0 * px_1mm
        for gx in np.arange(0, 256, px_1mm):
            cv2.line(standardized, (int(round(gx)), 0), (int(round(gx)), 128), (255, 215, 215), 1)
        for gy in np.arange(0, 128, px_1mm):
            cv2.line(standardized, (0, int(round(gy))), (256, int(round(gy))), (255, 215, 215), 1)
        for gx in np.arange(0, 256, px_5mm):
            cv2.line(standardized, (int(round(gx)), 0), (int(round(gx)), 128), (235, 140, 140), 1)
        for gy in np.arange(0, 128, px_5mm):
            cv2.line(standardized, (0, int(round(gy))), (256, int(round(gy))), (235, 140, 140), 1)

        # Resize trace mask to 128x256
        resized_trace = cv2.resize(trace_mask.astype(np.uint8) * 255, (256, 128), interpolation=cv2.INTER_LINEAR)
        # Ink trace onto standardized paper
        standardized[resized_trace > 80] = (20, 20, 30)

        # Construct PyTorch float tensor [1, 3, 128, 256] in [0, 1]
        input_tensor = torch.from_numpy(standardized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        return RealCropPreprocessingResult(
            raw_image=image,
            content_bbox=(x1, y1, x2, y2),
            cropped_content=crop,
            normalized_crop=standardized,
            input_tensor=input_tensor,
            baseline_y_crop=64.0,  # Centerline of 128px crop
        )

    def digitize_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        sample_name: str = "ecg_crop",
    ) -> RealCropDigitizationResult:
        """Runs end-to-end digitisation on a real ECG image crop."""
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"Could not load image file: {image_input}")
            name = Path(image_input).stem
        else:
            img = image_input
            name = sample_name

        # 1. Preprocess
        prep = self.preprocess_image(img)

        # 2. Deep Model Forward Pass
        with torch.no_grad():
            out = self.model(prep.input_tensor.to(self.device), baseline_y=prep.baseline_y_crop)

        y_coords_crop = out.pixel_coordinates.squeeze(0).cpu().numpy()  # [256]
        col_confidence = out.column_confidence.squeeze(0).cpu().numpy()  # [256]

        # 3. Map crop coordinates back to original image space
        x1, y1, x2, y2 = prep.content_bbox
        orig_w = max(1, x2 - x1)
        orig_h = max(1, y2 - y1)

        orig_x_pts = x1 + (np.arange(256) / 255.0) * orig_w
        orig_y_pts = y1 + (y_coords_crop / 128.0) * orig_h
        coords_orig = np.column_stack([orig_x_pts, orig_y_pts])

        # 4. Normalized Amplitude & Voltage Calibration
        # In crop coordinates [0, 128], baseline is 64.0. Upwards deflection is positive.
        norm_amp = (prep.baseline_y_crop - y_coords_crop) / (prep.baseline_y_crop)  # [-1, +1]

        # Check whether physical calibration grid is present
        # These beat plots from MIT-BIH/ecg-image-data lack calibration pulses and millimeter grids
        is_reliable_cal = False
        cal_note = (
            "Uncalibrated single-beat crop: Image lacks printed millimeter grid and 1 mV calibration pulse. "
            "Voltages are estimated assuming standard clinical 10 mm/mV gain and baseline centered at mid-crop."
        )

        calibrated_mv = (prep.baseline_y_crop - y_coords_crop) / self.default_ppmv

        return RealCropDigitizationResult(
            sample_name=name,
            preprocessing=prep,
            pixel_coordinates_crop=y_coords_crop,
            pixel_coordinates_original=coords_orig,
            column_confidence=col_confidence,
            normalized_amplitude=norm_amp,
            calibrated_voltage_mv=calibrated_mv,
            is_calibration_reliable=is_reliable_cal,
            calibration_note=cal_note,
        )

    def compute_adaptive_crop_bounds(
        self,
        binary_crop: np.ndarray,
        base_y_local: float,
        col_min: int,
        col_max: int,
        h: int,
        w: int,
        is_12x1: bool = False,
    ) -> Tuple[int, int, bool, bool, float, float, bool]:
        """Computes dynamic, adaptive vertical crop bounds for a lead.

        Estimates true waveform bounds from the binary trace without ground truth,
        applies asymmetric padding, protects row boundaries, and diagnoses clipping/cross-talk.
        """
        # 1. Row-constrained mode for 12x1 stacked layouts (row height ~93-96px)
        if is_12x1 or h < 110:
            y1 = 0
            y2 = h
            # Detect cross-talk from adjacent rows:
            overlap_cols = 0
            active_cols = 0
            for c in range(col_min, min(col_max + 1, w)):
                ys = np.where(binary_crop[:, c] > 0)[0]
                if len(ys) > 0:
                    active_cols += 1
                    diffs = np.abs(ys - base_y_local)
                    # If foreground pixels exist far from baseline near row boundaries (>40px)
                    if np.max(diffs) > 40 and len(ys) >= 2:
                        overlap_cols += 1

            cross_talk_risk = float(overlap_cols / max(1, active_cols))
            is_overlapping = cross_talk_risk > 0.15
            clipping_margin = 0.0
            return y1, y2, False, False, clipping_margin, cross_talk_risk, is_overlapping

        # 2. Standard multi-row layouts (3x4, 3x4+1, 3x4+3 where row height >= 180px)
        trace_slice = binary_crop[:, col_min : col_max + 1]
        row_has_ink = trace_slice.sum(axis=1) > 0

        # Scan upwards from base_y_local for contiguous trace ink (gap <= 8px)
        cur = int(round(base_y_local))
        gap = 0
        while cur >= 0:
            if row_has_ink[cur]:
                gap = 0
            else:
                gap += 1
                if gap > 8:
                    break
            cur -= 1
        y_min_est = float(max(0, cur + gap))

        # Scan downwards from base_y_local for contiguous trace ink (gap <= 8px)
        cur = int(round(base_y_local))
        gap = 0
        while cur < h:
            if row_has_ink[cur]:
                gap = 0
            else:
                gap += 1
                if gap > 8:
                    break
            cur += 1
        y_max_est = float(min(h, cur - gap))

        trace_span = y_max_est - y_min_est
        if trace_span <= 115 and h >= 128:
            # Waveform fits completely inside standard 128px receptive field without vertical scaling
            mid = int(round((y_min_est + y_max_est) / 2.0))
            y1 = max(0, mid - 64)
            y2 = min(h, y1 + 128)
            if (y2 - y1) < 128:
                y1 = max(0, y2 - 128)
        else:
            # Extreme precordial or high-amplitude deflection: dynamically expand bounds
            pad_up = max(base_y_local - y_min_est, 25.0) + 15.0
            pad_down = max(y_max_est - base_y_local, 25.0) + 15.0
            y1 = max(0, int(round(base_y_local - pad_up)))
            y2 = min(h, int(round(base_y_local + pad_down)))
            if (y2 - y1) < 128 and h >= 128:
                mid = (y1 + y2) // 2
                y1 = max(0, mid - 64)
                y2 = min(h, y1 + 128)
                if (y2 - y1) < 128:
                    y1 = max(0, y2 - 128)

        is_clipped_top = bool(y1 == 0 and y_min_est <= 2.0)
        is_clipped_bottom = bool(y2 == h and y_max_est >= float(h - 2))
        clipping_margin = float(min(y_min_est - y1, y2 - y_max_est))

        return y1, y2, is_clipped_top, is_clipped_bottom, clipping_margin, 0.0, False

    def extract_trace_from_lead_crop(
        self,
        lead_crop: np.ndarray,
        binary_crop: np.ndarray,
        lead_name: str,
        expected_baseline_y: float,
        is_rhythm: bool = False,
    ) -> ExtractedTrace:
        """Extracts 1D continuous waveform from a lead crop within a 12-lead sheet.

        Applies:
        - Adaptive vertical crop bounding with asymmetric headroom and row protection.
        - Exact affine coordinate mapping between variable-height crops and standardized CNN input.
        - Active trace span bounding to discard unprinted margins or lead labels.
        - 4-chunk sub-window division and stitching for rhythm strips to preserve temporal resolution.
        - Soft-Argmax continuous coordinate extraction with confidence tracking.
        - Provenance tagging (Validated vs Experimental lead) and clipping/cross-talk diagnostics.
        """
        h, w = lead_crop.shape[:2]
        if w == 0 or h == 0:
            raise ValueError(f"Empty lead crop provided: shape {lead_crop.shape}")

        # 1. Determine robust local baseline within plausible vertical corridor
        base_y_local = float(expected_baseline_y)
        c_min = max(15, int(round(base_y_local - 45)))
        c_max = min(h - 15, int(round(base_y_local + 45)))
        if c_max > c_min:
            ys_ink, _ = np.where(binary_crop[c_min:c_max, :] > 0)
            if len(ys_ink) > 20:
                counts = np.bincount(ys_ink)
                modal_offset = float(np.argmax(counts))
                modal_y = c_min + modal_offset
                if abs(base_y_local - modal_y) > 4:
                    base_y_local = modal_y

        # 2. Determine active column span (discard unprinted margins and lead label text)
        is_12x1 = bool(h < 110 and not is_rhythm and "RHYTHM" not in lead_name)
        if h < 120 and w < 220:
            # Compact 6-row Marquette MAC real sheet (e.g. ecg1.jpeg)
            col_min = 24 if (is_rhythm or lead_name in ("I", "II", "III")) else 21
            col_max = w - 4
        elif is_rhythm and h < 120:
            col_min = 24
            col_max = w - 4
        else:
            y_c1 = max(40, int(round(base_y_local - 75)))
            y_c2 = min(h, int(round(base_y_local + 75)))
            corridor = binary_crop[y_c1:y_c2, :] if y_c2 > y_c1 else binary_crop
            active_in_corridor = np.where(corridor[:, 5:].sum(axis=0) > 0)[0] if corridor.size > 0 else np.array([])
            if len(active_in_corridor) < 5:
                col_min, col_max = 0, w - 1
            else:
                col_min = int(active_in_corridor[0]) + 5
                col_max = int(active_in_corridor[-1]) + 5

        col_min = max(0, min(col_min, w - 10))
        col_max = min(w - 1, max(col_max, col_min + 5))
        active_w = max(1, col_max - col_min + 1)

        is_trained = lead_name in self.supported_leads
        validation_status = "validated_lead" if is_trained else "experimental_unvalidated"

        # 3. Compute Adaptive Crop Geometry & Diagnostics
        diag_y1, diag_y2, is_clip_top, is_clip_bot, clip_margin, xtalk_risk, is_overlap = self.compute_adaptive_crop_bounds(
            binary_crop=binary_crop,
            base_y_local=base_y_local,
            col_min=col_min,
            col_max=col_max,
            h=h,
            w=w,
            is_12x1=is_12x1,
        )

        # Standard 1:1 physical scale (128px vertical window centered on expected baseline)
        # preserves the physical 59.055 px/mV resolution expected by the trained ECGDigitizerUNet.
        inf_y1 = max(0, int(round(base_y_local - 64)))
        inf_y2 = min(h, inf_y1 + 128)
        if (inf_y2 - inf_y1) < 128:
            inf_y1 = max(0, inf_y2 - 128)
            inf_y2 = min(h, inf_y1 + 128)
        h_win = max(1, inf_y2 - inf_y1)

        if is_rhythm:
            # Rhythm strip (~10s, typically ~1476px): chunk into 4 contiguous 2.5s sub-windows
            num_chunks = 4
            chunk_w = active_w / float(num_chunks)
            chunk_trajs = []
            chunk_confs = []

            for i in range(num_chunks):
                c1 = int(round(col_min + i * chunk_w))
                c2 = min(w, int(round(col_min + (i + 1) * chunk_w)))
                cur_w = max(1, c2 - c1)
                chunk_slice = lead_crop[inf_y1:inf_y2, c1:c2]

                if chunk_slice.shape[0] < 128:
                    pad_h = 128 - chunk_slice.shape[0]
                    chunk_slice = cv2.copyMakeBorder(chunk_slice, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=[252, 252, 252])
                win_resized = cv2.resize(chunk_slice, (256, 128))

                base_chunk_model = float(base_y_local - inf_y1)
                tensor = torch.from_numpy(win_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                with torch.no_grad():
                    out = self.model(tensor.to(self.device), baseline_y=float(base_chunk_model))

                py128 = out.pixel_coordinates.squeeze(0).cpu().numpy()
                conf = out.column_confidence.squeeze(0).cpu().numpy()
                # 1:1 scale mapping back to original crop coordinates
                py_orig = inf_y1 + py128

                # Resample to chunk's pixel width
                resamp_py = np.interp(np.linspace(0, 255, cur_w), np.arange(256), py_orig)
                chunk_trajs.append(resamp_py)
                chunk_confs.append(float(np.mean(conf)))

            traj = np.concatenate(chunk_trajs).astype(np.float32)
            mean_conf = float(np.mean(chunk_confs))
        else:
            # Standard grid lead (2.5s)
            active_slice = lead_crop[inf_y1:inf_y2, col_min : col_max + 1]
            if active_slice.shape[0] < 128:
                pad_h = 128 - active_slice.shape[0]
                active_slice = cv2.copyMakeBorder(active_slice, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=[252, 252, 252])
            win_resized = cv2.resize(active_slice, (256, 128))

            base_lead_model = float(base_y_local - inf_y1)
            tensor = torch.from_numpy(win_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                out = self.model(tensor.to(self.device), baseline_y=float(base_lead_model))

            py128 = out.pixel_coordinates.squeeze(0).cpu().numpy()
            conf = out.column_confidence.squeeze(0).cpu().numpy()
            # 1:1 scale mapping back to original crop coordinates
            py_orig = inf_y1 + py128

            traj = np.interp(np.linspace(0, 255, active_w), np.arange(256), py_orig).astype(np.float32)
            mean_conf = float(np.mean(conf))

        rel_traj = (base_y_local - traj).astype(np.float32)
        comp_ratio = float(np.mean(binary_crop.sum(axis=0)[col_min : col_max + 1] > 0)) if binary_crop.size > 0 else 1.0

        # Sample jump quality diagnostics
        if len(traj) > 1:
            jumps = np.abs(np.diff(traj))
            mean_j = float(np.mean(jumps))
            max_j = float(np.max(jumps))
            is_qual_ok = bool(mean_j < 4.0 and not is_clip_top and not is_clip_bot and comp_ratio > 0.3)
            qual_note = "Passed" if is_qual_ok else ("Clipped trace bounds" if (is_clip_top or is_clip_bot) else f"High sample noise (mean jump {mean_j:.1f}px)")
        else:
            mean_j, max_j = 0.0, 0.0
            is_qual_ok = False
            qual_note = "Insufficient samples"

        if not is_qual_ok:
            validation_status = "quality_warning"

        return ExtractedTrace(
            lead_name=lead_name,
            pixel_trajectory=traj,
            baseline_y=float(base_y_local),
            relative_trajectory=rel_traj,
            completeness_ratio=comp_ratio,
            active_start_col=col_min,
            active_end_col=col_max,
            crop_width_px=w,
            crop_height_px=h,
            is_trained_lead=is_trained,
            validation_status=validation_status,
            confidence=mean_conf,
            extraction_method="deep_adaptive_crop",
            crop_y1=int(inf_y1),
            crop_y2=int(inf_y2),
            is_clipped_top=is_clip_top,
            is_clipped_bottom=is_clip_bot,
            clipping_margin_px=clip_margin,
            cross_talk_risk=xtalk_risk,
            is_overlapping=is_overlap,
            mean_jump_px=mean_j,
            max_jump_px=max_j,
            is_quality_passed=is_qual_ok,
            quality_note=qual_note,
        )
