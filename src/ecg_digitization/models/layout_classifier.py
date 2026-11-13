"""Separate Layout Classifier for ECG Sheet Architecture Detection.

Distinguishes between standard clinical ECG layouts:
1. single_lead: Single isolated lead or beat crop.
2. 3x4: Standard 12-lead sheet (4 columns x 3 rows) without rhythm strip.
3. 3x4_1: Standard clinical sheet (4 columns x 3 rows + 1 full-width rhythm strip at bottom).
4. 3x4_3: Marquette MAC-style sheet (4 columns x 3 rows + 3 rhythm strips at bottom).
5. 12x1: Stacked 12 full-width continuous rows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

LAYOUT_CLASSES: List[str] = [
    "single_lead",
    "3x4",
    "3x4_1",
    "3x4_3",
    "12x1",
]


class ECGLayoutCNN(nn.Module):
    """Lightweight 2D CNN for global ECG paper sheet layout classification."""

    def __init__(self, num_classes: int = len(LAYOUT_CLASSES), in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x96

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x48

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x24

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 3)),  # 128 x 2 x 3 = 768
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * 2 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


class LayoutDetectorService:
    """End-to-end layout detection combining geometric rules, projection periodicity,
    and the trained ECGLayoutCNN model."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: Optional[str] = None) -> None:
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.model: Optional[ECGLayoutCNN] = None
        self.classes = LAYOUT_CLASSES

        if model_path and Path(model_path).is_file():
            try:
                self.model = ECGLayoutCNN(num_classes=len(self.classes))
                ckpt = torch.load(model_path, map_location=self.device)
                state = ckpt.get("model_state", ckpt)
                self.model.load_state_dict(state)
                self.model.to(self.device).eval()
                logger.info("Loaded ECGLayoutCNN model from %s", model_path)
            except Exception as exc:
                logger.warning("Could not load layout model from %s: %s", model_path, exc)
                self.model = None

    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (128, 192)) -> torch.Tensor:
        """Preprocesses input ECG image for the layout classifier."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Invert so trace is bright (trace foreground = high activation)
        inverted = 255 - gray
        resized = cv2.resize(inverted, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        return tensor

    def detect_active_canvas(self, image: np.ndarray) -> Tuple[int, int, int, int, str]:
        """Detects the active ECG grid/waveform canvas bounding box, cleanly excluding
        non-grid patient metadata header tables, stamps, and footer annotation strips.

        Returns:
            Tuple of (y_min, y_max, x_min, x_max, canvas_mode).
        """
        h, w = image.shape[:2]
        aspect = w / float(h)
        if (w < 500 and h < 350 and aspect < 1.6) or aspect < 1.15:
            return 0, h, 0, w, "single_lead"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        has_color_grid = False
        y1_grid, y2_grid = 0, h

        # 1. Color grid bounding detection (for sheets with pink, tan, or cyan grids)
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            sat = np.abs(r.astype(int) - g.astype(int)) + np.abs(r.astype(int) - b.astype(int))
            if np.percentile(sat, 98) > 25:
                is_grid = (r.astype(int) - g.astype(int) >= 4) & (r.astype(int) - b.astype(int) >= 4) & (r > 150)
                row_grid = np.mean(is_grid, axis=1)
                grid_rows = np.where(row_grid > 0.05)[0]
                if len(grid_rows) > 0 and (grid_rows[-1] - grid_rows[0]) > 0.55 * h:
                    y1_grid = max(0, grid_rows[0] - 2)
                    y2_grid = min(h, grid_rows[-1] + 2)
                    has_color_grid = True

        # 2. Horizontal divider rules / header boundaries
        bin_inv = (gray < 150).astype(np.uint8) * 255
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.25)), 1))
        solid_hlines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, horiz_kernel)
        hlines_y = np.where(solid_hlines.sum(axis=1) > 0)[0]

        header_lines = [y for y in hlines_y if y < 0.25 * h]
        y1_rule = header_lines[-1] + 4 if header_lines else int(0.02 * h)

        footer_lines = [y for y in hlines_y if y > 0.88 * h]
        y2_rule = footer_lines[0] - 4 if footer_lines else int(0.98 * h)

        if has_color_grid:
            y1 = max(y1_grid, y1_rule) if header_lines else y1_grid
            y2 = min(y2_grid, y2_rule) if footer_lines else y2_grid
            mode = "color_grid"
        else:
            y1 = y1_rule
            y2 = y2_rule
            mode = "monochrome_rules"

        y1 = max(0, min(int(y1), int(0.35 * h)))
        y2 = min(h, max(int(y2), int(0.65 * h)))
        x1 = max(0, int(w * 0.02))
        x2 = min(w, int(w * 0.98))
        return y1, y2, x1, x2, mode

    def detect_layout_detailed(self, image: np.ndarray) -> Dict[str, Any]:
        """Detects layout using active canvas localization, solid line suppression,
        structural row periodicity, and deep CNN predictions.

        Returns:
            Dict containing predicted_layout, confidence, is_ambiguous,
            class_probabilities, n_rows, active_canvas, notes.
        """
        h, w = image.shape[:2]
        aspect = w / float(h)

        # 1. Fast geometric check for single-lead beat crops
        if aspect < 1.15 or (w < 450 and h < 300) or (w < 600 and h < 350 and aspect < 1.4):
            probs = {cls: (1.0 if cls == "single_lead" else 0.0) for cls in self.classes}
            return {
                "predicted_layout": "single_lead",
                "confidence": 1.0,
                "is_ambiguous": False,
                "class_probabilities": probs,
                "n_rows": 1,
                "active_canvas": (0, h, 0, w),
                "notes": "Single lead / beat crop detected via image dimensions and aspect ratio.",
            }

        # 2. Localize active waveform canvas
        y1, y2, x1, x2, canvas_mode = self.detect_active_canvas(image)
        active_h = max(10, y2 - y1)
        active_w = max(10, x2 - x1)

        # 3. Waveform ink extraction with solid horizontal line suppression
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        bin_inv = (gray < 150).astype(np.uint8) * 255
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.20)), 1))
        solid_hlines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, horiz_kernel)
        waveform_ink = cv2.subtract(bin_inv, solid_hlines)

        canvas_ink = waveform_ink[y1:y2, x1:x2]
        proj = np.sum(canvas_ink > 0, axis=1).astype(np.float32)

        win = max(7, int(active_h // 35))
        smooth = np.convolve(proj, np.ones(win) / float(win), mode="same")

        import scipy.signal
        peaks, _ = scipy.signal.find_peaks(
            smooth,
            distance=max(12, int(active_h / 18)),
            prominence=max(2.0, float(np.max(smooth)) * 0.08),
        )
        n_rows = len(peaks)

        # 4. CNN prediction on active canvas
        cnn_probs: Dict[str, float] = {}
        cnn_pred: Optional[str] = None
        cnn_conf: float = 0.0

        if self.model is not None:
            canvas_crop = image[y1:y2, x1:x2]
            tensor = self.preprocess_image(canvas_crop).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs_t = F.softmax(logits, dim=1).squeeze(0)
                probs_np = probs_t.cpu().numpy()
                cnn_probs = {self.classes[i]: float(probs_np[i]) for i in range(len(self.classes))}
                best_idx = int(np.argmax(probs_np))
                cnn_pred = self.classes[best_idx]
                cnn_conf = float(probs_np[best_idx])

        # 5. Hybrid Decision Rules combining physical structure and CNN
        is_ambiguous = False
        notes = ""

        if n_rows >= 10:
            pred = "12x1"
            conf = max(cnn_conf, 0.95) if cnn_pred == "12x1" else 0.92
            notes = f"Detected {n_rows} stacked waveform rows (12x1 layout)."
        elif n_rows in (5, 6, 7):
            pred = "3x4_3"
            conf = max(cnn_conf, 0.95) if cnn_pred == "3x4_3" else 0.92
            notes = f"Detected {n_rows} waveform rows (3 grid rows + 3 continuous rhythm strips, Marquette MAC format)."
        elif n_rows == 4:
            pred = "3x4_1"
            conf = max(cnn_conf, 0.96) if cnn_pred == "3x4_1" else 0.94
            notes = "Detected 4 waveform rows (3x4 grid + 1 continuous bottom rhythm strip)."
        elif n_rows == 3:
            pred = "3x4"
            conf = max(cnn_conf, 0.93) if cnn_pred == "3x4" else 0.90
            notes = "Detected 3 waveform rows (3x4 grid with no rhythm strip)."
        elif n_rows == 1:
            pred = "single_lead"
            conf = 0.95
            notes = "Single waveform row detected."
        else:
            # Ambiguous row count
            if cnn_pred is not None and cnn_conf > 0.65:
                pred = cnn_pred
                conf = cnn_conf * 0.85
                is_ambiguous = True
                notes = f"Row baseline projection inconclusive ({n_rows} peaks); CNN predicted {pred} with {cnn_conf*100:.0f}% confidence."
            elif aspect > 1.3:
                pred = "3x4_1"
                conf = 0.65
                is_ambiguous = True
                notes = f"Row baseline count ambiguous ({n_rows} peaks); defaulting to standard clinical 3x4_1."
            else:
                pred = "single_lead"
                conf = 0.65
                is_ambiguous = True
                notes = f"Ambiguous row structure ({n_rows} peaks); defaulting to single lead."

        # Calibrate posterior class probabilities using physical row evidence
        struct_prior = {cls: 0.02 for cls in self.classes}
        if n_rows >= 10:
            struct_prior["12x1"] = 0.92
        elif n_rows in (5, 6, 7):
            struct_prior["3x4_3"] = 0.92
        elif n_rows == 4:
            struct_prior["3x4_1"] = 0.92
        elif n_rows == 3:
            struct_prior["3x4"] = 0.92
        elif n_rows == 1:
            struct_prior["single_lead"] = 0.92

        tot_sp = sum(struct_prior.values())
        struct_prior = {k: v / tot_sp for k, v in struct_prior.items()}

        if not cnn_probs:
            cnn_probs = struct_prior
            conf = float(cnn_probs[pred])
        else:
            # Weighted mixture: 70% physical structural row evidence + 30% deep CNN
            alpha = 0.70
            fused_probs = {
                cls: float(alpha * struct_prior.get(cls, 0.0) + (1.0 - alpha) * cnn_probs.get(cls, 0.0))
                for cls in self.classes
            }
            cnn_probs = fused_probs
            conf = float(cnn_probs[pred])

        is_ambiguous = bool(conf < 0.60)

        return {
            "predicted_layout": pred,
            "confidence": float(conf),
            "is_ambiguous": is_ambiguous,
            "class_probabilities": cnn_probs,
            "n_rows": n_rows,
            "active_canvas": (y1, y2, x1, x2),
            "notes": notes,
        }

    def detect_layout(self, image: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Detects layout using hybrid geometric analysis, row projection peaks, and CNN predictions.

        Returns:
            Tuple of (layout_name, confidence, class_probabilities).
        """
        detailed = self.detect_layout_detailed(image)
        return detailed["predicted_layout"], detailed["confidence"], detailed["class_probabilities"]

    def _infer_layout_projection(self, image: np.ndarray) -> Tuple[str, float]:
        """Robust feature-based projection analysis separating text and waveforms."""
        h, w = image.shape[:2]
        aspect = w / float(h)
        if aspect < 1.15 or (w < 450 and h < 300):
            return "single_lead", 0.95

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        med = float(np.median(gray))
        thresh_val = min(150, max(70, int(med * 0.72)))
        ink = (gray < thresh_val).astype(np.float32)

        # Ignore outer 4% margins
        y1, y2 = int(0.04 * h), int(0.96 * h)
        x1, x2 = int(0.04 * w), int(0.96 * w)
        proj = np.sum(ink[y1:y2, x1:x2], axis=1)

        win = max(7, int(h // 40))
        smooth = np.convolve(proj, np.ones(win) / float(win), mode="same")

        import scipy.signal
        peaks, _ = scipy.signal.find_peaks(
            smooth,
            distance=max(12, int(h / 18)),
            prominence=max(10.0, float(np.max(smooth)) * 0.08),
        )

        wave_rows = []
        for p in peaks:
            abs_y = p + y1
            strip = ink[max(0, abs_y - 12):min(h, abs_y + 12), x1:x2]
            col_cov = float(np.mean(np.sum(strip, axis=0) > 0))
            if col_cov >= 0.15:
                wave_rows.append((abs_y, col_cov))

        n_r = len(wave_rows)
        if n_r >= 10:
            return "12x1", 0.92
        elif n_r in (5, 6, 7):
            return "3x4_3", 0.88
        elif n_r == 4:
            return "3x4_1", 0.90
        elif n_r == 3:
            return "3x4", 0.85

        # Fallback based on aspect ratio
        if aspect > 1.3:
            return "3x4_1", 0.70
        return "single_lead", 0.75
