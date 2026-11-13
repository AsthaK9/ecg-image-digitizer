"""ECG Image Preprocessing Pipeline.

Provides independently testable stages:
1. Image validation and loading
2. Grayscale & contrast enhancement (CLAHE)
3. Deskewing via Radon/Hough angle estimation
4. Grid detection and suppression (Color filtering + Morphological filtering)
5. Waveform binarization and morphological cleaning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Stores intermediate and final representations for inspection and downstream stages."""

    original_bgr: np.ndarray
    grayscale: np.ndarray
    contrast_enhanced: np.ndarray
    deskewed: np.ndarray
    grid_suppressed: np.ndarray
    binary_trace: np.ndarray
    skew_angle_deg: float


def load_and_validate_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Loads and validates an image from disk or an existing numpy array."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"Image file does not exist: {p}")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from file: {p}")
    elif isinstance(source, np.ndarray):
        img = source.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(source)}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {img.shape}; expected (H, W, 3)")

    if img.shape[0] < 50 or img.shape[1] < 50:
        raise ValueError(f"Image dimensions too small: {img.shape}")

    return img


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """Converts a BGR image to single-channel 8-bit grayscale."""
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def estimate_skew_angle(img_gray: np.ndarray, max_angle_deg: float = 15.0) -> float:
    """Estimates skew angle of grid lines or page margins using the Hough line transform.
    
    Returns the angle in degrees by which the image is tilted clockwise.
    A positive angle indicates clockwise tilt.
    """
    # Use horizontal morphological gradient to enhance horizontal grid lines over vertical QRS spikes
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    grad = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, horiz_kernel)
    edges = cv2.Canny(grad, 40, 120, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=15)

    if lines is None or len(lines) == 0:
        # Fallback to standard Canny
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return 0.0

    lines = lines.reshape(-1, 4)
    angles = []
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-4:
            continue
        angle_deg = np.degrees(np.arctan2(dy, dx))
        # Keep horizontal lines within [-max_angle, max_angle]
        if abs(angle_deg) <= max_angle_deg:
            angles.append(angle_deg)
        elif abs(angle_deg - 90.0) <= max_angle_deg:
            angles.append(angle_deg - 90.0)
        elif abs(angle_deg + 90.0) <= max_angle_deg:
            angles.append(angle_deg + 90.0)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    # Threshold gate: do not apply deskew to images that are already aligned within 0.2 deg
    if abs(median_angle) < 0.2:
        return 0.0

    return median_angle


def deskew_image(img: np.ndarray, angle_deg: float, border_value: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Rotates image to correct skew.
    
    A positive angle_deg indicates clockwise tilt; in OpenCV getRotationMatrix2D,
    a positive angle rotates counter-clockwise, which correctly cancels out the clockwise tilt.
    """
    if abs(angle_deg) < 0.1:
        return img

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    bv = border_value if img.ndim == 3 else border_value[0]

    return cv2.warpAffine(
        img,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bv,
    )


def detect_grid_presence(img_bgr: np.ndarray) -> bool:
    """Detects if a colored (pink/red/orange) clinical paper grid is present."""
    if img_bgr.ndim == 2:
        return False
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # Pink/red grid in HSV: hue near 0 or 180, moderate saturation, brightness above paper shadow
    pink_grid_mask = ((hue < 30) | (hue > 150)) & (sat > 16) & (val > 70)
    pct_pink = float(np.mean(pink_grid_mask) * 100.0)
    return pct_pink > 1.0


def suppress_grid(img_bgr: np.ndarray) -> np.ndarray:
    """Suppresses colored paper grid lines (pink/red/orange) while keeping black or colored trace ink.

    1. Checks if a colored paper grid is present. If absent (e.g. digital plot or plain crop),
       returns standard grayscale without risk of eroding trace lines.
    2. When grid is present:
       - Uses the Red channel where pink grid lines reflect strongly.
       - Suppresses pink grid pixels while strictly protecting dark trace ink (val < 110)
         and non-red colored traces.
    """
    if img_bgr.ndim == 2:
        return img_bgr.copy()

    has_grid = detect_grid_presence(img_bgr)
    if not has_grid:
        # No colored grid: return grayscale directly to preserve waveform completely
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Grid pixels: reddish/pinkish hue, moderate saturation, and MUST be bright (val >= 110)
    # Dark pixels (val < 110) are NEVER grid lines — they are ECG waveform ink!
    is_pink_grid = ((hue < 25) | (hue > 155)) & (sat > 25) & (val >= 110)

    # In addition, check that red channel is significantly brighter than green/blue
    is_red_dominant = (r.astype(np.int16) - g.astype(np.int16) > 15) & (r.astype(np.int16) - b.astype(np.int16) > 15)
    grid_mask = is_pink_grid & is_red_dominant

    suppressed = r.copy()
    suppressed[grid_mask] = 255
    return suppressed


def binarize_trace(img_gray: np.ndarray, block_size: int = 15, c: int = 5) -> np.ndarray:
    """Binarizes the ECG trace into a foreground mask (255 = trace, 0 = background).

    Uses adaptive Gaussian thresholding with contrast gating and area-based noise filtering.
    Preserves continuous, thin 1-pixel waveform lines without morphological erosion.
    """
    # Invert so trace is bright (background is dark)
    inv = cv2.bitwise_not(img_gray)

    # Adaptive Gaussian threshold
    if block_size % 2 == 0:
        block_size += 1
    adapt = cv2.adaptiveThreshold(
        inv,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )

    # Contrast gating: pixel must differ by at least 20 levels from pure background
    contrast_mask = (inv > 20).astype(np.uint8) * 255
    combined = cv2.bitwise_and(adapt, contrast_mask)

    # Remove only isolated 1-2 pixel salt-and-pepper noise specks via connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    cleaned = np.zeros_like(combined)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Keep any connected component of size >= 3 (preserves thin continuous traces)
        if area >= 3:
            cleaned[labels == i] = 255

    return cleaned


class PreprocessingPipeline:
    """Unified preprocessor executing deskew, grid suppression, and binarization."""

    def __init__(self, enable_deskew: bool = True, max_skew_angle: float = 15.0) -> None:
        self.enable_deskew = enable_deskew
        self.max_skew_angle = max_skew_angle

    def process(self, source: Union[str, Path, np.ndarray]) -> PreprocessingResult:
        """Runs full preprocessing pipeline and returns intermediate and final stages."""
        img_bgr = load_and_validate_image(source)
        gray = to_grayscale(img_bgr)
        enhanced = enhance_contrast(gray)

        # Deskewing: only estimate skew from grid lines if a grid is actually present.
        # On gridless beat crops or digital plots (e.g. N10007), Hough lines on the waveform trace
        # detect cardiac wave slopes and falsely rotate the image.
        skew_angle = 0.0
        deskewed_bgr = img_bgr
        if self.enable_deskew and detect_grid_presence(img_bgr):
            skew_angle = estimate_skew_angle(enhanced, max_angle_deg=self.max_skew_angle)
            if abs(skew_angle) >= 0.2:
                deskewed_bgr = deskew_image(img_bgr, skew_angle)

        # Grid suppression on deskewed BGR image
        grid_suppressed = suppress_grid(deskewed_bgr)

        # Binarize waveform trace
        binary_trace = binarize_trace(grid_suppressed)

        return PreprocessingResult(
            original_bgr=img_bgr,
            grayscale=gray,
            contrast_enhanced=enhanced,
            deskewed=deskewed_bgr,
            grid_suppressed=grid_suppressed,
            binary_trace=binary_trace,
            skew_angle_deg=skew_angle,
        )
