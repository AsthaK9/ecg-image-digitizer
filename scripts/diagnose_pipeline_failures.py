"""Reproducible Pipeline Diagnostics Script for Phase 0.

Reproduces and diagnoses:
1. Binary trace segmentation failure (blackout / lost traces / color traces).
2. Grid suppression trace erosion.
3. Skew estimation angle error and rotation direction.
4. Lead layout detection on 12-lead synthetic sheet vs single-lead beat.
5. End-to-end reconstruction vs ground truth.
Saves all intermediate and diagnostic images into data/interim/diagnostics/.
"""

import json
import logging
from pathlib import Path
import cv2
import numpy as np

from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import ECGRecordPaths
from ecg_digitization.preprocessing.enhancement import (
    PreprocessingPipeline,
    estimate_skew_angle,
    deskew_image,
    suppress_grid,
    binarize_trace,
)
from ecg_digitization.preprocessing.lead_detector import LeadLayoutDetector
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from ecg_digitization.synthetic.generator import SyntheticECGGenerator, DegradationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIAG_DIR = Path("data/interim/diagnostics")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def test_1_deskew_math():
    """Test deskew angle estimation and rotation on controlled images."""
    logger.info("=== TEST 1: Deskew Estimation & Direction ===")
    # Create an image with horizontal black grid lines on white background
    w, h = 1000, 800
    base = np.full((h, w, 3), 255, dtype=np.uint8)
    for y in range(50, h - 50, 40):
        cv2.line(base, (50, y), (w - 50, y), (0, 0, 0), 2)

    known_angles = [0.0, 0.8, 2.5, -3.0]
    results = {}
    for angle in known_angles:
        # Rotate image clockwise by angle
        center = (w / 2.0, h / 2.0)
        # In OpenCV: positive angle in getRotationMatrix2D rotates COUNTER-CLOCKWISE.
        # So a clockwise rotation is -angle:
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(base, rot_mat, (w, h), borderValue=(255, 255, 255))

        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        estimated_angle = estimate_skew_angle(gray)

        # Now test deskew_image on this rotated image
        deskewed = deskew_image(rotated, estimated_angle)
        gray_deskewed = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        residual_angle = estimate_skew_angle(gray_deskewed)

        results[f"angle_{angle}"] = {
            "known_angle_clockwise": angle,
            "estimated_angle": round(estimated_angle, 3),
            "residual_angle_after_deskew": round(residual_angle, 3),
        }
        logger.info(
            "Known Clockwise: %.2f deg | Estimated: %.2f deg | Residual after deskew: %.2f deg",
            angle,
            estimated_angle,
            residual_angle,
        )

    with open(DIAG_DIR / "test1_deskew_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2)

def test_2_preprocessing_failures():
    """Test binary segmentation and grid suppression on real and synthetic images."""
    logger.info("=== TEST 2: Preprocessing Failures (Grid Suppression & Binary Mask) ===")
    gen = SyntheticECGGenerator()
    parser = WFDBRecordParser()
    sig = parser.parse(
        ECGRecordPaths(
            record_id="00001_lr",
            header_path=Path("data/interim/ptbxl_sample/00001_lr.hea"),
            signal_path=Path("data/interim/ptbxl_sample/00001_lr.dat"),
        ),
        source_dataset="ptb_xl",
    )
    synth = gen.generate(sig, seed=42)
    synth_img = synth.image

    # 1. Test on Synthetic 12-lead sheet
    cv2.imwrite(str(DIAG_DIR / "synth_01_original.png"), synth_img)
    supp_synth = suppress_grid(synth_img)
    cv2.imwrite(str(DIAG_DIR / "synth_02_grid_suppressed.png"), supp_synth)
    bin_synth = binarize_trace(supp_synth)
    cv2.imwrite(str(DIAG_DIR / "synth_03_binary_trace.png"), bin_synth)

    white_pixel_pct_synth = float(np.mean(bin_synth == 255) * 100.0)
    logger.info("Synthetic 12-lead sheet: Binary trace white pixel %%: %.4f%%", white_pixel_pct_synth)

    # 2. Test on Real Single Lead N10007.png
    real_sample_path = Path("data/real_ecg_eval/samples/N10007.png")
    if real_sample_path.is_file():
        real_img = cv2.imread(str(real_sample_path))
        cv2.imwrite(str(DIAG_DIR / "real_01_original.png"), real_img)
        supp_real = suppress_grid(real_img)
        cv2.imwrite(str(DIAG_DIR / "real_02_grid_suppressed.png"), supp_real)
        bin_real = binarize_trace(supp_real)
        cv2.imwrite(str(DIAG_DIR / "real_03_binary_trace.png"), bin_real)
        white_pixel_pct_real = float(np.mean(bin_real == 255) * 100.0)
        logger.info("Real N10007.png: Binary trace white pixel %%: %.4f%%", white_pixel_pct_real)

    # 3. Test on Colored Trace (Cyan and Red trace)
    colored_cyan = synth_img.copy()
    # Replace black trace with cyan (BGR: 255, 255, 0)
    black_pixels = np.all(synth_img < 50, axis=-1)
    colored_cyan[black_pixels] = [255, 200, 0] # BGR cyan
    cv2.imwrite(str(DIAG_DIR / "cyan_01_original.png"), colored_cyan)
    supp_cyan = suppress_grid(colored_cyan)
    cv2.imwrite(str(DIAG_DIR / "cyan_02_grid_suppressed.png"), supp_cyan)
    bin_cyan = binarize_trace(supp_cyan)
    cv2.imwrite(str(DIAG_DIR / "cyan_03_binary_trace.png"), bin_cyan)
    white_pixel_pct_cyan = float(np.mean(bin_cyan == 255) * 100.0)
    logger.info("Cyan Trace: Binary trace white pixel %%: %.4f%%", white_pixel_pct_cyan)

def test_3_layout_detection():
    """Test layout detection on synthetic sheet and single lead."""
    logger.info("=== TEST 3: Layout Detection ===")
    detector = LeadLayoutDetector()
    gen = SyntheticECGGenerator()
    parser = WFDBRecordParser()
    sig = parser.parse(
        ECGRecordPaths(
            record_id="00001_lr",
            header_path=Path("data/interim/ptbxl_sample/00001_lr.hea"),
            signal_path=Path("data/interim/ptbxl_sample/00001_lr.dat"),
        ),
        source_dataset="ptb_xl",
    )
    synth = gen.generate(sig, seed=42)

    res_3x4 = detector.detect_layout(synth.image.shape, layout_hint="3x4_1")
    logger.info("3x4_1 detected leads count: %d, leads: %s", len(res_3x4.leads), list(res_3x4.leads.keys()))

    real_img = cv2.imread("data/real_ecg_eval/samples/N10007.png")
    if real_img is not None:
        res_single = detector.detect_layout(real_img.shape, layout_hint="single_lead")
        logger.info("single_lead detected leads count: %d, leads: %s", len(res_single.leads), list(res_single.leads.keys()))

if __name__ == "__main__":
    test_1_deskew_math()
    test_2_preprocessing_failures()
    test_3_layout_detection()
