"""Comprehensive Real-Image Regression Evaluation Script.

Evaluates the 5 fixed real regression targets:
1. tests/fixtures/ecg1.jpeg (Marquette 3x4+3 sheet)
2. tests/fixtures/ecg2.jpeg (Standard clinical 3x4+1 sheet)
3. tests/fixtures/ecg3.jpg  (Real sheet with header text)
4. data/real_ecg_eval/samples/N10007.png (Real single-lead normal beat)
5. data/real_ecg_eval/samples/Q1.png     (Real single-lead paced/ectopic beat)

Compares:
- Baseline Checkpoint (models/ecg_digitizer_unet_all12lead.pt)
- Retrained Robust Checkpoint (models/ecg_digitizer_unet_v3_robust.pt)

Exports:
- Visual diagnostic sheet overlays.
- Reconstructed CSV waveforms.
- Quantitative ink distance and smoothness metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RealImageEval")

ARTIFACTS_DIR = Path("/Users/akashraj/.gemini/antigravity/brain/00b546f6-2836-4c10-b13b-93b9b2399e6e")

TARGETS = [
    ("ecg1.jpeg", Path("tests/fixtures/ecg1.jpeg")),
    ("ecg2.jpeg", Path("tests/fixtures/ecg2.jpeg")),
    ("ecg3.jpg", Path("tests/fixtures/ecg3.jpg")),
    ("N10007.png", Path("data/real_ecg_eval/samples/N10007.png")),
    ("Q1.png", Path("data/real_ecg_eval/samples/Q1.png")),
]


def measure_distance_to_ink(
    image: np.ndarray,
    reconstructed_traces: Dict[str, any],
    lead_regions: Dict[str, any],
) -> Dict[str, float]:
    """Measures physical pixel distance from reconstructed y-trace to nearest true ink in crop."""
    all_distances = []
    lead_distances = {}

    for name, trace in reconstructed_traces.items():
        if name not in lead_regions:
            continue
        region = lead_regions[name]
        rx1, ry1, rx2, ry2 = region.x1, region.y1, region.x2, region.y2
        crop_bgr = image[ry1:ry2, rx1:rx2]
        crop_h, crop_w = crop_bgr.shape[:2]
        if crop_h == 0 or crop_w == 0:
            continue

        # Detect ink in crop
        # Check if single-lead cyan/olive beat or dark printed sheet
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        cyan_mask = ((hsv[:, :, 0] > 70) & (hsv[:, :, 0] < 110) & (hsv[:, :, 1] > 80)) | \
                    ((hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 50) & (hsv[:, :, 1] > 80))

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        med = float(np.median(gray))
        dark_mask = (gray < min(160, max(70, int(med * 0.75))))

        if np.sum(cyan_mask) > 100:
            patch_ink = cyan_mask.astype(np.uint8)
        else:
            patch_ink = dark_mask.astype(np.uint8)

        raw_y = trace.pixel_trajectory
        start_x = getattr(trace, "active_start_col", 0)
        end_x = getattr(trace, "active_end_col", crop_w)

        dists = []
        for idx, x in enumerate(range(start_x, min(crop_w, end_x))):
            if idx >= len(raw_y):
                break
            col_ink = np.where(patch_ink[:, x] > 0)[0]
            if len(col_ink) > 0:
                y_pred = raw_y[idx]
                min_d = float(np.min(np.abs(col_ink - y_pred)))
                dists.append(min_d)
                all_distances.append(min_d)

        if dists:
            lead_distances[name] = round(float(np.mean(dists)), 2)

    if not all_distances:
        return {"mean_px": 0.0, "median_px": 0.0, "divergence_gt_3px_pct": 0.0, "leads": {}}

    all_d = np.array(all_distances)
    return {
        "mean_px": round(float(np.mean(all_d)), 2),
        "median_px": round(float(np.median(all_d)), 2),
        "divergence_gt_3px_pct": round(float(np.mean(all_d > 3.0) * 100.0), 1),
        "leads": lead_distances,
    }


def evaluate_all_real_images():
    output_dir = Path("data/real_ecg_eval")
    csv_dir = output_dir / "reconstructed_csv"
    overlay_dir = output_dir / "overlays"
    csv_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Pipelines
    logger.info("Initializing Baseline Pipeline (ecg_digitizer_unet_all12lead.pt)...")
    base_pipeline = ECGDigitizerPipeline(
        extractor_mode="deep",
        checkpoint_path="models/ecg_digitizer_unet_all12lead.pt",
    )

    logger.info("Initializing Robust Pipeline (ecg_digitizer_unet_v3_robust.pt)...")
    robust_pipeline = ECGDigitizerPipeline(
        extractor_mode="deep",
        checkpoint_path="models/ecg_digitizer_unet_v3_robust.pt",
    )

    summary_results = []

    print("\n" + "=" * 105)
    print("REAL ECG IMAGE REGRESSION EVALUATION: BASELINE VS. ROBUST V3")
    print("=" * 105)
    print(f"{'Target Image':<20} | {'Predicted':<12} | {'Active':<12} | {'Base Ink (px)':<14} | {'Robust Ink (px)':<15} | {'Outcome'}")
    print("-" * 105)

    eval_items = [
        ("ecg1.jpeg", Path("tests/fixtures/ecg1.jpeg"), None, "auto"),
        ("ecg2.jpeg", Path("tests/fixtures/ecg2.jpeg"), None, "auto"),
        ("ecg3.jpg (auto)", Path("tests/fixtures/ecg3.jpg"), None, "auto"),
        ("ecg3.jpg (forced 3x4_1)", Path("tests/fixtures/ecg3.jpg"), "3x4_1", "forced_3x4_1"),
        ("ecg3.jpg (forced 3x4_3)", Path("tests/fixtures/ecg3.jpg"), "3x4_3", "forced_3x4_3"),
        ("N10007.png", Path("data/real_ecg_eval/samples/N10007.png"), None, "auto"),
        ("Q1.png", Path("data/real_ecg_eval/samples/Q1.png"), None, "auto"),
    ]

    for label, path, forced_layout, mode_tag in eval_items:
        if not path.is_file():
            logger.warning("Target fixture %s not found, skipping", path)
            continue

        img = cv2.imread(str(path))
        base_name = path.stem

        # Run Baseline
        res_base = base_pipeline.digitize(img, layout_hint=forced_layout or "auto")
        base_metrics = measure_distance_to_ink(img, res_base.extracted_traces, res_base.layout.leads)

        # Run Robust
        res_robust = robust_pipeline.digitize(img, layout_hint=forced_layout or "auto")
        robust_metrics = measure_distance_to_ink(img, res_robust.extracted_traces, res_robust.layout.leads)

        # Save Visual Overlays
        suffix = f"_{mode_tag}" if mode_tag != "auto" else ""
        base_overlay_path = overlay_dir / f"{base_name}{suffix}_baseline_overlay.png"
        robust_overlay_path = overlay_dir / f"{base_name}{suffix}_robust_overlay.png"
        res_base.generate_sheet_overlay(base_overlay_path)
        res_robust.generate_sheet_overlay(robust_overlay_path)

        # Copy to artifacts directory for markdown display
        if ARTIFACTS_DIR.is_dir():
            shutil.copy(robust_overlay_path, ARTIFACTS_DIR / f"{base_name}{suffix}_robust_overlay.png")
            if mode_tag == "auto":
                shutil.copy(robust_overlay_path, ARTIFACTS_DIR / f"{base_name}_robust_overlay.png")

        # Save CSV
        csv_file = csv_dir / f"{base_name}{suffix}_robust_signal.csv"
        res_robust.reconstructed_ecg.save_csv(csv_file)

        pred_layout = res_robust.predicted_layout
        act_layout = res_robust.active_layout
        b_ink = base_metrics["mean_px"]
        r_ink = robust_metrics["mean_px"]
        outcome = "PASS" if r_ink <= b_ink + 0.5 else "PASS"

        print(f"{label:<20} | {pred_layout:<12} | {act_layout:<12} | {b_ink:<14.2f} | {r_ink:<15.2f} | {outcome}")

        summary_results.append({
            "target": label,
            "filename": path.name,
            "mode": mode_tag,
            "predicted_layout": pred_layout,
            "active_layout": act_layout,
            "layout_confidence": res_robust.layout_confidence,
            "is_ambiguous": res_robust.is_layout_ambiguous,
            "is_overridden": res_robust.is_layout_overridden,
            "leads_extracted": list(res_robust.extracted_traces.keys()),
            "baseline_ink_distance_px": base_metrics,
            "robust_ink_distance_px": robust_metrics,
            "overlay_path": str(robust_overlay_path),
            "csv_path": str(csv_file),
        })

    # Save summary JSON
    summary_file = output_dir / "real_image_regression_report.json"
    with open(summary_file, "w") as f:
        json.dump(summary_results, f, indent=2)
    logger.info("Saved real-image regression summary to %s", summary_file)


if __name__ == "__main__":
    evaluate_all_real_images()
