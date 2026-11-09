"""Real ECG Image Inference Validation Script.

Processes real ECG beat images from ECG Image Data using the trained ECGDigitizerUNet model.
Exports:
1. Reconstructed CSV waveforms (normalized amplitudes and nominal voltages).
2. Visual overlay plots comparing original traces against model extractions.
3. Summary metrics and calibration limitation report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import zipfile
import cv2
import numpy as np

from ecg_digitization.inference.deep_digitizer import DeepCropDigitizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealInference")


def run_real_ecg_validation():
    # 1. Paths
    zip_path = Path("data/raw/ecg-image-data/ecg-image-data.zip")
    base_eval_dir = Path("data/real_ecg_eval")
    samples_dir = base_eval_dir / "samples"
    csv_dir = base_eval_dir / "reconstructed_csv"
    overlays_dir = base_eval_dir / "overlays"

    for d in [samples_dir, csv_dir, overlays_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Select 4 diverse real ECG images from the archive
    # Pick Normal (N), Premature Ventricular Contraction (V), Fusion beat (F), and Myocardial Infarction (M)
    selected_targets = [
        ("N10007.png", "ECG_Image_data/test/N/N10007.png", "Normal Sinus Rhythm (N)"),
        ("V1001.png", "ECG_Image_data/test/V/V1001.png", "Premature Ventricular Contraction (V)"),
        ("F0.png", "ECG_Image_data/test/F/F0.png", "Fusion Beat (F)"),
        ("M0.png", "ECG_Image_data/test/M/M0.png", "Myocardial Infarction / Abnormal Beat (M)"),
    ]

    extracted_samples = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name, zip_internal_path, category in selected_targets:
            dest_file = samples_dir / file_name
            with open(dest_file, "wb") as f:
                f.write(z.read(zip_internal_path))
            extracted_samples.append((dest_file, category))
            logger.info("Extracted %s (%s)", file_name, category)

    # 3. Initialize Deep Digitizer Engine
    digitizer = DeepCropDigitizer(checkpoint_path="models/ecg_digitizer_unet.pt")

    results_summary = []

    print("\n" + "=" * 90)
    print("REAL ECG IMAGE INFERENCE RESULTS (ECGDigitizerUNet)")
    print("=" * 90)
    print(f"{'Sample File':<14} | {'Category':<32} | {'Content BBox':<16} | {'Mean Conf':<10} | {'V_p-p (mV)':<11} | {'Calib Status'}")
    print("-" * 105)

    for img_path, category in extracted_samples:
        result = digitizer.digitize_image(img_path, sample_name=img_path.stem)

        # Export CSV
        csv_file = csv_dir / f"{img_path.stem}_digitized_waveform.csv"
        result.save_csv(csv_file)

        # Export Overlay Plot
        overlay_file = overlays_dir / f"{img_path.stem}_overlay_plot.png"
        result.save_overlay_plot(overlay_file, title=f"Real ECG Digitisation — {img_path.stem} ({category})")

        # Compute summary stats
        mean_conf = float(np.mean(result.column_confidence))
        min_conf = float(np.min(result.column_confidence))
        v_min = float(np.min(result.calibrated_voltage_mv))
        v_max = float(np.max(result.calibrated_voltage_mv))
        v_pp = v_max - v_min

        bbox = result.preprocessing.content_bbox
        bbox_str = f"({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})"

        print(
            f"{img_path.name:<14} | {category:<32} | {bbox_str:<16} | {mean_conf:8.2%}   | {v_pp:9.4f} mV | {'Uncalibrated Beat'}"
        )

        results_summary.append({
            "sample_name": img_path.name,
            "category": category,
            "sample_image_path": str(img_path),
            "csv_path": str(csv_file),
            "overlay_path": str(overlay_file),
            "content_bbox": bbox,
            "mean_confidence": round(mean_conf, 4),
            "min_confidence": round(min_conf, 4),
            "nominal_voltage_min_mv": round(v_min, 4),
            "nominal_voltage_max_mv": round(v_max, 4),
            "peak_to_peak_mv": round(v_pp, 4),
            "calibration_reliable": result.is_calibration_reliable,
            "calibration_note": result.calibration_note,
        })

    print("-" * 105)

    summary_file = base_eval_dir / "real_ecg_inference_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("Saved inference summary to: %s", summary_file)
    return results_summary


if __name__ == "__main__":
    run_real_ecg_validation()
