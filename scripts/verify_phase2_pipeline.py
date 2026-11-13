"""Phase 2 Hierarchical Pipeline Verification & Quantitative Benchmarking.

Evaluates ECGDigitizerPipeline across all cases in the Phase 2 small paired dataset:
1. case_01_clean_3x4_1
2. case_02_degraded_3x4_1
3. case_03_marquette_3x4_3
4. case_04_stacked_12x1
5. case_05_single_lead_n10007

Computes per-lead Pearson correlation r, RMSE (mV), SNR (dB), and confidence against ground truth.
Generates full diagnostic sheet overlay images.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd

from ecg_digitization.evaluation.metrics import (
    compute_lead_metrics,
    evaluate_digitization_fidelity,
)
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_lead_fidelity(reconstructed_mv: np.ndarray, gt_mv: np.ndarray, lead_name: str = "Lead") -> Dict[str, float]:
    """Computes numerical fidelity between reconstructed and ground-truth voltage arrays."""
    n_gt = len(gt_mv)
    n_rec = len(reconstructed_mv)
    if n_gt == 0 or n_rec == 0:
        return {"pearson_r": 0.0, "rmse_mv": 999.0, "snr_db": -99.0}

    # Resample reconstructed to match ground truth length
    rec_resamp = np.interp(np.linspace(0, 1, n_gt), np.linspace(0, 1, n_rec), reconstructed_mv)

    # Standard metrics
    metrics = compute_lead_metrics(ground_truth=gt_mv, reconstructed=rec_resamp, lead_name=lead_name)
    return {
        "pearson_r": round(float(metrics.pearson_r), 4),
        "rmse_mv": round(float(metrics.rmse_mv), 4),
        "snr_db": round(float(metrics.snr_db), 2),
    }


def main():
    dataset_dir = Path("data/interim/small_paired_dataset")
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}. Run generate_paired_evaluation_set.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pipe = ECGDigitizerPipeline()
    results_summary: List[Dict[str, Any]] = []

    artifact_dir = Path("/Users/akashraj/.gemini/antigravity/brain/00b546f6-2836-4c10-b13b-93b9b2399e6e")

    print("\n" + "=" * 80)
    print("ECG DIGIAI — PHASE 2 FULL 12-LEAD PIPELINE VERIFICATION")
    print("=" * 80)

    for case_id, meta in manifest["cases"].items():
        img_file = dataset_dir / meta["files"]["image"]
        img = cv2.imread(str(img_file))
        if img is None:
            logger.error("Could not load image: %s", img_file)
            continue

        layout_type = meta["layout_type"]
        is_single = (layout_type == "single_lead")
        logger.info("Processing %s (layout: %s, shape: %s)...", case_id, layout_type, img.shape)

        # Run hierarchical pipeline
        res = pipe.digitize(
            img,
            layout_hint=layout_type,
            extractor_mode="deep",
            is_calibration_reliable=not is_single,
        )

        # Generate and save diagnostic sheet overlay
        overlay_path = dataset_dir / f"{case_id}_overlay.png"
        res.generate_sheet_overlay(overlay_path)

        # Copy to artifact directory for presentation
        if artifact_dir.is_dir():
            shutil.copyfile(overlay_path, artifact_dir / f"{case_id}_overlay.png")

        # Save reconstructed multi-lead signal CSV
        rec_csv_path = dataset_dir / f"{case_id}_reconstructed_signal.csv"
        res.reconstructed_ecg.save_csv(rec_csv_path)

        # Evaluate against exact paired ground truth if available
        lead_metrics: Dict[str, Dict[str, float]] = {}
        mean_r = 0.0
        mean_rmse = 0.0

        if meta["files"].get("gt_signal_csv"):
            gt_csv = dataset_dir / meta["files"]["gt_signal_csv"]
            gt_df = pd.read_csv(gt_csv, index_col="sample_index")

            # Map leads
            r_list = []
            rmse_list = []

            for lead_name in meta["lead_names"]:
                if lead_name in res.reconstructed_ecg.leads:
                    rec_lead = res.reconstructed_ecg.leads[lead_name]
                    rec_voltages = rec_lead.voltage_mv

                    # Determine ground truth slice
                    if lead_name.endswith("_RHYTHM"):
                        src_lead_name = lead_name.split("_")[0]
                        if src_lead_name in gt_df.columns:
                            gt_lead_vals = gt_df[src_lead_name].values[:1000]
                        else:
                            continue
                    elif layout_type in ("3x4", "3x4_1", "3x4_3"):
                        # In 3x4 layouts, each column is 2.5s (250 samples at 100 Hz)
                        col_idx = 0
                        for c_i, c_leads in enumerate(synth_columns):
                            if lead_name in c_leads:
                                col_idx = c_i
                                break
                        idx1 = col_idx * 250
                        idx2 = (col_idx + 1) * 250
                        if lead_name in gt_df.columns:
                            gt_lead_vals = gt_df[lead_name].values[idx1:idx2]
                        else:
                            continue
                    elif layout_type == "12x1":
                        if lead_name in gt_df.columns:
                            gt_lead_vals = gt_df[lead_name].values[:1000]
                        else:
                            continue
                    else:
                        continue

                    m = evaluate_lead_fidelity(rec_voltages, gt_lead_vals, lead_name=lead_name)
                    lead_metrics[lead_name] = m
                    r_list.append(m["pearson_r"])
                    rmse_list.append(m["rmse_mv"])

            if r_list:
                mean_r = float(np.mean(r_list))
                mean_rmse = float(np.mean(rmse_list))

        status_str = f"Mean r = {mean_r:.4f}, Mean RMSE = {mean_rmse:.4f} mV" if lead_metrics else "Single-lead qualitative"
        print(f"\n[{case_id}] Layout: {layout_type} | Extracted: {len(res.extracted_traces)} leads | {status_str}")
        for l_name, m in lead_metrics.items():
            print(f"   Lead {l_name:10s} -> Pearson r = {m['pearson_r']:+.4f} | RMSE = {m['rmse_mv']:.4f} mV | SNR = {m['snr_db']:+5.1f} dB")

        results_summary.append({
            "case_id": case_id,
            "layout": layout_type,
            "leads_extracted": len(res.extracted_traces),
            "mean_pearson_r": mean_r,
            "mean_rmse_mv": mean_rmse,
            "lead_metrics": lead_metrics,
            "overlay_path": str(overlay_path),
        })

    # Save summary report
    summary_path = dataset_dir / "phase2_verification_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Phase 2 Verification Complete! Summary saved: {summary_path}")
    print("=" * 80)


synth_columns = [
    ["I", "II", "III"],
    ["AVR", "AVL", "AVF"],
    ["V1", "V2", "V3"],
    ["V4", "V5", "V6"],
]

if __name__ == "__main__":
    main()
