"""Gate 1 Remediations Evaluation and Before/After Benchmark Comparison.

Executes end-to-end evaluation across all 6 paired benchmark cases:
- case_01_clean_3x4_1
- case_02_degraded_3x4_1
- case_03_marquette_3x4_3
- case_04_stacked_12x1
- case_05_single_lead_n10007
- case_06_clean_3x4

Compares against Phase 2 pre-remediation metrics (from data/interim/phase2_leadwise_metrics.csv).
Generates:
- data/interim/gate1_before_after_metrics.csv
- Refreshed diagnostic overlay artifacts
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from ecg_digitization.evaluation.metrics import compute_lead_metrics
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYNTH_3X4_COLUMNS = [
    ["I", "II", "III"],
    ["AVR", "AVL", "AVF"],
    ["V1", "V2", "V3"],
    ["V4", "V5", "V6"],
]


def compute_time_lag(rec: np.ndarray, gt: np.ndarray, max_lag: int = 40) -> Tuple[int, float]:
    """Computes time lag in samples via normalized cross-correlation."""
    n = min(len(rec), len(gt))
    if n < 20:
        return 0, 0.0
    r_norm = rec[:n] - np.mean(rec[:n])
    g_norm = gt[:n] - np.mean(gt[:n])
    if np.std(r_norm) < 1e-6 or np.std(g_norm) < 1e-6:
        return 0, 0.0

    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            c = np.corrcoef(r_norm[: n + lag], g_norm[-lag:n])[0, 1]
        elif lag > 0:
            c = np.corrcoef(r_norm[lag:n], g_norm[: n - lag])[0, 1]
        else:
            c = np.corrcoef(r_norm, g_norm)[0, 1]
        corrs.append(c)

    best_idx = int(np.nanargmax(corrs))
    best_lag = lags[best_idx]
    best_corr = float(corrs[best_idx])
    return best_lag, best_corr


def main():
    dataset_dir = Path("data/interim/small_paired_dataset")
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Load baseline pre-remediation metrics
    baseline_csv = Path("data/interim/phase2_leadwise_metrics.csv")
    baseline_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if baseline_csv.is_file():
        df_base = pd.read_csv(baseline_csv)
        for _, row in df_base.iterrows():
            baseline_map[(str(row["case_id"]), str(row["lead_name"]))] = row.to_dict()

    pipe = ECGDigitizerPipeline()
    artifact_dir = Path("/Users/akashraj/.gemini/antigravity/brain/00b546f6-2836-4c10-b13b-93b9b2399e6e")

    comparison_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print("ECG DIGIAI — GATE 1 REMEDIATIONS BENCHMARK EVALUATION")
    print("=" * 90)

    for case_id, meta in manifest["cases"].items():
        img_file = dataset_dir / meta["files"]["image"]
        img = cv2.imread(str(img_file))
        if img is None:
            logger.error("Could not load image: %s", img_file)
            continue

        layout_type = meta["layout_type"]
        is_single = (layout_type == "single_lead")
        logger.info("Evaluating %s (layout: %s, shape: %s)...", case_id, layout_type, img.shape)

        res = pipe.digitize(
            img,
            layout_hint=layout_type,
            extractor_mode="deep",
            is_calibration_reliable=not is_single,
        )

        # Update diagnostic overlay
        overlay_path = dataset_dir / f"{case_id}_overlay.png"
        res.generate_sheet_overlay(overlay_path)
        if artifact_dir.is_dir():
            shutil.copyfile(overlay_path, artifact_dir / f"{case_id}_overlay.png")

        # Load ground truth if paired
        gt_df = None
        if meta["files"].get("gt_signal_csv"):
            gt_csv = dataset_dir / meta["files"]["gt_signal_csv"]
            gt_df = pd.read_csv(gt_csv, index_col="sample_index")

        # Map extracted traces by lead name for diagnostics
        trace_diag_map = res.extracted_traces

        for lead_name in meta["lead_names"]:
            base_row = baseline_map.get((case_id, lead_name), {})
            b_r = base_row.get("pearson_r", np.nan)
            b_rmse = base_row.get("rmse_mv", np.nan)
            b_lag_ms = base_row.get("lag_ms", np.nan)

            trace = trace_diag_map.get(lead_name)
            is_top_clipped = trace.is_clipped_top if trace else False
            is_bot_clipped = trace.is_clipped_bottom if trace else False
            clip_margin_px = round(trace.clipping_margin_px, 1) if trace else 0.0
            ct_risk = round(trace.cross_talk_risk, 3) if trace else 0.0
            is_overlapping = trace.is_overlapping if trace else False

            if lead_name not in res.reconstructed_ecg.leads:
                comparison_rows.append({
                    "case_id": case_id,
                    "layout": layout_type,
                    "lead_name": lead_name,
                    "status": "MISSING",
                    "before_r": b_r,
                    "after_r": np.nan,
                    "delta_r": np.nan,
                    "before_rmse": b_rmse,
                    "after_rmse": np.nan,
                    "delta_rmse": np.nan,
                    "before_lag_ms": b_lag_ms,
                    "after_lag_ms": np.nan,
                    "is_clipped_top": is_top_clipped,
                    "is_clipped_bottom": is_bot_clipped,
                    "clipping_margin_px": clip_margin_px,
                    "cross_talk_risk": ct_risk,
                    "is_overlapping": is_overlapping,
                })
                continue

            rec_lead = res.reconstructed_ecg.leads[lead_name]
            rec_voltages = rec_lead.voltage_mv
            rec_len = len(rec_voltages)

            if gt_df is not None:
                if lead_name.endswith("_RHYTHM"):
                    src_lead_name = lead_name.split("_")[0]
                    gt_vals = gt_df[src_lead_name].values[:1000]
                elif layout_type in ("3x4", "3x4_1", "3x4_3"):
                    col_idx = 0
                    for c_i, c_leads in enumerate(SYNTH_3X4_COLUMNS):
                        if lead_name in c_leads:
                            col_idx = c_i
                            break
                    idx1 = col_idx * 250
                    idx2 = (col_idx + 1) * 250
                    gt_vals = gt_df[lead_name].values[idx1:idx2]
                elif layout_type == "12x1":
                    gt_vals = gt_df[lead_name].values[:1000]
                else:
                    gt_vals = np.array([])

                n_gt = len(gt_vals)
                if n_gt > 0:
                    rec_resamp = np.interp(np.linspace(0, 1, n_gt), np.linspace(0, 1, rec_len), rec_voltages)
                    metrics = compute_lead_metrics(ground_truth=gt_vals, reconstructed=rec_resamp, lead_name=lead_name)
                    r_val = round(float(metrics.pearson_r), 4)
                    rmse_val = round(float(metrics.rmse_mv), 4)

                    lag_samples, _ = compute_time_lag(rec_resamp, gt_vals)
                    lag_ms = round(float(lag_samples * 10.0), 1)
                else:
                    r_val, rmse_val, lag_ms = np.nan, np.nan, 0.0
            else:
                r_val, rmse_val, lag_ms = np.nan, np.nan, 0.0

            delta_r = round(r_val - b_r, 4) if (pd.notna(r_val) and pd.notna(b_r)) else np.nan
            delta_rmse = round(rmse_val - b_rmse, 4) if (pd.notna(rmse_val) and pd.notna(b_rmse)) else np.nan

            comparison_rows.append({
                "case_id": case_id,
                "layout": layout_type,
                "lead_name": lead_name,
                "status": "EXTRACTED",
                "before_r": b_r,
                "after_r": r_val,
                "delta_r": delta_r,
                "before_rmse": b_rmse,
                "after_rmse": rmse_val,
                "delta_rmse": delta_rmse,
                "before_lag_ms": b_lag_ms,
                "after_lag_ms": lag_ms,
                "is_clipped_top": is_top_clipped,
                "is_clipped_bottom": is_bot_clipped,
                "clipping_margin_px": clip_margin_px,
                "cross_talk_risk": ct_risk,
                "is_overlapping": is_overlapping,
            })

    out_csv = Path("data/interim/gate1_before_after_metrics.csv")
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(out_csv, index=False)
    logger.info("Saved Gate 1 before/after comparison to: %s", out_csv)

    # Print summary by case
    print("\n" + "=" * 90)
    print("GATE 1 SUMMARY: MEAN PEARSON R AND RMSE BEFORE VS AFTER")
    print("=" * 90)
    for case_id in manifest["cases"].keys():
        sub = df_comp[df_comp["case_id"] == case_id]
        b_mean_r = sub["before_r"].dropna().mean()
        a_mean_r = sub["after_r"].dropna().mean()
        b_mean_rmse = sub["before_rmse"].dropna().mean()
        a_mean_rmse = sub["after_rmse"].dropna().mean()
        print(f"{case_id:28s} | Before r: {b_mean_r:.4f} -> After r: {a_mean_r:.4f} | Before RMSE: {b_mean_rmse:.4f} -> After RMSE: {a_mean_rmse:.4f}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
