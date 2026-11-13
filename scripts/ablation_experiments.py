"""Controlled Ablation Experiments for Gate 1 Remediation Investigation.

Runs 5 controlled ablation configurations across all 6 benchmark cases:
1. config_1_phase2_baseline: Original Phase 2 baseline (fixed 128px crop, static geometry).
2. config_2_adaptive_crop_only: Adaptive crop only + Phase 2 static geometry.
3. config_3_marquette_dynamic_only: Dynamic Marquette column geometry only + Phase 2 fixed crop.
4. config_4_12x1_constraints_only: 12x1 row constraints only + Phase 2 fixed crop.
5. config_5_all_gate1_combined: All Gate 1 changes together.

Computes for every lead:
- Pearson correlation (r)
- RMSE in mV
- MAE in mV
- Time lag (ms)
- Valid sample count
- Clipping rate (fraction of samples hitting vertical boundaries)

Outputs:
- data/interim/gate1_ablation_metrics.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

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


def extract_lead_trace_custom(
    model: torch.nn.Module,
    device: torch.device,
    lead_crop: np.ndarray,
    binary_crop: np.ndarray,
    lead_name: str,
    base_y_hint: float,
    is_rhythm: bool,
    crop_mode: str = "fixed_128",  # "fixed_128" or "adaptive"
    is_12x1: bool = False,
    col_trim: bool = False,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """Runs trace extraction with controlled ablation parameters."""
    h, w = lead_crop.shape[:2]
    base_y_local = float(base_y_hint)

    # 1. Baseline estimation
    c_min = max(15, int(round(base_y_local - 45)))
    c_max = min(h - 15, int(round(base_y_local + 45)))
    if c_max > c_min and binary_crop.size > 0:
        ys_ink, _ = np.where(binary_crop[c_min:c_max, :] > 0)
        if len(ys_ink) > 20:
            counts = np.bincount(ys_ink)
            modal_offset = float(np.argmax(counts))
            modal_y = c_min + modal_offset
            if abs(base_y_local - modal_y) > 4:
                base_y_local = modal_y

    # 2. Column boundaries
    if h < 120 and w < 220:
        col_min = 24 if (is_rhythm or lead_name in ("I", "II", "III")) else 21
        col_max = w - 4
    elif is_rhythm and h < 120:
        col_min = 24
        col_max = w - 4
    elif col_trim:
        # Active column trimming used in Gate 1
        y_c1 = max(40, int(round(base_y_local - 75)))
        y_c2 = min(h, int(round(base_y_local + 75)))
        corridor = binary_crop[y_c1:y_c2, :] if y_c2 > y_c1 else binary_crop
        active = np.where(corridor[:, 5:].sum(axis=0) > 0)[0] if corridor.size > 0 else np.array([])
        if len(active) < 5:
            col_min, col_max = 0, w - 1
        else:
            raw_c_min = int(active[0]) + 5
            col_min = max(raw_c_min, 12) if (raw_c_min < 20 and not is_rhythm) else raw_c_min
            col_max = int(active[-1]) + 5
    else:
        col_min, col_max = 0, w - 1

    col_min = max(0, min(col_min, w - 10))
    col_max = min(w - 1, max(col_max, col_min + 5))
    active_w = max(1, col_max - col_min + 1)

    # 3. Vertical Crop Window
    if crop_mode == "fixed_128":
        if is_12x1 or h < 110:
            y1 = 0
            y2 = h
        else:
            y1 = max(0, int(round(base_y_local - 64)))
            y2 = min(h, int(round(base_y_local + 64)))
            if (y2 - y1) < 128 and h >= 128:
                if y1 == 0:
                    y2 = 128
                elif y2 == h:
                    y1 = h - 128
    else:
        # Adaptive crop mode
        if is_12x1 or h < 110:
            y1, y2 = 0, h
        else:
            trace_slice = binary_crop[:, col_min : col_max + 1]
            row_has_ink = trace_slice.sum(axis=1) > 0
            cur = int(round(base_y_local))
            gap = 0
            while cur >= 0:
                if row_has_ink[cur]: gap = 0
                else:
                    gap += 1
                    if gap > 8: break
                cur -= 1
            y_min_est = float(max(0, cur + gap))

            cur = int(round(base_y_local))
            gap = 0
            while cur < h:
                if row_has_ink[cur]: gap = 0
                else:
                    gap += 1
                    if gap > 8: break
                cur += 1
            y_max_est = float(min(h, cur - gap))

            span = y_max_est - y_min_est
            if span <= 115 and h >= 128:
                mid = int(round((y_min_est + y_max_est) / 2.0))
                y1 = max(0, mid - 64)
                y2 = min(h, y1 + 128)
                if (y2 - y1) < 128:
                    y1 = max(0, y2 - 128)
            else:
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

    h_win = max(1, y2 - y1)

    # 4. Model inference
    if is_rhythm:
        num_chunks = 4
        chunk_w = active_w / float(num_chunks)
        trajs = []
        for i in range(num_chunks):
            c1 = int(round(col_min + i * chunk_w))
            c2 = min(w, int(round(col_min + (i + 1) * chunk_w)))
            cur_w = max(1, c2 - c1)
            chunk_slice = lead_crop[y1:y2, c1:c2]
            if chunk_slice.shape[0] < 128:
                pad_h = 128 - chunk_slice.shape[0]
                chunk_slice = cv2.copyMakeBorder(chunk_slice, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=[252, 252, 252])
            win_resized = cv2.resize(chunk_slice, (256, 128))
            base_model = (base_y_local - y1) * (128.0 / float(h_win))
            tensor = torch.from_numpy(win_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                out = model(tensor.to(device), baseline_y=float(base_model))
            py128 = out.pixel_coordinates.squeeze(0).cpu().numpy()
            py_orig = y1 + py128 * (float(h_win) / 128.0)
            trajs.append(np.interp(np.linspace(0, 255, cur_w), np.arange(256), py_orig))
        traj = np.concatenate(trajs).astype(np.float32)
    else:
        active_slice = lead_crop[y1:y2, col_min : col_max + 1]
        if active_slice.shape[0] < 128:
            pad_h = 128 - active_slice.shape[0]
            active_slice = cv2.copyMakeBorder(active_slice, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=[252, 252, 252])
        win_resized = cv2.resize(active_slice, (256, 128))
        base_model = (base_y_local - y1) * (128.0 / float(h_win))
        tensor = torch.from_numpy(win_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            out = model(tensor.to(device), baseline_y=float(base_model))
        py128 = out.pixel_coordinates.squeeze(0).cpu().numpy()
        py_orig = y1 + py128 * (float(h_win) / 128.0)
        traj = np.interp(np.linspace(0, 255, active_w), np.arange(256), py_orig).astype(np.float32)

    rel_traj = (base_y_local - traj).astype(np.float32)

    # Clipping rate: fraction of points near boundary (within 1px of y1 or y2)
    clip_hits = np.sum((traj <= y1 + 1.0) | (traj >= y2 - 1.0))
    clip_rate = float(clip_hits / max(1, len(traj)))

    diag = {
        "y1": y1,
        "y2": y2,
        "h_win": h_win,
        "base_y": base_y_local,
        "col_min": col_min,
        "col_max": col_max,
        "clip_rate": clip_rate,
    }
    return rel_traj, base_y_local, clip_rate, diag


def run_ablation():
    dataset_dir = Path("data/interim/small_paired_dataset")
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pipe = ECGDigitizerPipeline()
    model = pipe.deep_digitizer.model
    device = pipe.deep_digitizer.device

    configurations = [
        {"name": "config_1_phase2_baseline", "crop_mode": "fixed_128", "dynamic_marquette": False, "row_constrain_12x1": False, "col_trim": False},
        {"name": "config_2_adaptive_crop_only", "crop_mode": "adaptive", "dynamic_marquette": False, "row_constrain_12x1": False, "col_trim": False},
        {"name": "config_3_marquette_dynamic_only", "crop_mode": "fixed_128", "dynamic_marquette": True, "row_constrain_12x1": False, "col_trim": False},
        {"name": "config_4_12x1_constraints_only", "crop_mode": "fixed_128", "dynamic_marquette": False, "row_constrain_12x1": True, "col_trim": False},
        {"name": "config_5_all_gate1_combined", "crop_mode": "adaptive", "dynamic_marquette": True, "row_constrain_12x1": True, "col_trim": True},
    ]

    all_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print("ECG DIGIAI — CONTROLLED ABLATION EXPERIMENTS (5 CONFIGURATIONS)")
    print("=" * 90)

    for cfg in configurations:
        cfg_name = cfg["name"]
        logger.info("Running ablation: %s ...", cfg_name)

        for case_id, meta in manifest["cases"].items():
            img_file = dataset_dir / meta["files"]["image"]
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            layout_type = meta["layout_type"]
            is_single = (layout_type == "single_lead")
            prep = pipe.preprocessor.process(img)

            # Determine lead layout based on config
            det_img = prep.deskewed if cfg["dynamic_marquette"] else None
            layout_res = pipe.lead_detector.detect_layout(
                image=det_img,
                image_shape=img.shape,
                layout_hint=layout_type,
            )

            # Load ground truth if paired
            gt_df = None
            if meta["files"].get("gt_signal_csv"):
                gt_csv = dataset_dir / meta["files"]["gt_signal_csv"]
                gt_df = pd.read_csv(gt_csv, index_col="sample_index")

            for lead_name in meta["lead_names"]:
                if lead_name not in layout_res.leads:
                    continue

                lr = layout_res.leads[lead_name]
                lead_crop = prep.deskewed[lr.y1:lr.y2, lr.x1:lr.x2]
                crop_bin = prep.binary_trace[lr.y1:lr.y2, lr.x1:lr.x2]

                is_12x1_case = (layout_type == "12x1" and cfg["row_constrain_12x1"])
                rel_traj, base_y_est, clip_rate, diag = extract_lead_trace_custom(
                    model=model,
                    device=device,
                    lead_crop=lead_crop,
                    binary_crop=crop_bin,
                    lead_name=lead_name,
                    base_y_hint=float(lr.baseline_y_hint),
                    is_rhythm=lr.is_rhythm,
                    crop_mode=cfg["crop_mode"],
                    is_12x1=is_12x1_case,
                    col_trim=cfg["col_trim"],
                )

                # Calibrate to mV
                duration = 10.0 if (lr.is_rhythm or layout_type == "12x1") else 2.5
                ppmv = 59.055
                raw_mv = rel_traj / ppmv
                n_samples = int(round(duration * 100))
                rec_resamp = np.interp(np.linspace(0, 1, n_samples), np.linspace(0, 1, len(raw_mv)), raw_mv)

                # Evaluate against ground truth
                r_val, rmse_val, mae_val, lag_ms = np.nan, np.nan, np.nan, 0.0
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
                        gt_vals = gt_df[lead_name].values[col_idx*250 : (col_idx+1)*250]
                    elif layout_type == "12x1":
                        gt_vals = gt_df[lead_name].values[:1000]
                    else:
                        gt_vals = np.array([])

                    n_gt = len(gt_vals)
                    if n_gt > 0 and len(rec_resamp) > 0:
                        rec_aligned = np.interp(np.linspace(0, 1, n_gt), np.linspace(0, 1, len(rec_resamp)), rec_resamp)
                        metrics = compute_lead_metrics(ground_truth=gt_vals, reconstructed=rec_aligned, lead_name=lead_name)
                        r_val = round(float(metrics.pearson_r), 4)
                        rmse_val = round(float(metrics.rmse_mv), 4)
                        mae_val = round(float(np.mean(np.abs(rec_aligned - gt_vals))), 4)
                        lag_samples, _ = compute_time_lag(rec_aligned, gt_vals)
                        lag_ms = round(float(lag_samples * 10.0), 1)

                all_rows.append({
                    "config": cfg_name,
                    "case_id": case_id,
                    "layout": layout_type,
                    "lead_name": lead_name,
                    "pearson_r": r_val,
                    "rmse_mv": rmse_val,
                    "mae_mv": mae_val,
                    "lag_ms": lag_ms,
                    "valid_samples": len(rec_resamp),
                    "clipping_rate": round(clip_rate, 4),
                    "crop_y1": diag["y1"],
                    "crop_y2": diag["y2"],
                    "h_win": diag["h_win"],
                })

    df_out = pd.DataFrame(all_rows)
    out_csv = Path("data/interim/gate1_ablation_metrics.csv")
    df_out.to_csv(out_csv, index=False)
    logger.info("Saved ablation metrics to: %s", out_csv)

    # Print Summary Table
    print("\n" + "=" * 90)
    print("ABLATION SUMMARY TABLE: MEAN PEARSON R AND RMSE ACROSS CASES")
    print("=" * 90)
    summary_table = []
    cases = list(manifest["cases"].keys())
    for cfg in configurations:
        c_name = cfg["name"]
        sub_cfg = df_out[df_out["config"] == c_name]
        row_str = f"{c_name:32s}"
        for case in cases:
            sub_c = sub_cfg[sub_cfg["case_id"] == case]
            mean_r = sub_c["pearson_r"].dropna().mean()
            row_str += f" | {case[:12]}: {mean_r:.4f}"
        print(row_str)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_ablation()
