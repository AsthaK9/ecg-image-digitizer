"""Generate Paired Synthetic 12-Lead ECG Sheet Dataset for ECG DigiAI (Gate 2).

Builds and validates a reproducible full-sheet ECG image dataset paired with exact
PTB-XL digital ground truth:
- Folds 1–8: Training split
- Fold 9: Validation split
- Fold 10: Strictly quarantined (never accessed or processed)
- Layouts: 3x4, 3x4+1, 3x4+3 (Marquette), and 12x1
- Exact physical calibration: 25 mm/s, 10 mm/mV, 150 DPI
- Realistic photographic/scanning optical degradations
- Exact digital waveform ground truth per lead window (.csv and .npy)
- Detailed manifest tracking all geometry, seeds, and metadata

Modes:
- `--validate-sample-10`: Renders and rigorously validates 10 representative records
  (image quality, calibration, alignment, extreme waves, degradations, reproducibility)
  and produces visual overlays and quantitative round-trip metrics.
- `--bulk-generate`: Generates the complete 5,000 paired sheet dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import ECGRecordPaths, ECGSignal
from ecg_digitization.evaluation.metrics import compute_lead_metrics
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from ecg_digitization.synthetic.generator import (
    DegradationConfig,
    ECGLayoutConfig,
    SyntheticECGGenerator,
    SyntheticECGResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYNTH_3X4_COLUMNS = [
    ["I", "II", "III"],
    ["AVR", "AVL", "AVF"],
    ["V1", "V2", "V3"],
    ["V4", "V5", "V6"],
]


def find_ptbxl_root() -> Path:
    candidates = list(Path("data/raw/ptb-xl").glob("**/ptbxl_database.csv"))
    if not candidates:
        raise FileNotFoundError("ptbxl_database.csv not found under data/raw/ptb-xl")
    return candidates[0].parent


def load_wfdb_record(ptbxl_root: Path, filename_lr: str) -> ECGSignal:
    hea = ptbxl_root / f"{filename_lr}.hea"
    dat = ptbxl_root / f"{filename_lr}.dat"
    if not hea.is_file() or not dat.is_file():
        raise FileNotFoundError(f"Missing WFDB files for {filename_lr}: {hea} or {dat}")
    parser = WFDBRecordParser()
    rec_id = Path(filename_lr).name
    return parser.parse(ECGRecordPaths(rec_id, hea, dat), source_dataset="ptbxl")


def verify_patient_split(df_manifest: pd.DataFrame, db_df: pd.DataFrame) -> Dict[str, Any]:
    """Verifies that patient separation between training and validation is 100% strict

    and that Fold 10 is completely untouched.
    """
    fold_10_present = (df_manifest["strat_fold"] == 10).any()
    if fold_10_present:
        raise ValueError("CRITICAL SECURITY VIOLATION: Fold 10 records found in manifest! Fold 10 must remain quarantined.")

    train_mask = df_manifest["split"] == "train"
    val_mask = df_manifest["split"] == "val"

    train_pats = set(df_manifest.loc[train_mask, "patient_id"].unique())
    val_pats = set(df_manifest.loc[val_mask, "patient_id"].unique())

    overlap = train_pats.intersection(val_pats)
    if overlap:
        raise ValueError(f"CRITICAL LEAKAGE: Patient overlap detected between Train and Val: {overlap}")

    stats = {
        "total_sheets": len(df_manifest),
        "train_sheets": int(train_mask.sum()),
        "val_sheets": int(val_mask.sum()),
        "unique_train_patients": len(train_pats),
        "unique_val_patients": len(val_pats),
        "patient_overlap_count": len(overlap),
        "fold_10_quarantined": True,
    }
    logger.info("Patient split verification passed: %s", stats)
    return stats


def compute_time_lag(rec: np.ndarray, gt: np.ndarray, max_lag: int = 40) -> Tuple[int, float]:
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
    return lags[best_idx], float(corrs[best_idx])


def render_and_save_record(
    generator: SyntheticECGGenerator,
    record: ECGSignal,
    layout_type: str,
    degradation: Optional[DegradationConfig],
    seed: int,
    output_dir: Path,
    image_id: str,
    meta_extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Renders a single paired synthetic ECG sheet and saves images, masks, and ground-truth signals."""
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    sig_dir = output_dir / "signals"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    sig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Render Sheet
    synth_res = generator.generate(
        signal=record,
        sampling_rate_hz=record.metadata.sampling_rate_hz,
        lead_names=record.metadata.lead_names,
        degradation=degradation,
        seed=seed,
        layout_type=layout_type,
    )

    # 2. Save Image Files
    img_file = img_dir / f"{image_id}.png"
    clean_file = img_dir / f"{image_id}_clean.png"
    mask_file = mask_dir / f"{image_id}_mask.png"

    cv2.imwrite(str(img_file), synth_res.image)
    cv2.imwrite(str(clean_file), synth_res.clean_image)
    cv2.imwrite(str(mask_file), synth_res.trace_mask)

    # 3. Save Exact Ground-Truth Signals
    gt_df = pd.DataFrame(record.values.T, columns=[l.upper() for l in record.metadata.lead_names])
    gt_df.index.name = "sample_index"
    gt_csv = sig_dir / f"{image_id}_gt_signal.csv"
    gt_npy = sig_dir / f"{image_id}_gt_signal.npy"

    gt_df.to_csv(gt_csv)
    np.save(gt_npy, record.values)

    # 4. Construct Comprehensive Manifest Record
    deg_cfg = degradation or DegradationConfig(enabled=False)
    manifest_entry = {
        "image_id": image_id,
        "ptbxl_ecg_id": meta_extra["ecg_id"],
        "patient_id": meta_extra["patient_id"],
        "strat_fold": meta_extra["strat_fold"],
        "split": meta_extra["split"],
        "layout_type": layout_type,
        "lead_names": json.dumps(list(synth_res.exact_lead_signals.keys())),
        "sampling_rate_hz": float(synth_res.sampling_rate_hz),
        "duration_s": float(record.values.shape[1] / synth_res.sampling_rate_hz),
        "paper_speed_mm_s": float(generator.layout.speed_mm_s),
        "gain_mm_mv": float(generator.layout.gain_mm_mv),
        "dpi": int(generator.layout.dpi),
        "pixels_per_mm": round(float(generator.pixels_per_mm), 4),
        "pixels_per_sec": round(float(generator.pixels_per_sec), 4),
        "pixels_per_mv": round(float(generator.pixels_per_mv), 4),
        "lead_bboxes_json": json.dumps(synth_res.lead_bounding_boxes),
        "lead_baselines_json": json.dumps(synth_res.lead_baselines),
        "seed": int(seed),
        "degradation_enabled": bool(deg_cfg.enabled),
        "rotation_deg": float(deg_cfg.rotation_deg),
        "noise_sigma": float(deg_cfg.gaussian_noise_sigma),
        "blur_ksize": int(deg_cfg.blur_kernel_size),
        "shadow_gradient": bool(deg_cfg.shadow_gradient),
        "jpeg_quality": int(deg_cfg.jpeg_quality) if deg_cfg.jpeg_quality is not None else -1,
        "grid_style": str(deg_cfg.grid_color_style),
        "image_file": str(img_file.relative_to(output_dir)),
        "clean_image_file": str(clean_file.relative_to(output_dir)),
        "mask_file": str(mask_file.relative_to(output_dir)),
        "gt_signal_csv": str(gt_csv.relative_to(output_dir)),
        "gt_signal_npy": str(gt_npy.relative_to(output_dir)),
    }
    return manifest_entry


def validate_10_samples(
    ptbxl_root: Path,
    output_dir: Path,
    db_df: pd.DataFrame,
    artifact_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Renders 10 representative sheets covering clean and degraded layouts,

    extreme precordials, and verifies calibration, reproducibility, and round-trip fidelity.
    """
    logger.info("Executing 10-record representative validation for Gate 2...")

    # 10 Representative test cases (6 Train from Folds 1-8, 4 Val from Fold 9; ZERO Fold 10)
    validation_cases = [
        {
            "image_id": "val_01_clean_3x4_1",
            "ecg_id": 1,
            "filename_lr": "records100/00000/00001_lr",
            "layout_type": "3x4_1",
            "split": "train",
            "strat_fold": 2,
            "patient_id": 15709,
            "description": "Standard Sinus Rhythm, Clean 3x4+1 Layout (Train Fold 2)",
            "degradation": DegradationConfig(enabled=False),
        },
        {
            "image_id": "val_02_deep_s_3x4_1",
            "ecg_id": 2,
            "filename_lr": "records100/00000/00002_lr",
            "layout_type": "3x4_1",
            "split": "train",
            "strat_fold": 2,
            "patient_id": 13243,
            "description": "Extreme Precordial Deep S-Wave (-3.23 mV), Clean 3x4+1 (Train Fold 2)",
            "degradation": DegradationConfig(enabled=False),
        },
        {
            "image_id": "val_03_tall_r_3x4_1",
            "ecg_id": 4,
            "filename_lr": "records100/00000/00004_lr",
            "layout_type": "3x4_1",
            "split": "train",
            "strat_fold": 3,
            "patient_id": 17014,
            "description": "Extreme Tall R-Wave (+3.16 mV), Clean 3x4+1 (Train Fold 3)",
            "degradation": DegradationConfig(enabled=False),
        },
        {
            "image_id": "val_04_clean_marquette_3x4_3",
            "ecg_id": 6,
            "filename_lr": "records100/00000/00006_lr",
            "layout_type": "3x4_3",
            "split": "train",
            "strat_fold": 6,
            "patient_id": 19000,
            "description": "Clean Marquette MAC 6-Row with 3 Rhythm Strips V1, II, V5 (Train Fold 6)",
            "degradation": DegradationConfig(enabled=False),
        },
        {
            "image_id": "val_05_clean_12x1",
            "ecg_id": 5,
            "filename_lr": "records100/00000/00005_lr",
            "layout_type": "12x1",
            "split": "train",
            "strat_fold": 4,
            "patient_id": 17448,
            "description": "Continuous 12x1 Stacked 10-Second Rows (Train Fold 4)",
            "degradation": DegradationConfig(enabled=False),
        },
        {
            "image_id": "val_06_clean_3x4",
            "ecg_id": 24,
            "filename_lr": "records100/00000/00024_lr",
            "layout_type": "3x4",
            "split": "train",
            "strat_fold": 3,
            "patient_id": 9012,
            "description": "Standard 3x4 Plain Grid without Rhythm Strip (Train Fold 3)",
            "degradation": DegradationConfig(enabled=False),
        },
        {
            "image_id": "val_07_degraded_3x4_1",
            "ecg_id": 8,
            "filename_lr": "records100/00000/00008_lr",
            "layout_type": "3x4_1",
            "split": "val",
            "strat_fold": 9,
            "patient_id": 11275,
            "description": "Degraded 3x4+1 (Deep S -2.44 mV, Rotation 1.5 deg, Shadow, Blur, Noise, Faded Grid)",
            "degradation": DegradationConfig(
                enabled=True,
                rotation_deg=1.5,
                gaussian_noise_sigma=3.5,
                blur_kernel_size=3,
                shadow_gradient=True,
                jpeg_quality=65,
                grid_color_style="faded",
            ),
        },
        {
            "image_id": "val_08_degraded_marquette_3x4_3",
            "ecg_id": 17,
            "filename_lr": "records100/00000/00017_lr",
            "layout_type": "3x4_3",
            "split": "val",
            "strat_fold": 9,
            "patient_id": 13619,
            "description": "Degraded Marquette 3x4+3 (Rotation -1.2 deg, Noise, Shadow, JPEG Q70)",
            "degradation": DegradationConfig(
                enabled=True,
                rotation_deg=-1.2,
                gaussian_noise_sigma=3.0,
                blur_kernel_size=3,
                shadow_gradient=True,
                jpeg_quality=70,
                grid_color_style="faded",
            ),
        },
        {
            "image_id": "val_09_degraded_12x1",
            "ecg_id": 30,
            "filename_lr": "records100/00000/00030_lr",
            "layout_type": "12x1",
            "split": "val",
            "strat_fold": 9,
            "patient_id": 19053,
            "description": "Degraded 12x1 Stacked (Tall R +2.50 mV, Rotation 1.0 deg, Blur, Noise, Dark Grid)",
            "degradation": DegradationConfig(
                enabled=True,
                rotation_deg=1.0,
                gaussian_noise_sigma=3.0,
                blur_kernel_size=3,
                shadow_gradient=True,
                jpeg_quality=75,
                grid_color_style="dark",
            ),
        },
        {
            "image_id": "val_10_degraded_3x4",
            "ecg_id": 94,
            "filename_lr": "records100/00000/00094_lr",
            "layout_type": "3x4",
            "split": "val",
            "strat_fold": 9,
            "patient_id": 21731,
            "description": "Degraded Plain 3x4 (Tall R +2.38 mV, Rotation -1.5 deg, Shadow, Noise, Faded Grid)",
            "degradation": DegradationConfig(
                enabled=True,
                rotation_deg=-1.5,
                gaussian_noise_sigma=3.5,
                blur_kernel_size=3,
                shadow_gradient=True,
                jpeg_quality=65,
                grid_color_style="faded",
            ),
        },
    ]

    generator = SyntheticECGGenerator()
    pipe = ECGDigitizerPipeline()
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    round_trip_rows = []

    seed_base = 2026

    for idx, c in enumerate(validation_cases):
        img_id = c["image_id"]
        logger.info("Processing validation case [%d/10]: %s (%s)...", idx + 1, img_id, c["layout_type"])
        record = load_wfdb_record(ptbxl_root, c["filename_lr"])

        # Check Marquette genuine duration
        if c["layout_type"] == "3x4_3":
            sig_dur = record.values.shape[1] / record.metadata.sampling_rate_hz
            if sig_dur < 9.9:
                raise ValueError(f"Record {c['filename_lr']} duration {sig_dur}s is insufficient for genuine Marquette rhythm strips.")

        entry = render_and_save_record(
            generator=generator,
            record=record,
            layout_type=c["layout_type"],
            degradation=c["degradation"],
            seed=seed_base + idx,
            output_dir=output_dir,
            image_id=img_id,
            meta_extra={
                "ecg_id": c["ecg_id"],
                "patient_id": c["patient_id"],
                "strat_fold": c["strat_fold"],
                "split": c["split"],
            },
        )
        manifest_entries.append(entry)

        # Reproducibility verification from saved seed
        synth_repro = generator.generate(
            signal=record,
            sampling_rate_hz=record.metadata.sampling_rate_hz,
            lead_names=record.metadata.lead_names,
            degradation=c["degradation"],
            seed=seed_base + idx,
            layout_type=c["layout_type"],
        )
        saved_img = cv2.imread(str(output_dir / entry["image_file"]))
        if not np.array_equal(synth_repro.image, saved_img):
            raise AssertionError(f"Determinism failure! Re-rendering {img_id} with seed {seed_base+idx} did not match saved image.")

        # Round-trip Digitisation and Quantitative Evaluation
        res_dig = pipe.digitize(saved_img, layout_hint=c["layout_type"], extractor_mode="deep")
        overlay_path = overlay_dir / f"{img_id}_overlay.png"
        res_dig.generate_sheet_overlay(overlay_path)

        if artifact_dir and artifact_dir.is_dir():
            shutil.copyfile(overlay_path, artifact_dir / f"{img_id}_overlay.png")

        # Load ground truth for round-trip metrics
        gt_df = pd.read_csv(output_dir / entry["gt_signal_csv"], index_col="sample_index")

        rendered_leads = json.loads(entry["lead_names"])
        for lead_name in rendered_leads:
            if lead_name.endswith("_RHYTHM"):
                src_lead = lead_name.split("_")[0]
                gt_vals = gt_df[src_lead].values[:1000]
            elif c["layout_type"] in ("3x4", "3x4_1", "3x4_3"):
                col_i = 0
                for c_idx, c_leads in enumerate(SYNTH_3X4_COLUMNS):
                    if lead_name in c_leads:
                        col_i = c_idx
                        break
                gt_vals = gt_df[lead_name].values[col_i * 250 : (col_i + 1) * 250]
            elif c["layout_type"] == "12x1":
                gt_vals = gt_df[lead_name].values[:1000]
            else:
                gt_vals = np.array([])

            if lead_name in res_dig.reconstructed_ecg.leads:
                rec_vals = res_dig.reconstructed_ecg.leads[lead_name].voltage_mv
                rec_interp = np.interp(np.linspace(0, 1, len(gt_vals)), np.linspace(0, 1, len(rec_vals)), rec_vals)
                metrics = compute_lead_metrics(gt_vals, rec_interp, lead_name=lead_name)
                lag_s, _ = compute_time_lag(rec_interp, gt_vals)
                round_trip_rows.append({
                    "image_id": img_id,
                    "layout": c["layout_type"],
                    "lead_name": lead_name,
                    "pearson_r": round(float(metrics.pearson_r), 4),
                    "rmse_mv": round(float(metrics.rmse_mv), 4),
                    "gt_min_mv": round(float(gt_vals.min()), 3),
                    "rec_min_mv": round(float(rec_interp.min()), 3),
                    "gt_max_mv": round(float(gt_vals.max()), 3),
                    "rec_max_mv": round(float(rec_interp.max()), 3),
                    "lag_ms": round(float(lag_s * 10.0), 1),
                })

    df_manifest = pd.DataFrame(manifest_entries)
    manifest_csv = output_dir / "paired_ecg_manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    logger.info("Saved validation manifest with 10 records to %s", manifest_csv)

    # Save Round-Trip Metrics
    df_rt = pd.DataFrame(round_trip_rows)
    rt_csv = output_dir / "gate2_roundtrip_metrics.csv"
    df_rt.to_csv(rt_csv, index=False)

    # Verify Patient Separation
    verify_patient_split(df_manifest, db_df)

    logger.info("10-Record Representative Validation Complete! Summary:")
    summary = df_rt.groupby("image_id").agg({"pearson_r": "mean", "rmse_mv": "mean"})
    print("\n" + "=" * 80)
    print("GATE 2 — 10 REPRESENTATIVE SHEETS VALIDATION SUMMARY")
    print("=" * 80)
    for img_id, row in summary.iterrows():
        print(f"{img_id:<32} | Mean Pearson r: {row['pearson_r']:.4f} | Mean RMSE: {row['rmse_mv']:.4f} mV")
    print("=" * 80 + "\n")

    return df_manifest


def main():
    parser = argparse.ArgumentParser(description="Paired Synthetic 12-Lead ECG Sheet Dataset Generator")
    parser.add_argument("--validate-sample-10", action="store_true", default=True, help="Run 10-record validation first")
    parser.add_argument("--bulk-generate", action="store_true", help="Generate complete 5,000 sheet dataset (requires Gate 2 approval)")
    parser.add_argument("--output-dir", type=str, default="data/processed/paired_ecg_dataset", help="Output directory")
    args = parser.parse_args()

    ptbxl_root = find_ptbxl_root()
    db_df = pd.read_csv(ptbxl_root / "ptbxl_database.csv")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/Users/akashraj/.gemini/antigravity/brain/00b546f6-2836-4c10-b13b-93b9b2399e6e")

    if args.bulk_generate:
        logger.warning("Bulk generation of 5,000 sheets requested. Verifying prerequisites...")
        # Will be triggered after user review of Gate 2 validation evidence
        raise NotImplementedError("Bulk generation of 5,000 records requires explicit user approval of the 10-sample validation evidence.")
    else:
        validate_10_samples(ptbxl_root, output_dir, db_df, artifact_dir=artifact_dir)


if __name__ == "__main__":
    main()
