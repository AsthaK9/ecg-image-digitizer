"""Generate Small Deterministic Paired Evaluation Dataset for Phase 2.

Produces a verified paired evaluation set using strictly PTB-XL Validation Fold 9 records
(zero patient overlap with Training Folds 1-8):
1. case_01_clean_3x4_1: Clean clinical 3x4 + 1 rhythm strip (record 00008_lr, patient 11275)
2. case_02_degraded_3x4_1: Degraded 3x4 + 1 rhythm strip (record 00010_lr, patient 9456)
3. case_03_marquette_3x4_3: Marquette MAC 6-row format with 3 rhythm strips (record 00017_lr, patient 13619)
4. case_04_stacked_12x1: 12 continuous stacked horizontal rows (record 00008_lr, patient 11275)
5. case_05_single_lead_n10007: Single-beat qualitative real image reference (N10007.png)

Outputs images, ground-truth masks, numpy signals, CSVs, and master manifest.json.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd

from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import ECGRecordPaths
from ecg_digitization.synthetic.generator import (
    DegradationConfig,
    SyntheticECGGenerator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_ptbxl_root() -> Path:
    candidates = list(Path("data/raw/ptb-xl").glob("**/ptbxl_database.csv"))
    if not candidates:
        raise FileNotFoundError("ptbxl_database.csv not found under data/raw/ptb-xl")
    return candidates[0].parent


def load_record_by_id(ptbxl_root: Path, record_filename: str):
    hea_path = ptbxl_root / f"{record_filename}.hea"
    dat_path = ptbxl_root / f"{record_filename}.dat"
    if not hea_path.is_file() or not dat_path.is_file():
        raise FileNotFoundError(f"Missing WFDB files: {hea_path} or {dat_path}")
    rec_id = Path(record_filename).name
    parser = WFDBRecordParser()
    return parser.parse(ECGRecordPaths(rec_id, hea_path, dat_path), source_dataset="ptbxl")


def verify_patient_isolation(db_df: pd.DataFrame, eval_patient_ids: set) -> None:
    train_df = db_df[db_df["strat_fold"].isin(range(1, 9))]
    train_patients = set(train_df["patient_id"].unique())
    overlap = eval_patient_ids.intersection(train_patients)
    if overlap:
        raise ValueError(f"CRITICAL: Patient data leakage detected! Overlapping patients: {overlap}")
    logger.info("Patient isolation verified: 0 overlap between evaluation set and training folds 1-8.")


def main():
    output_dir = Path("data/interim/small_paired_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    ptbxl_root = get_ptbxl_root()
    db_df = pd.read_csv(ptbxl_root / "ptbxl_database.csv", index_col="ecg_id")

    # Define the 4 Fold 9 test cases
    cases_def = [
        {
            "case_id": "case_01_clean_3x4_1",
            "ecg_id": 8,
            "filename_lr": "records100/00000/00008_lr",
            "layout_type": "3x4_1",
            "description": "Clean 12-lead sheet (3x4 + Lead II rhythm strip) with 1 mV pulse and pink grid",
            "degradation": DegradationConfig(enabled=False),
        },
        {
            "case_id": "case_02_degraded_3x4_1",
            "ecg_id": 10,
            "filename_lr": "records100/00000/00010_lr",
            "layout_type": "3x4_1",
            "description": "Degraded 12-lead sheet (rotation 1.2 deg, noise, blur, shadow, JPEG Q60, faded grid)",
            "degradation": DegradationConfig(
                enabled=True,
                rotation_deg=1.2,
                gaussian_noise_sigma=3.5,
                blur_kernel_size=3,
                shadow_gradient=True,
                jpeg_quality=60,
                grid_color_style="faded",
            ),
        },
        {
            "case_id": "case_03_marquette_3x4_3",
            "ecg_id": 17,
            "filename_lr": "records100/00000/00017_lr",
            "layout_type": "3x4_3",
            "description": "Marquette MAC format (6 rows: 3x4 grid + 3 rhythm strips V1, II, V5)",
            "degradation": DegradationConfig(enabled=False),
        },
        {
            "case_id": "case_04_stacked_12x1",
            "ecg_id": 8,
            "filename_lr": "records100/00000/00008_lr",
            "layout_type": "12x1",
            "description": "Continuous 12x1 stacked horizontal rows (10s continuous per lead)",
            "degradation": DegradationConfig(enabled=False),
        },
    ]

    eval_patient_ids = set()
    for c in cases_def:
        row = db_df.loc[c["ecg_id"]]
        p_id = row["patient_id"]
        s_fold = int(row["strat_fold"])
        assert s_fold == 9, f"Record {c['ecg_id']} is not in Fold 9! (Fold: {s_fold})"
        c["patient_id"] = float(p_id)
        c["strat_fold"] = s_fold
        eval_patient_ids.add(p_id)

    verify_patient_isolation(db_df, eval_patient_ids)

    synth = SyntheticECGGenerator()
    manifest_entries: Dict[str, Any] = {}

    for c in cases_def:
        case_id = c["case_id"]
        logger.info("Generating %s (layout: %s, record: %s)...", case_id, c["layout_type"], c["filename_lr"])
        sig = load_record_by_id(ptbxl_root, c["filename_lr"])

        res = synth.generate(
            signal=sig,
            degradation=c["degradation"],
            seed=42,
            layout_type=c["layout_type"],
        )

        img_path = output_dir / f"{case_id}.png"
        clean_path = output_dir / f"{case_id}_clean.png"
        mask_path = output_dir / f"{case_id}_mask.png"
        sig_path = output_dir / f"{case_id}_gt_signal.npy"
        csv_path = output_dir / f"{case_id}_gt_signal.csv"
        meta_path = output_dir / f"{case_id}_meta.json"

        cv2.imwrite(str(img_path), res.image)
        cv2.imwrite(str(clean_path), res.clean_image)
        cv2.imwrite(str(mask_path), res.trace_mask)
        np.save(str(sig_path), res.source_signal)

        # Save tabular ground truth CSV
        gt_df = pd.DataFrame(res.source_signal.T, columns=res.lead_names)
        gt_df.index.name = "sample_index"
        gt_df["time_sec"] = np.arange(len(gt_df)) / res.sampling_rate_hz
        gt_df.to_csv(csv_path)

        meta = {
            "case_id": case_id,
            "source_dataset": "PTB-XL",
            "strat_fold": c["strat_fold"],
            "patient_id": c["patient_id"],
            "ptbxl_filename": c["filename_lr"],
            "layout_type": c["layout_type"],
            "description": c["description"],
            "image_shape": list(res.image.shape),
            "sampling_rate_hz": res.sampling_rate_hz,
            "num_leads": len(res.lead_bounding_boxes),
            "lead_names": list(res.lead_bounding_boxes.keys()),
            "lead_bounding_boxes": res.lead_bounding_boxes,
            "lead_baselines": res.lead_baselines,
            "calibration": res.calibration,
            "degradation": {
                "enabled": c["degradation"].enabled,
                "rotation_deg": c["degradation"].rotation_deg,
                "gaussian_noise_sigma": c["degradation"].gaussian_noise_sigma,
                "blur_kernel_size": c["degradation"].blur_kernel_size,
                "shadow_gradient": c["degradation"].shadow_gradient,
                "jpeg_quality": c["degradation"].jpeg_quality,
                "grid_color_style": c["degradation"].grid_color_style,
            },
            "files": {
                "image": str(img_path.name),
                "clean_image": str(clean_path.name),
                "mask": str(mask_path.name),
                "gt_signal_npy": str(sig_path.name),
                "gt_signal_csv": str(csv_path.name),
            },
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        manifest_entries[case_id] = meta

    # Add Case 5: Single-lead real reference
    case_05_id = "case_05_single_lead_n10007"
    src_single = Path("data/real_ecg_eval/samples/N10007.png")
    if src_single.is_file():
        dst_img = output_dir / f"{case_05_id}.png"
        shutil.copyfile(src_single, dst_img)
        img_single = cv2.imread(str(dst_img))
        manifest_entries[case_05_id] = {
            "case_id": case_05_id,
            "source_dataset": "ECG Image Data (MIT-BIH Beat Class N)",
            "strat_fold": None,
            "patient_id": None,
            "ptbxl_filename": None,
            "layout_type": "single_lead",
            "description": "Real single-beat plot (N10007.png) from ECG Image Data without printed grid lines",
            "image_shape": list(img_single.shape),
            "sampling_rate_hz": 100.0,
            "num_leads": 1,
            "lead_names": ["II"],
            "lead_bounding_boxes": {"II": [0, 0, img_single.shape[1], img_single.shape[0]]},
            "lead_baselines": {"II": img_single.shape[0] // 2},
            "calibration": {"is_calibration_reliable": False, "note": "Uncalibrated single beat plot"},
            "degradation": {"enabled": False},
            "files": {
                "image": str(dst_img.name),
                "clean_image": str(dst_img.name),
                "mask": None,
                "gt_signal_npy": None,
                "gt_signal_csv": None,
            },
        }

    manifest = {
        "dataset_name": "ECG DigiAI Phase 2 Deterministic Paired Evaluation Dataset",
        "description": "Paired benchmark set containing full 12-lead clinical sheets (clean, degraded, Marquette 6-row, 12x1 stacked) and 1 real single-lead reference. All synthetic cases derived strictly from PTB-XL Fold 9 with zero training patient leakage.",
        "num_cases": len(manifest_entries),
        "ptbxl_validation_fold": 9,
        "cases": manifest_entries,
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Successfully generated small deterministic paired evaluation dataset at: %s", output_dir)
    logger.info("Master manifest saved: %s (%d test cases)", manifest_file, len(manifest_entries))


if __name__ == "__main__":
    main()
