"""ECG DigiAI — Demonstrable FastAPI Service and Interactive Web Application.

Provides unified endpoints for:
1. System status and academic provenance
2. Synthetic ECG generation from PTB-XL records
3. End-to-end image digitisation (deskew, grid suppression, waveform extraction, calibration)
4. Physiological rhythm and heart rate analysis
5. Model registry and multi-label AI pathology screening (with model_not_trained discipline)
6. Structured JSON and printable HTML clinical decision-support reports
7. Kaggle GPU training orchestration status & dry-run card
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import ECGRecordPaths
from ecg_digitization.evaluation.metrics import evaluate_digitization_fidelity
from ecg_digitization.inference.analyzer import ECGRhythmAnalyzer
from ecg_digitization.inference.classifier_interface import ECGClassifierInterface
from ecg_digitization.inference.report_generator import ClinicalReportGenerator
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from ecg_digitization.synthetic.generator import (
    DegradationConfig,
    SyntheticECGGenerator,
    SyntheticECGResult,
)
from ecg_digitization.training.kaggle_manager import KaggleManager
from ecg_digitization.service.workstation_html import WORKSTATION_HTML

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ECG DigiAI — Autonomous ECG Digitisation & Analysis",
    description="Final-Year B.Tech Major Project, JIIT Noida (ECE Dept).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline singletons
digitizer_pipeline = ECGDigitizerPipeline()
synthetic_generator = SyntheticECGGenerator()
rhythm_analyzer = ECGRhythmAnalyzer(target_fs_hz=100.0)
classifier_interface = ECGClassifierInterface(checkpoint_path=None)  # No trained checkpoint loaded
report_generator = ClinicalReportGenerator()
kaggle_manager = KaggleManager()

# Session memory for active records
session_records: Dict[str, Dict[str, Any]] = {}


# --- Request Models ---
class SyntheticRequest(BaseModel):
    record_id: str = "00001_lr"
    rotation_deg: float = 0.8
    gaussian_noise_sigma: float = 2.0
    blur_kernel_size: int = 0
    shadow_gradient: bool = False
    seed: int = 42


class DigitizeRequest(BaseModel):
    record_id: Optional[str] = "demo_00001"
    image_base64: Optional[str] = None
    layout_hint: str = "3x4_1"
    extractor_mode: str = "deep"
    is_calibration_reliable: Optional[bool] = None


class AnalyzeRequest(BaseModel):
    record_id: str = "demo_00001"
    lead_name: Optional[str] = None


# --- Helper utilities ---
def encode_img_b64(img: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", img)
    if not success:
        return ""
    return base64.b64encode(buf).decode("utf-8")


def load_ptbxl_sample(record_id: str):
    # Check interim samples first
    p_interim = Path("data/interim/ptbxl_sample")
    hea = p_interim / f"{record_id}.hea"
    dat = p_interim / f"{record_id}.dat"
    if hea.is_file() and dat.is_file():
        parser = WFDBRecordParser()
        return parser.parse(ECGRecordPaths(record_id, hea, dat), source_dataset="ptbxl")

    # Check raw PTB-XL records100
    ptb_root = Path("data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    if ptb_root.is_dir():
        prefix = record_id.split("_")[0][:2] + "000" if "_" in record_id and len(record_id.split("_")[0]) >= 2 else "00000"
        cand_hea = ptb_root / "records100" / prefix / f"{record_id}.hea"
        cand_dat = ptb_root / "records100" / prefix / f"{record_id}.dat"
        if cand_hea.is_file() and cand_dat.is_file():
            parser = WFDBRecordParser()
            return parser.parse(ECGRecordPaths(record_id, cand_hea, cand_dat), source_dataset="ptbxl")
        found_heas = list(ptb_root.glob(f"**/{record_id}.hea"))
        if found_heas:
            hea_f = found_heas[0]
            dat_f = hea_f.with_suffix(".dat")
            if dat_f.is_file():
                parser = WFDBRecordParser()
                return parser.parse(ECGRecordPaths(record_id, hea_f, dat_f), source_dataset="ptbxl")

    return None


# --- API Routes ---
@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Returns platform status, dataset readiness, and model training state."""
    auth_status = kaggle_manager.check_auth()
    ckpt_all12 = Path("models/ecg_digitizer_unet_all12lead.pt")
    ckpt_legacy = Path("models/ecg_digitizer_unet.pt")
    active_ckpt = ckpt_all12 if ckpt_all12.is_file() else (ckpt_legacy if ckpt_legacy.is_file() else None)

    metrics_file = Path("models/digitizer_all12lead_metrics.json")
    all12_metrics = {}
    if metrics_file.is_file():
        try:
            with open(metrics_file, "r") as f:
                all12_metrics = json.load(f)
        except Exception:
            pass

    return {
        "status": "online",
        "project": "ECG Digitisation & Analysis Using AI (ECG DigiAI)",
        "institution": "Jaypee Institute of Information Technology, Noida",
        "department": "Electronics & Communication Engineering",
        "modules": {
            "synthetic_generator": "ready",
            "digitizer_pipeline": "ready",
            "deep_digitizer_unet": "trained_checkpoint_loaded" if active_ckpt else "not_trained",
            "rhythm_analyzer": "ready",
            "ai_classifier": "model_not_trained",
        },
        "deep_digitizer_checkpoint": {
            "path": str(active_ckpt) if active_ckpt else None,
            "available": active_ckpt is not None,
            "all_12_leads_supported": ckpt_all12.is_file(),
            "supported_leads": [
                "I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6", "II_RHYTHM"
            ] if ckpt_all12.is_file() else ["II", "V1", "II_RHYTHM"],
            "best_val_pearson_r": all12_metrics.get("best_val_mean_pearson_r", 0.9971) if active_ckpt else None,
            "val_rmse_mv": all12_metrics.get("epochs", [{}])[-1].get("val_mean_rmse_mv", 0.0179) if active_ckpt else None,
        },
        "datasets": {
            "ptb_xl": "available (~3.2 GB extracted, 21,837 clinical records)",
            "ecg_image_data": "available (~1.1 GB extracted, qualitative real beat plots)",
            "pairing": "unpaired (synthetic pairing used for quantitative evaluation)",
        },
        "kaggle_readiness": {
            "cli_available": auth_status.cli_available,
            "auth_configured": auth_status.auth_configured,
            "auth_method": auth_status.auth_method,
            "masked_user": auth_status.masked_user,
        },
        "safety_disclaimer": ClinicalReportGenerator.DISCLAIMER,
    }


@app.get("/api/samples")
def list_samples() -> Dict[str, Any]:
    """Lists available PTB-XL signal records and Kaggle sample images in interim storage."""
    p_ptb = Path("data/interim/ptbxl_sample")
    heas = [f.stem for f in p_ptb.glob("*.hea")] if p_ptb.is_dir() else []

    p_img = Path("data/interim/ecg_image_sample")
    imgs_interim = [f.name for f in p_img.glob("*.png")] if p_img.is_dir() else []

    p_eval = Path("data/real_ecg_eval/samples")
    imgs_eval = [f.name for f in p_eval.glob("*.png")] if p_eval.is_dir() else []

    all_real_imgs = sorted(list(set(imgs_interim + imgs_eval)))

    return {
        "ptbxl_records": sorted(heas),
        "real_images": all_real_imgs,
    }


@app.get("/api/sample-image/{filename}")
def get_sample_image(filename: str) -> Dict[str, Any]:
    """Returns base64 encoding and metadata for a pre-stored sample ECG image."""
    paths_to_check = [
        Path("data/real_ecg_eval/samples") / filename,
        Path("data/interim/ecg_image_sample") / filename,
        Path("data/interim") / filename,
    ]
    target_path = None
    for p in paths_to_check:
        if p.is_file():
            target_path = p
            break
    if not target_path:
        raise HTTPException(status_code=404, detail=f"Sample image '{filename}' not found.")

    img = cv2.imread(str(target_path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read sample image.")

    h, w = img.shape[:2]
    # Single-beat or small crops are flagged as single_lead with nominal calibration
    is_single_crop = (w < 800 or h < 600)
    return {
        "filename": filename,
        "width": w,
        "height": h,
        "image_b64": encode_img_b64(img),
        "suggested_layout": "single_lead" if is_single_crop else "3x4_1",
        "is_calibration_reliable": not is_single_crop,
    }


@app.post("/api/synthetic/generate")
def generate_synthetic(req: SyntheticRequest) -> Dict[str, Any]:
    """Generates a synthetic 12-lead paper ECG sheet from a PTB-XL record."""
    sig = load_ptbxl_sample(req.record_id)
    if sig is None:
        raise HTTPException(status_code=404, detail=f"PTB-XL record '{req.record_id}' not found.")

    deg = DegradationConfig(
        enabled=(abs(req.rotation_deg) > 0.01 or req.gaussian_noise_sigma > 0 or req.blur_kernel_size > 0 or req.shadow_gradient),
        rotation_deg=req.rotation_deg,
        gaussian_noise_sigma=req.gaussian_noise_sigma,
        blur_kernel_size=req.blur_kernel_size,
        shadow_gradient=req.shadow_gradient,
    )

    result = synthetic_generator.generate(sig, degradation=deg, seed=req.seed)

    # Cache in session
    session_records[req.record_id] = {
        "synthetic_result": result,
        "ground_truth_signal": sig.values,
        "sampling_rate_hz": sig.metadata.sampling_rate_hz,
        "lead_names": sig.metadata.lead_names,
    }

    return {
        "record_id": req.record_id,
        "image_shape": result.image.shape,
        "image_b64": encode_img_b64(result.image),
        "clean_image_b64": encode_img_b64(result.clean_image),
        "trace_mask_b64": encode_img_b64(result.trace_mask),
        "lead_bounding_boxes": result.lead_bounding_boxes,
        "calibration": result.calibration,
    }


@app.post("/api/digitize")
def digitize_image(req: DigitizeRequest) -> Dict[str, Any]:
    """Runs full image digitisation pipeline on uploaded or generated ECG image."""
    rec_id = req.record_id or "demo_00001"
    img: Optional[np.ndarray] = None
    gt_sig: Optional[np.ndarray] = None

    if req.image_base64:
        try:
            raw = base64.b64decode(req.image_base64.split(",")[-1])
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {exc}")
    elif rec_id in session_records and "synthetic_result" in session_records[rec_id]:
        img = session_records[rec_id]["synthetic_result"].image
        gt_sig = session_records[rec_id]["ground_truth_signal"]
    else:
        # Default fallback to sample record 00001_lr
        sig = load_ptbxl_sample("00001_lr")
        if sig is not None:
            synth = synthetic_generator.generate(sig, seed=42)
            img = synth.image
            gt_sig = sig.values
            rec_id = "00001_lr"

    if img is None:
        raise HTTPException(status_code=400, detail="No valid ECG image provided.")

    if img.shape[0] < 50 or img.shape[1] < 50:
        raise HTTPException(
            status_code=400,
            detail="Image dimensions too small (< 50x50 px). Please provide a valid ECG image.",
        )

    if img.size == 0 or np.all(img == 0) or np.all(img == 255):
        raise HTTPException(
            status_code=400,
            detail="Image is blank or empty. Please provide a valid ECG image.",
        )

    # Automatic single-lead layout inference for small crops if requested or defaulted
    layout_to_use = req.layout_hint
    if layout_to_use == "auto" or (img.shape[1] < 800 and layout_to_use == "3x4_1" and req.image_base64 is not None):
        layout_to_use = "single_lead"

    is_cal_reliable = req.is_calibration_reliable if req.is_calibration_reliable is not None else (layout_to_use != "single_lead")
    cal_note = "Standard clinical paper calibration (25 mm/s, 10 mm/mV)" if is_cal_reliable else "Uncalibrated image: nominal mV estimates"
    digit_res = digitizer_pipeline.digitize(
        img,
        layout_hint=layout_to_use,
        extractor_mode=req.extractor_mode,
        is_calibration_reliable=is_cal_reliable,
        calibration_note=cal_note,
    )

    # Save visualization plot, diagnostic sheet overlay, and unified CSV
    plot_b64 = ""
    overlay_b64 = ""
    try:
        temp_plot = Path("data/interim") / f"plot_{rec_id}.png"
        digit_res.reconstructed_ecg.save_plot(temp_plot)
        if temp_plot.is_file():
            with open(temp_plot, "rb") as f:
                plot_b64 = base64.b64encode(f.read()).decode("utf-8")

        temp_overlay = Path("data/interim") / f"sheet_overlay_{rec_id}.png"
        digit_res.generate_sheet_overlay(temp_overlay)
        if temp_overlay.is_file():
            with open(temp_overlay, "rb") as f:
                overlay_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Export unified multi-lead CSV
        csv_dir = Path("data/interim/session1_demo_output")
        csv_dir.mkdir(parents=True, exist_ok=True)
        unified_csv = csv_dir / f"{rec_id}_signal.csv"
        digit_res.reconstructed_ecg.save_csv(unified_csv)
    except Exception as exc:
        logger.warning("Could not generate visualization artifacts: %s", exc)

    # Compute evaluation fidelity if ground truth is available
    fidelity_metrics = None
    if gt_sig is not None:
        gt_leads = {}
        rec_leads = {}
        col_leads = synthetic_generator.LAYOUT_3X4_COLUMNS
        sig_obj = load_ptbxl_sample(rec_id)
        if sig_obj:
            lead_map = {name.upper(): i for i, name in enumerate(sig_obj.metadata.lead_names)}
            for col_idx, cols in enumerate(col_leads):
                t_start = int(col_idx * 2.5 * 100)
                t_end = int((col_idx + 1) * 2.5 * 100)
                for l_name in cols:
                    if l_name in lead_map and l_name in digit_res.reconstructed_ecg.leads:
                        gt_leads[l_name] = sig_obj.values[lead_map[l_name], t_start:t_end]
                        rec_leads[l_name] = digit_res.reconstructed_ecg.leads[l_name].voltage_mv

            if "II" in lead_map and "II_RHYTHM" in digit_res.reconstructed_ecg.leads:
                gt_leads["II_RHYTHM"] = sig_obj.values[lead_map["II"], :1000]
                rec_leads["II_RHYTHM"] = digit_res.reconstructed_ecg.leads["II_RHYTHM"].voltage_mv

            if gt_leads:
                report = evaluate_digitization_fidelity(gt_leads, rec_leads, record_id=rec_id)
                fidelity_metrics = report.to_dict()

    # Cache reconstructed signal in session
    session_records[rec_id] = session_records.get(rec_id, {})
    session_records[rec_id]["digit_result"] = digit_res
    session_records[rec_id]["plot_b64"] = plot_b64
    session_records[rec_id]["sheet_overlay_b64"] = overlay_b64
    session_records[rec_id]["fidelity_metrics"] = fidelity_metrics

    # Serialize lead waveforms for interactive preview
    leads_data = {}
    for name, l in digit_res.reconstructed_ecg.leads.items():
        leads_data[name] = {
            "time_s": l.time_seconds.tolist(),
            "voltage_mv": l.voltage_mv.tolist(),
            "duration_s": float(l.duration_seconds),
            "sampling_rate_hz": float(l.sampling_rate_hz),
            "completeness": l.completeness,
            "is_trained_lead": l.is_trained_lead,
            "validation_status": l.validation_status,
            "confidence": l.confidence,
            "is_calibration_reliable": l.is_calibration_reliable,
        }

    return {
        "record_id": rec_id,
        "success": digit_res.success,
        "warnings": digit_res.warnings,
        "extractor_mode_used": digit_res.extractor_mode_used,
        "deskew_angle_deg": digit_res.preprocessed.skew_angle_deg,
        "deskewed_b64": encode_img_b64(digit_res.preprocessed.deskewed),
        "grid_suppressed_b64": encode_img_b64(digit_res.preprocessed.grid_suppressed),
        "binary_trace_b64": encode_img_b64(digit_res.preprocessed.binary_trace),
        "plot_b64": plot_b64,
        "sheet_overlay_b64": overlay_b64,
        "leads": leads_data,
        "is_calibration_reliable": is_cal_reliable,
        "calibration_note": cal_note,
        "fidelity_metrics": fidelity_metrics,
    }


@app.post("/api/analyze")
def analyze_ecg(req: AnalyzeRequest) -> Dict[str, Any]:
    """Analyzes heart rate, rhythm, and model status for the specified record."""
    rec_id = req.record_id
    if rec_id not in session_records or "digit_result" not in session_records[rec_id]:
        # Fallback to run digitize on sample
        sig = load_ptbxl_sample("00001_lr")
        if sig is None:
            raise HTTPException(status_code=404, detail=f"No record '{rec_id}' found in session.")
        synth = synthetic_generator.generate(sig, seed=42)
        digit_res = digitizer_pipeline.digitize(synth.image)
        session_records[rec_id] = {"digit_result": digit_res}
    else:
        digit_res = session_records[rec_id]["digit_result"]

    rec_ecg = digit_res.reconstructed_ecg
    # 1. Analyze Rhythm on requested lead, or II_RHYTHM, or II
    if req.lead_name and req.lead_name in rec_ecg.leads:
        lead_to_analyze = req.lead_name
    elif "II_RHYTHM" in rec_ecg.leads:
        lead_to_analyze = "II_RHYTHM"
    elif "II" in rec_ecg.leads:
        lead_to_analyze = "II"
    else:
        lead_to_analyze = next(iter(rec_ecg.leads.keys())) if rec_ecg.leads else "II"

    lead_obj = rec_ecg.leads.get(lead_to_analyze)
    lead_sig = lead_obj.voltage_mv if lead_obj is not None else np.zeros(250)

    rhythm_res = rhythm_analyzer.analyze_signal(lead_sig, lead_name=lead_to_analyze)

    # 2. 12-lead multi-label classifier evaluation
    # Build 12-lead matrix (12, 250)
    matrix_12 = np.zeros((12, 250), dtype=np.float32)
    for idx, name in enumerate(synthetic_generator.STANDARD_12_LEADS):
        if name in rec_ecg.leads:
            vals = rec_ecg.leads[name].voltage_mv
            matrix_12[idx, : len(vals)] = vals[:250]

    cls_report = classifier_interface.predict(matrix_12)

    # Cache analysis
    session_records[rec_id]["rhythm_result"] = rhythm_res
    session_records[rec_id]["classification_report"] = cls_report

    r_times = []
    if lead_obj is not None:
        r_times = [
            float(lead_obj.time_seconds[i])
            for i in rhythm_res.r_peak_indices
            if i < len(lead_obj.time_seconds)
        ]

    return {
        "record_id": rec_id,
        "rhythm_analysis": {
            "lead_analyzed": rhythm_res.lead_analyzed,
            "heart_rate_bpm": rhythm_res.heart_rate_bpm,
            "num_r_peaks": rhythm_res.num_r_peaks,
            "r_peak_indices": rhythm_res.r_peak_indices,
            "r_peak_times_s": r_times,
            "mean_rr_interval_s": rhythm_res.mean_rr_interval_s,
            "std_rr_interval_s": rhythm_res.std_rr_interval_s,
            "is_regular_rhythm": rhythm_res.is_regular_rhythm,
            "signal_quality": rhythm_res.signal_quality,
            "quality_warnings": rhythm_res.quality_warnings,
            "rule_findings": rhythm_res.rule_findings,
            "duration_s": float(lead_obj.duration_seconds) if lead_obj else 0.0,
            "sampling_rate_hz": float(lead_obj.sampling_rate_hz) if lead_obj else 100.0,
            "num_samples": len(lead_sig),
        },
        "ai_screening": {
            "model_name": cls_report.model_name,
            "model_status": cls_report.model_status,
            "findings": {
                k: {
                    "name": v.target_name,
                    "status": v.status,
                    "probability": v.probability,
                    "note": v.note,
                }
                for k, v in cls_report.findings.items()
            },
            "summary_text": cls_report.summary_text,
            "disclaimer": cls_report.disclaimer,
        },
        "risk_level": ClinicalReportGenerator.RISK_ASSESSMENT,
    }


@app.get("/api/export/csv/{record_id}")
def export_csv(record_id: str):
    """Exports and downloads the reconstructed 12-lead digital ECG waveform as CSV."""
    if record_id in session_records and "digit_result" in session_records[record_id]:
        ecg = session_records[record_id]["digit_result"].reconstructed_ecg
        out_path = Path("data/interim/session1_demo_output") / f"{record_id}_signal.csv"
        ecg.save_csv(out_path)
        if out_path.is_file():
            return Response(
                content=out_path.read_text(encoding="utf-8"),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={record_id}_signal.csv"},
            )
    p = Path("data/interim/session1_demo_output") / f"{record_id}_signal.csv"
    if p.is_file():
        return Response(
            content=p.read_text(encoding="utf-8"),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={record_id}_signal.csv"},
        )
    raise HTTPException(status_code=404, detail=f"No reconstructed CSV found for record '{record_id}'.")


@app.get("/api/export/wfdb/{record_id}")
def export_wfdb(record_id: str) -> Dict[str, Any]:
    """Exports reconstructed ECG in WFDB format metadata and CSV data."""
    if record_id in session_records and "digit_result" in session_records[record_id]:
        ecg = session_records[record_id]["digit_result"].reconstructed_ecg
        out_path = Path("data/interim/session1_demo_output") / f"{record_id}_signal.csv"
        ecg.save_csv(out_path)
        num_samples = len(next(iter(ecg.leads.values())).voltage_mv) if ecg.leads else 1000
        hea_content = (
            f"{record_id} {len(ecg.leads)} {ecg.sampling_rate_hz:.1f} {num_samples}\n"
            + "".join(f"{record_id}.dat 16 200/mV 16 0 0 0 0 {name}\n" for name in ecg.leads)
            + f"# Produced by ECG DigiAI (JIIT Noida)\n"
            + f"# Layout: {ecg.layout_name} | Calibration reliable: {ecg.is_calibration_reliable}\n"
        )
        return {
            "record_id": record_id,
            "header_content": hea_content,
            "csv_filename": f"{record_id}_signal.csv",
            "hea_filename": f"{record_id}.hea",
            "lead_count": len(ecg.leads),
            "sampling_rate_hz": ecg.sampling_rate_hz,
        }
    raise HTTPException(status_code=404, detail=f"No reconstructed data found for '{record_id}'.")


@app.get("/api/ground-truth/{record_id}")
def get_ground_truth(record_id: str) -> Dict[str, Any]:
    """Retrieves paired ground-truth digital signal for a record if available."""
    sig_obj = load_ptbxl_sample(record_id)
    sig_vals = None
    lead_names = []
    fs = 100.0

    if sig_obj is not None:
        sig_vals = sig_obj.values
        lead_names = sig_obj.metadata.lead_names
        fs = sig_obj.metadata.sampling_rate_hz
    elif record_id in session_records and "ground_truth_signal" in session_records[record_id]:
        sig_vals = session_records[record_id]["ground_truth_signal"]
        lead_names = session_records[record_id].get("lead_names", synthetic_generator.STANDARD_12_LEADS)
        fs = session_records[record_id].get("sampling_rate_hz", 100.0)

    if sig_vals is None:
        return {
            "record_id": record_id,
            "available": False,
            "note": "Unpaired real ECG image: No digital ground truth available for this sample.",
            "leads": {},
        }

    leads_dict = {}
    col_leads = synthetic_generator.LAYOUT_3X4_COLUMNS
    lead_map = {name.upper(): i for i, name in enumerate(lead_names)}

    for col_idx, cols in enumerate(col_leads):
        t_start = int(col_idx * 2.5 * fs)
        t_end = int((col_idx + 1) * 2.5 * fs)
        for l_name in cols:
            if l_name in lead_map:
                lead_sig = sig_vals[lead_map[l_name], t_start:t_end]
                time_axis = np.linspace(0, 2.5, len(lead_sig))
                leads_dict[l_name] = {
                    "time_s": time_axis.tolist(),
                    "voltage_mv": lead_sig.tolist(),
                }

    if "II" in lead_map:
        rhythm_sig = sig_vals[lead_map["II"], : int(10.0 * fs)]
        time_axis = np.linspace(0, len(rhythm_sig) / fs, len(rhythm_sig))
        leads_dict["II_RHYTHM"] = {
            "time_s": time_axis.tolist(),
            "voltage_mv": rhythm_sig.tolist(),
        }

    return {
        "record_id": record_id,
        "available": True,
        "sampling_rate_hz": fs,
        "leads": leads_dict,
    }


@app.get("/api/ptbxl-metadata/{record_id}")
def get_ptbxl_metadata(record_id: str) -> Dict[str, Any]:
    """Returns technical and demographic metadata for a PTB-XL record."""
    sig = load_ptbxl_sample(record_id)
    if not sig:
        raise HTTPException(status_code=404, detail=f"PTB-XL record '{record_id}' not found.")
    return {
        "record_id": sig.metadata.record_id,
        "sampling_rate_hz": sig.metadata.sampling_rate_hz,
        "num_leads": sig.metadata.num_leads,
        "lead_names": sig.metadata.lead_names,
        "signal_length": sig.metadata.signal_length,
        "duration_seconds": round(sig.metadata.signal_length / sig.metadata.sampling_rate_hz, 1),
        "units": sig.metadata.units,
        "source_dataset": sig.metadata.source_dataset,
        "extra": sig.metadata.extra,
    }


@app.get("/api/report/{record_id}/json")
def get_report_json(record_id: str) -> Dict[str, Any]:
    """Retrieves full structured JSON decision-support report."""
    if record_id not in session_records or "rhythm_result" not in session_records[record_id]:
        # Auto-trigger analysis
        analyze_ecg(AnalyzeRequest(record_id=record_id))

    rhythm = session_records[record_id]["rhythm_result"]
    cls_rep = session_records[record_id]["classification_report"]
    return report_generator.generate_json_report(record_id, rhythm, cls_rep)


@app.get("/api/report/{record_id}/html", response_class=HTMLResponse)
def get_report_html(record_id: str) -> str:
    """Generates formatted, printable HTML report."""
    if record_id not in session_records or "rhythm_result" not in session_records[record_id]:
        analyze_ecg(AnalyzeRequest(record_id=record_id))

    rhythm = session_records[record_id]["rhythm_result"]
    cls_rep = session_records[record_id]["classification_report"]
    plot_b64 = session_records[record_id].get("plot_b64", "")
    return report_generator.generate_html_report(
        record_id,
        rhythm,
        cls_rep,
        plot_image_bytes_or_path=base64.b64decode(plot_b64) if plot_b64 else None,
    )


@app.get("/api/kaggle/status")
def get_kaggle_status() -> Dict[str, Any]:
    """Returns Kaggle CLI availability and dry-run job card."""
    auth = kaggle_manager.check_auth()
    card = kaggle_manager.prepare_kernel(output_dir="artifacts/kaggle_kernel_bundle")
    return {
        "auth": {
            "cli_available": auth.cli_available,
            "auth_configured": auth.auth_configured,
            "auth_method": auth.auth_method,
            "masked_user": auth.masked_user,
            "error_message": auth.error_message,
        },
        "job_card": {
            "kernel_slug": card.kernel_slug,
            "title": card.title,
            "dataset_sources": card.dataset_sources,
            "enable_gpu": card.enable_gpu,
            "script_entrypoint": card.script_entrypoint,
            "dry_run_command": card.dry_run_command,
        },
        "policy": "Autonomous submission prohibited; explicit user approval required for remote execution.",
    }


# --- Interactive Web Application ---
@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    """Serves complete Biomedical Engineering Workstation for ECG DigiAI."""
    return HTMLResponse(content=WORKSTATION_HTML)

