"""ECG DigiAI — End-to-End Three-Scenario Verification Suite.

Exercises and verifies the three required real-world clinical workstation scenarios:
1. Scenario A — Synthetic Paired ECG:
   Select PTB-XL Record -> Generate degraded 12-lead paper ECG -> Digitize with U-Net ->
   Compare against exact ground truth -> Compute per-lead metrics -> Export CSV & WFDB.
2. Scenario B — Real ECG Image:
   Upload real ECG image (Kaggle sample) -> Preprocess & Normalize -> Digitize ->
   Inspect waveform & verify calibration_reliable == False (qualitative transparency).
3. Scenario C — Partial/Single Lead:
   Upload single-lead crop -> Extract available lead only -> Verify zero fabricated missing leads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import cv2
import numpy as np

from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import ECGRecordPaths
from ecg_digitization.evaluation.metrics import compute_lead_metrics
from ecg_digitization.inference.deep_digitizer import DeepCropDigitizer
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from ecg_digitization.synthetic.generator import DegradationConfig, SyntheticECGGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScenarioVerification")


def run_scenario_a() -> dict:
    logger.info("=== [SCENARIO A] Synthetic Paired 12-Lead ECG Demonstration ===")
    out_dir = Path("data/interim/verification_scenario_a")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load real PTB-XL record
    hea_path = Path("data/interim/ptbxl_sample/00001_lr.hea")
    dat_path = Path("data/interim/ptbxl_sample/00001_lr.dat")
    parser = WFDBRecordParser()
    sig = parser.parse(ECGRecordPaths("00001_lr", hea_path, dat_path), source_dataset="ptbxl")

    # 2. Render paper ECG image with synthetic generator
    generator = SyntheticECGGenerator()
    degradations = DegradationConfig(
        enabled=True,
        rotation_deg=0.0,
        gaussian_noise_sigma=1.5,
        blur_kernel_size=0,
    )
    synth_result = generator.generate(sig, degradation=degradations, seed=42)
    clean_img_path = out_dir / "00001_lr_synthetic_paper.png"
    cv2.imwrite(str(clean_img_path), synth_result.image)
    logger.info("Generated synthetic 12-lead paper image: %s (Dimensions: %s)", clean_img_path, synth_result.image.shape)

    # 3. Digitize using Deep U-Net Pipeline
    pipeline = ECGDigitizerPipeline(extractor_mode="deep")
    digit_res = pipeline.digitize(
        image_source=synth_result.image,
        layout_hint="3x4_1",
        extractor_mode="deep",
        is_calibration_reliable=True,
    )
    assert digit_res.success, "Digitization failed for Scenario A"
    logger.info("Digitization completed successfully. Leads reconstructed: %s", list(digit_res.reconstructed_ecg.leads.keys()))

    # 4. Save artifacts & 12-lead diagnostic overlay
    artifacts = digit_res.save_artifacts(out_dir, base_name="scenario_a_00001_lr")
    logger.info("Saved sheet overlay: %s", artifacts["sheet_overlay"])

    # 5. Quantitative Per-Lead Comparison against Ground Truth
    per_lead_results = {}
    for l_name in digit_res.reconstructed_ecg.leads:
        if l_name in synth_result.exact_lead_signals:
            gt_v = synth_result.exact_lead_signals[l_name]
            rec_v = np.array(digit_res.reconstructed_ecg.leads[l_name].voltage_mv)

            # Align lengths if necessary
            min_len = min(len(gt_v), len(rec_v))
            gt_crop = gt_v[:min_len]
            rec_crop = rec_v[:min_len]

            m = compute_lead_metrics(gt_crop, rec_crop, lead_name=l_name)
            per_lead_results[l_name] = {
                "pearson_r": round(m.pearson_r, 4),
                "rmse_mv": round(m.rmse_mv, 4),
                "snr_db": round(m.snr_db, 2),
                "completeness": round(m.completeness, 4),
                "is_validated_lead": digit_res.extracted_traces[l_name].is_trained_lead if l_name in digit_res.extracted_traces else True,
            }

    mean_r = float(np.mean([v["pearson_r"] for v in per_lead_results.values()]))
    mean_rmse = float(np.mean([v["rmse_mv"] for v in per_lead_results.values()]))
    logger.info("Scenario A Results — Mean Pearson r: %.4f | Mean RMSE: %.4f mV", mean_r, mean_rmse)

    return {
        "scenario": "A_synthetic_paired",
        "record_id": "00001_lr",
        "mean_pearson_r": round(mean_r, 4),
        "mean_rmse_mv": round(mean_rmse, 4),
        "per_lead_results": per_lead_results,
        "csv_path": str(artifacts["csv"]),
        "sheet_overlay_path": str(artifacts["sheet_overlay"]),
    }


def run_scenario_b() -> dict:
    logger.info("=== [SCENARIO B] Real ECG Image Qualitative Inference ===")
    out_dir = Path("data/interim/verification_scenario_b")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_img_path = Path("data/real_ecg_eval/samples/N10007.png")
    assert sample_img_path.is_file(), f"Sample image {sample_img_path} not found"
    img = cv2.imread(str(sample_img_path))

    deep_digitizer = DeepCropDigitizer()
    res = deep_digitizer.digitize_image(img, sample_name="N10007")

    csv_path = res.save_csv(out_dir / "N10007_reconstructed.csv")
    overlay_path = res.save_overlay_plot(out_dir / "N10007_diagnostic_overlay.png")

    assert res.is_calibration_reliable is False, "Real beat crop must be flagged calibration_reliable == False"
    logger.info(
        "Scenario B Complete — Mean Confidence: %.2f%% | Calibration Reliable: %s | CSV: %s",
        float(np.mean(res.column_confidence)) * 100.0,
        res.is_calibration_reliable,
        csv_path,
    )

    return {
        "scenario": "B_real_ecg_image",
        "sample_name": "N10007.png",
        "mean_confidence": round(float(np.mean(res.column_confidence)), 4),
        "is_calibration_reliable": res.is_calibration_reliable,
        "calibration_note": res.calibration_note,
        "csv_path": str(csv_path),
        "overlay_plot_path": str(overlay_path),
    }


def run_scenario_c() -> dict:
    logger.info("=== [SCENARIO C] Partial / Single-Lead Upload ===")
    out_dir = Path("data/interim/verification_scenario_c")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load single-lead sample (e.g. Q1.png or N10007.png)
    single_lead_path = Path("data/real_ecg_eval/samples/Q1.png")
    assert single_lead_path.is_file(), f"Single lead sample {single_lead_path} not found"
    img = cv2.imread(str(single_lead_path))

    pipeline = ECGDigitizerPipeline(extractor_mode="deep")
    res = pipeline.digitize(
        image_source=img,
        layout_hint="single_lead",
        extractor_mode="deep",
        is_calibration_reliable=False,
    )

    assert res.success, "Single lead digitization failed"
    # Strict verification: ONLY Lead II or detected single lead is reconstructed
    # Leads I, III, aVR, aVL, aVF, V1-V6 must NOT be fabricated!
    reconstructed_leads = list(res.reconstructed_ecg.leads.keys())
    logger.info("Reconstructed leads: %s", reconstructed_leads)

    assert "I" not in reconstructed_leads, "Lead I should NOT be fabricated for single-lead image"
    assert "V6" not in reconstructed_leads, "Lead V6 should NOT be fabricated for single-lead image"
    assert len(reconstructed_leads) == 1, f"Expected exactly 1 lead, got {len(reconstructed_leads)}"

    artifacts = res.save_artifacts(out_dir, base_name="scenario_c_single_lead_Q1")
    logger.info("Scenario C Verified: Only single available lead extracted without fabricating missing leads.")

    return {
        "scenario": "C_single_lead",
        "sample": "Q1.png",
        "reconstructed_leads": reconstructed_leads,
        "missing_leads_fabricated": False,
        "csv_path": str(artifacts["csv"]),
    }


def main():
    logger.info("Executing 3-scenario verification suite...")
    res_a = run_scenario_a()
    res_b = run_scenario_b()
    res_c = run_scenario_c()

    summary = {
        "scenario_a": res_a,
        "scenario_b": res_b,
        "scenario_c": res_c,
        "status": "all_passed",
    }

    out_file = Path("data/interim/three_scenarios_verification_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("All 3 scenarios verified and summary saved to: %s", out_file)
    print("\n=== SUMMARY OF 3-SCENARIO VERIFICATION ===")
    print(f"Scenario A (Paired Synthetic): Mean Pearson r = {res_a['mean_pearson_r']:.4f}, Mean RMSE = {res_a['mean_rmse_mv']:.4f} mV")
    print(f"Scenario B (Real ECG Image): Mean Confidence = {res_b['mean_confidence']*100:.2f}%, Calibration Reliable = {res_b['is_calibration_reliable']}")
    print(f"Scenario C (Single Lead): Reconstructed = {res_c['reconstructed_leads']}, Missing Leads Fabricated = {res_c['missing_leads_fabricated']}")


if __name__ == "__main__":
    main()
