"""Verification script for Session 1 End-to-End Pipeline:
PTB-XL signal -> Synthetic ECG image -> Preprocessing -> Lead Detection
-> Waveform Extraction -> Reconstructed Digital Signal -> Ground Truth Comparison
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import ECGRecordPaths
from ecg_digitization.synthetic.generator import SyntheticECGGenerator, DegradationConfig
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from ecg_digitization.evaluation.metrics import evaluate_digitization_fidelity


def main() -> None:
    # 1. Load sample PTB-XL record
    parser = WFDBRecordParser()
    paths = ECGRecordPaths(
        record_id="00001_lr",
        header_path=Path("data/interim/ptbxl_sample/00001_lr.hea"),
        signal_path=Path("data/interim/ptbxl_sample/00001_lr.dat"),
    )
    sig = parser.parse(paths, source_dataset="ptbxl")
    print(f"1. Loaded PTB-XL record {sig.metadata.record_id}: shape {sig.values.shape}, fs={sig.metadata.sampling_rate_hz} Hz")

    # 2. Generate Synthetic ECG Image with photographic/scan degradation
    gen = SyntheticECGGenerator()
    deg = DegradationConfig(enabled=True, rotation_deg=0.8, gaussian_noise_sigma=2.0)
    synth_res = gen.generate(sig, degradation=deg, seed=42)
    print(f"2. Rendered synthetic 12-lead ECG sheet: {synth_res.image.shape}, masks: {synth_res.trace_mask.shape}")

    # 3. Execute End-to-End Digitisation Pipeline
    pipeline = ECGDigitizerPipeline()
    digit_res = pipeline.digitize(synth_res.image, layout_hint="3x4_1")
    print(f"3. Digitisation completed: success={digit_res.success}, warnings={digit_res.warnings}")

    # 4. Save visual and numerical artifacts
    out_dir = Path("data/interim/session1_demo_output")
    saved = digit_res.save_artifacts(out_dir, base_name="demo_00001")
    print("4. Saved artifacts:")
    for k, v in saved.items():
        print(f"   - {k}: {v}")

    # 5. Quantitative Evaluation against ground truth
    gt_leads = {}
    rec_leads = {}
    lead_idx_map = {name.upper(): i for i, name in enumerate(sig.metadata.lead_names)}

    for col_idx, col_leads in enumerate(gen.LAYOUT_3X4_COLUMNS):
        t_start = int(col_idx * 2.5 * 100)
        t_end = int((col_idx + 1) * 2.5 * 100)
        for lead_name in col_leads:
            if lead_name in lead_idx_map and lead_name in digit_res.reconstructed_ecg.leads:
                gt_seg = sig.values[lead_idx_map[lead_name], t_start:t_end]
                rec_seg = digit_res.reconstructed_ecg.leads[lead_name].voltage_mv
                gt_leads[lead_name] = gt_seg
                rec_leads[lead_name] = rec_seg

    # Add rhythm strip
    if "II" in lead_idx_map and "II_RHYTHM" in digit_res.reconstructed_ecg.leads:
        gt_leads["II_RHYTHM"] = sig.values[lead_idx_map["II"], :1000]
        rec_leads["II_RHYTHM"] = digit_res.reconstructed_ecg.leads["II_RHYTHM"].voltage_mv

    report = evaluate_digitization_fidelity(gt_leads, rec_leads, record_id="demo_00001")
    print("\n" + "=" * 55)
    print("      QUANTITATIVE GROUND-TRUTH FIDELITY REPORT      ")
    print("=" * 55)
    print(f"  Mean Pearson Correlation (r) : {report.mean_pearson_r:.4f}")
    print(f"  Mean Root Mean Squared Error  : {report.mean_rmse_mv:.4f} mV")
    print(f"  Mean Signal-to-Noise Ratio    : {report.mean_snr_db:.2f} dB")
    print(f"  Mean Trace Completeness       : {report.mean_completeness * 100:.1f} %")
    print("-" * 55)
    for lead_name, m in report.per_lead.items():
        print(f"  Lead {lead_name:11s} | r: {m.pearson_r:+.4f} | RMSE: {m.rmse_mv:.4f} mV | SNR: {m.snr_db:5.1f} dB")
    print("=" * 55)

    # Save metrics JSON
    with open(out_dir / "reconstruction_metrics.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print("Metrics written to data/interim/session1_demo_output/reconstruction_metrics.json")


if __name__ == "__main__":
    main()
