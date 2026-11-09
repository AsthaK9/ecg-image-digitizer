"""Explainable Clinical Decision-Support Report Generator.

Produces:
1. Structured machine-readable JSON reports.
2. Formatted, printable clinical HTML reports.

Adheres strictly to academic guidelines:
- Never fabricates probabilities when model is untrained.
- Explicitly flags unsupported labels ('Not supported by available labels').
- Sets Risk Level to 'Not assessed'.
- Mandates institutional provenance and decision-support disclaimers.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ecg_digitization.inference.analyzer import RhythmAnalysisResult
from ecg_digitization.inference.classifier_interface import ClassificationReport

logger = logging.getLogger(__name__)


class ClinicalReportGenerator:
    """Generates structured JSON and printable HTML diagnostic reports."""

    INSTITUTION = "Jaypee Institute of Information Technology, Noida"
    DEPARTMENT = "Electronics & Communication Engineering"
    PROJECT = "ECG Digitisation & Analysis Using AI (ECG DigiAI)"

    DISCLAIMER = (
        "DECISION-SUPPORT RESEARCH NOTICE: ECG DigiAI is an academic research and clinical decision-support "
        "system developed at JIIT Noida. This report does NOT constitute an autonomous clinical diagnosis. "
        "All findings, rate estimations, and waveforms must be verified by a licensed physician before any "
        "diagnostic or therapeutic decision."
    )

    RISK_ASSESSMENT = "Not assessed (Automated risk stratification requires prospective clinical validation)"

    def generate_json_report(
        self,
        record_id: str,
        rhythm_result: RhythmAnalysisResult,
        classification_report: ClassificationReport,
        lead_completeness: Optional[Dict[str, float]] = None,
        plot_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Generates a complete structured JSON report."""
        report = {
            "metadata": {
                "report_id": f"REP-{record_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "record_id": record_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "institution": self.INSTITUTION,
                "department": self.DEPARTMENT,
                "project": self.PROJECT,
            },
            "physiological_measurements": {
                "heart_rate_bpm": rhythm_result.heart_rate_bpm,
                "num_r_peaks_detected": rhythm_result.num_r_peaks,
                "mean_rr_interval_s": rhythm_result.mean_rr_interval_s,
                "std_rr_interval_s": rhythm_result.std_rr_interval_s,
                "is_regular_rhythm": rhythm_result.is_regular_rhythm,
                "signal_quality": rhythm_result.signal_quality,
                "quality_warnings": rhythm_result.quality_warnings,
            },
            "findings": {
                "rhythm_and_rate": rhythm_result.rule_findings,
                "model_status": classification_report.model_status,
                "model_name": classification_report.model_name,
                "checkpoint": classification_report.checkpoint_path,
                "pathologies": {
                    code: {
                        "name": item.target_name,
                        "status": item.status,
                        "probability": item.probability,
                        "prediction": item.prediction,
                        "note": item.note,
                    }
                    for code, item in classification_report.findings.items()
                },
            },
            "clinical_risk_level": self.RISK_ASSESSMENT,
            "waveform_plot": str(plot_path) if plot_path else None,
            "lead_completeness": lead_completeness or {},
            "disclaimer": self.DISCLAIMER,
        }
        return report

    def generate_html_report(
        self,
        record_id: str,
        rhythm_result: RhythmAnalysisResult,
        classification_report: ClassificationReport,
        lead_completeness: Optional[Dict[str, float]] = None,
        plot_image_bytes_or_path: Optional[Union[str, Path, bytes]] = None,
    ) -> str:
        """Generates a styled, printable HTML clinical decision-support report."""
        # Convert image to base64 if available
        img_tag = ""
        if plot_image_bytes_or_path:
            try:
                if isinstance(plot_image_bytes_or_path, (str, Path)):
                    p = Path(plot_image_bytes_or_path)
                    if p.is_file():
                        with open(p, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        img_tag = f'<img src="data:image/png;base64,{b64}" alt="Reconstructed ECG" style="max-width:100%; height:auto; border:1px solid #ccc; border-radius:4px; margin-top:10px;" />'
                elif isinstance(plot_image_bytes_or_path, bytes):
                    b64 = base64.b64encode(plot_image_bytes_or_path).decode("utf-8")
                    img_tag = f'<img src="data:image/png;base64,{b64}" alt="Reconstructed ECG" style="max-width:100%; height:auto; border:1px solid #ccc; border-radius:4px; margin-top:10px;" />'
            except Exception as exc:
                logger.warning("Could not embed image in HTML report: %s", exc)

        hr_str = f"{rhythm_result.heart_rate_bpm:.1f} BPM" if rhythm_result.heart_rate_bpm else "Indeterminate"
        rr_str = f"{rhythm_result.mean_rr_interval_s:.3f} s" if rhythm_result.mean_rr_interval_s else "N/A"
        quality_badge = {
            "good": '<span style="background:#e6f4ea; color:#137333; padding:3px 8px; border-radius:4px; font-weight:bold;">GOOD</span>',
            "acceptable": '<span style="background:#fef7e0; color:#b06000; padding:3px 8px; border-radius:4px; font-weight:bold;">ACCEPTABLE</span>',
            "low": '<span style="background:#fce8e6; color:#c5221f; padding:3px 8px; border-radius:4px; font-weight:bold;">LOW QUALITY</span>',
        }.get(rhythm_result.signal_quality, rhythm_result.signal_quality)

        # Build pathology rows
        pathology_rows = []
        for code, f in classification_report.findings.items():
            if f.status == "model_not_trained":
                status_html = '<span style="color:#e37400; font-weight:bold;">model_not_trained</span>'
                prob_html = '<span style="color:#777;">— (untrained)</span>'
            elif f.status == "Not supported by available labels":
                status_html = '<span style="color:#888; font-style:italic;">Not supported by available labels</span>'
                prob_html = '<span style="color:#777;">—</span>'
            else:
                prob_val = f.probability if f.probability is not None else 0.0
                status_html = f'<span style="color:{"#d93025" if f.prediction else "#188038"}; font-weight:bold;">{"DETECTED" if f.prediction else "NEGATIVE"}</span>'
                prob_html = f"<strong>{prob_val * 100:.1f}%</strong>"

            pathology_rows.append(f"""
                <tr>
                    <td style="padding:8px; border-bottom:1px solid #eee;"><strong>{f.target_name}</strong> ({code})</td>
                    <td style="padding:8px; border-bottom:1px solid #eee;">{status_html}</td>
                    <td style="padding:8px; border-bottom:1px solid #eee;">{prob_html}</td>
                    <td style="padding:8px; border-bottom:1px solid #eee; color:#555; font-size:12px;">{f.note}</td>
                </tr>
            """)
        pathology_table = "\n".join(pathology_rows)

        rule_findings_html = "".join(f"<li>{r}</li>" for r in rhythm_result.rule_findings)
        warnings_html = "".join(f"<li style='color:#c5221f;'>{w}</li>" for w in rhythm_result.quality_warnings) if rhythm_result.quality_warnings else "<li>None reported.</li>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ECG DigiAI Report — {record_id}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #202124; background: #fff; }}
    .header {{ border-bottom: 2px solid #1a73e8; padding-bottom: 12px; margin-bottom: 20px; }}
    .title {{ font-size: 22px; font-weight: bold; color: #1a73e8; margin: 0; }}
    .subtitle {{ font-size: 13px; color: #5f6368; margin-top: 4px; }}
    .meta-box {{ display: flex; justify-content: space-between; background: #f8f9fa; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; font-size: 13px; }}
    .section-title {{ font-size: 15px; font-weight: bold; color: #202124; border-bottom: 1px solid #dadce0; padding-bottom: 6px; margin-top: 24px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 15px; }}
    .card {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
    th {{ background: #f1f3f4; text-align: left; padding: 8px; font-weight: 600; color: #3c4043; border-bottom: 1px solid #dadce0; }}
    .disclaimer {{ background: #fff3e0; border-left: 4px solid #f57c00; padding: 12px; font-size: 12px; color: #bf360c; margin-top: 30px; border-radius: 2px; line-height: 1.5; }}
    @media print {{ body {{ margin: 15px; }} .no-print {{ display: none; }} }}
</style>
</head>
<body>

<div class="header">
    <div class="title">ECG DigiAI — Clinical Decision-Support Report</div>
    <div class="subtitle">{self.INSTITUTION} | {self.DEPARTMENT} | {self.PROJECT}</div>
</div>

<div class="meta-box">
    <div><strong>Record ID:</strong> {record_id}</div>
    <div><strong>Generated (UTC):</strong> {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}</div>
    <div><strong>Clinical Risk Assessment:</strong> <span style="color:#d93025; font-weight:bold;">Not assessed</span></div>
</div>

<div class="section-title">1. Physiological Rhythm & Rate Assessment</div>
<div class="grid-2">
    <div class="card">
        <div style="font-size:12px; color:#5f6368;">Heart Rate</div>
        <div style="font-size:24px; font-weight:bold; color:#1a73e8; margin-top:4px;">{hr_str}</div>
        <div style="margin-top:8px; font-size:12px;">Mean RR Interval: <strong>{rr_str}</strong></div>
        <div style="margin-top:4px; font-size:12px;">R-Peaks Detected: <strong>{rhythm_result.num_r_peaks}</strong></div>
        <div style="margin-top:4px; font-size:12px;">Rhythm Regularity: <strong>{"Regular" if rhythm_result.is_regular_rhythm else "Irregular"}</strong></div>
    </div>
    <div class="card">
        <div style="font-size:12px; color:#5f6368;">Signal Quality & Observations</div>
        <div style="margin-top:6px;">Quality Index: {quality_badge}</div>
        <div style="margin-top:8px; font-size:12px;"><strong>Findings:</strong></div>
        <ul style="margin:4px 0 0 18px; padding:0; font-size:12px;">
            {rule_findings_html}
        </ul>
        <div style="margin-top:6px; font-size:12px;"><strong>Quality Notices:</strong></div>
        <ul style="margin:4px 0 0 18px; padding:0; font-size:12px;">
            {warnings_html}
        </ul>
    </div>
</div>

<div class="section-title">2. AI Pathology Target Screening</div>
<div style="font-size:12px; color:#5f6368; margin-bottom:8px;">
    Architecture: <strong>{classification_report.model_name}</strong> | Status: <strong>{classification_report.model_status}</strong>
</div>
<table>
    <thead>
        <tr>
            <th>Pathology Target</th>
            <th>Model Status</th>
            <th>Confidence / Output</th>
            <th>Clinical & Dataset Context</th>
        </tr>
    </thead>
    <tbody>
        {pathology_table}
    </tbody>
</table>

<div class="section-title">3. Digitized 12-Lead Waveform Visualization</div>
{img_tag or '<div style="font-size:13px; color:#777; font-style:italic;">No waveform image provided.</div>'}

<div class="disclaimer">
    <strong>MANDATORY ACADEMIC & CLINICAL NOTICE:</strong><br>
    {self.DISCLAIMER}
</div>

</body>
</html>
"""
        return html
