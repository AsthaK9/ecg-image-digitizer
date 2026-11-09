"""Evaluation Metrics for Digitized ECG Reconstruction against Ground Truth.

Computes:
1. Pearson Correlation Coefficient (r)
2. Root Mean Squared Error (RMSE in mV)
3. Signal-to-Noise Ratio (SNR in dB)
4. Trace Completeness Ratio (%)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LeadMetrics:
    """Quantitative fidelity metrics for a single reconstructed ECG lead."""

    lead_name: str
    pearson_r: float
    rmse_mv: float
    snr_db: float
    completeness: float
    num_samples: int


@dataclass
class DigitizationEvaluationReport:
    """Aggregate evaluation report over all leads in an ECG record."""

    record_id: str
    per_lead: Dict[str, LeadMetrics]
    mean_pearson_r: float
    mean_rmse_mv: float
    mean_snr_db: float
    mean_completeness: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "mean_pearson_r": round(self.mean_pearson_r, 4),
            "mean_rmse_mv": round(self.mean_rmse_mv, 4),
            "mean_snr_db": round(self.mean_snr_db, 2),
            "mean_completeness": round(self.mean_completeness, 4),
            "per_lead": {
                k: {
                    "pearson_r": round(v.pearson_r, 4),
                    "rmse_mv": round(v.rmse_mv, 4),
                    "snr_db": round(v.snr_db, 2),
                    "completeness": round(v.completeness, 4),
                }
                for k, v in self.per_lead.items()
            },
        }


def compute_lead_metrics(
    ground_truth: np.ndarray,
    reconstructed: np.ndarray,
    lead_name: str,
    completeness: float = 1.0,
) -> LeadMetrics:
    """Computes Pearson r, RMSE, and SNR between paired ground truth and reconstructed lead."""
    gt = np.asarray(ground_truth, dtype=np.float64).flatten()
    rec = np.asarray(reconstructed, dtype=np.float64).flatten()

    # Align lengths to common minimum length
    min_len = min(len(gt), len(rec))
    if min_len < 2:
        return LeadMetrics(
            lead_name=lead_name,
            pearson_r=0.0,
            rmse_mv=float("nan"),
            snr_db=float("-inf"),
            completeness=0.0,
            num_samples=min_len,
        )

    gt = gt[:min_len]
    rec = rec[:min_len]

    # Pearson correlation r
    gt_std = float(np.std(gt))
    rec_std = float(np.std(rec))
    if gt_std < 1e-6 or rec_std < 1e-6:
        # Constant or near-flat signal
        r = 1.0 if abs(np.mean(gt) - np.mean(rec)) < 1e-3 else 0.0
    else:
        r = float(np.corrcoef(gt, rec)[0, 1])
        if np.isnan(r):
            r = 0.0

    # RMSE
    diff = gt - rec
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # SNR in dB: 10 * log10( P_signal / P_noise )
    p_signal = float(np.mean(gt ** 2))
    p_noise = float(np.mean(diff ** 2))
    eps = 1e-8
    snr = 10.0 * np.log10((p_signal + eps) / (p_noise + eps))

    return LeadMetrics(
        lead_name=lead_name,
        pearson_r=float(np.clip(r, -1.0, 1.0)),
        rmse_mv=rmse,
        snr_db=float(snr),
        completeness=float(completeness),
        num_samples=min_len,
    )


def evaluate_digitization_fidelity(
    ground_truth_leads: Dict[str, np.ndarray],
    reconstructed_leads: Dict[str, np.ndarray],
    record_id: str = "synthetic_sample",
) -> DigitizationEvaluationReport:
    """Evaluates full multi-lead digital reconstruction fidelity against exact ground truth."""
    per_lead: Dict[str, LeadMetrics] = {}

    common_leads = [k for k in ground_truth_leads.keys() if k in reconstructed_leads]
    if not common_leads:
        raise ValueError("No matching lead names between ground truth and reconstructed signals.")

    for lead_name in common_leads:
        gt = ground_truth_leads[lead_name]
        rec = reconstructed_leads[lead_name]
        metrics = compute_lead_metrics(gt, rec, lead_name=lead_name)
        per_lead[lead_name] = metrics

    mean_r = float(np.mean([m.pearson_r for m in per_lead.values()]))
    mean_rmse = float(np.mean([m.rmse_mv for m in per_lead.values()]))
    mean_snr = float(np.mean([m.snr_db for m in per_lead.values()]))
    mean_comp = float(np.mean([m.completeness for m in per_lead.values()]))

    return DigitizationEvaluationReport(
        record_id=record_id,
        per_lead=per_lead,
        mean_pearson_r=mean_r,
        mean_rmse_mv=mean_rmse,
        mean_snr_db=mean_snr,
        mean_completeness=mean_comp,
    )
