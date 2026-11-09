"""Model Registry and Inference Interface for ECG Classification.

Enforces clinical decision-support safety:
- When a trained model checkpoint is not provided, returns explicit 'model_not_trained'.
- When a target pathology has insufficient labels in the dataset (e.g. STE_),
  returns 'Not supported by available labels'.
- Never invents probabilities or synthetic diagnostic scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ecg_digitization.models.classifier_1d import ECGResNet1D

logger = logging.getLogger(__name__)


@dataclass
class TargetFinding:
    """Clinical finding for a specific target pathology."""

    target_code: str
    target_name: str
    status: str                       # "model_not_trained", "evaluated", "Not supported by available labels"
    probability: Optional[float]      # Real calibrated probability only if model is trained
    prediction: Optional[bool]
    note: str


@dataclass
class ClassificationReport:
    """Consolidated classification and model status report."""

    model_name: str
    model_status: str                 # "model_not_trained" or "ready"
    checkpoint_path: Optional[str]
    findings: Dict[str, TargetFinding]
    summary_text: str
    disclaimer: str


class ECGClassifierInterface:
    """Inference interface managing model loading, forward passes, and clinical safety."""

    TARGET_METADATA = {
        "AFIB": {"name": "Atrial Fibrillation", "supported": True},
        "SBRAD": {"name": "Sinus Bradycardia", "supported": True},
        "STACH": {"name": "Sinus Tachycardia", "supported": True},
        "STE_": {"name": "ST Elevation", "supported": False, "reason": "Not supported by available labels (sparse direct labels in PTB-XL)"},
        "STD_": {"name": "ST Depression", "supported": True},
        "BBB": {"name": "Bundle Branch Block", "supported": True},
        "MI": {"name": "Myocardial Infarction", "supported": True},
    }

    DISCLAIMER = (
        "DECISION-SUPPORT NOTICE: This system provides research decision-support only. "
        "Outputs do NOT constitute independent clinical diagnoses and must be reviewed "
        "by a qualified medical practitioner."
    )

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.model: Optional[ECGResNet1D] = None
        self.is_trained = False

        if self.checkpoint_path and Path(self.checkpoint_path).is_file():
            self._load_checkpoint(self.checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        try:
            model = ECGResNet1D(in_channels=12, num_classes=len(ECGResNet1D.TARGET_CLASSES))
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            # Remove prefix if present
            clean_state = {k.replace("model.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state)
            model.to(self.device)
            model.eval()
            self.model = model
            self.is_trained = True
            logger.info("Successfully loaded ECG model checkpoint from: %s", path)
        except Exception as exc:
            logger.warning("Failed to load checkpoint from %s: %s", path, exc)
            self.model = None
            self.is_trained = False

    def predict(self, ecg_signal_12lead: np.ndarray) -> ClassificationReport:
        """Evaluates 12-lead ECG signal across targets or returns honest model_not_trained."""
        sig = np.asarray(ecg_signal_12lead, dtype=np.float32)

        # Validate shape
        if sig.ndim != 2 or sig.shape[0] != 12:
            raise ValueError(f"Expected 12-lead signal of shape (12, N), got shape {sig.shape}")

        findings: Dict[str, TargetFinding] = {}

        if not self.is_trained or self.model is None:
            # Model is not trained
            for code, meta in self.TARGET_METADATA.items():
                if not meta.get("supported", True):
                    findings[code] = TargetFinding(
                        target_code=code,
                        target_name=meta["name"],
                        status="Not supported by available labels",
                        probability=None,
                        prediction=None,
                        note=meta.get("reason", "Not supported by available labels"),
                    )
                else:
                    findings[code] = TargetFinding(
                        target_code=code,
                        target_name=meta["name"],
                        status="model_not_trained",
                        probability=None,
                        prediction=None,
                        note="No trained neural network checkpoint loaded; training scheduled on GPU.",
                    )

            return ClassificationReport(
                model_name="ECGResNet1D-Baseline",
                model_status="model_not_trained",
                checkpoint_path=self.checkpoint_path,
                findings=findings,
                summary_text="Deep-learning classifier checkpoint absent (model_not_trained). Rule-based rhythm findings remain active.",
                disclaimer=self.DISCLAIMER,
            )

        # If a real trained model exists:
        with torch.no_grad():
            tensor = torch.from_numpy(sig).unsqueeze(0).to(self.device)  # [1, 12, N]
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        for idx, code in enumerate(ECGResNet1D.TARGET_CLASSES):
            meta = self.TARGET_METADATA.get(code, {"name": code, "supported": True})
            if not meta.get("supported", True):
                findings[code] = TargetFinding(
                    target_code=code,
                    target_name=meta["name"],
                    status="Not supported by available labels",
                    probability=None,
                    prediction=None,
                    note=meta.get("reason", "Not supported by available labels"),
                )
            else:
                prob = float(probs[idx])
                pred = prob >= 0.50
                findings[code] = TargetFinding(
                    target_code=code,
                    target_name=meta["name"],
                    status="evaluated",
                    probability=round(prob, 4),
                    prediction=pred,
                    note=f"Model output probability: {prob*100:.1f}%",
                )

        return ClassificationReport(
            model_name="ECGResNet1D-Baseline",
            model_status="ready",
            checkpoint_path=self.checkpoint_path,
            findings=findings,
            summary_text="Model evaluation complete across supported target pathologies.",
            disclaimer=self.DISCLAIMER,
        )
