"""Unit tests and CPU smoke tests for classifier, rhythm analysis, and inference safety."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from ecg_digitization.inference.analyzer import ECGRhythmAnalyzer
from ecg_digitization.inference.classifier_interface import ECGClassifierInterface
from ecg_digitization.models.classifier_1d import ECGResNet1D


def test_rhythm_analyzer_normal_and_bradycardia() -> None:
    analyzer = ECGRhythmAnalyzer(target_fs_hz=100.0)

    # 1. Bradycardia signal: 50 bpm -> 1 beat every 1.2s -> peaks at 0, 120, 240, 360, 480 samples
    t = np.linspace(0, 5, 500)
    sig = np.zeros(500, dtype=np.float32)
    # Put sharp R peaks
    for idx in [50, 170, 290, 410]:
        sig[idx] = 1.5

    res = analyzer.analyze_signal(sig, lead_name="II")
    assert res.num_r_peaks == 4
    assert res.heart_rate_bpm is not None
    assert res.heart_rate_bpm < 60.0
    assert any("Bradycardia" in f for f in res.rule_findings)


def test_classifier_untrained_safety() -> None:
    # Interface with NO checkpoint
    interface = ECGClassifierInterface(checkpoint_path=None)
    dummy_signal = np.zeros((12, 1000), dtype=np.float32)

    report = interface.predict(dummy_signal)
    assert report.model_status == "model_not_trained"

    # Supported classes must be model_not_trained with probability None
    assert report.findings["AFIB"].status == "model_not_trained"
    assert report.findings["AFIB"].probability is None

    # STE_ must be 'Not supported by available labels'
    assert report.findings["STE_"].status == "Not supported by available labels"
    assert report.findings["STE_"].probability is None


def test_classifier_cpu_smoke_test_and_checkpoint_roundtrip() -> None:
    # 1. Forward pass
    batch_size = 2
    in_channels = 12
    seq_len = 500
    x = torch.randn(batch_size, in_channels, seq_len)
    targets = torch.randint(0, 2, (batch_size, 7)).float()

    model = ECGResNet1D(in_channels=12, num_classes=7)
    logits = model(x)
    assert logits.shape == (batch_size, 7)

    # 2. Finite loss check
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, targets)
    assert torch.isfinite(loss).item() is True
    loss.backward()

    # 3. Checkpoint creation
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "smoke_checkpoint.pt"
        torch.save({"state_dict": model.state_dict()}, ckpt_path)
        assert ckpt_path.is_file()

        # 4. Checkpoint loading via ECGClassifierInterface
        interface = ECGClassifierInterface(checkpoint_path=ckpt_path)
        assert interface.is_trained is True
        assert interface.model is not None

        # Predict with loaded model
        eval_sig = np.zeros((12, 500), dtype=np.float32)
        eval_report = interface.predict(eval_sig)
        assert eval_report.model_status == "ready"
        assert eval_report.findings["AFIB"].status == "evaluated"
        assert isinstance(eval_report.findings["AFIB"].probability, float)
        report_ste = eval_report.findings["STE_"]
        assert report_ste.status == "Not supported by available labels"
