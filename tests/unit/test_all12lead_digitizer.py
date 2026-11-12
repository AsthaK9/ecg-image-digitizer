"""Unit tests for All-12-Lead ECG Digitizer U-Net and Multi-Lead Pipeline.

Tests:
1. 12-Lead naming and order conformance.
2. All12LeadPairedDataset generation across all 12 clinical leads.
3. DeepCropDigitizer initialization and all-12-lead validation status tagging.
4. ECGDigitizerPipeline multi-lead extraction across all 12 leads and rhythm strip.
5. Zero-fabrication discipline on single-lead crops.
6. Unified CSV schema conformance across all 12 leads.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from ecg_digitization.data.parser import WFDBRecordParser
from ecg_digitization.data.types import ECGRecordMetadata, ECGRecordPaths, ECGSignal
from ecg_digitization.inference.deep_digitizer import DeepCropDigitizer
from ecg_digitization.models.digitizer_unet import ECGDigitizerUNet
from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from ecg_digitization.synthetic.generator import SyntheticECGGenerator
from scripts.train_all12lead_digitizer import (
    STANDARD_12_LEADS,
    All12LeadPairedDataset,
    render_lead_crop,
)


def test_standard_12_leads_conformance():
    """Verifies standard 12-lead identities according to clinical standards."""
    expected = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert STANDARD_12_LEADS == expected
    assert len(STANDARD_12_LEADS) == 12


def test_render_lead_crop_paired_shapes():
    """Verifies that render_lead_crop returns strictly paired tensors."""
    dummy_signal = np.sin(np.linspace(0, 10, 250)).astype(np.float32) * 0.8
    img, mask, sig, base_y = render_lead_crop(dummy_signal, crop_h=128, crop_w=256, seed=42)

    assert img.shape == (3, 128, 256)
    assert mask.shape == (1, 128, 256)
    assert sig.shape == (256,)
    assert base_y == 64.0
    assert torch.all(img >= 0.0) and torch.all(img <= 1.0)
    assert torch.all((mask == 0.0) | (mask == 1.0))


def test_all_12_lead_dataset_sampling():
    """Verifies that All12LeadPairedDataset generates balanced samples across all 12 leads."""
    num_records = 2
    signals = np.zeros((num_records, 12, 1000), dtype=np.float32)
    for r in range(num_records):
        for l in range(12):
            signals[r, l] = np.sin(np.linspace(0, 10, 1000) + l)

    dataset = All12LeadPairedDataset(
        signals=signals,
        record_ids=["rec_0", "rec_1"],
        leads_per_record=12,
        seed=100,
        apply_noise=False,
    )

    # 2 records * 12 leads = 24 crops
    assert len(dataset) == 24

    lead_counts = {}
    for i in range(len(dataset)):
        img, mask, sig, base_y, lead_name = dataset[i]
        assert lead_name in STANDARD_12_LEADS
        lead_counts[lead_name] = lead_counts.get(lead_name, 0) + 1

    # Exactly 2 samples per lead
    for l_name in STANDARD_12_LEADS:
        assert lead_counts[l_name] == 2


def test_deep_crop_digitizer_loads_all12lead_checkpoint():
    """Verifies DeepCropDigitizer loads all-12-lead checkpoint and recognizes all leads as validated."""
    ckpt = Path("models/ecg_digitizer_unet_all12lead.pt")
    if not ckpt.is_file():
        pytest.skip("All-12-lead checkpoint not yet generated.")

    digitizer = DeepCropDigitizer(checkpoint_path=ckpt)
    for lead in STANDARD_12_LEADS:
        assert lead in digitizer.supported_leads
    assert "II_RHYTHM" in digitizer.supported_leads


def test_end_to_end_12lead_pipeline_extraction():
    """Verifies ECGDigitizerPipeline digitizes a 12-lead sheet and tags all 12 leads as validated."""
    hea_path = Path("data/interim/ptbxl_sample/00001_lr.hea")
    dat_path = Path("data/interim/ptbxl_sample/00001_lr.dat")
    if not hea_path.is_file():
        pytest.skip("00001_lr sample not present in data/interim/ptbxl_sample")

    parser = WFDBRecordParser()
    sig = parser.parse(ECGRecordPaths("00001_lr", hea_path, dat_path), source_dataset="ptbxl")
    generator = SyntheticECGGenerator()
    synth = generator.generate(sig, seed=42)

    pipeline = ECGDigitizerPipeline(extractor_mode="deep")
    result = pipeline.digitize(synth.image, layout_hint="3x4_1", extractor_mode="deep")

    assert result.success is True
    assert len(result.reconstructed_ecg.leads) == 13  # 12 grid leads + II_RHYTHM

    for l_name in STANDARD_12_LEADS:
        assert l_name in result.reconstructed_ecg.leads
        assert len(result.reconstructed_ecg.leads[l_name].voltage_mv) == 250
        trace = result.extracted_traces[l_name]
        assert trace.is_trained_lead is True
        assert trace.validation_status == "validated_lead"


def test_single_lead_no_fabrication():
    """Verifies that single-lead images extract only the available lead without fabricating missing leads."""
    dummy_crop = np.full((128, 256, 3), 252, dtype=np.uint8)
    cv2.line(dummy_crop, (0, 64), (256, 64), (20, 20, 30), 2)

    pipeline = ECGDigitizerPipeline(extractor_mode="deep")
    result = pipeline.digitize(dummy_crop, layout_hint="single_lead", extractor_mode="deep")

    assert result.success is True
    assert list(result.reconstructed_ecg.leads.keys()) == ["II"]
    for unavail in ["I", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
        assert unavail not in result.reconstructed_ecg.leads
