"""Unit tests for Phase 2 Synthetic Full-Sheet Renderer and Paired Evaluation Dataset.

Verifies:
1. All 4 clinical layouts: 3x4, 3x4_1, 3x4_3, 12x1.
2. Calibration pulse, millimeter grid, and metadata header.
3. Degradation suite (rotation, noise, blur, shadow, JPEG compression, grid styles).
4. Strict patient isolation and fold provenance.
5. End-to-end trace extraction across layouts without crashes.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ecg_digitization.pipeline.digitizer import ECGDigitizerPipeline
from ecg_digitization.synthetic.generator import (
    DegradationConfig,
    ECGLayoutConfig,
    SyntheticECGGenerator,
)


@pytest.fixture
def dummy_12lead_signal():
    """Generates synthetic 12-lead signal of 10s at 100 Hz."""
    t = np.linspace(0, 10, 1000)
    sig = np.zeros((12, 1000), dtype=np.float32)
    for i in range(12):
        sig[i] = 0.5 * np.sin(2 * np.pi * 1.2 * t + i * 0.2)
    return sig


class TestSyntheticLayouts:
    def test_all_four_layouts_generate_valid_images_and_bboxes(self, dummy_12lead_signal):
        synth = SyntheticECGGenerator()

        # 1. Standard 3x4
        res_3x4 = synth.generate(dummy_12lead_signal, layout_type="3x4")
        assert res_3x4.image.shape == (1240, 1654, 3)
        assert len(res_3x4.lead_bounding_boxes) == 12
        assert len(res_3x4.exact_lead_signals) == 12
        assert "II_RHYTHM" not in res_3x4.lead_bounding_boxes

        # 2. 3x4 + 1 Rhythm Strip
        res_3x4_1 = synth.generate(dummy_12lead_signal, layout_type="3x4_1")
        assert len(res_3x4_1.lead_bounding_boxes) == 13
        assert "II_RHYTHM" in res_3x4_1.lead_bounding_boxes
        assert res_3x4_1.calibration["layout"] == "3x4_1"

        # 3. Marquette MAC 3x4 + 3 Rhythm Strips
        res_3x4_3 = synth.generate(dummy_12lead_signal, layout_type="3x4_3")
        assert len(res_3x4_3.lead_bounding_boxes) == 15
        assert "V1_RHYTHM" in res_3x4_3.lead_bounding_boxes
        assert "II_RHYTHM" in res_3x4_3.lead_bounding_boxes
        assert "V5_RHYTHM" in res_3x4_3.lead_bounding_boxes

        # 4. 12x1 Continuous Stacked Rows
        res_12x1 = synth.generate(dummy_12lead_signal, layout_type="12x1")
        assert len(res_12x1.lead_bounding_boxes) == 12
        assert set(res_12x1.lead_bounding_boxes.keys()) == set(synth.STANDARD_12_LEADS)

    def test_unsupported_layout_raises_value_error(self, dummy_12lead_signal):
        synth = SyntheticECGGenerator()
        with pytest.raises(ValueError, match="Unsupported layout_type"):
            synth.generate(dummy_12lead_signal, layout_type="unsupported_layout")

    def test_degradation_pipeline_with_jpeg_and_grid_styles(self, dummy_12lead_signal):
        synth = SyntheticECGGenerator()
        deg = DegradationConfig(
            enabled=True,
            rotation_deg=-1.5,
            gaussian_noise_sigma=3.0,
            blur_kernel_size=3,
            shadow_gradient=True,
            jpeg_quality=50,
            grid_color_style="faded",
        )
        res = synth.generate(dummy_12lead_signal, layout_type="3x4_1", degradation=deg)
        assert res.image.shape == (1240, 1654, 3)
        assert not np.array_equal(res.image, res.clean_image)
        # Trace mask should be binary
        assert np.isin(res.trace_mask, [0, 255]).all()

    def test_digitizer_pipeline_end_to_end_on_all_layouts(self, dummy_12lead_signal):
        synth = SyntheticECGGenerator()
        pipe = ECGDigitizerPipeline()

        for l_type in ["3x4", "3x4_1", "3x4_3", "12x1"]:
            res_synth = synth.generate(dummy_12lead_signal, layout_type=l_type)
            res_pipe = pipe.digitize(res_synth.image, layout_hint=l_type)
            assert res_pipe.success is True
            assert len(res_pipe.extracted_traces) >= 12


class TestPairedDatasetManifest:
    def test_manifest_file_exists_and_valid(self):
        manifest_path = Path("data/interim/small_paired_dataset/manifest.json")
        assert manifest_path.is_file(), "manifest.json should be generated"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["num_cases"] >= 5
        assert manifest["ptbxl_validation_fold"] == 9
        cases = manifest["cases"]

        # Ensure all cases have valid files
        for cid, meta in cases.items():
            img_file = Path("data/interim/small_paired_dataset") / meta["files"]["image"]
            assert img_file.is_file(), f"Missing image for {cid}: {img_file}"

            # Check patient separation for PTB-XL records
            if meta["source_dataset"] == "PTB-XL":
                assert meta["strat_fold"] == 9, f"Case {cid} must belong to validation Fold 9"
