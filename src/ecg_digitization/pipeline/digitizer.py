"""End-to-End ECG Image Digitisation Pipeline.

Integrates:
1. Preprocessing (deskew, grid suppression, contrast enhancement, binarization)
2. Lead layout detection (3x4 + 1 rhythm strip or 12x1)
3. Waveform extraction (column-wise centroid/median tracking)
4. Signal calibration and reconstruction (pixel-to-mV/time mapping, resampling)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ecg_digitization.preprocessing.enhancement import (
    PreprocessingPipeline,
    PreprocessingResult,
)
from ecg_digitization.preprocessing.lead_detector import (
    LeadDetectionResult,
    LeadLayoutDetector,
)
from ecg_digitization.reconstruction.calibrator import (
    CalibrationConfig,
    ReconstructedECG,
    SignalReconstructor,
)
from ecg_digitization.reconstruction.extractor import (
    ExtractedTrace,
    WaveformExtractor,
)

logger = logging.getLogger(__name__)


@dataclass
class DigitizationPipelineResult:
    """Full results from end-to-end digitisation of an ECG image."""

    preprocessed: PreprocessingResult
    layout: LeadDetectionResult
    extracted_traces: Dict[str, ExtractedTrace]
    reconstructed_ecg: ReconstructedECG
    success: bool
    warnings: List[str]

    def save_artifacts(self, output_dir: Union[str, Path], base_name: str = "digitized_ecg") -> Dict[str, Path]:
        """Saves intermediate outputs, reconstructed CSV, and diagnostic plots."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        saved = {}

        # 1. Save Preprocessing Stages
        prep_dir = out / "preprocessing"
        prep_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(prep_dir / f"{base_name}_01_deskewed.png"), self.preprocessed.deskewed)
        cv2.imwrite(str(prep_dir / f"{base_name}_02_grid_suppressed.png"), self.preprocessed.grid_suppressed)
        cv2.imwrite(str(prep_dir / f"{base_name}_03_binary_trace.png"), self.preprocessed.binary_trace)
        saved["binary_trace"] = prep_dir / f"{base_name}_03_binary_trace.png"

        # 2. Save Reconstructed CSV and Plot
        csv_path = out / f"{base_name}_signal.csv"
        self.reconstructed_ecg.save_csv(csv_path)
        saved["csv"] = csv_path

        plot_path = out / f"{base_name}_plot.png"
        self.reconstructed_ecg.save_plot(plot_path, title=f"ECG DigiAI Reconstructed — {base_name}")
        saved["plot"] = plot_path

        # 3. Save Lead Bounding Boxes Visualization
        bbox_img = self.preprocessed.deskewed.copy()
        for name, region in self.layout.leads.items():
            color = (0, 0, 255) if region.is_rhythm else (0, 200, 0)
            cv2.rectangle(bbox_img, (region.x1, region.y1), (region.x2, region.y2), color, 2)
            cv2.putText(
                bbox_img,
                name,
                (region.x1 + 5, region.y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        bbox_path = out / f"{base_name}_lead_regions.png"
        cv2.imwrite(str(bbox_path), bbox_img)
        saved["lead_regions"] = bbox_path

        return saved


class ECGDigitizerPipeline:
    """Orchestrates end-to-end ECG paper image digitisation."""

    def __init__(
        self,
        calibration: Optional[CalibrationConfig] = None,
        enable_deskew: bool = True,
        extraction_method: str = "median",
    ) -> None:
        self.calibration = calibration or CalibrationConfig()
        self.preprocessor = PreprocessingPipeline(enable_deskew=enable_deskew)
        self.lead_detector = LeadLayoutDetector()
        self.extractor = WaveformExtractor(method=extraction_method)
        self.reconstructor = SignalReconstructor(calibration=self.calibration)

    def digitize(
        self,
        image_source: Union[str, Path, np.ndarray],
        layout_hint: str = "3x4_1",
    ) -> DigitizationPipelineResult:
        """Executes full digitisation on an input ECG image."""
        warnings: List[str] = []

        # 1. Preprocess image
        prep = self.preprocessor.process(image_source)
        if abs(prep.skew_angle_deg) > 0.5:
            warnings.append(f"Detected skew angle of {prep.skew_angle_deg:.1f} degrees; image deskewed.")

        # 2. Detect lead layout
        layout = self.lead_detector.detect_layout(prep.binary_trace.shape, layout_hint=layout_hint)
        warnings.extend(layout.warnings)

        if not layout.is_supported:
            dummy_ecg = self.reconstructor.reconstruct_ecg({}, layout_name=layout_hint)
            return DigitizationPipelineResult(
                preprocessed=prep,
                layout=layout,
                extracted_traces={},
                reconstructed_ecg=dummy_ecg,
                success=False,
                warnings=warnings,
            )

        # 3. Extract waveform per lead
        extracted_traces: Dict[str, ExtractedTrace] = {}
        for lead_name, region in layout.leads.items():
            # Crop lead region from binary trace
            lead_crop = prep.binary_trace[region.y1 : region.y2, region.x1 : region.x2]
            if lead_crop.size == 0:
                warnings.append(f"Empty crop for lead '{lead_name}'.")
                continue

            # In standard layout, baseline is approximately centered vertically in each box
            expected_baseline_y = lead_crop.shape[0] / 2.0
            trace = self.extractor.extract_from_crop(
                lead_mask=lead_crop,
                lead_name=lead_name,
                baseline_y=expected_baseline_y,
            )
            extracted_traces[lead_name] = trace

        # 4. Calibrate and reconstruct digital ECG
        reconstructed = self.reconstructor.reconstruct_ecg(
            extracted_traces=extracted_traces,
            layout_name=layout_hint,
        )

        return DigitizationPipelineResult(
            preprocessed=prep,
            layout=layout,
            extracted_traces=extracted_traces,
            reconstructed_ecg=reconstructed,
            success=True,
            warnings=warnings,
        )
