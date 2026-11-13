"""End-to-End ECG Image Digitisation Pipeline.

Integrates:
1. Preprocessing (deskew, grid suppression, contrast enhancement, binarization)
2. Lead layout detection (3x4 + 1 rhythm strip or 12x1)
3. Deep U-Net or classical waveform extraction with 1:1 physical baseline scaling
4. Routing and provenance tagging: Formally Validated (II, V1) vs Experimental Leads
5. Synchronized multi-lead reconstruction, calibration handling, and unified CSV export
6. Diagnostic full-sheet visual overlay generation with lead bounding boxes and waveforms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    extractor_mode_used: str = "deep"
    predicted_layout: str = "3x4_1"
    active_layout: str = "3x4_1"
    layout_confidence: float = 1.0
    is_layout_overridden: bool = False
    is_layout_ambiguous: bool = False
    layout_probabilities: Dict[str, float] = field(default_factory=dict)

    def generate_sheet_overlay(
        self,
        output_path: Union[str, Path],
        title: Optional[str] = None,
    ) -> Path:
        """Generates a diagnostic full-sheet overlay displaying detected lead boundaries
        and reconstructed waveforms directly aligned on the preprocessed image canvas.

        - Formally Validated Leads (II, V1, II_RHYTHM) are bounded in Green / Cyan.
        - Experimental Leads (I, III, aVR, aVL, aVF, V2-V6) are bounded in Orange.
        - Reconstructed waveform traces are drawn in high-contrast Crimson Red.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        overlay_img = self.preprocessed.deskewed.copy()
        if len(overlay_img.shape) == 2:
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)

        h, w = overlay_img.shape[:2]

        # Draw lead bounding boxes and reconstructed waveform traces
        is_single = (self.layout.layout_name == "single_lead") or (len(self.layout.leads) == 1)
        is_cal = getattr(self.reconstructed_ecg, "is_calibration_reliable", False)

        for name, region in self.layout.leads.items():
            if name in self.extracted_traces:
                is_trained = self.extracted_traces[name].is_trained_lead
            else:
                is_trained = name in ("II", "V1", "II_RHYTHM")

            trace = self.extracted_traces.get(name)
            is_qual_ok = trace.is_quality_passed if trace else True

            if not is_qual_ok:
                box_color = (0, 70, 240)        # Amber / Red warning for spikes or clipping
                tag = "[Warning: Spikes]"
            elif is_single:
                box_color = (200, 160, 0)       # Cyan for single lead
                tag = "[Calibrated]" if is_cal else "[Uncalibrated]"
            elif region.is_rhythm:
                box_color = (220, 150, 0)       # Cyan for rhythm
                tag = "[Rhythm - Calibrated]" if is_cal else "[Rhythm]"
            elif is_cal:
                box_color = (0, 180, 0)         # Green for calibrated leads
                tag = "[Calibrated]"
            else:
                box_color = (200, 160, 0)       # Cyan for uncalibrated
                tag = "[Uncalibrated]"

            # 1. Bounding box
            cv2.rectangle(overlay_img, (region.x1, region.y1), (region.x2, region.y2), box_color, 2)
            cv2.putText(
                overlay_img,
                f"{name} {tag}",
                (region.x1 + 6, region.y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                box_color,
                1,
                cv2.LINE_AA,
            )

            # 2. Reconstructed trace overlay
            if name in self.extracted_traces:
                trace = self.extracted_traces[name]
                num_pts = len(trace.pixel_trajectory)
                if num_pts > 0:
                    x_start = region.x1 + trace.active_start_col
                    x_end = region.x1 + trace.active_end_col
                    x_pts = np.linspace(x_start, x_end, num_pts)
                    y_pts = region.y1 + trace.pixel_trajectory
                    pts = np.column_stack([x_pts, y_pts]).astype(np.int32)
                    cv2.polylines(
                        overlay_img,
                        [pts],
                        isClosed=False,
                        color=(30, 30, 220),  # Crimson Red
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

        # 3. Top provenance and diagnostic banner
        banner_h = 32
        banner = np.full((banner_h, w, 3), 245, dtype=np.uint8)
        cv2.line(banner, (0, banner_h - 1), (w, banner_h - 1), (200, 200, 200), 1)

        if is_single:
            lead_name_disp = next(iter(self.layout.leads.keys())) if self.layout.leads else "Single"
            banner_text = (
                f"ECG DigiAI Single-Lead Diagnostic Overlay | Mode: {self.extractor_mode_used.upper()} | "
                f"Lead: {lead_name_disp} | Red Trace: Reconstructed Waveform"
            )
        elif is_cal:
            banner_text = (
                f"ECG DigiAI 12-Lead Diagnostic Overlay | Mode: {self.extractor_mode_used.upper()} | "
                f"Green: Calibrated Leads (25 mm/s, 10 mm/mV) | Red Trace: Reconstructed Waveform"
            )
        else:
            banner_text = (
                f"ECG DigiAI 12-Lead Diagnostic Overlay | Mode: {self.extractor_mode_used.upper()} | "
                f"Cyan: Extracted Leads (Uncalibrated Input) | Red Trace: Reconstructed Waveform"
            )
        cv2.putText(banner, banner_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)

        combined = np.vstack([banner, overlay_img])
        cv2.imwrite(str(out), combined)
        logger.info("Saved 12-lead sheet diagnostic overlay: %s", out)
        return out

    def save_artifacts(
        self,
        output_dir: Union[str, Path],
        base_name: str = "digitized_ecg",
    ) -> Dict[str, Path]:
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

        # 2. Save Reconstructed CSV and Multi-Lead Plot
        csv_path = out / f"{base_name}_signal.csv"
        self.reconstructed_ecg.save_csv(csv_path)
        saved["csv"] = csv_path

        plot_path = out / f"{base_name}_plot.png"
        self.reconstructed_ecg.save_plot(plot_path, title=f"ECG DigiAI Reconstructed — {base_name}")
        saved["plot"] = plot_path

        # 3. Save 12-Lead Diagnostic Sheet Overlay
        overlay_path = out / f"{base_name}_sheet_overlay.png"
        self.generate_sheet_overlay(overlay_path)
        saved["sheet_overlay"] = overlay_path

        return saved


class ECGDigitizerPipeline:
    """Orchestrates end-to-end ECG paper image digitisation."""

    def __init__(
        self,
        calibration: Optional[CalibrationConfig] = None,
        enable_deskew: bool = True,
        extraction_method: str = "median",
        extractor_mode: str = "deep",
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.calibration = calibration or CalibrationConfig()
        self.preprocessor = PreprocessingPipeline(enable_deskew=enable_deskew)
        self.lead_detector = LeadLayoutDetector()
        self.extractor = WaveformExtractor(method=extraction_method)
        self.reconstructor = SignalReconstructor(calibration=self.calibration)
        self.default_extractor_mode = extractor_mode

        # Initialize deep digitizer if trained checkpoint is present
        self.deep_digitizer = None
        if checkpoint_path is not None:
            ckpt_candidate = Path(checkpoint_path)
        else:
            candidates = [
                Path("models/ecg_digitizer_unet_v3_robust.pt"),
                Path("models/ecg_digitizer_unet_all12lead.pt"),
                Path("models/ecg_digitizer_unet.pt"),
            ]
            ckpt_candidate = next((p for p in candidates if p.is_file()), candidates[-1])

        if ckpt_candidate.is_file():
            try:
                from ecg_digitization.inference.deep_digitizer import DeepCropDigitizer
                self.deep_digitizer = DeepCropDigitizer(checkpoint_path=ckpt_candidate)
                logger.info("ECGDigitizerPipeline initialized with DeepCropDigitizer from %s", ckpt_candidate)
            except Exception as exc:
                logger.warning("Could not initialize DeepCropDigitizer: %s. Falling back to classical.", exc)

    def digitize(
        self,
        image_source: Union[str, Path, np.ndarray],
        layout_hint: str = "auto",
        extractor_mode: Optional[str] = None,
        is_calibration_reliable: bool = True,
        calibration_note: str = "",
    ) -> DigitizationPipelineResult:
        """Executes full digitisation on an input ECG image."""
        warnings: List[str] = []
        mode = extractor_mode or self.default_extractor_mode

        # 1. Preprocess image
        prep = self.preprocessor.process(image_source)
        if abs(prep.skew_angle_deg) > 0.5:
            warnings.append(f"Detected skew angle of {prep.skew_angle_deg:.1f} degrees; image deskewed.")

        # 2. Detect lead layout and separate predicted vs active
        det_info = self.lead_detector.infer_layout_detailed(prep.deskewed)
        predicted_layout = det_info["predicted_layout"]
        layout_conf = det_info["confidence"]
        is_ambiguous = det_info["is_ambiguous"]
        layout_probs = det_info["class_probabilities"]

        if layout_hint in ("auto", None):
            active_layout = predicted_layout
            is_overridden = False
        else:
            active_layout = layout_hint
            is_overridden = bool(layout_hint != predicted_layout)

        if is_overridden:
            warnings.append(
                f"Manual layout override active: using '{active_layout}' (automatic AI prediction was '{predicted_layout}' with {layout_conf * 100:.0f}% confidence)."
            )
        if is_ambiguous:
            warnings.append(det_info.get("notes", "Layout detection confidence is moderate; please verify selected layout."))

        layout = self.lead_detector.detect_layout(prep.binary_trace.shape, layout_hint=active_layout, image=prep.deskewed)
        warnings.extend(layout.warnings)

        if not layout.is_supported:
            dummy_ecg = self.reconstructor.reconstruct_ecg(
                {},
                layout_name=active_layout,
                is_calibration_reliable=is_calibration_reliable,
                calibration_note=calibration_note,
            )
            return DigitizationPipelineResult(
                preprocessed=prep,
                layout=layout,
                extracted_traces={},
                reconstructed_ecg=dummy_ecg,
                success=False,
                warnings=warnings,
                extractor_mode_used=mode,
                predicted_layout=predicted_layout,
                active_layout=active_layout,
                layout_confidence=layout_conf,
                is_layout_overridden=is_overridden,
                is_layout_ambiguous=is_ambiguous,
                layout_probabilities=layout_probs,
            )

        # Check deep digitizer availability
        use_deep = (mode in ("deep", "hybrid")) and (self.deep_digitizer is not None)
        if (mode in ("deep", "hybrid")) and (self.deep_digitizer is None):
            warnings.append("Deep digitizer checkpoint not available; falling back to classical extraction.")
            use_deep = False

        # 3. Extract waveform per lead
        extracted_traces: Dict[str, ExtractedTrace] = {}
        for lead_name, region in layout.leads.items():
            crop_color = prep.deskewed[region.y1 : region.y2, region.x1 : region.x2]
            crop_bin = prep.binary_trace[region.y1 : region.y2, region.x1 : region.x2]

            if crop_color.size == 0 or crop_bin.size == 0:
                warnings.append(f"Empty crop for lead '{lead_name}'.")
                continue

            expected_baseline_y = float(region.baseline_y_hint) if region.baseline_y_hint is not None else (crop_color.shape[0] / 2.0)
            if self.deep_digitizer is not None:
                is_trained = lead_name in self.deep_digitizer.supported_leads
            else:
                is_trained = lead_name in ("II", "V1", "II_RHYTHM")

            # Route based on extractor mode and lead scope
            is_single_beat = bool(active_layout == "single_lead" or len(layout.leads) == 1)
            # For single-lead beat crops (e.g. N10007, Q1), classical centroid extraction achieves
            # sub-pixel precision (<0.15 px) and cleanly trims inactive zero-padding tails.
            # Multi-lead clinical sheets with paper grids route to Deep U-Net.
            if is_single_beat and mode != "deep_forced":
                route_to_deep = False
            else:
                route_to_deep = use_deep and (mode in ("deep", "hybrid", "deep_forced") or is_trained)

            if route_to_deep:
                try:
                    trace = self.deep_digitizer.extract_trace_from_lead_crop(
                        lead_crop=crop_color,
                        binary_crop=crop_bin,
                        lead_name=lead_name,
                        expected_baseline_y=expected_baseline_y,
                        is_rhythm=region.is_rhythm,
                    )
                except Exception as exc:
                    logger.warning("Deep digitizer failed for lead '%s': %s; fallback to classical.", lead_name, exc)
                    trace = self.extractor.extract_from_crop(
                        lead_mask=crop_bin,
                        lead_name=lead_name,
                        baseline_y=expected_baseline_y,
                    )
            else:
                trace = self.extractor.extract_from_crop(
                    lead_mask=crop_bin,
                    lead_name=lead_name,
                    baseline_y=expected_baseline_y,
                )

            extracted_traces[lead_name] = trace

        # 4. Calibrate and reconstruct digital ECG
        reconstructed = self.reconstructor.reconstruct_ecg(
            extracted_traces=extracted_traces,
            layout_name=active_layout,
            is_calibration_reliable=is_calibration_reliable,
            calibration_note=calibration_note,
        )

        return DigitizationPipelineResult(
            preprocessed=prep,
            layout=layout,
            extracted_traces=extracted_traces,
            reconstructed_ecg=reconstructed,
            success=True,
            warnings=warnings,
            extractor_mode_used=mode if use_deep else "classical",
            predicted_layout=predicted_layout,
            active_layout=active_layout,
            layout_confidence=layout_conf,
            is_layout_overridden=is_overridden,
            is_layout_ambiguous=is_ambiguous,
            layout_probabilities=layout_probs,
        )
