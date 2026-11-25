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

    def generate_qrs_overlay(self, output_path: Union[str, Path]) -> Path:
        """Generates dedicated QRS Detection Overlay with detected complexes, R-peaks, and Q/S boundaries."""
        from ecg_digitization.inference.analyzer import ECGRhythmAnalyzer

        analyzer = ECGRhythmAnalyzer(target_fs_hz=100.0)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        overlay_img = self.preprocessed.deskewed.copy()
        if len(overlay_img.shape) == 2:
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)

        h, w = overlay_img.shape[:2]
        qrs_layer = overlay_img.copy()

        for name, region in self.layout.leads.items():
            rec_lead = self.reconstructed_ecg.leads.get(name)
            trace = self.extracted_traces.get(name)
            if rec_lead is None or trace is None or len(rec_lead.voltage_mv) < 15:
                continue

            duration_s = max(0.1, float(rec_lead.duration_seconds))
            x_start = region.x1 + trace.active_start_col
            x_end = region.x1 + trace.active_end_col
            col_span = max(1, x_end - x_start)

            complexes = analyzer.detect_qrs_complexes(rec_lead.voltage_mv, lead_name=name)

            for c in complexes:
                r_col = int(round(x_start + (c.r_time_s / duration_s) * col_span))
                r_col = max(region.x1, min(region.x2, r_col))

                trace_idx = int(round((c.r_time_s / duration_s) * (len(trace.pixel_trajectory) - 1)))
                trace_idx = max(0, min(len(trace.pixel_trajectory) - 1, trace_idx))
                r_y = int(round(region.y1 + trace.pixel_trajectory[trace_idx]))
                r_y = max(region.y1, min(region.y2, r_y))

                # Shaded QRS interval corridor
                if c.q_time_s is not None and c.s_time_s is not None:
                    q_col = int(round(x_start + (c.q_time_s / duration_s) * col_span))
                    s_col = int(round(x_start + (c.s_time_s / duration_s) * col_span))
                else:
                    q_col = int(round(x_start + (max(0.0, c.r_time_s - 0.04) / duration_s) * col_span))
                    s_col = int(round(x_start + (min(duration_s, c.r_time_s + 0.04) / duration_s) * col_span))

                q_col = max(region.x1, min(region.x2, q_col))
                s_col = max(region.x1, min(region.x2, s_col))

                shade_color = (200, 245, 200) if c.confidence > 0.8 else (240, 240, 200)
                cv2.rectangle(qrs_layer, (q_col, region.y1), (s_col, region.y2), shade_color, -1)

                # Mark Q onset (Amber)
                if c.q_idx is not None and c.q_time_s is not None:
                    q_c = int(round(x_start + (c.q_time_s / duration_s) * col_span))
                    q_tr = max(0, min(len(trace.pixel_trajectory) - 1, int(round((c.q_time_s / duration_s) * (len(trace.pixel_trajectory) - 1)))))
                    q_y = int(round(region.y1 + trace.pixel_trajectory[q_tr]))
                    cv2.circle(overlay_img, (q_c, q_y), 3, (0, 165, 255), -1, cv2.LINE_AA)
                    cv2.putText(overlay_img, "Q", (q_c - 5, q_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 140, 220), 1, cv2.LINE_AA)

                # Mark S offset (Blue)
                if c.s_idx is not None and c.s_time_s is not None:
                    s_c = int(round(x_start + (c.s_time_s / duration_s) * col_span))
                    s_tr = max(0, min(len(trace.pixel_trajectory) - 1, int(round((c.s_time_s / duration_s) * (len(trace.pixel_trajectory) - 1)))))
                    s_y = int(round(region.y1 + trace.pixel_trajectory[s_tr]))
                    cv2.circle(overlay_img, (s_c, s_y), 3, (255, 120, 0), -1, cv2.LINE_AA)
                    cv2.putText(overlay_img, "S", (s_c + 2, s_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 100, 0), 1, cv2.LINE_AA)

                # Mark R-peak
                peak_label = "QS" if c.is_inverted else "R"
                cv2.circle(overlay_img, (r_col, r_y), 5, (20, 20, 230), -1, cv2.LINE_AA)
                cv2.circle(overlay_img, (r_col, r_y), 7, (255, 255, 255), 1, cv2.LINE_AA)
                lbl_y = r_y + 14 if c.is_inverted else r_y - 8
                cv2.putText(overlay_img, peak_label, (r_col - 4, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 230), 1, cv2.LINE_AA)

        cv2.addWeighted(qrs_layer, 0.30, overlay_img, 0.70, 0, overlay_img)

        # Draw lead region bounding frames
        for name, region in self.layout.leads.items():
            cv2.rectangle(overlay_img, (region.x1, region.y1), (region.x2, region.y2), (180, 180, 180), 1)
            cv2.putText(overlay_img, name, (region.x1 + 4, region.y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

        banner_h = 32
        banner = np.full((banner_h, w, 3), 245, dtype=np.uint8)
        cv2.line(banner, (0, banner_h - 1), (w, banner_h - 1), (200, 200, 200), 1)
        banner_text = "ECG DigiAI QRS Detection Overlay | Red: R-Peak / Apex | Amber: Q-Onset | Blue: S-Offset | Shaded: QRS Region"
        cv2.putText(banner, banner_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)

        combined = np.vstack([banner, overlay_img])
        cv2.imwrite(str(out), combined)
        logger.info("Saved QRS detection overlay: %s", out)
        return out

    def generate_uncertainty_overlay(
        self,
        output_path: Union[str, Path],
        selected_lead: Optional[str] = None,
    ) -> Path:
        """Generates dedicated Trace Fidelity & Uncertainty Overlay.

        Visualizes directly observed ink vs. interpolated segments, highlights
        multi-path ambiguous regions, and shows QRS detections with confidence.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        overlay_img = self.preprocessed.deskewed.copy()
        if len(overlay_img.shape) == 2:
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)

        h, w = overlay_img.shape[:2]
        highlight_layer = overlay_img.copy()

        # 1. First pass: highlight ambiguous multi-path regions
        for name, region in self.layout.leads.items():
            if selected_lead and name != selected_lead:
                continue
            trace = self.extracted_traces.get(name)
            if not trace or not trace.fidelity_diagnostics:
                continue
            diag = trace.fidelity_diagnostics
            x_start = region.x1 + trace.active_start_col

            for r_st, r_en in diag.ambiguous_regions:
                x_a = max(region.x1, min(region.x2, x_start + r_st))
                x_b = max(region.x1, min(region.x2, x_start + r_en + 1))
                cv2.rectangle(highlight_layer, (x_a, region.y1), (x_b, region.y2), (0, 215, 255), -1)

        cv2.addWeighted(highlight_layer, 0.28, overlay_img, 0.72, 0, overlay_img)

        # 2. Second pass: render lead bounding boxes, observed vs interpolated trajectory segments
        for name, region in self.layout.leads.items():
            is_active_target = (selected_lead is None or name == selected_lead)
            trace = self.extracted_traces.get(name)

            box_col = (0, 160, 240) if (selected_lead and name == selected_lead) else (180, 180, 180)
            box_thick = 2 if (selected_lead and name == selected_lead) else 1
            cv2.rectangle(overlay_img, (region.x1, region.y1), (region.x2, region.y2), box_col, box_thick)

            lbl = f"{name} [Focus]" if (selected_lead and name == selected_lead) else name
            cv2.putText(overlay_img, lbl, (region.x1 + 4, region.y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)

            if not trace or len(trace.pixel_trajectory) == 0:
                continue

            diag = trace.fidelity_diagnostics
            traj = trace.pixel_trajectory
            n_pts = len(traj)
            x_start = region.x1 + trace.active_start_col
            x_end = region.x1 + trace.active_end_col
            x_coords = np.linspace(x_start, x_end, n_pts)
            y_coords = region.y1 + traj

            # Trace segments: green for directly observed ink, amber/orange for interpolated gaps
            obs_mask = diag.observed_mask if (diag and len(diag.observed_mask) == n_pts) else [True] * n_pts

            for i in range(n_pts - 1):
                pt1 = (int(round(x_coords[i])), int(round(y_coords[i])))
                pt2 = (int(round(x_coords[i + 1])), int(round(y_coords[i + 1])))
                is_obs = obs_mask[i] and obs_mask[i + 1]

                seg_color = (0, 180, 20) if is_obs else (0, 120, 255)  # Green vs Amber
                thick = 2 if is_active_target else 1
                cv2.line(overlay_img, pt1, pt2, seg_color, thick, cv2.LINE_AA)

            # 3. Third pass: QRS detections with confidence
            if diag and is_active_target:
                for c in diag.qrs_complexes:
                    r_x = int(round(x_start + c.peak_col))
                    r_y = int(round(region.y1 + c.peak_y))
                    r_x = max(region.x1, min(region.x2, r_x))
                    r_y = max(region.y1, min(region.y2, r_y))

                    qrs_color = (20, 20, 230) if c.is_high_confidence else (0, 140, 255)
                    cv2.circle(overlay_img, (r_x, r_y), 4, qrs_color, -1, cv2.LINE_AA)
                    cv2.circle(overlay_img, (r_x, r_y), 6, (255, 255, 255), 1, cv2.LINE_AA)

                    conf_str = f"R {int(c.confidence * 100)}%" if c.is_high_confidence else f"R? {int(c.confidence * 100)}%"
                    cv2.putText(overlay_img, conf_str, (r_x - 12, r_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, qrs_color, 1, cv2.LINE_AA)

        banner_h = 32
        banner = np.full((banner_h, w, 3), 245, dtype=np.uint8)
        cv2.line(banner, (0, banner_h - 1), (w, banner_h - 1), (200, 200, 200), 1)

        sel_str = f" [Focus: Lead {selected_lead}]" if selected_lead else ""
        banner_text = f"ECG DigiAI Trace Fidelity Overlay{sel_str} | Green: Observed Ink | Amber: Interpolated Gap | Shaded: Multi-Path Ambiguity"
        cv2.putText(banner, banner_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)

        combined = np.vstack([banner, overlay_img])
        cv2.imwrite(str(out), combined)
        logger.info("Saved trace fidelity overlay: %s", out)
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
