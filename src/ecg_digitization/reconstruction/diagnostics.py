"""Trace Fidelity Diagnostics module for ECG DigiAI.

Computes comprehensive per-lead physical ink recovery, ambiguity,
interpolation, QRS peak detection confidence, and localized uncertainty intervals
derived directly from intermediate segmentation masks and trajectory coordinates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UncertainInterval:
    """Localized interval of waveform extraction uncertainty."""

    start_col: int
    end_col: int
    reason: str
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_col": self.start_col,
            "end_col": self.end_col,
            "reason": self.reason,
            "start_time_s": round(self.start_time_s, 3) if self.start_time_s is not None else None,
            "end_time_s": round(self.end_time_s, 3) if self.end_time_s is not None else None,
            "description": self.description,
        }


@dataclass
class QRSComplexInfo:
    """Detection provenance and morphological confidence for a single QRS complex."""

    peak_col: int
    peak_y: float
    confidence: float
    is_high_confidence: bool
    q_col: Optional[int] = None
    s_col: Optional[int] = None
    r_time_s: Optional[float] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_col": self.peak_col,
            "peak_y": round(float(self.peak_y), 2),
            "confidence": round(float(self.confidence), 3),
            "is_high_confidence": self.is_high_confidence,
            "q_col": self.q_col,
            "s_col": self.s_col,
            "r_time_s": round(float(self.r_time_s), 3) if self.r_time_s is not None else None,
            "note": self.note,
        }


@dataclass
class TraceFidelityDiagnostics:
    """Per-lead trace fidelity and physical ink extraction diagnostic object.

    Strictly separates:
    - Measured values (directly counted from binary mask pixels and sample coordinates).
    - Estimated values (statistically inferred from ink clustering or signal analysis).
    - Scientifically unavailable values (explicitly flagged as N/A).
    """

    lead_name: str

    # Overall Status
    overall_status: str                     # "HIGH_FIDELITY", "ACCEPTABLE_FIDELITY", "UNCERTAIN_EXTRACTION"
    overall_status_note: str

    # A. Ink Recovery Metrics
    total_trace_ink_pixels: int             # [Measured] Total foreground ink pixels in active lead crop bounding box
    trajectory_supporting_ink_pixels: int   # [Measured] Ink pixels falling within trace envelope (<= 3px)
    rejected_non_trace_pixels: int          # [Estimated] Foreground pixels in crop rejected as grid, text, or noise
    ink_support_percent: float              # [Measured] Columns with direct observed ink supporting trajectory (%)
    ink_recovery_note: str                  # Scientific definition: trajectory column support ratio

    # B. Ambiguity Metrics
    ambiguous_columns_count: int            # [Measured] Columns with >= 2 disconnected vertical ink clusters (gap > 2px)
    ambiguous_columns_percent: float        # [Measured] Ambiguous columns / total active columns (%)
    ambiguous_regions: List[Tuple[int, int]] # [Measured] Contiguous [start_col, end_col] multi-path intervals

    # C. Interpolation Metrics
    total_samples: int                      # [Measured] Total trajectory points in active span
    observed_samples: int                   # [Measured] Samples directly grounded on observed ink locus
    interpolated_samples: int               # [Measured] Samples synthesized across ink dropouts
    interpolated_percent: float             # [Measured] Interpolated samples / total samples (%)
    longest_interpolation_gap: int          # [Measured] Max consecutive unobserved columns (px)
    observed_mask: List[bool]               # [Measured] Per-sample boolean array (True = observed, False = interpolated)

    # D. Peak Detection Metrics
    qrs_detected_count: int                 # [Measured] Number of physiological QRS complexes detected
    qrs_high_confidence_count: int          # [Measured] Detections with confidence >= 0.75
    qrs_uncertain_count: int                # [Measured] Candidate detections with confidence < 0.75
    qrs_complexes: List[QRSComplexInfo]     # Per-complex coordinates and confidence

    # E. Extraction Uncertainty Intervals
    uncertain_intervals: List[UncertainInterval] # Localized intervals with cause descriptions

    def to_dict(self) -> Dict[str, Any]:
        """Serializes diagnostics to a clean JSON-compatible dictionary."""
        return {
            "lead_name": self.lead_name,
            "overall_status": self.overall_status,
            "overall_status_note": self.overall_status_note,
            "ink_recovery": {
                "total_trace_ink_pixels": self.total_trace_ink_pixels,
                "trajectory_supporting_ink_pixels": self.trajectory_supporting_ink_pixels,
                "rejected_non_trace_pixels": self.rejected_non_trace_pixels,
                "ink_support_percent": round(self.ink_support_percent, 1),
                "ink_recovery_note": self.ink_recovery_note,
            },
            "ambiguity": {
                "ambiguous_columns_count": self.ambiguous_columns_count,
                "ambiguous_columns_percent": round(self.ambiguous_columns_percent, 1),
                "ambiguous_regions": self.ambiguous_regions,
            },
            "interpolation": {
                "total_samples": self.total_samples,
                "observed_samples": self.observed_samples,
                "interpolated_samples": self.interpolated_samples,
                "interpolated_percent": round(self.interpolated_percent, 1),
                "longest_interpolation_gap": self.longest_interpolation_gap,
                "observed_mask": self.observed_mask,
            },
            "peak_detection": {
                "qrs_detected_count": self.qrs_detected_count,
                "qrs_high_confidence_count": self.qrs_high_confidence_count,
                "qrs_uncertain_count": self.qrs_uncertain_count,
                "qrs_complexes": [c.to_dict() for c in self.qrs_complexes],
            },
            "uncertain_intervals": [u.to_dict() for u in self.uncertain_intervals],
        }


class TraceFidelityComputer:
    """Computes comprehensive trace fidelity diagnostics from raw crop segmentation and trajectory."""

    @classmethod
    def compute(
        cls,
        binary_crop: np.ndarray,
        pixel_trajectory: np.ndarray,
        active_start_col: int,
        active_end_col: int,
        lead_name: str,
        sampling_rate_hz: float = 100.0,
        voltage_mv: Optional[np.ndarray] = None,
        crop_height_px: int = 128,
        is_clipped_top: bool = False,
        is_clipped_bottom: bool = False,
    ) -> TraceFidelityDiagnostics:
        total_pts = len(pixel_trajectory)
        if total_pts == 0 or binary_crop.size == 0:
            return cls._empty_diagnostics(lead_name)

        h_crop, w_crop = binary_crop.shape[:2]
        c_min = max(0, min(active_start_col, w_crop - 1))
        c_max = max(c_min, min(active_end_col, w_crop - 1))
        span_w = c_max - c_min + 1

        # Resample or align trajectory to active column range
        if total_pts != span_w:
            traj_aligned = np.interp(np.linspace(0, total_pts - 1, span_w), np.arange(total_pts), pixel_trajectory)
        else:
            traj_aligned = pixel_trajectory

        crop_active_ink = binary_crop[:, c_min : c_max + 1]
        total_ink_px = int(np.sum(crop_active_ink > 0))

        observed_flags: List[bool] = []
        ambiguous_flags: List[bool] = []
        supporting_ink_count = 0

        for offset in range(span_w):
            c_img = c_min + offset
            y_traj = float(traj_aligned[offset])
            col_ink_y = np.where(binary_crop[:, c_img] > 0)[0]

            if len(col_ink_y) == 0:
                observed_flags.append(False)
                ambiguous_flags.append(False)
            else:
                # Support: ink locus within 2.0 px of trajectory
                dist_to_traj = np.abs(col_ink_y - y_traj)
                is_obs = bool(np.min(dist_to_traj) <= 2.0)
                observed_flags.append(is_obs)

                # Count ink pixels within envelope (<= 4.0 px) as supporting waveform
                supporting_ink_count += int(np.sum(dist_to_traj <= 4.0))

                # Ambiguity: multiple vertical clusters separated by > 2px gap
                splits = np.where(np.diff(col_ink_y) > 2)[0] + 1
                ambiguous_flags.append(bool(len(splits) >= 1))

        observed_count = sum(observed_flags)
        interpolated_count = span_w - observed_count
        interpolated_pct = float(interpolated_count / span_w * 100.0) if span_w > 0 else 0.0
        ink_support_pct = float(observed_count / span_w * 100.0) if span_w > 0 else 0.0

        ambiguous_count = sum(ambiguous_flags)
        ambiguous_pct = float(ambiguous_count / span_w * 100.0) if span_w > 0 else 0.0

        # Longest consecutive interpolation gap
        longest_gap = 0
        current_gap = 0
        for is_obs in observed_flags:
            if not is_obs:
                current_gap += 1
                if current_gap > longest_gap:
                    longest_gap = current_gap
            else:
                current_gap = 0

        # Ambiguous regions (contiguous blocks of length >= 2)
        ambiguous_regions: List[Tuple[int, int]] = []
        in_region = False
        reg_start = 0
        for idx, is_amb in enumerate(ambiguous_flags):
            if is_amb and not in_region:
                in_region = True
                reg_start = idx
            elif not is_amb and in_region:
                in_region = False
                if idx - reg_start >= 2:
                    ambiguous_regions.append((reg_start, idx - 1))
        if in_region and span_w - reg_start >= 2:
            ambiguous_regions.append((reg_start, span_w - 1))

        # Rejected non-trace pixels
        rejected_ink = max(0, total_ink_px - supporting_ink_count)

        # Peak detection metrics via ECGRhythmAnalyzer
        qrs_complexes: List[QRSComplexInfo] = []
        qrs_high_conf = 0
        qrs_uncert = 0

        if voltage_mv is not None and len(voltage_mv) >= 15:
            try:
                from ecg_digitization.inference.analyzer import ECGRhythmAnalyzer
                analyzer = ECGRhythmAnalyzer(target_fs_hz=sampling_rate_hz)
                detected = analyzer.detect_qrs_complexes(voltage_mv, lead_name=lead_name)
                duration_s = max(0.1, float(len(voltage_mv) / sampling_rate_hz))

                for c in detected:
                    is_high = bool(c.confidence >= 0.75)
                    if is_high:
                        qrs_high_conf += 1
                    else:
                        qrs_uncert += 1

                    peak_col_local = int(round((c.r_time_s / duration_s) * (span_w - 1)))
                    peak_col_local = max(0, min(span_w - 1, peak_col_local))
                    p_y = float(traj_aligned[peak_col_local])

                    q_col_local = None
                    if c.q_time_s is not None:
                        q_col_local = int(round((c.q_time_s / duration_s) * (span_w - 1)))
                        q_col_local = max(0, min(span_w - 1, q_col_local))

                    s_col_local = None
                    if c.s_time_s is not None:
                        s_col_local = int(round((c.s_time_s / duration_s) * (span_w - 1)))
                        s_col_local = max(0, min(span_w - 1, s_col_local))

                    note = "High physiological confidence" if is_high else "Uncertain candidate morphology"
                    qrs_complexes.append(
                        QRSComplexInfo(
                            peak_col=peak_col_local,
                            peak_y=p_y,
                            confidence=float(c.confidence),
                            is_high_confidence=is_high,
                            q_col=q_col_local,
                            s_col=s_col_local,
                            r_time_s=float(c.r_time_s),
                            note=note,
                        )
                    )
            except Exception as exc:
                logger.debug("Could not detect QRS complexes for diagnostics in lead %s: %s", lead_name, exc)

        # Localized uncertain intervals
        uncertain_intervals: List[UncertainInterval] = []
        duration_total_s = float(span_w / sampling_rate_hz)

        # 1. Missing ink gaps (>= 3 consecutive unobserved columns)
        gap_start = None
        for i, is_obs in enumerate(observed_flags):
            if not is_obs and gap_start is None:
                gap_start = i
            elif is_obs and gap_start is not None:
                if i - gap_start >= 3:
                    t_st = (gap_start / span_w) * duration_total_s
                    t_en = (i / span_w) * duration_total_s
                    uncertain_intervals.append(
                        UncertainInterval(
                            start_col=gap_start,
                            end_col=i - 1,
                            reason="missing_ink",
                            start_time_s=t_st,
                            end_time_s=t_en,
                            description=f"Trace dropout spanning {i - gap_start} columns; reconstructed via interpolation.",
                        )
                    )
                gap_start = None
        if gap_start is not None and span_w - gap_start >= 3:
            t_st = (gap_start / span_w) * duration_total_s
            t_en = duration_total_s
            uncertain_intervals.append(
                UncertainInterval(
                    start_col=gap_start,
                    end_col=span_w - 1,
                    reason="missing_ink",
                    start_time_s=t_st,
                    end_time_s=t_en,
                    description=f"Terminal trace dropout spanning {span_w - gap_start} columns.",
                )
            )

        # 2. Multi-path ambiguity intervals (>= 3 consecutive ambiguous columns)
        for r_st, r_en in ambiguous_regions:
            if r_en - r_st + 1 >= 3:
                t_st = (r_st / span_w) * duration_total_s
                t_en = (r_en / span_w) * duration_total_s
                uncertain_intervals.append(
                    UncertainInterval(
                        start_col=r_st,
                        end_col=r_en,
                        reason="ambiguous_stroke",
                        start_time_s=t_st,
                        end_time_s=t_en,
                        description=f"Multi-path ink interference ({r_en - r_st + 1} cols) from overlapping text, grid or cross-talk.",
                    )
                )

        # 3. Crop boundary proximity / clipping warnings
        near_top = np.where(traj_aligned <= 2.0)[0]
        near_bot = np.where(traj_aligned >= h_crop - 2.0)[0]
        if len(near_top) > 0 or is_clipped_top:
            uncertain_intervals.append(
                UncertainInterval(
                    start_col=int(near_top[0]) if len(near_top) > 0 else 0,
                    end_col=int(near_top[-1]) if len(near_top) > 0 else span_w - 1,
                    reason="boundary_proximity",
                    description="Waveform touches upper crop boundary; potential peak clipping risk.",
                )
            )
        if len(near_bot) > 0 or is_clipped_bottom:
            uncertain_intervals.append(
                UncertainInterval(
                    start_col=int(near_bot[0]) if len(near_bot) > 0 else 0,
                    end_col=int(near_bot[-1]) if len(near_bot) > 0 else span_w - 1,
                    reason="boundary_proximity",
                    description="Waveform touches lower crop boundary; potential trough clipping risk.",
                )
            )

        # Overall Status
        if interpolated_pct < 5.0 and ambiguous_pct < 10.0 and longest_gap <= 3 and not is_clipped_top and not is_clipped_bottom:
            status = "HIGH_FIDELITY"
            status_note = "Continuous trace ink directly observed; minimal interpolation or ambiguity."
        elif interpolated_pct <= 18.0 and longest_gap <= 10 and not is_clipped_top and not is_clipped_bottom:
            status = "ACCEPTABLE_FIDELITY"
            status_note = "Reliable waveform reconstruction with minor localized interpolation or ambiguity."
        else:
            status = "UNCERTAIN_EXTRACTION"
            status_note = "Elevated interpolation, multi-path ink ambiguity, or boundary proximity detected."

        ink_note = (
            "Measured as direct trajectory ink support ratio (columns with observed ink within <= 2px / active span). "
            "True pixel ground-truth recovery is scientifically indeterminate without independent vector reference."
        )

        return TraceFidelityDiagnostics(
            lead_name=lead_name,
            overall_status=status,
            overall_status_note=status_note,
            total_trace_ink_pixels=total_ink_px,
            trajectory_supporting_ink_pixels=supporting_ink_count,
            rejected_non_trace_pixels=rejected_ink,
            ink_support_percent=round(ink_support_pct, 1),
            ink_recovery_note=ink_note,
            ambiguous_columns_count=ambiguous_count,
            ambiguous_columns_percent=round(ambiguous_pct, 1),
            ambiguous_regions=ambiguous_regions,
            total_samples=span_w,
            observed_samples=observed_count,
            interpolated_samples=interpolated_count,
            interpolated_percent=round(interpolated_pct, 1),
            longest_interpolation_gap=longest_gap,
            observed_mask=observed_flags,
            qrs_detected_count=len(qrs_complexes),
            qrs_high_confidence_count=qrs_high_conf,
            qrs_uncertain_count=qrs_uncert,
            qrs_complexes=qrs_complexes,
            uncertain_intervals=uncertain_intervals,
        )

    @classmethod
    def _empty_diagnostics(cls, lead_name: str) -> TraceFidelityDiagnostics:
        return TraceFidelityDiagnostics(
            lead_name=lead_name,
            overall_status="UNCERTAIN_EXTRACTION",
            overall_status_note="No trace ink or trajectory samples available.",
            total_trace_ink_pixels=0,
            trajectory_supporting_ink_pixels=0,
            rejected_non_trace_pixels=0,
            ink_support_percent=0.0,
            ink_recovery_note="No trajectory samples available.",
            ambiguous_columns_count=0,
            ambiguous_columns_percent=0.0,
            ambiguous_regions=[],
            total_samples=0,
            observed_samples=0,
            interpolated_samples=0,
            interpolated_percent=0.0,
            longest_interpolation_gap=0,
            observed_mask=[],
            qrs_detected_count=0,
            qrs_high_confidence_count=0,
            qrs_uncertain_count=0,
            qrs_complexes=[],
            uncertain_intervals=[],
        )
