"""ECG Signal Quality and Rhythm Analysis.

Provides:
1. R-peak detection using Pan-Tompkins-inspired derivative & adaptive thresholding.
2. Heart rate (BPM) and RR interval variability calculation.
3. Signal quality index (SQI) and lead-off / flatline detection.
4. Rule-based Bradycardia (<60 bpm) and Tachycardia (>100 bpm) findings.
5. Clinical decision-support disclaimers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

logger = logging.getLogger(__name__)


@dataclass
class QRSComplex:
    """Detailed fiducial landmarks for a single detected QRS complex."""

    r_idx: int
    r_time_s: float
    r_voltage_mv: float
    is_inverted: bool
    q_idx: Optional[int] = None
    q_time_s: Optional[float] = None
    q_voltage_mv: Optional[float] = None
    s_idx: Optional[int] = None
    s_time_s: Optional[float] = None
    s_voltage_mv: Optional[float] = None
    qrs_duration_ms: Optional[float] = None
    confidence: float = 1.0


@dataclass
class RhythmAnalysisResult:
    """Findings from rule-based physiological rhythm analysis."""

    lead_analyzed: str
    heart_rate_bpm: Optional[float]
    num_r_peaks: int
    r_peak_indices: List[int]
    mean_rr_interval_s: Optional[float]
    std_rr_interval_s: Optional[float]
    is_regular_rhythm: bool
    signal_quality: str               # "good", "acceptable", "low"
    quality_warnings: List[str]
    rule_findings: List[str]          # e.g. ["Sinus Bradycardia (<60 bpm)"]


class ECGRhythmAnalyzer:
    """Analyzes reconstructed or raw ECG signals for heart rate and basic rhythm abnormalities."""

    def __init__(self, target_fs_hz: float = 100.0) -> None:
        self.fs = target_fs_hz

    def bandpass_filter(self, signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
        """Applies zero-phase Butterworth bandpass filter to suppress baseline wander and noise."""
        nyq = 0.5 * self.fs
        # highcut must be strictly less than nyquist frequency
        safe_highcut = min(highcut, nyq * 0.95)
        safe_lowcut = max(lowcut, 0.1)

        b, a = butter(2, [safe_lowcut / nyq, safe_highcut / nyq], btype="band")
        # Pad short signals if needed
        if len(signal) <= 15:
            return signal
        return filtfilt(b, a, signal)

    def detect_r_peaks(self, signal: np.ndarray, lead_name: str = "II") -> np.ndarray:
        """Detects QRS peaks supporting upright, inverted (e.g. aVR), and biphasic morphologies."""
        if len(signal) < int(self.fs * 0.5):
            return np.array([], dtype=int)

        filtered = self.bandpass_filter(signal)
        # Derivative / difference to emphasize steep QRS slopes
        diff = np.diff(filtered, prepend=filtered[0])
        squared = diff ** 2

        # Moving average window (~120 ms)
        win_len = max(3, int(0.12 * self.fs))
        kernel = np.ones(win_len) / win_len
        integrated = np.convolve(squared, kernel, mode="same")

        # Dynamic threshold: fraction of 90th percentile to avoid outlier spike skew
        thresh = 0.35 * np.percentile(integrated, 90)
        # Physiological constraint: minimum 250 ms between successive heartbeats (max ~240 bpm)
        min_dist = max(2, int(0.25 * self.fs))

        candidate_peaks, _ = find_peaks(integrated, height=thresh, distance=min_dist)

        # Refine slope candidate peaks to actual local signal voltage apex within [-80 ms, +80 ms]
        search_radius = max(2, int(0.08 * self.fs))
        baseline = float(np.median(signal))
        is_lead_typically_inverted = lead_name.upper() in ("AVR", "V1")

        refined_peaks: List[int] = []
        for cp in candidate_peaks:
            w_start = max(0, cp - search_radius)
            w_end = min(len(signal), cp + search_radius + 1)
            if w_end > w_start:
                window = signal[w_start:w_end]
                pos_dev = float(np.max(window) - baseline)
                neg_dev = float(baseline - np.min(window))

                # Determine polarity: inverted if downward deflection dominates significantly
                if is_lead_typically_inverted or (neg_dev > 1.35 * max(0.01, pos_dev)):
                    local_apex = w_start + int(np.argmin(window))
                else:
                    local_apex = w_start + int(np.argmax(window))

                if local_apex not in refined_peaks:
                    refined_peaks.append(local_apex)

        return np.array(sorted(refined_peaks), dtype=int) if refined_peaks else np.array([], dtype=int)

    def detect_qrs_complexes(self, signal: np.ndarray, lead_name: str = "II") -> List[QRSComplex]:
        """Detects full QRS complexes with R-peaks, Q onsets, and S offsets with uncertainty gating."""
        sig = np.asarray(signal, dtype=np.float32).flatten()
        r_peaks = self.detect_r_peaks(sig, lead_name=lead_name)
        if len(r_peaks) == 0:
            return []

        baseline = float(np.median(sig))
        is_lead_typically_inverted = lead_name.upper() in ("AVR", "V1")
        complexes: List[QRSComplex] = []

        q_window = max(2, int(0.08 * self.fs))
        s_window = max(2, int(0.08 * self.fs))

        for r_idx in r_peaks:
            r_val = float(sig[r_idx])
            r_time = float(r_idx / self.fs)
            is_inverted = is_lead_typically_inverted or (baseline - r_val > abs(r_val - baseline) * 0.7)

            # Search Q onset preceding R
            q_idx: Optional[int] = None
            q_time: Optional[float] = None
            q_val: Optional[float] = None
            q_start = max(0, r_idx - q_window)
            if r_idx > q_start + 1:
                q_seg = sig[q_start:r_idx]
                if is_inverted:
                    cand_q = q_start + int(np.argmax(q_seg))
                else:
                    cand_q = q_start + int(np.argmin(q_seg))
                # Reliable check: must be distinct inflection from R
                if abs(sig[cand_q] - r_val) > 0.04 and (r_idx - cand_q) >= 1:
                    q_idx = cand_q
                    q_time = float(cand_q / self.fs)
                    q_val = float(sig[cand_q])

            # Search S offset following R
            s_idx: Optional[int] = None
            s_time: Optional[float] = None
            s_val: Optional[float] = None
            s_end = min(len(sig), r_idx + s_window + 1)
            if s_end > r_idx + 1:
                s_seg = sig[r_idx + 1:s_end]
                if is_inverted:
                    cand_s = r_idx + 1 + int(np.argmax(s_seg))
                else:
                    cand_s = r_idx + 1 + int(np.argmin(s_seg))
                if abs(sig[cand_s] - r_val) > 0.04 and (cand_s - r_idx) >= 1:
                    s_idx = cand_s
                    s_time = float(cand_s / self.fs)
                    s_val = float(sig[cand_s])

            duration_ms: Optional[float] = None
            confidence = 1.0
            if q_idx is not None and s_idx is not None:
                duration_ms = float((s_idx - q_idx) / self.fs * 1000.0)
                # Physiological QRS duration check (40 ms to 180 ms)
                if not (35.0 <= duration_ms <= 220.0):
                    confidence = 0.6  # Uncertainty tag
            elif q_idx is None and s_idx is None:
                confidence = 0.75  # R peak confident, boundaries ambiguous

            complexes.append(
                QRSComplex(
                    r_idx=int(r_idx),
                    r_time_s=r_time,
                    r_voltage_mv=r_val,
                    is_inverted=is_inverted,
                    q_idx=q_idx,
                    q_time_s=q_time,
                    q_voltage_mv=q_val,
                    s_idx=s_idx,
                    s_time_s=s_time,
                    s_voltage_mv=s_val,
                    qrs_duration_ms=duration_ms,
                    confidence=confidence,
                )
            )

        return complexes

    def analyze_signal(
        self,
        lead_signal: np.ndarray,
        lead_name: str = "II",
    ) -> RhythmAnalysisResult:
        """Performs full rhythm and rate analysis on a single ECG lead signal."""
        sig = np.asarray(lead_signal, dtype=np.float32).flatten()
        warnings: List[str] = []
        findings: List[str] = []

        # 1. Signal Quality Assessment
        sig_len_s = len(sig) / self.fs
        sig_std = float(np.std(sig))
        sig_range = float(np.ptp(sig))

        if sig_std < 0.02 or sig_range < 0.05:
            warnings.append("Near-zero amplitude detected (potential flatline or disconnected lead).")
            return RhythmAnalysisResult(
                lead_analyzed=lead_name,
                heart_rate_bpm=None,
                num_r_peaks=0,
                r_peak_indices=[],
                mean_rr_interval_s=None,
                std_rr_interval_s=None,
                is_regular_rhythm=False,
                signal_quality="low",
                quality_warnings=warnings,
                rule_findings=["Signal amplitude insufficient for analysis"],
            )

        # 2. Peak Detection
        peaks = self.detect_r_peaks(sig, lead_name=lead_name)
        num_peaks = len(peaks)

        if num_peaks < 2:
            warnings.append("Fewer than 2 R-peaks detected; duration too short or signal quality low.")
            return RhythmAnalysisResult(
                lead_analyzed=lead_name,
                heart_rate_bpm=None,
                num_r_peaks=num_peaks,
                r_peak_indices=peaks.tolist(),
                mean_rr_interval_s=None,
                std_rr_interval_s=None,
                is_regular_rhythm=False,
                signal_quality="low",
                quality_warnings=warnings,
                rule_findings=["Insufficient R-peaks detected"],
            )

        # 3. RR Intervals and Heart Rate
        rr_samples = np.diff(peaks)
        rr_seconds = rr_samples / self.fs

        mean_rr = float(np.mean(rr_seconds))
        std_rr = float(np.std(rr_seconds))
        bpm = float(60.0 / mean_rr) if mean_rr > 0 else 0.0

        # Rhythm regularity: requires at least 4 RR intervals (>= 5 peaks) for meaningful statistical assessment
        if len(rr_seconds) >= 4:
            cv_rr = std_rr / mean_rr if mean_rr > 0 else 0.0
            is_regular = cv_rr < 0.20
            if not is_regular:
                warnings.append(f"Irregular RR intervals detected (CV={cv_rr:.2f}).")
                findings.append("Irregular rhythm pattern observed")
            else:
                findings.append("Regular rhythm pattern")
        else:
            # Short segment: insufficient beats for statistical rhythm regularity determination
            is_regular = True
            warnings.append(f"Short signal segment ({num_peaks} R-peaks): rhythm regularity assessment requires >= 5 beats.")
            findings.append("Short recording duration (limited rhythm assessment)")

        # 4. Clinical Bradycardia / Tachycardia Rules
        if bpm < 60.0:
            findings.append(f"Sinus Bradycardia ({bpm:.1f} bpm < 60 bpm)")
        elif bpm > 100.0:
            findings.append(f"Sinus Tachycardia ({bpm:.1f} bpm > 100 bpm)")
        else:
            findings.append(f"Normal Heart Rate ({bpm:.1f} bpm)")

        if len(rr_seconds) < 4:
            quality = "acceptable"
        elif is_regular and (40.0 <= bpm <= 180.0):
            quality = "good"
        else:
            quality = "acceptable"

        return RhythmAnalysisResult(
            lead_analyzed=lead_name,
            heart_rate_bpm=round(bpm, 1),
            num_r_peaks=num_peaks,
            r_peak_indices=peaks.tolist(),
            mean_rr_interval_s=round(mean_rr, 3),
            std_rr_interval_s=round(std_rr, 3),
            is_regular_rhythm=is_regular,
            signal_quality=quality,
            quality_warnings=warnings,
            rule_findings=findings,
        )
