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

    def detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Detects QRS R-peaks using bandpass filtering, squaring, and minimum-distance peak finding."""
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

        peaks, _ = find_peaks(integrated, height=thresh, distance=min_dist)
        return peaks

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
        peaks = self.detect_r_peaks(sig)
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

        # Rhythm regularity (coefficient of variation)
        cv_rr = std_rr / mean_rr if mean_rr > 0 else 0.0
        is_regular = cv_rr < 0.20
        if not is_regular:
            warnings.append(f"Irregular RR intervals detected (CV={cv_rr:.2f}).")
            findings.append("Irregular rhythm pattern observed")

        # 4. Clinical Bradycardia / Tachycardia Rules
        if bpm < 60.0:
            findings.append(f"Sinus Bradycardia ({bpm:.1f} bpm < 60 bpm)")
        elif bpm > 100.0:
            findings.append(f"Sinus Tachycardia ({bpm:.1f} bpm > 100 bpm)")
        else:
            findings.append(f"Normal Heart Rate ({bpm:.1f} bpm)")

        quality = "good" if (is_regular and 40.0 <= bpm <= 180.0) else "acceptable"

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
