"""Signal Calibration, Digital Reconstruction and Export.

Maps pixel displacements into physical physiological units (mV and seconds),
resamples to target sampling frequencies (e.g. 100 Hz or 500 Hz),
and exports reconstructed signals to CSV and diagnostic plots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ecg_digitization.reconstruction.extractor import ExtractedTrace

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Clinical ECG calibration constants."""

    speed_mm_s: float = 25.0       # Standard paper speed: 25 mm/s
    gain_mm_mv: float = 10.0       # Standard voltage gain: 10 mm/mV
    dpi: int = 150                 # Scanning resolution in dots per inch
    pixels_per_mm: Optional[float] = None
    target_fs_hz: float = 100.0    # Target digital sampling rate in Hz

    def __post_init__(self) -> None:
        if self.pixels_per_mm is None:
            self.pixels_per_mm = self.dpi / 25.4
        self.pixels_per_sec = self.speed_mm_s * self.pixels_per_mm
        self.pixels_per_mv = self.gain_mm_mv * self.pixels_per_mm


@dataclass
class ReconstructedLead:
    """Calibrated digital signal for a single lead."""

    lead_name: str
    voltage_mv: np.ndarray        # 1D array of voltages in mV
    time_seconds: np.ndarray      # 1D array of timestamps in seconds
    sampling_rate_hz: float
    duration_seconds: float
    completeness: float


@dataclass
class ReconstructedECG:
    """Multi-lead reconstructed digital ECG."""

    leads: Dict[str, ReconstructedLead]
    sampling_rate_hz: float
    layout_name: str
    calibration: CalibrationConfig

    def to_dataframe(self) -> pd.DataFrame:
        """Converts synchronized leads into a pandas DataFrame."""
        max_len = max(len(lead.voltage_mv) for lead in self.leads.values())
        time_axis = np.arange(max_len) / self.sampling_rate_hz

        data: Dict[str, np.ndarray] = {"time_s": time_axis}
        for name, lead in self.leads.items():
            if len(lead.voltage_mv) == max_len:
                data[name] = lead.voltage_mv
            else:
                padded = np.full(max_len, np.nan, dtype=np.float32)
                padded[: len(lead.voltage_mv)] = lead.voltage_mv
                data[name] = padded

        return pd.DataFrame(data)

    def save_csv(self, output_path: Union[str, Path]) -> Path:
        """Exports digital ECG waveforms to a CSV file."""
        df = self.to_dataframe()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, float_format="%.4f")
        return out

    def save_plot(self, output_path: Union[str, Path], title: str = "Reconstructed 12-Lead ECG") -> Path:
        """Exports a multi-panel plot of the reconstructed ECG leads."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lead_keys = [k for k in self.leads.keys() if k != "II_RHYTHM"]
        num_leads = len(lead_keys)

        fig, axes = plt.subplots(num_leads, 1, figsize=(10, 1.2 * max(num_leads, 1)), sharex=True)
        if num_leads == 1:
            axes = [axes]

        for ax, name in zip(axes, lead_keys):
            lead = self.leads[name]
            ax.plot(lead.time_seconds, lead.voltage_mv, color="#1f77b4", linewidth=1.0)
            ax.set_ylabel(f"{name}\n(mV)", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_ylim(-1.5, 2.0)

        axes[-1].set_xlabel("Time (seconds)", fontsize=9)
        fig.suptitle(title, fontsize=11, y=0.99)
        plt.tight_layout()
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out


class SignalReconstructor:
    """Calibrates pixel trajectories into digital voltages and resamples to target Hz."""

    def __init__(self, calibration: Optional[CalibrationConfig] = None) -> None:
        self.cal = calibration or CalibrationConfig()

    def reconstruct_lead(self, trace: ExtractedTrace, lead_duration_s: Optional[float] = None) -> ReconstructedLead:
        """Calibrates a single extracted lead trace into physical mV and resampled Hz."""
        raw_mv = trace.relative_trajectory / self.cal.pixels_per_mv
        num_px = len(raw_mv)

        # Time duration based on pixel width
        actual_duration_s = num_px / self.cal.pixels_per_sec
        duration_s = lead_duration_s if lead_duration_s is not None else actual_duration_s

        # Target sample points
        target_num_samples = int(round(duration_s * self.cal.target_fs_hz))
        if target_num_samples <= 1:
            target_num_samples = 2

        orig_time = np.linspace(0, duration_s, num_px)
        target_time = np.linspace(0, duration_s, target_num_samples)

        interp_func = interp1d(orig_time, raw_mv, kind="linear", fill_value="extrapolate")
        resampled_mv = interp_func(target_time).astype(np.float32)

        return ReconstructedLead(
            lead_name=trace.lead_name,
            voltage_mv=resampled_mv,
            time_seconds=target_time,
            sampling_rate_hz=self.cal.target_fs_hz,
            duration_seconds=duration_s,
            completeness=trace.completeness_ratio,
        )

    def reconstruct_ecg(
        self,
        extracted_traces: Dict[str, ExtractedTrace],
        layout_name: str = "3x4_1",
    ) -> ReconstructedECG:
        """Reconstructs all leads in an ECG record into synchronized physical units."""
        leads: Dict[str, ReconstructedLead] = {}

        for name, trace in extracted_traces.items():
            # In standard 3x4 layout, grid leads are 2.5 seconds each, rhythm strip is 10.0 seconds
            if layout_name == "3x4_1":
                duration = 10.0 if "RHYTHM" in name.upper() else 2.5
            else:
                duration = None

            lead = self.reconstruct_lead(trace, lead_duration_s=duration)
            leads[name] = lead

        return ReconstructedECG(
            leads=leads,
            sampling_rate_hz=self.cal.target_fs_hz,
            layout_name=layout_name,
            calibration=self.cal,
        )
