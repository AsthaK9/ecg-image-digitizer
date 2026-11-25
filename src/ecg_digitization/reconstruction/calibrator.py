"""Signal Calibration, Digital Reconstruction and Export.

Maps pixel displacements into physical physiological units (mV and seconds),
resamples to target sampling frequencies (e.g. 100 Hz or 500 Hz),
and exports reconstructed signals to CSV and diagnostic plots.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator

from ecg_digitization.data.types import CanonicalECGSignal, LeadSignal
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
    is_trained_lead: bool = False
    validation_status: str = "unvalidated"
    confidence: float = 1.0
    is_calibration_reliable: bool = True
    calibration_note: str = ""


@dataclass
class ReconstructedECG:
    """Multi-lead reconstructed digital ECG."""

    leads: Dict[str, ReconstructedLead]
    sampling_rate_hz: float
    layout_name: str
    calibration: CalibrationConfig
    is_calibration_reliable: bool = True
    calibration_note: str = ""

    # Standard clinical lead sequence
    STANDARD_LEAD_ORDER = [
        "I", "II", "III",
        "AVR", "AVL", "AVF",
        "V1", "V2", "V3",
        "V4", "V5", "V6",
        "II_RHYTHM",
    ]

    def to_canonical(self, record_id: str = "reconstructed_ecg", source_type: str = "digitized_ecg") -> CanonicalECGSignal:
        """Converts ReconstructedECG to CanonicalECGSignal container."""
        canonical_leads: Dict[str, LeadSignal] = {}
        for name, lead in self.leads.items():
            canonical_leads[name] = LeadSignal(
                lead_name=lead.lead_name,
                voltage=lead.voltage_mv,
                time_seconds=lead.time_seconds,
                sampling_rate_hz=lead.sampling_rate_hz,
                units="mV" if lead.is_calibration_reliable else "relative_units",
                is_calibrated=lead.is_calibration_reliable,
                completeness=lead.completeness,
                is_trained_lead=lead.is_trained_lead,
                validation_status=lead.validation_status,
                confidence=lead.confidence,
            )
        return CanonicalECGSignal(
            record_id=record_id,
            leads=canonical_leads,
            sampling_rate_hz=self.sampling_rate_hz,
            layout_name=self.layout_name,
            units="mV" if self.is_calibration_reliable else "relative_units",
            is_calibration_reliable=self.is_calibration_reliable,
            calibration_standard="Calibrated Physical Voltage (mV)" if self.is_calibration_reliable else "Uncalibrated / Normalized Amplitude",
            calibration_note=self.calibration_note,
            source_type=source_type,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Converts synchronized leads into a pandas DataFrame in standard clinical ordering."""
        max_len = max(len(lead.voltage_mv) for lead in self.leads.values()) if self.leads else 0
        time_axis = np.arange(max_len) / self.sampling_rate_hz if max_len > 0 else np.array([])

        data: Dict[str, np.ndarray] = {"time_s": time_axis}
        ordered_keys = [k for k in self.STANDARD_LEAD_ORDER if k in self.leads] + [
            k for k in self.leads if k not in self.STANDARD_LEAD_ORDER
        ]

        for name in ordered_keys:
            lead = self.leads[name]
            if len(lead.voltage_mv) == max_len:
                data[name] = lead.voltage_mv
            else:
                padded = np.full(max_len, np.nan, dtype=np.float32)
                padded[: len(lead.voltage_mv)] = lead.voltage_mv
                data[name] = padded

        return pd.DataFrame(data)

    def save_csv(self, output_path: Union[str, Path]) -> Path:
        """Exports digital ECG waveforms to a unified CSV file with provenance metadata."""
        out = Path(output_path)
        export_to_csv(self, output_path=out)
        return out

    def save_plot(self, output_path: Union[str, Path], title: str = "Reconstructed 12-Lead ECG") -> Path:
        """Exports a multi-panel plot of the reconstructed ECG leads."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        ordered_keys = [k for k in self.STANDARD_LEAD_ORDER if k in self.leads] + [
            k for k in self.leads if k not in self.STANDARD_LEAD_ORDER
        ]
        num_leads = len(ordered_keys)

        # Proportional figure sizing: single lead gets 4.5 inches height, multi-lead gets 1.5 in/lead
        fig_height = 4.5 if num_leads == 1 else max(3.5, 1.5 * num_leads)
        fig, axes = plt.subplots(num_leads, 1, figsize=(11, fig_height), sharex=False)
        if num_leads == 1:
            axes = [axes]

        cal_status_str = "Physical Calibrated (mV)" if self.is_calibration_reliable else "Nominal Voltage (mV equiv)"

        for ax, name in zip(axes, ordered_keys):
            lead = self.leads[name]
            trace_color = "#0f766e" if lead.is_trained_lead else "#1a73e8"
            ax.plot(lead.time_seconds, lead.voltage_mv, color=trace_color, linewidth=1.3)

            tag = "[Validated]" if lead.is_trained_lead else "[Experimental]"
            ax.set_ylabel(f"{name}\n{tag}", fontsize=8, color="#0f172a", fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")

            # Adaptive y-limits preserving zero baseline and signal extrema without clipping
            vals = lead.voltage_mv
            if len(vals) > 0:
                v_min = float(np.min(vals))
                v_max = float(np.max(vals))
                span = max(0.5, v_max - v_min)
                margin = max(0.15, span * 0.12)
                ax.set_ylim(min(v_min - margin, -0.2), max(v_max + margin, 0.2))
            else:
                ax.set_ylim(-1.0, 1.0)

            # Mark zero baseline
            ax.axhline(0, color="#94a3b8", linestyle="-", linewidth=0.8, alpha=0.7)

        axes[-1].set_xlabel(f"Time (seconds) — {cal_status_str}", fontsize=9, color="#475569")
        full_title = f"{title}\nGreen: Formally Validated Leads (II, V1) | Blue: Experimental Leads | Status: {cal_status_str}"
        fig.suptitle(full_title, fontsize=10, y=0.995, color="#0f172a")
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

        target_time = np.arange(target_num_samples, dtype=np.float32) / self.cal.target_fs_hz

        if num_px == target_num_samples:
            resampled_mv = raw_mv.astype(np.float32)
        elif num_px > 2:
            orig_time = np.linspace(0.0, float(duration_s), num_px, dtype=np.float64)
            pchip = PchipInterpolator(orig_time, raw_mv.astype(np.float64), extrapolate=True)
            resampled_mv = pchip(target_time.astype(np.float64)).astype(np.float32)
        else:
            orig_time = np.linspace(0.0, float(duration_s), num_px, dtype=np.float32)
            interp_func = interp1d(orig_time, raw_mv, kind="linear", fill_value="extrapolate")
            resampled_mv = interp_func(target_time).astype(np.float32)

        return ReconstructedLead(
            lead_name=trace.lead_name,
            voltage_mv=resampled_mv,
            time_seconds=target_time,
            sampling_rate_hz=self.cal.target_fs_hz,
            duration_seconds=duration_s,
            completeness=trace.completeness_ratio,
            is_trained_lead=trace.is_trained_lead,
            validation_status=trace.validation_status,
            confidence=trace.confidence,
            is_calibration_reliable=getattr(self.cal, "is_reliable", True),
        )

    def reconstruct_ecg(
        self,
        extracted_traces: Dict[str, ExtractedTrace],
        layout_name: str = "3x4_1",
        is_calibration_reliable: bool = True,
        calibration_note: str = "",
    ) -> ReconstructedECG:
        """Reconstructs all leads in an ECG record into synchronized physical units."""
        leads: Dict[str, ReconstructedLead] = {}

        for name, trace in extracted_traces.items():
            # In standard 3x4 layouts, grid leads are 2.5 seconds each, rhythm strips and 12x1 rows are 10.0 seconds
            if layout_name in ("3x4", "3x4_1", "3x4_3"):
                duration = 10.0 if "RHYTHM" in name.upper() else 2.5
            elif layout_name == "12x1":
                duration = 10.0
            else:
                duration = None

            lead = self.reconstruct_lead(trace, lead_duration_s=duration)
            lead.is_calibration_reliable = is_calibration_reliable
            lead.calibration_note = calibration_note
            leads[name] = lead

        return ReconstructedECG(
            leads=leads,
            sampling_rate_hz=self.cal.target_fs_hz,
            layout_name=layout_name,
            calibration=self.cal,
            is_calibration_reliable=is_calibration_reliable,
            calibration_note=calibration_note,
        )


def export_to_csv(
    signal: Union[CanonicalECGSignal, ReconstructedECG],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Exports a CanonicalECGSignal or ReconstructedECG to a unified CSV format with provenance metadata."""
    if isinstance(signal, ReconstructedECG):
        canonical = signal.to_canonical()
    else:
        canonical = signal

    df = canonical.to_dataframe()
    validated_leads = [k for k, l in canonical.leads.items() if l.is_trained_lead]

    header_lines = [
        "# ECG DigiAI — Unified Reconstructed ECG Waveform Record\n",
        f"# Layout: {canonical.layout_name} | Sampling Rate: {canonical.sampling_rate_hz:.1f} Hz | Calibration Reliable: {canonical.is_calibration_reliable}\n",
        f"# Calibration Standard: {canonical.calibration_standard}\n",
        f"# Calibration Note: {canonical.calibration_note or ('Standard clinical paper calibration (25 mm/s, 10 mm/mV)' if canonical.is_calibration_reliable else 'Uncalibrated input: voltage represents relative deflection')}\n",
        f"# Validated Leads: {', '.join(validated_leads) if validated_leads else 'None'}\n",
        f"# Leads Present ({len(canonical.leads)}): {', '.join(canonical.lead_names)}\n",
    ]

    csv_body = df.to_csv(index=False, float_format="%.4f")
    full_text = "".join(header_lines) + csv_body

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(full_text, encoding="utf-8")
        logger.info("Saved unified multi-lead CSV: %s", out)

    return full_text


def import_from_csv(
    source: Union[str, Path, io.StringIO, io.TextIOBase],
    record_id: Optional[str] = None,
) -> CanonicalECGSignal:
    """Imports an ECG signal CSV file or text with exact round-trip preservation."""
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
        rec_id = record_id or source.stem.replace("_signal", "")
    elif isinstance(source, str):
        p = Path(source)
        if len(source) < 500 and "\n" not in source and p.is_file():
            text = p.read_text(encoding="utf-8")
            rec_id = record_id or p.stem.replace("_signal", "")
        else:
            text = source
            rec_id = record_id or "imported_ecg"
    else:
        text = source.read()
        rec_id = record_id or "imported_ecg"

    lines = text.splitlines()
    header_meta: Dict[str, Any] = {}
    csv_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            content = stripped.lstrip("#").strip()
            if "|" in content:
                parts = [p.strip() for p in content.split("|")]
                for part in parts:
                    if ":" in part:
                        k, v = part.split(":", 1)
                        header_meta[k.strip().lower()] = v.strip()
            elif ":" in content:
                k, v = content.split(":", 1)
                header_meta[k.strip().lower()] = v.strip()
        else:
            if stripped:
                csv_lines.append(line)

    if not csv_lines:
        raise ValueError("CSV contains no data rows.")

    csv_data = "\n".join(csv_lines)
    df = pd.read_csv(io.StringIO(csv_data))

    layout_name = header_meta.get("layout", "auto")
    fs_val = 100.0
    if "sampling rate" in header_meta:
        m = re.search(r"([\d\.]+)", header_meta["sampling rate"])
        if m:
            fs_val = float(m.group(1))

    is_cal_reliable = True
    if "calibration reliable" in header_meta:
        is_cal_reliable = header_meta["calibration reliable"].lower() in ("true", "1", "yes")

    cal_standard = header_meta.get(
        "calibration standard",
        "Calibrated Physical Voltage (mV)" if is_cal_reliable else "Uncalibrated / Normalized Amplitude",
    )
    cal_note = header_meta.get("calibration note", "Imported from CSV")

    validated_leads_str = header_meta.get("validated leads", "")
    validated_leads_set = {
        l.strip()
        for l in validated_leads_str.split(",")
        if l.strip() and l.strip().lower() != "none"
    }

    # Extract time column
    time_col = None
    for c in ("time_s", "time", "t_s", "t", "timestamp"):
        if c in df.columns:
            time_col = c
            break

    if time_col is not None:
        time_axis = df[time_col].to_numpy(dtype=np.float32)
        if len(time_axis) > 1 and "sampling rate" not in header_meta:
            dt = np.median(np.diff(time_axis))
            if dt > 0:
                fs_val = round(1.0 / float(dt), 1)
        lead_cols = [c for c in df.columns if c != time_col]
    else:
        time_axis = (np.arange(len(df), dtype=np.float32) / fs_val)
        lead_cols = list(df.columns)

    if not lead_cols:
        raise ValueError("No lead columns found in CSV.")

    if layout_name in ("auto", ""):
        layout_name = "single_lead" if len(lead_cols) == 1 else "3x4_1"

    leads: Dict[str, LeadSignal] = {}
    for col_name in lead_cols:
        raw_vals = df[col_name].to_numpy(dtype=np.float32)
        valid_mask = ~np.isnan(raw_vals)
        if not np.any(valid_mask):
            continue
        last_valid = int(np.where(valid_mask)[0][-1])
        vals = raw_vals[: last_valid + 1]
        t_lead = time_axis[: len(vals)]

        is_trained = col_name in validated_leads_set if validated_leads_set else True
        units = "mV" if is_cal_reliable else "relative_units"

        leads[col_name] = LeadSignal(
            lead_name=col_name,
            voltage=vals,
            time_seconds=t_lead,
            sampling_rate_hz=fs_val,
            units=units,
            is_calibrated=is_cal_reliable,
            completeness=float(np.mean(valid_mask[: len(vals)])),
            is_trained_lead=is_trained,
            validation_status="calibrated_lead" if is_cal_reliable else "uncalibrated_lead",
            confidence=1.0,
        )

    return CanonicalECGSignal(
        record_id=rec_id,
        leads=leads,
        sampling_rate_hz=fs_val,
        layout_name=layout_name,
        units="mV" if is_cal_reliable else "relative_units",
        is_calibration_reliable=is_cal_reliable,
        calibration_standard=cal_standard,
        calibration_note=cal_note,
        source_type="imported_csv",
        metadata={"header_comments": header_meta},
    )
