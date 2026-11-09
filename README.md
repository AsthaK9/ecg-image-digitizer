# ECG DigiAI — ECG Digitisation & Analysis Using AI

**Institution:** Jaypee Institute of Information Technology, Noida  
**Department:** Electronics & Communication Engineering  
**Project:** Final-Year B.Tech Major Project  
**Authors & Maintainers:** Astha K. & Senior Implementation Team  

---

## Overview

**ECG DigiAI** is an end-to-end clinical decision-support and computer vision platform designed to bridge the gap between legacy paper electrocardiograms and modern digital cardiology workflows. The platform provides:

1. **Synthetic ECG Image Generation**: Renders realistic 12-lead paper ECG sheets ($25\text{ mm/s}$, $10\text{ mm/mV}$, $150\text{ DPI}$) with precise millimeter grids (1 mm minor, 5 mm major), calibration pulses, and configurable photographic degradations (rotation skew, perspective distortion, blur, noise, and shadow gradients) while preserving paired ground-truth signals and binary trace masks.
2. **ECG Image Digitisation Pipeline**:
   - Automated Hough line deskewing and orientation correction.
   - Dual-domain colored grid suppression (Red channel extraction + HSV color gating) to remove pink/red paper grid lines while preserving black trace ink.
   - Geometry-based lead region localization (Standard 3x4 + 1 continuous Lead II rhythm strip and 12x1 layouts).
   - Active-span column-wise centroid and median waveform extraction with gap interpolation.
   - Digital calibration, physical unit conversion (mV and seconds), and resampling ($100\text{ Hz}$).
3. **Physiological Rhythm & Rate Analysis**:
   - Bandpass filtering (0.5–40 Hz) and Pan-Tompkins QRS R-peak detection.
   - Heart rate (BPM) and RR interval variability ($CV_{RR}$) estimation.
   - Rule-based detection of Sinus Bradycardia ($<60\text{ bpm}$) and Sinus Tachycardia ($>100\text{ bpm}$) with signal quality index warnings.
4. **AI Multi-Label Pathology Screening Interface**:
   - `ECGResNet1D` 12-lead deep residual network baseline.
   - Clinical decision-support safety discipline: returns explicit `model_not_trained` status until trained weights are loaded.
   - Unsupported target discipline: flags sparse classes (`STE_`) as `Not supported by available labels`.
   - Never fabricates synthetic predictions or confidence scores.
5. **Explainable Decision-Support Reports**:
   - Structured machine-readable JSON reports.
   - Printable clinical HTML reports with embedded multi-lead plots, findings, and regulatory disclaimers.
   - Clinical Risk Level explicitly marked `Not assessed` pending prospective clinical validation.
6. **Kaggle GPU Training Infrastructure**:
   - Automated packaging of `kernel-metadata.json` and standalone training script for multi-label GPU training without autonomous submission risks.

---

## Quickstart & Demo Commands (Windows PowerShell)

Ensure `.venv` is activated:

```powershell
# 1. Run all unit and integration tests (87 passing tests, 85% coverage)
.venv\Scripts\pytest

# 2. Run the end-to-end verification script with ground-truth fidelity report
.venv\Scripts\python scripts\verify_session1_pipeline.py

# 3. Start the interactive ECG DigiAI Web Application & FastAPI backend
.venv\Scripts\python -m uvicorn ecg_digitization.service.app:app --host 127.0.0.1 --port 8000
```

Once running, navigate to **`http://127.0.0.1:8000`** in your browser to access the 6-screen interactive dashboard.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Single-page interactive web application |
| `GET` | `/api/status` | System health, module status, and Kaggle readiness |
| `GET` | `/api/samples` | Lists available PTB-XL records and test images |
| `POST` | `/api/synthetic/generate` | Generates synthetic ECG image with ground-truth mask |
| `POST` | `/api/digitize` | Runs deskew, grid suppression, and waveform extraction |
| `POST` | `/api/analyze` | Computes heart rate, rhythm rules, and AI screening |
| `GET` | `/api/report/{id}/json` | Generates structured JSON clinical report |
| `GET` | `/api/report/{id}/html` | Generates printable clinical HTML report |
| `GET` | `/api/kaggle/status` | Verifies Kaggle CLI and generates dry-run push command |

---

## Verification & Reconstruction Metrics

Evaluated on PTB-XL Record `00001_lr` under photographic degradation (rotation skew + Gaussian noise):

- **Lead I Pearson Correlation ($r$)**: **$+0.8754$**
- **Lead II Pearson Correlation ($r$)**: **$+0.9058$**
- **Lead V2 Pearson Correlation ($r$)**: **$+0.6396$**
- **Mean Root Mean Squared Error (RMSE)**: **$0.2145\text{ mV}$**
- **Mean Trace Completeness**: **$100.0\%$**

---

## Academic Notice & Clinical Disclaimer

> **DECISION-SUPPORT NOTICE:** ECG DigiAI is an academic research and clinical decision-support system developed at the Jaypee Institute of Information Technology, Noida. This system does NOT provide autonomous medical diagnoses. All outputs, rate calculations, and reconstructed waveforms must be independently reviewed and verified by a licensed medical practitioner.
