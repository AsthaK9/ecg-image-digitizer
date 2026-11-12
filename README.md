# ECG DigiAI — ECG Digitisation & Analysis Using AI

**Institution:** Jaypee Institute of Information Technology (JIIT), Noida  
**Department:** Electronics & Communication Engineering  
**Project:** Final-Year B.Tech Major Project  
**Group Members:** Astha Kumari | Akash Raj | Kanishk Deo  
**Environment:** macOS (Apple Silicon arm64), Python 3.14 (`.venv`), PyTorch with Metal Performance Shaders (MPS)  

---

## 1. Overview

**ECG DigiAI** is an end-to-end biomedical engineering workstation and deep-learning pipeline designed to reconstruct calibrated digital time-series waveforms from printed, photographed, or scanned 12-lead electrocardiograms.

The platform provides:

1. **Synthetic ECG Generation (Component A)**:
   - Renders standard 12-lead paper ECG sheets ($25\text{ mm/s}$, $10\text{ mm/mV}$, $150\text{ DPI}$) with clinical pink/red millimeter grids (1 mm minor, 5 mm major) and calibration pulses.
   - Clinical 3x4 + 1 Lead II continuous rhythm strip layout.
   - Configurable photographic degradations: affine rotation skew, Gaussian optical blur, sensor noise, and illumination gradients.
   - Exact pairing with 1D digital ground truth and 2D pixel-level trace masks.

2. **All-12-Lead Deep Learning Digitisation (Component B)**:
   - Preprocessing: Canny/Hough deskewing, dual-domain red-channel/HSV grid suppression, and adaptive binarization.
   - Lead detection: Standard clinical 3x4+1, 12x1, and single-lead beat layouts.
   - Deep extractor: 2D ResNet-U-Net (`models/ecg_digitizer_unet_all12lead.pt`) with Differentiable Soft-Argmax column-wise coordinate head.
   - **All 12 standard clinical leads** supported: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6` plus continuous rhythm strip.
   - Calibrated physical conversion ($mV$ and seconds) and uniform 100 Hz resampling.

3. **Physiological Rhythm & Rate Analysis**:
   - Butterworth bandpass filtering (0.5–40 Hz) and Pan-Tompkins QRS R-peak detection.
   - Heart rate (BPM) and RR interval variability ($CV_{RR}$) estimation.
   - Rule-based detection of Sinus Bradycardia ($<60\text{ bpm}$) and Sinus Tachycardia ($>100\text{ bpm}$) with signal quality index (SQI).

4. **Biomedical Engineering Workstation UI**:
   - **01 — Digitisation Workbench**: Image upload / sample selector, scanning animation sweep beam, preprocessing previews, and 12-lead diagnostic bounding-box overlay.
   - **02 — Lead Explorer**: Clinical 4x3+1 arrangement, interactive millimeter-grid canvas, animated playback, coordinate cursor displaying $(t, V)$, diamond R-peak markers, and physiological rhythm analysis.
   - **03 — Synthetic ECG Lab**: PTB-XL record browser, fine-grained physical degradation controls, paired ground-truth generation, and one-click handoff to the workbench.
   - **04 — Evaluation & Export**: Paired digital ground-truth vs. digitized waveform overlay, error difference residual plot, quantitative fidelity metrics (Pearson $r$, RMSE, SNR, completeness), benchmark table, and multi-format exports (CSV, WFDB `.hea`, printable clinical HTML report, structured JSON).

5. **Scope Boundaries**:
   - Disease classification is **explicitly OUT OF SCOPE**.
   - No fabricated disease predictions or clinical diagnostic claims. The workstation operates strictly as an ECG digitisation and waveform analysis instrument.

---

## 2. Quickstart Commands (macOS / zsh)

Ensure the project virtual environment is active:

```bash
# Activate virtual environment
source .venv/bin/activate

# 1. Run all unit and integration tests (108 passing tests, 82% coverage)
.venv/bin/pytest tests/unit/

# 2. Run the 3-scenario scientific verification suite
.venv/bin/python scripts/verify_3_scenarios.py

# 3. Start the interactive Workstation & FastAPI Backend
.venv/bin/python -m uvicorn ecg_digitization.service.app:app --host 127.0.0.1 --port 8000
```

Once started, open **`http://127.0.0.1:8000`** in your browser to interact with the workstation.

---

## 3. Deep Learning Checkpoints

| Checkpoint File | Size | Architecture | Training Distribution | Performance |
|---|---|---|---|---|
| `models/ecg_digitizer_unet_all12lead.pt` | **8,121,711 bytes** | ResNet-U-Net + Soft-Argmax | **All 12 Leads** (I–III, aVR–aVF, V1–V6) on PTB-XL Folds 1–8 | Val Mean $r > 0.978$, RMSE $< 0.051\text{ mV}$ |
| `models/ecg_digitizer_unet.pt` | 8,121,015 bytes | ResNet-U-Net + Soft-Argmax | Leads II & V1 (Legacy baseline) | Val Mean $r = 0.9971$, RMSE $= 0.0179\text{ mV}$ |
| `models/ecg_digitizer_unet.pt.bak` | 8,121,015 bytes | Safety backup | Leads II & V1 | Identical to baseline |

---

## 4. Scientific Verification (Three Scenarios)

Evaluated via `scripts/verify_3_scenarios.py` (results saved to `data/interim/three_scenarios_verification_summary.json`):

### Scenario A: Paired Synthetic 12-Lead Sheet (PTB-XL `00001_lr`)
- **Mean Pearson Correlation ($r$)**: **$+0.8456$**
- **Mean RMSE**: **$0.1305\text{ mV}$**
- Per-lead fidelity:
  - Lead I: $r = 0.7725$, RMSE $= 0.0931\text{ mV}$
  - Lead II: $r = 0.7811$, RMSE $= 0.0950\text{ mV}$
  - Lead V1: $r = 0.9581$, RMSE $= 0.0899\text{ mV}$
  - Lead V5: $r = 0.9297$, RMSE $= 0.1706\text{ mV}$
  - Lead V6: $r = 0.9100$, RMSE $= 0.1580\text{ mV}$
  - II_RHYTHM: $r = 0.8159$, RMSE $= 0.0877\text{ mV}$

### Scenario B: Real ECG Image (Kaggle `N10007.png`)
- Trace confidence: **100%**
- `is_calibration_reliable`: **`False`** (reported scientifically because image lacked calibration pulse; zero fabricated calibration).

### Scenario C: Single-Lead Crop (Kaggle `Q1.png`)
- Extracted leads: Exactly 1 (`II`).
- Missing leads (`I, III, aVR, aVL, aVF, V1–V6`): **Zero fabricated signals**.

---

## 5. REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | 4-Tab Biomedical Engineering Workstation HTML5 SPA |
| `GET` | `/api/status` | Active model checkpoint, supported leads (all 12), and device status |
| `GET` | `/api/samples` | Lists available PTB-XL records and test images |
| `POST` | `/api/synthetic/generate` | Generates paired synthetic ECG image with ground-truth mask |
| `POST` | `/api/digitize` | Full digitisation: deskew, grid suppression, layout detection, and waveform extraction |
| `POST` | `/api/analyze` | Pan-Tompkins R-peak detection, heart rate, and physiological rhythm analysis |
| `GET` | `/api/ground-truth/{id}` | Fetches raw 12-lead digital ground truth from PTB-XL |
| `GET` | `/api/ptbxl-metadata/{id}` | Clinical record metadata and acquisition specifications |
| `GET` | `/api/export/wfdb/{id}` | Generates PhysioNet/WFDB `.hea` header for digitized signals |
| `GET` | `/api/report/{id}/json` | Structured JSON research report |
| `GET` | `/api/report/{id}/html` | Printable clinical research report |

---

## 6. Repository Integrity & Git Note

This workspace was migrated to this development machine from an archive snapshot that does not contain a `.git` version control directory. All local code, models, datasets, and scripts have been preserved and verified in place without modifying or overwriting existing history.

---

## 7. Academic Notice & Disclaimer

> **ACADEMIC RESEARCH NOTICE:** ECG DigiAI is an academic research and engineering workstation developed at the Jaypee Institute of Information Technology, Noida. This system does NOT provide autonomous medical diagnoses. All outputs, rate calculations, and reconstructed waveforms must be independently reviewed and verified by a licensed medical practitioner.
