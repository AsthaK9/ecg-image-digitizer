# ECG DigiAI — Live Demonstration & Verification Guide

Exact Windows PowerShell instructions to demonstrate the complete ECG DigiAI system.

---

## 1. Prerequisites Check

From the project root (`C:\Projects\ecg-image-digitizer`):

```powershell
# Verify Python version (Python 3.14.x)
.venv\Scripts\python.exe --version

# Verify Git
git status
```

---

## 2. Automated Test Suite Execution

Execute the full suite of 87 unit and integration tests (validating Hydra configs, data modules, synthetic generation, preprocessing, reconstruction, classifier safety, and API service handlers):

```powershell
.venv\Scripts\pytest
```

*Expected Result: `87 passed, 177 warnings` in approximately 10-12 seconds with ~85% code coverage.*

---

## 3. End-to-End Pipeline Verification Script

Execute the full verification script that processes PTB-XL record `00001_lr`, renders a degraded synthetic 12-lead paper ECG sheet, runs deskewing, colored grid suppression, lead detection, waveform extraction, digital reconstruction, and quantitative ground-truth comparison:

```powershell
.venv\Scripts\python scripts\verify_session1_pipeline.py
```

*Expected Outputs Saved:*
- `data/interim/session1_demo_output/preprocessing/demo_00001_03_binary_trace.png`
- `data/interim/session1_demo_output/demo_00001_signal.csv`
- `data/interim/session1_demo_output/demo_00001_lead_regions.png`
- `data/interim/session1_demo_output/demo_00001_plot.png`
- `data/interim/session1_demo_output/reconstruction_metrics.json`

---

## 4. Starting the Interactive Web Application

Start the local FastAPI service on port 8000:

```powershell
.venv\Scripts\python -m uvicorn ecg_digitization.service.app:app --host 127.0.0.1 --port 8000
```

Open your browser to:
**`http://127.0.0.1:8000`**

### Demonstration Walkthrough:
1. **Screen 1 (Dashboard)**: Review project identity (JIIT Noida), active pipeline status, and ground-truth verification fidelity metrics ($r \approx 0.90$).
2. **Screen 2 (Synthetic Generator)**: Select `00001_lr`, adjust rotation skew and noise, and click **Render Synthetic ECG Paper Sheet**. Click **Send to Digitizer**.
3. **Screen 3 (ECG Digitisation)**: Click **Run End-to-End Digitisation**. Observe the intermediate stages (Deskewed, Grid Suppressed, Binary Trace) and detected lead bounding boxes.
4. **Screen 4 (Reconstructed Waveforms)**: View the 12-lead synchronized ECG plot and live ground-truth comparison metrics. Click **Download Reconstructed CSV**.
5. **Screen 5 (AI Analysis & Screening)**: Click **Run Clinical Analysis**. Inspect the calculated Heart Rate (BPM), rhythm regularity, rule findings (Bradycardia / Tachycardia), and the multi-label table showing `model_not_trained` (no fake confidence scores) and `Not supported by available labels`.
6. **Screen 6 (Clinical Report)**: Inspect the embedded clinical decision-support report. Click **Print / Save HTML** or **Download JSON Report**.

---

## 5. Kaggle GPU Training Package Dry-Run

Verify the prepared Kaggle GPU training submission package without pushing to remote:

```powershell
.venv\Scripts\python -c "from ecg_digitization.training.kaggle_manager import KaggleManager; km = KaggleManager(); card = km.prepare_kernel('artifacts/kaggle_kernel_bundle'); print('Dry-run submission command:'); print(card.dry_run_command)"
```
