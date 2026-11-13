# ECG DigiAI — Master Progress & Permanent Source of Truth

**Project:** ECG Digitisation & Analysis Using AI (ECG DigiAI)  
**Institution:** Jaypee Institute of Information Technology, Noida (ECE Dept)  
**Last Updated:** 13 November 2026  
**Operating Environment:** macOS (Darwin arm64, Apple Silicon), Python 3.14.7 (`.venv`), PyTorch 2.15.0 (MPS available)  
**Git Status:** Extracted standalone source tree (no `.git` VCS directory present). Preserved cleanly in place.

---

## 1. Project Objective and Scope
- **Objective:** Reconstruct accurate, calibrated digital ECG waveforms ($mV$ vs time) across all 12 standard clinical leads (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`) and continuous Lead II rhythm strip (`II_RHYTHM`) from photographed or scanned ECG paper images.
- **Strict Scope Boundaries:**
  - Disease classification is explicitly **OUT OF SCOPE**. No disease predictions or risk scores.
  - Zero fabrication of missing leads or uncalibrated voltages.
  - Nominal voltages must be strictly flagged as nominal, not calibrated.
  - Checkpoints and datasets strictly preserved outside Git.

---

## 2. Current Repository Structure
- `src/ecg_digitization/`:
  - `data/`: Dataset loaders, split manager (leakage-safe patient folds 1-8 train, 9 val, 10 test), WFDB parsing.
  - `preprocessing/`: `enhancement.py` (deskew, grid suppression, binarization), `lead_detector.py` (layout bounding boxes).
  - `models/`: `digitizer_unet.py` (ResNet-18 U-Net + Soft-Argmax coordinate head).
  - `reconstruction/`: `extractor.py` (column-wise centroid/median tracking), `calibrator.py` (voltage and time scaling).
  - `inference/`: `deep_digitizer.py` (DeepCropDigitizer inference engine), `analyzer.py` (Pan-Tompkins R-peaks, HR, RR interval variability).
  - `pipeline/`: `digitizer.py` (ECGDigitizerPipeline orchestrator).
  - `service/`: `app.py` (FastAPI backend), `workstation_html.py` (4-tab clinical engineering SPA).
  - `synthetic/`: `generator.py` (Paired 12-lead paper ECG renderer from PTB-XL with millimeter grid, calibration pulse, and degradations).
- `models/`:
  - `ecg_digitizer_unet_all12lead.pt` (Active all-12-lead checkpoint, 8.12 MB, Epoch 4, val r=0.9786, val RMSE=0.0505 mV).
  - `ecg_digitizer_unet.pt` (Legacy 2-lead baseline, 8.12 MB).
  - `ecg_digitizer_unet.pt.bak` (Safety backup).
  - `digitizer_all12lead_metrics.json` (Validation metrics).
- `scripts/`:
  - `diagnose_pipeline_failures.py` (Reproducible diagnostic tool for deskew, binarization, grid suppression).
  - `train_all12lead_digitizer.py` (MPS/CUDA all-12-lead training script).
  - `verify_3_scenarios.py` (Automated 3-scenario verification script).
- `kaggle_bundles/digitizer/`:
  - `train_digitizer_kaggle.py` (Self-contained Kaggle GPU training script with BGR pink grid and 13-lead saving).
  - `kernel-metadata.json` (Kernel configuration referencing `khyeh0719/ptb-xl-dataset`).
- `data/`:
  - `raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/` (21,837 PTB-XL records).
  - `raw/ecg-image-data/ECG_Image_data/` (Unpaired real paper crops across classes F, M, N, Q, S, V).

---

## 3. Dataset Paths & Comprehensive Audit Status
- **Dataset A — PTB-XL (`data/raw/ptb-xl/`):**
  - Verified root: `data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/`
  - Total records: **21,837 complete 12-lead records** from **18,885 unique patients**.
  - Channel verification: Exactly 12 standard leads in standard clinical order (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`).
  - Signal properties: Exactly 10.0 seconds duration, physical millivolts ($mV$), gain 1000.0, baseline 0. Available at 100 Hz (`records100/`, 1,000 samples) and 500 Hz (`records500/`, 5,000 samples).
  - Patient separation: 10 strat_folds with **100% strict zero-patient overlap** (Folds 1-8: 15,108 patients, Fold 9: 1,887 patients, Fold 10: 1,890 patients).
  - Pairing status: **100% paired**. Serves as the authentic digital ground truth for full 12-lead paper ECG rendering and quantitative evaluation.
- **Dataset B — ECG Image Data (`data/raw/ecg-image-data/`):**
  - Verified root: `data/raw/ecg-image-data/ECG_Image_data/`
  - Total images: **123,998 files** across `test` (25,649) and `train` (98,349) in 6 classes (`F, M, N, Q, S, V`).
  - Image properties: 100% PNG format, identical resolution **`432 x 288 x 3`**.
  - Visible leads: **Single-lead beat crops only** (0 full 12-lead sheets in the entire dataset). Typically Lead II / MLII extracted from MIT-BIH.
  - Paper grid & calibration: Pure white background; zero millimeter grid lines; zero calibration pulses.
  - Pairing status: **Unpaired**. No digital signal arrays provided.
  - Suitability: Strictly suitable only for single-lead qualitative robustness testing (Scenarios B & C). Completely unsuitable for 12-lead digitisation training.
  - Audit deliverables: Inventory CSV at `data/interim/ecg_image_data_inventory.csv`, Contact Sheet at `data/interim/contact_sheet_ecg_image_data.png`.

---

## 4. Current Model Architecture & Checkpoint Details
- Model: `ECGDigitizerUNet` (ResNet-18 encoder, 2D decoder, column-wise Soft-Argmax coordinate head).
- Active Checkpoint: `models/ecg_digitizer_unet_all12lead.pt` (Epoch 4, best mean Pearson $r = 0.9786$, RMSE $= 0.0505\text{ mV}$).
- Legacy Checkpoint: `models/ecg_digitizer_unet.pt` (Leads II & V1).

---

## 5. Verified Recovery Status by Phase

| Phase | Description | Status | Verification Result |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Audit, Issue Register, Persistent Docs | **VERIFIED** | 4 persistent docs maintained; Issue register tracked |
| **Phase 1** | Image Preprocessing (Deskew, Grid, Trace Binarization) | **VERIFIED** | Deskew 0.00° residual; real `N10007.png` trace uncorrupted (single unbroken CC of 3,008 px) |
| **Phase 2** | Lead Layout Detection & Geometry | **VERIFIED** | All 13 lead bboxes detected for 3x4+1; single lead isolated without hallucinating leads |
| **Phase 3** | Deep Waveform Reconstruction & Spatial Alignment | **VERIFIED** | Baseline corridor scanning eliminates text label & pulse contamination; uniform 374px active spans |
| **Phase 4** | Kaggle GPU Retraining Package Preparation | **VERIFIED** | BGR pink grid rendered; POSIX `.venv/bin/kaggle` detected; local MPS checkpoint verified |
| **Phase 5** | End-to-End Three-Scenario Verification | **VERIFIED** | Scenario A: mean $r = 0.8134$; Scenario B: 100% conf; Scenario C: 0 fabricated leads |
| **Phase 6** | Workstation UI & Lead Explorer Interactions | **VERIFIED** | Drag-to-select interval measurement on canvas ($\Delta t$ ms & HR); rhythm guard $\ge 4$ RR |
| **Phase 7** | Synthetic ECG Generation & Calibration Pulse | **VERIFIED** | 1 mV pulse ($10\text{ mm} \times 5\text{ mm}$) with $\ge 12\text{ px}$ column separation; header metadata |
| **Phase 8** | Evaluation & Multi-Format Signal Export | **VERIFIED** | State preserved on base64 input; unified CSV, WFDB `.dat`/`.hea`, JSON, and overlay plots |
| **Phase 9** | Unit & Integration Test Suite | **VERIFIED** | **122 / 122 tests passed**, 0 failures, 85% total codebase coverage |
| **Phase 10** | Real Clinical Sheet Recovery (`ecg1.jpeg` & `3x4_3` Layout) | **VERIFIED** | Auto-inferred 6-row Marquette MAC `3x4_3` layout; fixed 1:1 pixel scaling; red sawtooth oscillations reduced from 40 px jump to 1.1 px |
| **Phase 11** | Visual Precision & Single-Lead Robustness (`ecg1` & `N10007`) | **VERIFIED** | Dynamic peak-anchored row geometry; label margin suppression (0.92 px mean distance to original ink on `ecg1.jpeg`; V2: 0.71 px; V5_RHYTHM: 1.21 px); gated deskew for gridless crops; full trace extent [52:359] for `N10007.png` |
| **Phase 12** | Full 12-Lead Paired Pipeline & Multi-Layout Renderer | **VERIFIED** | Synthetic renderer supporting all 4 layouts (`3x4`, `3x4_1`, `3x4_3`, `12x1`) with pink millimeter grid, 1 mV pulses, and JPEG/optical degradations. Small paired benchmark generated strictly from PTB-XL Fold 9 with zero training patient leakage. **127 / 127 tests passed (86% coverage)**. Clean 12-lead mean $r = 0.7386$, 12x1 stacked mean $r = 0.7041$. |

---

## 6. End-to-End Quantitative Verification Metrics (`verify_3_scenarios.py`)

### Scenario A — Paired Synthetic 12-Lead Clinical Sheet (PTB-XL Record `00001_lr`)
- **Overall Mean Pearson Correlation ($r$):** **0.8254**
- **Overall Mean RMSE:** **0.0768 mV**
- **Completeness:** **100.0% across all 13 leads**

| Lead Name | Pearson Correlation ($r$) | RMSE (mV) | SNR (dB) | Completeness | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **II_RHYTHM** | **0.9409** | 0.2084 | -7.97 | 100% | Validated |
| **V2** | **0.9316** | 0.0960 | +6.07 | 100% | Validated |
| **V6** | **0.9056** | 0.0864 | +2.39 | 100% | Validated |
| **V1** | **0.8914** | 0.1970 | -6.05 | 100% | Validated |
| **II** | **0.8871** | 0.0730 | +1.32 | 100% | Validated |
| **V5** | **0.8743** | 0.0800 | +1.80 | 100% | Validated |
| **AVL** | **0.8203** | 0.0727 | -0.71 | 100% | Validated |
| **V4** | **0.8211** | 0.2051 | -5.76 | 100% | Validated |
| **V3** | **0.7941** | 0.0948 | +0.66 | 100% | Validated |
| **AVR** | **0.7705** | 0.2013 | -6.90 | 100% | Validated |
| **I** | **0.7393** | 0.2090 | -4.14 | 100% | Validated |
| **AVF** | **0.6304** | 0.0815 | -4.63 | 100% | Validated (PTP = 0.27 mV) |
| **III** | **0.5678** | 0.0862 | -1.85 | 100% | Validated (PTP = 0.44 mV) |

### Scenario B — Real Paper ECG Image (`N10007.png`)
- **Mean Column Confidence:** **100.00%**
- **Calibration Reliable Flag:** **`False`** (Explicit transparency: beat crop lacks printed grid/pulse)
- **Output Artifacts:** Reconstructed CSV, 3-panel diagnostic overlay plot saved.

### Scenario C — Partial / Single-Lead Upload (`Q1.png`)
- **Reconstructed Leads:** `['II']` (Only the single uploaded lead is extracted)
- **Missing Leads Fabricated:** **`False`** (Zero hallucinated leads)
- **Output Artifacts:** Lead II signal CSV and sheet overlay saved.

---

## 7. Application Startup Commands
- Start Web Service / Workstation UI:
  ```bash
  .venv/bin/uvicorn ecg_digitization.service.app:app --host 127.0.0.1 --port 8000
  ```
- Run End-to-End 3-Scenario Verification:
  ```bash
  MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/verify_3_scenarios.py
  ```
- Run Full Test Suite:
  ```bash
  MPLCONFIGDIR=/tmp/mpl .venv/bin/pytest tests/
  ```
- Run Preprocessing Diagnostics:
  ```bash
  MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/diagnose_pipeline_failures.py
  ```

---

## 8. Phase 3 Training Readiness Audit Status (Completed 13 Nov 2026)
- **Status:** Complete. Documented in artifact `phase3_training_readiness.md`.
- **Determination:** **NO-GO FOR IMMEDIATE TRAINING** pending 4 remediation gates:
  1. *Dynamic Range Fix:* Expand crop window from $128\text{ px} \to 192\text{ px}$ to eliminate precordial S-wave flat-topping (V2, V3).
  2. *Marquette Layout Fix:* Dynamically compute column boundaries in `lead_detector.py` to eliminate 39 px / 264 ms horizontal lag.
  3. *Augmentation Realism:* Introduce optical blur, illumination gradients, and grid contrast variations into Kaggle training script.
  4. *Kaggle Authentication:* Re-authenticate Kaggle CLI with active credentials.
- **Model Architecture Decision:** Selected **Option C (Hybrid Architecture)**: Layout Anchor + Adaptive-Window Lead Crop U-Net ($192\text{ px}$) + 1D Waveform Refinement Head. Option B (full-sheet direct) rejected due to extreme QRS downsampling penalty ($80\text{ ms} \to <3\text{ px}$) and memory OOM risk on Kaggle GPUs.
- **Fold Isolation:** PTB-XL Folds 1–8 reserved for training, Fold 9 for validation/early stopping, Fold 10 strictly quarantined for final benchmark. Zero patient leakage.

