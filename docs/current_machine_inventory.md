# ECG DigiAI — Current Machine & Environment Inventory

**Date:** 12 November 2026  
**Environment:** macOS (Apple Silicon arm64)  
**Project:** ECG Digitisation & Analysis Using AI (ECG DigiAI)  
**Institution:** Jaypee Institute of Information Technology, Noida (ECE Dept)  

---

## 1. Operating System & Runtime Environment

- **Operating System:** macOS (Darwin arm64, Apple Silicon).
- **Python Virtual Environment:** `.venv` at project root.
- **Python Version:** 3.14.7 (`.venv/bin/python`).
- **Core Deep Learning Framework:** PyTorch 2.15.0.
  - MPS (Metal Performance Shaders) built: **True**
  - MPS available: **True** (`torch.backends.mps.is_available() == True`).
- **Computer Vision & Scientific Computing:**
  - OpenCV: 5.0.0.93 (headless)
  - Albumentations: 2.0.8
  - SciPy: 1.18.1
  - NumPy: 2.5.4
  - Pandas: 3.1.1
  - WFDB: 4.3.1
- **API & Web Framework:**
  - FastAPI: 0.162.2
  - Starlette: 1.10.0
  - Uvicorn: 0.54.0
  - Pydantic: 2.14.3
- **Testing & Quality Assurance:**
  - pytest: 9.2.0
  - pytest-cov: 7.1.0
  - pytest-mock: 3.17.0

---

## 2. Source Control & Repository State

- **Git Directory (`.git`):** Absent. The workspace was extracted directly from a GitHub repository ZIP release (`AsthaK9/ecg-image-digitizer`).
- **Integrity Rule:** No git clone or overwrite performed. All local code, models, and tests are preserved with absolute file-level safety.

---

## 3. Dataset Discovery & Storage Inventory

### A. PTB-XL Dataset (Digital 12-Lead ECGs)
- **Kaggle Source:** `khyeh0719/ptb-xl-dataset`
- **Discovered Local Path:** `data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/`
- **Total Records:** 21,837 clinical 12-lead ECG recordings.
- **Metadata Files:**
  - `ptbxl_database.csv`: Contains demographics, diagnostic codes (`scp_codes`), sampling paths, and patient-stratified cross-validation folds (`strat_fold` 1–10).
  - `scp_statements.csv`: Clinical statement diagnostic mappings.
- **Record Directories:**
  - `records100/`: 100 Hz recordings (10-second duration = 1,000 samples per lead).
  - `records500/`: 500 Hz high-resolution recordings (5,000 samples per lead).
- **Interim Development Samples:**
  - `data/interim/ptbxl_sample/`: Verified authentic WFDB pairs (`00001_lr` – `00099_lr`) for deterministic offline testing and real-time generation.

### B. ECG Image Data (Unpaired Real Paper Crops)
- **Kaggle Source:** `erhmrai/ecg-image-data`
- **Discovered Local Path:** `data/raw/ecg-image-data/ECG_Image_data/`
- **Classes:** 6 cardiac beat classes (`F`, `M`, `N`, `Q`, `S`, `V`) across `train/` and `test/` splits.
- **Data Pairing Discipline:** These images contain no digital ground-truth signals and are utilized strictly for qualitative visual overlay inspection and real-world preprocessing robustness evaluation.

### C. Curated Real ECG Evaluation Suite
- **Directory:** `data/real_ecg_eval/samples/`
- **Samples:** `N10007.png` (Normal), `V1001.png` (PVC), `F0.png` (Fusion), `M0.png` (MI), `S100.png` (SVPC), `Q1.png` (Paced).

---

## 4. Model Checkpoint Inventory

- **Active All-12-Lead Digitizer Checkpoint:** `models/ecg_digitizer_unet_all12lead.pt` (8,121,711 bytes).
  - Architecture: 2D ResNet-U-Net with Differentiable Soft-Argmax Coordinate Head.
  - Coverage: **All 12 standard clinical leads** (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`) and rhythm strip.
  - Validation Provenance: Best Validation Mean Pearson $r > 0.978$, Mean RMSE $< 0.051\text{ mV}$.
- **Legacy Digitizer Baseline Checkpoint:** `models/ecg_digitizer_unet.pt` (8,121,015 bytes).
  - Validation Provenance: Best Validation Pearson correlation $r = 0.9971$, RMSE $= 0.0179\text{ mV}$ (Leads II and V1).
- **Safety Backup Checkpoint:** `models/ecg_digitizer_unet.pt.bak` (8,121,015 bytes, identical to legacy baseline).

---

## 5. Baseline Test Status

- **Command:** `.venv/bin/pytest tests/unit/`
- **Status:** **108 passed**, 0 failures, 82% code coverage.
