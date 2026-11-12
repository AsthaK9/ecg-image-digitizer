# ECG DigiAI — Dataset Pairing Analysis & Ground-Truth Discipline

**Project:** ECG Digitisation & Analysis Using AI (ECG DigiAI)  
**Institution:** Jaypee Institute of Information Technology, Noida (ECE Dept)  
**Date:** 12 November 2026  

---

## 1. Overview of Raw Datasets Discovered on Local Machine

| Dataset | Discovered Local Path | Format & Contents | Ground Truth Signals? | Paper Images? | Direct Pairing? | Role in ECG DigiAI |
|---|---|---|---|---|---|---|
| **PTB-XL (v1.0.1)** | `data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/` | 21,837 WFDB records (`.hea`, `.dat`), 100 Hz (`records100/`) and 500 Hz (`records500/`), `ptbxl_database.csv`, `scp_statements.csv` | **Yes** (Authentic 12-lead clinical voltage arrays, $mV$) | **No** (pure digital vector waveforms) | **No** | Primary source for digital signals, all-12-lead paired synthetic paper generation, and ground-truth benchmark evaluation |
| **ECG Image Data** | `data/raw/ecg-image-data/ECG_Image_data/` | 123,998 PNG image crops across classes `F`, `M`, `N`, `Q`, `S`, `V` in `train/` and `test/` | **No** (unpaired image crops, no calibration grids, no $mV$ scales) | **Yes** (single-beat matplotlib plots) | **No** | Qualitative preprocessing robustness, deskewing stress-testing, and real-world image inference |

---

## 2. Rigorous Data Pairing Audit & Scientific Discipline

### A. The Non-Pairing Reality
A critical inspection of the filesystem and metadata confirms:
1. `khyeh0719/ptb-xl-dataset` contains raw clinical digital recordings from 18,885 unique patients across 12 standard leads (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`) and duration of 10.0 seconds.
2. `erhmrai/ecg-image-data` contains isolated single-beat crops rendered from MIT-BIH Arrhythmia Database extracts without original timing headers, physical calibration markers (e.g. 10 mm/mV pulses), or 12-lead layout structure.
3. **Conclusion**: There is **zero intrinsic pairing** between these two datasets. Attempting to match records between them or computing supervised error metrics between arbitrary pairs would be scientifically invalid and fabricated.

### B. Ground-Truth Preservation Strategy
To enable mathematically sound, verifiable supervised deep learning for image-to-signal digitisation:
1. **Source Clinical Signals**: Authentic 12-lead recordings drawn strictly from PTB-XL.
2. **Clinical Grid Simulation**: Synthetic paper rendering using authentic clinical specifications:
   - Minor grid: 1 mm pink lines ($0.04\text{ s}$ horizontal at $25\text{ mm/s}$, $0.1\text{ mV}$ vertical at $10\text{ mm/mV}$).
   - Major grid: 5 mm red lines ($0.20\text{ s}$ horizontal, $0.5\text{ mV}$ vertical).
   - Standard scanning resolution: $150\text{ DPI}$ ($59.055\text{ pixels/mV}$, $147.638\text{ pixels/second}$).
   - Optional $1\text{ mV}$ calibration pulse ($10\text{ mm}$ height $\times 5\text{ mm}$ width).
3. **Exact Pixel-to-Signal Pairing**:
   - For every synthetic sample, the renderer produces simultaneously:
     - 2D degraded/clean image $I \in \mathbb{R}^{H \times W \times 3}$.
     - Exact 2D binary trace mask $M \in \{0, 1\}^{H \times W}$.
     - Exact 1D digital target voltage vector $V(t) \in \mathbb{R}^{L}$ in physical $mV$.
     - Exact lead bounding boxes $(x_1, y_1, x_2, y_2)$ and isoelectric baseline coordinates $y_{\text{base}}$.

### C. Patient-Safe Stratification Discipline
All data loading strictly adheres to the official PTB-XL stratified 10-fold patient assignment:
- **Folds 1–8**: Training split (17,441 records). Zero patient overlap with validation or test.
- **Fold 9**: Validation split (2,193 records) used for checkpoint selection and hyperparameter tuning.
- **Fold 10**: Held-out testing split (2,203 records) strictly quarantined from training.

---

## 3. Real ECG Image Role & Scope Definition

Real ECG images (`data/real_ecg_eval/samples/`) are processed through the complete workstation pipeline:
1. Active content bounding box localization (stripping whitespace borders).
2. Ink color normalization (handling cyan, green, and black ink traces).
3. Rescaling to U-Net input canvas ($128 \times 256 \times 3$).
4. Soft-Argmax waveform extraction and overlay generation.
5. Export to CSV and 3-panel visual verification overlays (`data/real_ecg_eval/overlays/`).
6. **Calibration Transparency**: Because real beat crops lack printed calibration pulses or millimeter grids, output voltages are explicitly tagged with `is_calibration_reliable = False` and labeled as nominal deflections.
