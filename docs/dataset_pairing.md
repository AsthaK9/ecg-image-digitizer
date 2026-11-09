# Dataset Pairing Analysis & Ground-Truth Discipline

## 1. Overview of Available Datasets

| Dataset | Source / Reference | Format | Ground Truth Signal? | Purpose in ECG DigiAI |
|---|---|---|---|---|
| **PTB-XL (v1.0.1)** | PhysioNet / `ptb-xl-dataset.zip` (~1.76 GB) | 12-lead WFDB (`.hea`, `.dat`), 100 Hz / 500 Hz, 21,837 records | **Yes** (12-lead clinical signals + SCP codes) | Primary source for digital signals, synthetic image generation, and multi-label arrhythmia/pathology classification |
| **ECG Image Data** | Kaggle (`erhmrai/ecg-image-data.zip`, ~858 MB) | 123,998 PNG images (432x288), classes: F, M, N, Q, S, V | **No** (Unpaired beat/lead plot images) | Qualitative preprocessing, noise robustness, and real-world image evaluation |

---

## 2. Pairing Analysis & Findings

1. **Unpaired Status**:
   - The Kaggle `ecg-image-data` archive contains cropped single-heartbeat/rhythm plots categorized according to AAMI classes (`F`, `M`, `N`, `Q`, `S`, `V`). It does **not** provide the underlying digital vector waveforms or calibration metadata.
   - The PTB-XL archive contains comprehensive 12-lead raw signals and SCP clinical diagnosis labels, but no paper-scanned images.
   - **Conclusion**: There is **no intrinsic pairing** between the Kaggle image archive and the PTB-XL digital records.

2. **Ground-Truth Preservation Strategy**:
   - To quantitatively evaluate digitisation fidelity (Pearson correlation $r$, Root Mean Squared Error, SNR, completeness), we generate paired **Synthetic ECG Images** directly from PTB-XL digital signals.
   - Every synthetic image records:
     - Exact source signal ($mV$ vs time $s$)
     - Pixel-to-physical calibration scales ($25\text{ mm/s}$, $10\text{ mm/mV}$, DPI)
     - Pixel trace ground truth masks
     - Lead layout bounding boxes (standard 12-lead 3x4 + rhythm strip)
   - Real Kaggle images are reserved strictly for qualitative stress-testing of preprocessing pipelines (deskewing, binarization, contrast adjustment). No fake image-to-signal metrics are calculated on unpaired real images.

3. **Split Discipline**:
   - Classification and evaluation respect patient-level splits (`patient_id` in PTB-XL) and the official 10-fold stratified split (`strat_fold` 1–8 train, 9 val, 10 test) to prevent data leakage across patients.
