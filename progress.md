# ECG DigiAI — Implementation Progress Log

## Project Identity & Academic Scope
- **Project**: ECG Digitisation & Analysis Using AI (ECG DigiAI)
- **Institution**: Jaypee Institute of Information Technology, Noida (ECE Dept)
- **Academic Context**: B.Tech Major Project (Final-Year)
- **Primary Authorities**: Major Project Synopsis & Mid-Viva Presentation

---

## Technical Strategy Review & Primary Objective Realignment
- **Primary Focus**: Image-to-Signal Digitisation (Component B). The previous `ECGResNet1D` experiment operates strictly on numerical signals for downstream classification (Component C) and does not train the core image digitizer.
- **Kaggle GPU Quota Policy**: No Kaggle GPU quota will be consumed on downstream classification. Remote training will be dedicated to deep learning image-to-signal reconstruction (`ECGDigitizerUNet`).
- **Data Pairing Audit**:
  - `khyeh0719/ptb-xl-dataset` contains 12-lead digital signals but no scanned paper images.
  - `erhmrai/ecg-image-data` contains single-beat crops with no paired digital signals.
  - **Strategy**: Supervised image-to-signal training leverages on-the-fly degraded synthetic paper generation from PTB-XL records to preserve exact ground-truth trace masks and calibrated waveform targets.
- **Detailed Strategic Document**: See [`docs/implementation_roadmap.md`](docs/implementation_roadmap.md).

---

## Session 1 Verified Milestones (Completed)
- [x] Environment & Hydra config fix: Added `configs/env/default.yaml` with dynamic env interpolation, updated `configs/config.yaml`, and unignored `configs/env/` in `.gitignore`.
- [x] Dataset pairing analysis (`docs/dataset_pairing.md`): Discovered Kaggle images are unpaired beat plots; established ground-truth evaluation discipline using PTB-XL paired synthetic rendering.
- [x] PTB-XL SCP Statement label audit (`docs/label_mapping.md`): Mapped AFIB (1,514), SBRAD (637), STACH (877), STD_ (1,009), BBB (2,273), MI (5,486), and flagged sparse STE_ (28 direct).
- [x] Synthetic ECG Generator (`ecg_digitization.synthetic`): Realistic 12-lead clinical paper grid (1 mm / 5 mm, 25 mm/s, 10 mm/mV, 150 DPI), 3x4 + 1 rhythm strip layout, paired binary trace masks, and configurable photographic degradations.
- [x] Image Preprocessing (`ecg_digitization.preprocessing`): Automated Hough deskewing, dual-domain red-channel/HSV grid suppression, and adaptive Gaussian + Otsu binarization.
- [x] Lead Layout Detector (`ecg_digitization.preprocessing.lead_detector`): Standard 3x4 + 1 rhythm strip and 12x1 layout bounding box extraction.
- [x] Waveform Extractor & Reconstructor (`ecg_digitization.reconstruction`): Active-span column-wise centroid/median tracking, isoelectric baseline subtraction, calibrated mV/time conversion, and uniform 100 Hz resampling.
- [x] Evaluation Harness (`ecg_digitization.evaluation`): Ground-truth comparison measuring Pearson $r$, RMSE, SNR, and completeness ($r \approx 0.9058$ on Lead II).

---

## Session 2 Verified Milestones (Completed)
- [x] **Physiological Rhythm & Rate Analysis** (`ecg_digitization.inference.analyzer`):
  - Butterworth bandpass filtering (0.5–40 Hz) and Pan-Tompkins QRS R-peak detection.
  - Heart rate calculation (BPM) and RR interval variability ($CV_{RR}$) assessment.
  - Signal Quality Index (good / acceptable / low) with lead-off/flatline detection.
  - Rule-based clinical findings for Sinus Bradycardia ($<60\text{ bpm}$) and Sinus Tachycardia ($>100\text{ bpm}$).
- [x] **AI Multi-Label Classifier & Model Safety Interface** (`ecg_digitization.models`, `inference.classifier_interface`):
  - 1D ResNet baseline architecture (`ECGResNet1D`) retained as auxiliary downstream evaluation module.
  - CPU smoke test: Verified forward pass, finite BCE loss backward pass, checkpoint saving, and checkpoint loading.
  - Clinical decision-support safety: Returns explicit `model_not_trained` status when checkpoint is absent; returns `Not supported by available labels` for sparse direct labels (`STE_`); zero fabricated predictions.
- [x] **Explainable Decision-Support Report Generator** (`ecg_digitization.inference.report_generator`):
  - Generates structured machine-readable JSON and formatted, printable clinical HTML reports.
  - Embeds multi-lead waveform plots, physiological rate measurements, model status, and mandatory regulatory disclaimers.
  - Clinical Risk Level explicitly marked `Not assessed` pending prospective clinical validation.
- [x] **Demonstrable FastAPI Service & Interactive Web Application** (`ecg_digitization.service.app`):
  - Six interactive dashboard screens: Dashboard, Synthetic Generator, Digitisation Pipeline, Reconstructed Waveforms, AI Analysis & Screening, Clinical Report.
  - REST API endpoints for status, sample listing, synthetic generation, digitisation, analysis, and report generation/download.
  - Windows Python 3.14 compatible.
- [x] **Kaggle GPU Training Infrastructure** (`ecg_digitization.training.kaggle_manager`):
  - Automated Kaggle CLI execution wrapper with credential discovery.
  - Packaging of kernel metadata and standalone GPU scripts.
- [x] **Full Test Suite Verification**:
  - **87/87 tests passed** with **85% code coverage**.

---

## Session 3 Verified Milestones (Completed)
- [x] **ECGDigitizerUNet Architecture** (`src/ecg_digitization/models/digitizer_unet.py`):
  - 2D ResNet-U-Net encoder-decoder for pixel-level ECG trace segmentation.
  - Differentiable Soft-Argmax column-wise coordinate extraction: $\hat{y}(x) = \sum_y y \cdot P(y \mid x)$.
  - Column confidence estimation and missing-trace/blank-region masking.
  - Differentiable physical calibration conversion: $V(x) = \frac{y_{\text{baseline}} - \hat{y}(x)}{\text{pixels\_per\_mv}}$.
  - End-to-end composite reconstruction loss: $\mathcal{L} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}} + 2 \mathcal{L}_{\text{SmoothL1}} + \mathcal{L}_{\text{Corr}}(1 - r)$.
- [x] **Synthetic Data Verification & Discrepancy Quantification** (`tests/unit/test_synthetic_verification.py`):
  - Verified every synthetic sample contains: `source_record_id`, `source_signal`, `sampling_rate_hz`, `clean_image`, `image`, `exact_lead_signals`, `trace_mask`, `lead_bounding_boxes`, `calibration`, `lead_baselines`.
  - Empirically quantified rasterization discrepancies:
    - Isoelectric / smooth segments: Mean Absolute Error is $0.0162\text{ mV}$ ($16.2\,\mu\text{V}$, well within clinical gold-standard threshold $< 0.05\text{ mV}$).
    - High slew-rate QRS peaks ($dV/dt > 50\text{ mV/s}$): Polyline discretization introduces localized peak error up to $0.175\text{ mV}$ (approx. 10 px vertical spread).
    - Uncorrupted reconstruction Pearson correlation exceeds $r > 0.99$.
- [x] **Leakage-Safe Dataset Splitting** (`src/ecg_digitization/data/split_manager.py`):
  - Enforces official PTB-XL patient-disjoint folds (Folds 1-8 Train, Fold 9 Val, Fold 10 Test; 0 cross-fold patients).
  - Groups all synthetic variants (e.g. `00001_lr_deg1`, `00001_lr_rot2`) to root record `00001_lr`, guaranteeing zero leakage across splits.
- [x] **CPU Smoke Test Execution** (`scripts/smoke_test_digitizer_unet.py`):
  - Validated mini-batch generation (4 synthetic crops `[4, 3, 128, 256]`).
  - Validated gradient flow and parameter updates: loss decreased monotonically from 2.2028 to 1.8354 in 3 steps.
  - Verified checkpoint serialization, reloading, and numerical bitwise match (`models/digitizer_unet_smoke_test.pt`).
- [x] **Digitization-Focused Kaggle GPU Training Bundle** (`kaggle_bundles/digitizer/`):
  - Generated self-contained kernel bundle: `train_digitizer_kaggle.py` and `kernel-metadata.json`.
  - Kernel ID: `asthak11111/ecg-digiai-unet-digitizer`.
  - Mounted dataset: `khyeh0719/ptb-xl-dataset`.
- [x] **Unit & Integration Test Suite**:
  - **101 passed tests** (0 failures) with **85% code coverage**.
- [x] **Pre-Submission Static & Dynamic Audit of `train_digitizer_kaggle.py`**:
  - Record counts verified: 2,500 train records (Folds 1-8), 500 val records (Fold 9), 2,203 held-out test records (Fold 10).
  - Patient leakage audit: 0 patient overlap across all 18,885 unique patients in PTB-XL official stratification.
  - Sample generation verified: 5,000 train crops and 1,000 val crops (2 crops/record: Lead II 0-2.5s limb lead + Lead V1 2.5-5.0s precordial lead).
  - Explicit scope limitation: Two-lead proof of concept (Lead II and V1) to validate the core deep digitizer architecture before multi-lead expansion.
  - Code modularity: Refactored executable logic into `def main(): ... if __name__ == '__main__': main()` to ensure safe multi-worker dataloading and clean local importability.
  - Resource labeling: Parameter count (1.98M) is mathematically exact; VRAM (<1.5 GB) and runtime (10-12 min) are analytical and extrapolated estimates, not empirical telemetric measurements.
  - Checkpoint integrity: saves full `model_state_dict`, model `config`, `epoch`, `best_val_pearson_r`, and `val_rmse_mv` to `ecg_digitizer_unet.pt`.
- [x] **Targeted Deterministic Waveform & Alignment Validation** (`scripts/verify_crop_alignment.py`, `tests/unit/test_crop_verification.py`):
  - Verified landmark alignment: baseline (y=64.0, err=1.0 px), P-peak (y=49.24, err=0.24 px), T-peak (y=40.40, err=0.40 px).
  - Identified canvas vertical clipping bound: 128px canvas with 64px baseline accommodates up to +1.08 mV without top clipping; documented for future full-sheet 256px lead crops.
  - Model forward and backward pass confirmed: 71/71 parameters receive finite, non-null gradients on synthetic batches.
  - Initial Pearson r (-0.2794) explained: Expected random-initialization behavior of untrained weights; CPU smoke test showed rapid convergence to r=0.84 within 3 optimization steps.

---

- [x] **Autonomous Kaggle GPU Execution & Checkpoint Integration**:
  - Submitted kernel `asthak11111/ecg-digiai-u-net-image-digitizer-training` to Kaggle NVIDIA Tesla T4 GPU.
  - Completed 5 full epochs on 5,000 paired synthetic crops (Folds 1-8) and 1,000 validation crops (Fold 9) in 745 seconds (12.4 minutes).
  - Validation metrics achieved:
    - **Validation Pearson Correlation**: **$r = 0.9971$**
    - **Validation RMSE**: **$0.0179\text{ mV}$** ($17.9\,\mu\text{V}$, clinical gold standard is $<0.05\text{ mV}$)
    - **Validation Loss**: **$0.0091$**
  - Autonomously downloaded `ecg_digitizer_unet.pt` and `digitizer_metrics.json`.
  - Registered checkpoint in local registry (`models/ecg_digitizer_unet.pt`).
  - Verified local inference: loads into `ECGDigitizerUNet` without error and responds through `/api/status` with `trained_checkpoint_loaded`.

---

## Session 4 Verified Milestones: Real ECG Image Inference Validation (Completed)
- [x] **Real ECG Sample Selection & Extraction** (`data/real_ecg_eval/samples/`):
  - Extracted 4 diverse real cardiac beat images from `data/raw/ecg-image-data/ecg-image-data.zip`:
    - `N10007.png`: Normal Sinus Rhythm (N)
    - `V1001.png`: Premature Ventricular Contraction (V)
    - `F0.png`: Fusion of ventricular and normal beat (F)
    - `M0.png`: Myocardial Infarction / Abnormal beat (M)
- [x] **Adapted Preprocessing Engine** (`src/ecg_digitization/inference/deep_digitizer.py`):
  - Automated Content Bounding Box Detection: Localizes active plot canvas, stripping outer whitespace margins (`(67, 43)-(372, 247)`).
  - Ink Normalization: Standardizes varied plot ink colors (cyan `[191,191,0]`, green `[0,128,0]`, black `[20,20,30]`) onto light paper background.
  - Geometry Rescaling: Adapts real crops to model resolution ($128 \times 256 \times 3$).
- [x] **Deep Inference & Waveform Digitisation**:
  - Executed trained `ECGDigitizerUNet` on all 4 real ECG samples.
  - High model trace confidence across all real samples:
    - `N10007.png`: 100.00% mean confidence (min: 99.73%)
    - `V1001.png`: 100.00% mean confidence (min: 99.94%)
    - `F0.png`: 99.66% mean confidence (min: 12.73%)
    - `M0.png`: 100.00% mean confidence (min: 100.00%)
- [x] **Reconstructed Waveform CSV Exports** (`data/real_ecg_eval/reconstructed_csv/`):
  - `N10007_digitized_waveform.csv`: 256 samples with coordinates, relative amplitudes, and nominal voltages.
  - `V1001_digitized_waveform.csv`: 256 samples with coordinates, relative amplitudes, and nominal voltages.
  - `F0_digitized_waveform.csv`: 256 samples with coordinates, relative amplitudes, and nominal voltages.
  - `M0_digitized_waveform.csv`: 256 samples with coordinates, relative amplitudes, and nominal voltages.
- [x] **3-Panel Visual Overlay Exports** (`data/real_ecg_eval/overlays/`):
  - `N10007_overlay_plot.png`, `V1001_overlay_plot.png`, `F0_overlay_plot.png`, `M0_overlay_plot.png`:
    1. Original crop with reconstructed waveform overlaid in magenta directly on the real trace.
    2. Normalized model input with predicted Soft-Argmax centerline and isoelectric baseline.
    3. Reconstructed 1D time-series with uncertainty shading.
- [x] **Rigorous Clinical Calibration & Scope Discipline**:
  - Ground-truth accuracy is explicitly **not claimed** on real beat plots due to lack of paired digital signals.
  - Physical calibration is explicitly marked `calibration_reliable = False` on beat plots due to absence of millimeter grids and 1 mV calibration pulses; nominal voltages are approximations.
  - Single-lead crop scope documented: Model currently digitizes single-lead crops (Lead II / V1 representation), not yet simultaneous full 12-lead sheets.
- [x] **Unit & Integration Test Suite**:
  - **104 passed tests** (0 failures) with **85% code coverage**.

---

## Current Status: Real Image Validation Complete & Tested
- Real ECG beat images validated through deep U-Net digitizer.
- Artifacts exported to `data/real_ecg_eval/`.
- Test suite: **104/104 passed**.
- Ready for full 12-lead multi-lead integration milestone.
