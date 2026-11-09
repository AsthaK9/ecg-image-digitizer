# Academic Traceability Matrix — ECG DigiAI

Institution: Jaypee Institute of Information Technology, Noida  
Department: Electronics & Communication Engineering  
Project: ECG Digitisation & Analysis Using AI (B.Tech Major Project)

| # | Academic Objective | Source Authority | Implemented Module | Verification Evidence | Status |
|---|---|---|---|---|---|
| **A.1** | Synthetic ECG Image Generation: Realistic paper grids (1 mm minor, 5 mm major) | Synopsis / Mid-Viva | `ecg_digitization.synthetic.generator` | `tests/unit/test_synthetic.py`, `scripts/verify_session1_pipeline.py` | **Done** |
| **A.2** | 12-Lead layout rendering (3x4 + Lead II rhythm strip) with 25 mm/s & 10 mm/mV | Synopsis / Mid-Viva | `ecg_digitization.synthetic.generator` | Generated sample `data/interim/session1_demo_output/` | **Done** |
| **A.3** | Ground-truth preservation: paired signals & binary trace masks | Synopsis / Mid-Viva | `SyntheticECGResult`, `trace_mask` | `test_synthetic_ecg_generator_clean` | **Done** |
| **A.4** | Photographic/Scanning Degradation (blur, noise, rotation skew, lighting) | Synopsis / Mid-Viva | `DegradationConfig` in `generator.py` | `test_synthetic_ecg_generator_degradation` | **Done** |
| **B.1** | Image loading, validation, grayscale, and CLAHE contrast enhancement | Synopsis / Mid-Viva | `ecg_digitization.preprocessing.enhancement` | `test_preprocessing.py` | **Done** |
| **B.2** | Image deskewing and orientation correction | Synopsis / Mid-Viva | `estimate_skew_angle`, `deskew_image` | `test_preprocessing.py`, verified skew rotation | **Done** |
| **B.3** | Grid detection and color-based / morphological suppression | Synopsis / Mid-Viva | `suppress_grid` in `enhancement.py` | `test_grid_suppression`, binary trace visual artifact | **Done** |
| **B.4** | Lead-region detection and ordering (3x4 + 1 and 12x1 templates) | Synopsis / Mid-Viva | `ecg_digitization.preprocessing.lead_detector` | `test_lead_layout_detector`, `demo_00001_lead_regions.png` | **Done** |
| **B.5** | Waveform extraction (column-wise centroid/median + interpolation) | Synopsis / Mid-Viva | `ecg_digitization.reconstruction.extractor` | `test_waveform_extractor`, active-span bounding | **Done** |
| **B.6** | Pixel-to-time and pixel-to-voltage calibration & resampling | Synopsis / Mid-Viva | `ecg_digitization.reconstruction.calibrator` | `SignalReconstructor`, `ReconstructedECG` | **Done** |
| **B.7** | Digital reconstruction export (CSV, multi-lead plot) | Synopsis / Mid-Viva | `ReconstructedECG.save_csv`, `save_plot` | `demo_00001_signal.csv`, `demo_00001_plot.png` | **Done** |
| **B.8** | Reconstruction evaluation harness (Pearson r, RMSE, SNR, completeness) | Synopsis / Mid-Viva | `ecg_digitization.evaluation.metrics` | `reconstruction_metrics.json` (r up to 0.9058) | **Done** |
| **C.1** | R-peak & heart rate estimation (Bradycardia / Tachycardia rules) | Synopsis / Mid-Viva | Scheduled for Session 2 | Implementation planned in Step 1 of Session 2 | **Pending** |
| **C.2** | AI multi-label pathology classifier (AFIB, BBB, MI, STD, STE) | Synopsis / Mid-Viva | Scheduled for Session 2 | Model architecture & registry scaffold ready | **Pending** |
| **C.3** | Clinical decision-support explainability report (JSON + HTML) | Synopsis / Mid-Viva | Scheduled for Session 2 | Implementation planned in Step 2 of Session 2 | **Pending** |
| **C.4** | Demonstrable web UI & FastAPI backend | Synopsis / Mid-Viva | Scheduled for Session 2 | Implementation planned in Step 3 of Session 2 | **Pending** |
| **C.5** | Kaggle GPU training orchestration wrapper | Synopsis / Mid-Viva | Scheduled for Session 2 | Implementation planned in Step 4 of Session 2 | **Pending** |
