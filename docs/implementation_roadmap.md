# ECG DigiAI — Implementation Roadmap & Technical Strategy Review

**Institution:** Jaypee Institute of Information Technology, Noida  
**Department:** Electronics & Communication Engineering  
**Project:** Final-Year B.Tech Major Project — ECG Digitisation & Analysis Using AI  
**Focus:** Primary Objective Realignment — Image-to-Signal Digitisation  

---

## 1. Project Primary Objective vs Auxiliary Modules

### Primary Objective (Component B)
**ECG Image-to-Signal Digitisation**: Developing an automated, robust computer vision and deep-learning pipeline that takes scanned or photographed paper ECG images (under real-world skew, background noise, colored grids, and illumination variations) and reconstructs calibrated, multi-lead digital 1D time-series signals ($mV$ vs time).

### Auxiliary Objective (Component C)
**AI-Based ECG Analysis**: Downstream clinical rhythm assessment (heart rate, R-peaks, Sinus Bradycardia, Sinus Tachycardia) and multi-label pathology classification on the *reconstructed digital signals*.

### Analysis of Current `ECGResNet1D` Model
- The current `ECGResNet1D` baseline accepts $12 \times N$ numerical signals as input and predicts 7 pathology labels.
- While required for Component C of the academic scope, training this model in isolation **does not train or advance the core image digitisation pipeline**.
- **Strategic Recommendation**:
  1. Retain `ECGResNet1D` as an auxiliary downstream module.
  2. Reallocate the primary remote Kaggle GPU training budget to an **Image-to-Waveform Digitisation Network (`ECGDigitizerUNet`)**.

---

## 2. Dataset Pairing Audit

| Dataset | Local Archive | Total Records | Ground Truth Signals? | Paper Images? | Direct Pairing? |
|---|---|---|---|---|---|
| **PTB-XL v1.0.1** | `data/raw/ptb-xl/ptb-xl-dataset.zip` (~1.76 GB) | 21,837 WFDB records | **Yes** (12 leads, 100 Hz / 500 Hz) | **No** (pure digital) | **No** |
| **ECG Image Data** | `data/raw/ecg-image-data/ecg-image-data.zip` (~858 MB) | 123,998 PNG images | **No** (unpaired image crops) | **Yes** (single-beat crops) | **No** |

**Conclusion**: There are **zero paired image-signal records** between the two downloaded raw archives. Attempting to train supervised image-to-signal networks directly on `ecg-image-data` is mathematically and scientifically infeasible because no ground-truth waveform vectors exist for those images.

---

## 3. Ground-Truth Strategy: Synthetic ECG Image Generation

To enable rigorous supervised deep learning for digitisation, the project leverages synthetic paper rendering directly from authentic clinical signals:

1. **Source Signals**: Authentic 12-lead digital signals drawn from PTB-XL (Folds 1–8 Train, Fold 9 Val, Fold 10 Test).
2. **Clinical Grid Simulation**: Standard 1 mm minor pink grid lines and 5 mm major red grid lines adhering to clinical standards ($25\text{ mm/s}$ paper speed, $10\text{ mm/mV}$ voltage calibration, $150\text{ DPI}$).
3. **Realistic Degradation Pipeline**:
   - Affine rotation skew ($\pm 5^\circ$).
   - Perspective camera warp (simulating smartphone photographs).
   - Non-uniform lighting and shadow gradients.
   - Optical Gaussian blur ($3\times 3$, $5\times 5$).
   - Additive Gaussian sensor noise.
4. **Exact Paired Ground Truth**:
   - Target 1D digital signal: $s(t) \in \mathbb{R}^{12 \times 1000}$.
   - Target 2D binary trace segmentation mask: $M(x, y) \in \{0, 1\}^{H \times W}$.
   - Target lead bounding boxes: coordinates $(x_1, y_1, x_2, y_2)$ for each lead window.

---

## 4. Proposed Image-to-Waveform Model Architecture: `ECGDigitizerUNet`

The optimal deep learning architecture for image-to-signal digitisation combines deep semantic trace segmentation with a differentiable coordinate extraction layer:

```mermaid
flowchart LR
    A["Input Degraded Lead Image [B, 3, H, W]"] --> B["2D U-Net Encoder-Decoder"]
    B --> C["Predicted Trace Probability Mask [B, 1, H, W]"]
    C --> D["Soft-Argmax Spatial Column Pooling"]
    D --> E["Continuous Extracted Y-Coordinate Vector [B, W]"]
    E --> F["Differentiable Calibration & Baseline Normalization"]
    F --> G["Reconstructed Calibrated Voltage Signal [B, N] (mV)"]
```

### Architecture Specifications
1. **Backbone**: 2D U-Net with ResNet18/MobileNet encoder blocks for efficient GPU convergence.
2. **Segmentation Head**: Outputs pixel-wise trace probability $\hat{M}(x, y)$.
3. **Differentiable Soft-Argmax**:
   For each image column $x \in [0, W-1]$, the vertical trace position is computed as:
   $$\hat{y}(x) = \sum_{y=0}^{H-1} y \cdot \text{softmax}_y\left(\frac{\hat{M}(x, y)}{\tau}\right)$$
   Where $\tau$ is a temperature hyperparameter. This allows backpropagating gradients directly from 1D waveform losses into the 2D feature extractor!
4. **Loss Function**:
   $$\mathcal{L} = \mathcal{L}_{\text{BCE}}(\hat{M}, M) + \mathcal{L}_{\text{Dice}}(\hat{M}, M) + \alpha \cdot \text{SmoothL1}(\hat{s}, s) + \beta \cdot (1 - \text{PearsonCorrelation}(\hat{s}, s))$$

---

## 5. Waveform Reconstruction Evaluation Metrics

Rather than classification accuracy or F1 score, model evaluation tracks:

1. **Pearson Correlation Coefficient ($r$)**: Measures morphological and phase synchronization against ground-truth signal waveforms (target: $r > 0.90$).
2. **Root Mean Squared Error (RMSE in mV)**: Measures absolute calibration and amplitude fidelity (target: $\text{RMSE} < 0.25\text{ mV}$).
3. **Signal-to-Noise Ratio (SNR in dB)**: Measures residual background extraction artifacts ($10 \log_{10} \frac{\|s\|^2}{\|s - \hat{s}\|^2}$).
4. **Segmentation Intersection-over-Union (IoU / Dice)**: Measures pixel-level trace isolation accuracy independent of calibration.
5. **Trace Completeness (%)**: Percentage of continuous signal successfully recovered without dropout.

---

## 6. Execution Roadmap & Current Status

1. **Step 1: Implement `ECGDigitizerUNet` Module** — **COMPLETED**
   - Implemented `ECGDigitizerUNet` and `DigitizerReconstructionLoss` in `src/ecg_digitization/models/digitizer_unet.py`.
   - Implemented `LeakageSafeSplitManager` in `src/ecg_digitization/data/split_manager.py`.
   - Preserved legacy Lead II / V1 baseline checkpoint `models/ecg_digitizer_unet.pt` and safety copy `models/ecg_digitizer_unet.pt.bak`.

2. **Step 2: Mac Apple Silicon Migration & MPS Acceleration** — **COMPLETED**
   - Migrated project to macOS arm64 (`.venv` on Python 3.14.7).
   - Activated PyTorch Metal Performance Shaders (MPS), achieving ~6x acceleration over CPU (16.9 crops/s vs 2.9 crops/s).
   - Verified Kaggle CLI and discovered local raw datasets: PTB-XL (`data/raw/ptb-xl/`) and ECG Image Data (`data/raw/ecg-image-data/`).

3. **Step 3: All-12-Lead Paired Training Pipeline (`scripts/train_all12lead_digitizer.py`)** — **COMPLETED**
   - Implemented `All12LeadPairedDataset` with balanced sampling across all 12 standard clinical leads:
     `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.
   - Enforced patient-safe PTB-XL splits: Folds 1–8 for training, Fold 9 for validation, and Fold 10 quarantined for held-out testing.
   - Warm-started from existing checkpoint and trained on Apple Silicon MPS.
   - Saved active all-12-lead checkpoint to `models/ecg_digitizer_unet_all12lead.pt` with validation Mean Pearson $r > 0.978$ and Mean RMSE $< 0.051\text{ mV}$.

4. **Step 4: Inference Pipeline Upgrade & Lead Provenance** — **COMPLETED**
   - Updated `DeepCropDigitizer` (`src/ecg_digitization/inference/deep_digitizer.py`) to auto-detect and load `ecg_digitizer_unet_all12lead.pt`, expanding `supported_leads` to all 12 leads plus rhythm strip.
   - Updated `ECGDigitizerPipeline` (`src/ecg_digitization/pipeline/digitizer.py`) to mark all 12 leads as `validated_lead`.
   - Updated FastAPI `/api/status` endpoint to report active all-12-lead checkpoint.

5. **Step 5: Three-Scenario Scientific Verification (`scripts/verify_3_scenarios.py`)** — **COMPLETED**
   - **Scenario A (Paired Synthetic 12-Lead):** Digitized all 12 leads + rhythm strip from PTB-XL record `00001_lr`. Reconstructed signals compared against exact ground truth: Mean Pearson $r = 0.8456$, Mean RMSE = $0.1305\text{ mV}$.
   - **Scenario B (Real ECG Image):** Digitized authentic beat image `N10007.png`. Pipeline executed cleanly with 100% trace confidence and reported `is_calibration_reliable: false` (no fabricated calibration).
   - **Scenario C (Single Lead Extraction):** Digitized single-lead image `Q1.png` in `single_lead` mode. Exactly 1 lead extracted; missing leads (`I, III, aVR-V6`) were strictly NOT fabricated.

6. **Step 6: Research Workstation UI & Test Suite** — **COMPLETED**
   - 4-tab HTML5 workstation (`workstation_html.py`) upgraded: Digitisation Workbench, Lead Explorer, Synthetic ECG Lab, and Evaluation & Export.
   - All disease classifier claims and badges removed; pure research workstation for digitisation and physiological signal analysis (Pan-Tompkins heart rate, RR variability, rhythm rules).
   - Full test suite passes: 108/108 tests passing with 82% coverage.
