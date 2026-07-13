# Architecture Overview

## Pipeline

```
Raw Image
   -> [1] Preprocessing: deskew, grid detection/calibration, panel localization
   -> [2] Segmentation: trace-mask extraction per lead panel
   -> [3] Reconstruction: column-wise scan, gap-filling, pixel->mV/time conversion
   -> [4] Postprocessing: resample, assemble 12-lead array, format as WFDB/JSON
   -> Digitized Signal
```

Each stage is independently trainable, testable, and swappable. Deterministic
logic (calibration math, column-scan extraction) is kept out of learned
components so it can be unit tested without a GPU and without training data.

## Module Ownership

| Module | Owns | Does not own |
|---|---|---|
| `data/wfdb_parser.py` | Reading `.hea`/`.dat` into calibrated arrays | Image logic |
| `data/calibration.py` | Pure pixel<->mm<->mV<->time math | I/O, model calls |
| `preprocessing/*` | Deskew, grid detection, panel localization | Signal reconstruction |
| `models/*` | `nn.Module` architecture definitions | Training loop, data loading |
| `reconstruction/*` | Mask -> signal, deterministic | Learned weights |
| `training/trainer.py` | Generic train/val loop | Dataset- or model-specific logic |
| `evaluation/metrics.py` | SNR, RMSE, DTW, clinical intervals | Plotting, I/O |
| `inference/predictor.py` | Full pipeline orchestration for one input | Training code |
| `serving/api/` | Thin HTTP wrapper | Business/signal logic |

## Why a Hybrid Pipeline (not end-to-end regression)

Grid calibration (mm-per-pixel, paper speed, gain) is a known, exact physical
relationship printed on the ECG paper itself — not something worth asking a
neural network to re-derive from scratch. A hybrid approach (learned
segmentation + deterministic reconstruction) lets the model focus its
capacity on the genuinely hard perceptual problem (separating trace from
grid/noise/occlusion) while keeping the well-understood geometry and unit
conversion as auditable, testable code. See `docs/decisions/0001-hybrid-vs-e2e.md`.

## Configuration

All experiment configuration is composed via Hydra from independent groups
(`data`, `model`, `training`, `logging`, `env`, `experiment`) under `configs/`.
See the root `README.md` for usage and `docs/data_contract.md` for the exact
schemas each stage expects/produces.
