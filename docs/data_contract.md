# Data Contract

> Status: draft — to be finalized once the raw Kaggle/PhysioNet dataset is
> downloaded and inspected directly (folder names, CSV columns, and image
> naming conventions must be verified against the actual data before this
> doc is treated as ground truth).

## Record Identity

A **record** is identified by a `record_id` (string) shared across:

- `data/raw/<split>/<record_id>.hea` — WFDB header
- `data/raw/<split>/<record_id>.dat` — WFDB signal (ground truth; train/val only)
- `data/raw/<split>/<record_id>-*.png|.jpg` — one or more rendered/scanned images

## `.hea` fields relied upon

| Field | Used by |
|---|---|
| Sampling frequency (Hz) | `data/calibration.py`, resampling in `reconstruction/postprocess.py` |
| Number of signals / lead names | `data/wfdb_parser.py`, output assembly |
| Signal length (samples) | Target output shape |
| Gain / ADC resolution | Converting raw signal to physical mV |
| Demographics (age, sex) — optional | Not used by the digitization task itself; may be absent in test split |

## Split Policy

Splitting is performed **by `record_id`, never by image**. Multiple images of
the same underlying signal (e.g. a clean render + an augmented/occluded
render) must land in the same split to avoid ground-truth leakage between
train and validation.

## Output Contract (model prediction)

Per record: a `[num_leads, num_samples]` array at the record's declared
sampling rate, in millivolts, aligned to the record's declared signal length
— i.e. shape- and unit-compatible with the corresponding `.dat` file, so
evaluation (`evaluation/metrics.py`) can diff prediction against ground truth
directly.

## Open Items

- [ ] Confirm exact Kaggle folder/CSV names against the downloaded archive
- [ ] Confirm `sample_submission.csv` schema
- [ ] Confirm whether test-split images include any real (non-synthetic) scans
