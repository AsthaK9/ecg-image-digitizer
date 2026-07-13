# ECG Image Digitization

Production-grade pipeline for converting scanned/photographed paper ECG images
into digitized WFDB time-series signals.

> **Status:** Project scaffold. Configuration system, logging/tracking stack,
> folder structure, and tooling are in place and verified end-to-end. Model
> architectures and the training/inference pipeline are not yet implemented —
> see [Roadmap](#roadmap).

## Stack

- **Python 3.11**
- **PyTorch** / **Torchvision** — modeling
- **OpenCV** / **Albumentations** — image preprocessing & augmentation
- **Hydra** — hierarchical, composable experiment configuration
- **Weights & Biases** + **TensorBoard** — experiment tracking (both enabled by default, independently toggleable)

## Project Structure

```
ecg-digitization/
├── configs/                  # Hydra config groups (data, model, training, logging, env, experiment)
├── src/ecg_digitization/      # Installable package — all production code
│   ├── data/                  # Dataset classes, WFDB parsing, calibration, transforms
│   ├── preprocessing/         # Deskew, grid detection, panel localization
│   ├── models/                 # Model architectures (empty — see Roadmap)
│   ├── reconstruction/          # Mask → signal extraction, gap-filling
│   ├── pipeline/                 # Orchestrates preprocessing → model → reconstruction
│   ├── training/                  # Generic trainer, callbacks, metrics
│   ├── inference/                  # Prediction entrypoints
│   ├── evaluation/                  # Evaluation suite, error analysis
│   └── utils/                        # Seeding, logging/tracking, I/O helpers
├── scripts/                   # CLI entrypoints (train.py, ...)
├── tests/                      # unit / integration / fixtures
├── notebooks/                    # EDA only — never production code
├── docs/                          # architecture notes, data contract, ADRs, model cards
└── data/                            # git-ignored; raw/interim/processed/external
```

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline design
and rationale, and [`docs/decisions/`](docs/decisions/) for architecture
decision records.

## Setup

```bash
# Runtime only
make setup

# Runtime + dev tooling (lint, test, pre-commit hooks)
make setup-dev
```

Copy `.env.example` to `.env` and fill in your data root, artifacts root, and
`WANDB_API_KEY`.

## Configuration System

All experiment configuration is managed by **Hydra**, composed from independent
config groups under `configs/`:

| Group | Purpose |
|---|---|
| `data/` | Dataset paths, image/signal specs, split strategy, augmentation toggles |
| `model/` | Architecture selection + hyperparameters (placeholder until a model exists) |
| `training/` | Optimizer, scheduler, loss, checkpointing, early stopping |
| `logging/` | TensorBoard / W&B toggles and settings, log level |
| `env/` | Machine-specific paths and device settings — kept separate so experiment configs stay portable |
| `experiment/` | Composes overrides across the groups above for a single named run |

Run with defaults:

```bash
python scripts/train.py
```

Override individual values from the CLI:

```bash
python scripts/train.py data.dataloader.batch_size=64 training.epochs=10
```

Run a named experiment (composes a full override set):

```bash
python scripts/train.py experiment=exp001_baseline
```

Sweep a hyperparameter:

```bash
python scripts/train.py --multirun training.optimizer.lr=1e-3,5e-4,1e-4
```

Every run logs its **fully resolved** config (after all overrides) alongside
its outputs — see `hydra.run.dir` in `configs/config.yaml`.

## Experiment Tracking

Both backends are wired through a single `ExperimentLogger`
(`src/ecg_digitization/utils/logging_utils.py`) so training code never calls
`wandb` or `SummaryWriter` directly:

- **TensorBoard**: local, zero-config, logs to `artifacts/logs/<run_name>/tensorboard`
- **Weights & Biases**: remote, requires `WANDB_API_KEY`; toggle with `logging.wandb.enabled=false`

## Development

```bash
make format      # black + isort
make lint         # ruff
make typecheck      # mypy
make test             # pytest (unit + integration)
make test-cov           # pytest with coverage report
```

Pre-commit hooks run the same checks automatically on every commit after
`make setup-dev`.

## Roadmap

The scaffold currently proves out (and is tested against): config composition,
seeding, structured logging, and dual experiment-tracking backends, with
`scripts/train.py` runnable end-to-end and exiting cleanly at the point where
a model would attach.

Not yet implemented (see `docs/decisions/` for the planned approach):

1. WFDB parsing + pixel/mm/mV/time calibration module (`data/`, `preprocessing/`)
2. Synthetic data generation wrapper (ECG-Image-Kit)
3. Segmentation model + trainer wiring
4. Deterministic column-scan signal reconstruction (`reconstruction/`)
5. End-to-end inference pipeline (`pipeline/`, `inference/`)
6. Evaluation suite (SNR, clinical interval metrics)
7. Serving layer (FastAPI + Docker)

## License

MIT — see [`LICENSE`](LICENSE).
