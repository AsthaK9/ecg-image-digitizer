# Data Layer

This document describes `src/ecg_digitization/data/` — the module responsible
for discovering ECG records on disk, parsing their ground-truth WFDB
signals, and exposing them to PyTorch as a lazily-loaded `Dataset` /
`DataLoader` pair driven entirely by the project's Hydra config.

For the on-disk file conventions this layer assumes, see
[`data_contract.md`](data_contract.md). For the pipeline this layer feeds
into, see [`architecture.md`](architecture.md).

## Module Map

```
src/ecg_digitization/data/
├── types.py         # dataclasses, Protocols, exception hierarchy — no I/O
├── parser.py         # WFDB (.hea/.dat) -> ECGSignal, + parser registry
├── io.py               # filesystem discovery (RecordLocator) + image loading, + locator registry
├── dataset.py            # ECGImageDataset (PyTorch Dataset) + collate_ecg_batch
└── datamodule.py            # ECGDataModule: Hydra config -> datasets -> DataLoaders
```

Each module has exactly one responsibility, and dependencies point in one
direction only:

```
datamodule.py --> dataset.py --> types.py
      |                             ^
      v                             |
  io.py, parser.py -----------------+
```

`dataset.py` never imports `io.py` or `parser.py` directly — it depends only
on the `RecordLocator` / `RecordParser` / `ImageLoader` **protocols**
declared in `types.py`. Concrete implementations are injected via
constructor arguments. This is what makes the dataset class itself
dataset-source-agnostic.

## Design Principles Applied

- **Dependency Inversion**: `ECGImageDataset` and `ECGDataModule` depend on
  the abstractions in `types.py` (`RecordLocator`, `RecordParser`,
  `ImageLoader`), never on concrete classes in `io.py` / `parser.py`.
- **Open/Closed**: adding a new ECG dataset source means implementing those
  three protocols and calling `io.register_locator()` /
  `parser.register_parser()` — no existing file needs to change.
- **Single Responsibility**: `io.py` only touches the filesystem and image
  codecs; `parser.py` only understands WFDB; `dataset.py` only orchestrates
  lazy loading and tensor conversion; `datamodule.py` only bridges Hydra
  config to the above.
- **Fail loud, fail with context**: every error surfaces as a subclass of
  `ECGDataError` (`RecordNotFoundError`, `RecordParsingError`,
  `ImageLoadError`, `EmptyDatasetError`, `UnsupportedDatasetSourceError`)
  carrying the offending `record_id` and enough context to debug without a
  stack trace deep-dive.

## Core Types (`types.py`)

| Type | Purpose |
|---|---|
| `ECGRecordPaths` | Resolved file locations for one record (`header_path`, `signal_path`, `image_paths`). `header_path`/`signal_path` are `None` for image-only records. |
| `ECGRecordMetadata` | Sampling rate, lead names, signal length, units, source dataset, and a free-form `extra` dict for source-specific fields. |
| `ECGSignal` | Validated `(values, metadata)` pair — construction itself enforces shape consistency, so an invalid `ECGSignal` cannot exist. |
| `ECGSample` | Typed, immutable view of a single dataset item (mirrors the dict `ECGImageDataset.__getitem__` returns). |
| `DatasetSplit` | `StrEnum` of `"train" / "val" / "test"`. |

Protocols (`RecordLocator`, `RecordParser`, `ImageLoader`) are structural —
a class satisfies them by implementing the right methods, no explicit
inheritance required.

## Supporting the PhysioNet/Kaggle Layout

`io.PhysioNetRecordLocator` and `parser.WFDBRecordParser` implement the
current data source:

```
<root>/<record_id>.hea       # WFDB header (ground truth, may be absent)
<root>/<record_id>.dat       # WFDB signal (ground truth, may be absent)
<root>/<record_id>-0.png     # one or more rendered/scanned images
<root>/<record_id>-1.png
```

`PhysioNetRecordLocator.list_record_ids()` treats a record as existing if it
has a `.hea` file **or** at least one image — this is what lets an
image-only held-out test split (no ground truth shipped) use the exact same
locator as a fully-labeled train/val split.

`WFDBRecordParser` wraps `wfdb.rdrecord`, transposes the signal to this
project's `(num_leads, num_samples)` convention, and validates it (non-empty,
finite, lead count matches declared metadata) before returning it — callers
never see a malformed `ECGSignal`.

## Adding a New ECG Dataset Source

No changes to `dataset.py` or `datamodule.py` are required. Steps:

1. Implement a class satisfying `RecordLocator` (a `list_record_ids()` and
   `get_paths(record_id)` method) for the new dataset's on-disk layout.
2. If its ground truth isn't WFDB, implement a class satisfying
   `RecordParser` (a `parse(paths, source_dataset)` method returning an
   `ECGSignal`). If it *is* WFDB, reuse `WFDBRecordParser` as-is.
3. Register both:

   ```python
   from ecg_digitization.data.io import register_locator
   from ecg_digitization.data.parser import register_parser

   register_locator("my_new_dataset", MyDatasetLocator)
   register_parser("my_new_dataset", MyDatasetParser)  # or WFDBRecordParser
   ```

4. Point the config at it: `data.source: my_new_dataset` (falls back to
   `"physionet"` if unset — the existing `configs/data/base.yaml` needed no
   changes to keep working).

`ECGDataModule` resolves `data.source` against these registries at
construction time and raises `UnsupportedDatasetSourceError` immediately —
with the list of currently-registered sources — if it doesn't recognize the
name, rather than failing later mid-training.

## `ECGImageDataset` (`dataset.py`)

A standard `torch.utils.data.Dataset`. Key contract:

- **Lazy**: construction only stores the record ID list and injected
  collaborators. No file is opened, no image decoded, no signal parsed
  until `__getitem__(i)` is called for that specific index. Verified in
  `tests/unit/test_dataset.py::TestECGImageDataset::test_construction_does_not_touch_locator_or_parser`.
- **Returns a dict of tensors**: `record_id`, `image` (`FloatTensor [C,H,W]`
  in `[0, 1]`), `signal` (`FloatTensor [num_leads, length]` or `None`),
  `sampling_rate_hz`, `lead_names`.
- **`require_signal`**: `True` for train/val (a missing signal is a hard
  error), `False` for test/inference splits (a missing signal yields
  `signal=None` rather than raising).
- **`validate()`**: eagerly attempts every record and returns
  `(record_id, error)` pairs for failures instead of raising — intended for
  a one-off data-quality pass in CI or before a long training run, not for
  use in the training loop itself.

### Batching: `collate_ecg_batch`

The default PyTorch collate cannot batch samples with `signal=None` or
signals of differing length. Use `collate_ecg_batch` as the `DataLoader`'s
`collate_fn` (already wired into `ECGDataModule`):

- If every sample in the batch has a signal, they are zero-padded to the
  batch's longest signal and a `signal_length` tensor is returned so the
  padding can be masked downstream.
- If no sample has a signal, `signal` / `signal_length` are omitted from the
  batch entirely.
- If a batch mixes samples with and without a signal, it raises — this
  should never happen in practice (a dataset's `require_signal` is uniform
  across all its samples) and indicates a configuration bug if it does.

## `ECGDataModule` (`datamodule.py`)

The Hydra integration point. Construction takes the **root** resolved
config (`cfg`, not `cfg.data`) and resolves everything needed to build
datasets and dataloaders:

1. **Source resolution**: `cfg.data.get("source", "physionet")` looked up
   against `io.LOCATOR_REGISTRY` / `parser.PARSER_REGISTRY`.
2. **Record discovery**, in priority order:
   - **Pre-split layout**: if `data.train_dir`, `data.val_dir`, and
     `data.test_dir` *each* independently resolve to at least one record,
     they're used directly, as-is.
   - **Single-pool split**: otherwise, all records under `data.raw_dir` (or
     `data.root_dir`) are discovered and split by `record_id` — **never by
     image** — according to `data.split.train_ratio` / `val_ratio` /
     `test_ratio`, shuffled deterministically with `data.split.random_seed`
     (falling back to the top-level `seed`). Splitting by `record_id`
     guarantees multiple rendered images of the same ground-truth signal
     can never land in different splits.
3. **Dataset construction**: one `ECGImageDataset` per split, with
   `require_signal=False` only for the `test` split.
4. **DataLoader construction**: `train_dataloader()` / `val_dataloader()` /
   `test_dataloader()` read `data.dataloader.*` (batch size, workers,
   pin_memory, shuffle, drop_last) and always use `collate_ecg_batch`.

No changes were made to `configs/data/base.yaml` to support this — every
key `ECGDataModule` reads (`root_dir`, `raw_dir`, `train_dir`, `val_dir`,
`test_dir`, `image.target_size`, `split.*`, `dataloader.*`) already existed
in the scaffold. The only new, optional key is `data.source`, which is
absent from the existing config and therefore defaults to `"physionet"` —
fully backward compatible.

### Usage

```python
from ecg_digitization.data import ECGDataModule

datamodule = ECGDataModule(cfg)   # cfg = the resolved root Hydra config
datamodule.setup()

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

for batch in train_loader:
    images = batch["image"]              # [B, C, H, W]
    signals = batch["signal"]            # [B, num_leads, max_len] (if present)
    lengths = batch["signal_length"]     # [B] (if signals present)
```

## Error Handling Reference

| Exception | Raised when |
|---|---|
| `RecordNotFoundError` | A locator finds no header and no image for a `record_id`. |
| `RecordParsingError` | A WFDB record can't be parsed, fails validation (non-finite values, empty signal, missing lead names), or ground truth is required but absent. |
| `ImageLoadError` | An image file is missing or fails to decode. |
| `EmptyDatasetError` | A resolved split (pre-split or pooled) contains zero records. |
| `UnsupportedDatasetSourceError` | `data.source` has no registered locator/parser. |

All inherit from `ECGDataError`, so calling code can catch data-layer
problems broadly (`except ECGDataError:`) without also swallowing
unrelated bugs.

## Testing

- `tests/unit/test_parser.py` — exercises `WFDBRecordParser` against real
  WFDB files written via `wfdb.wrsamp` (not mocks — the parser's entire job
  is correctly interfacing with the `wfdb` library).
- `tests/unit/test_dataset.py` — exercises `ECGImageDataset` and
  `collate_ecg_batch` against lightweight in-memory protocol fakes (fast,
  isolated), plus `PhysioNetRecordLocator` and `OpenCVImageLoader` against a
  real temp filesystem (their entire job is filesystem/codec interaction,
  which fakes can't meaningfully verify).
- `tests/integration/test_datamodule.py` — exercises `ECGDataModule`
  end-to-end against a real composed Hydra config and real synthetic WFDB
  records, including split determinism and a no-leakage-across-splits check.

Run the data layer's tests specifically:

```bash
pytest tests/unit/test_parser.py tests/unit/test_dataset.py tests/integration/test_datamodule.py -v
```
