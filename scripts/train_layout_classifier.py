"""Training script for ECG Sheet Layout Classifier.

Trains ECGLayoutCNN to distinguish:
0: single_lead
1: 3x4
2: 3x4_1
3: 3x4_3
4: 12x1

Uses PTB-XL:
- Folds 1-8 for training
- Fold 9 for validation
- Fold 10 strictly held out (quarantined)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wfdb

from ecg_digitization.models.layout_classifier import (
    ECGLayoutCNN,
    LAYOUT_CLASSES,
    LayoutDetectorService,
)
from ecg_digitization.synthetic.generator import (
    DegradationConfig,
    SyntheticECGGenerator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainLayoutClassifier")


class SyntheticLayoutDataset(Dataset):
    """Dataset that generates varied ECG sheet layouts on the fly from PTB-XL records."""

    def __init__(
        self,
        record_paths: List[Path],
        layouts: List[str] = LAYOUT_CLASSES,
        target_size: Tuple[int, int] = (128, 192),
        augment: bool = True,
        samples_per_layout: int = 40,
        seed: int = 42,
    ) -> None:
        self.record_paths = record_paths
        self.layouts = layouts
        self.target_size = target_size
        self.augment = augment
        self.generator = SyntheticECGGenerator()
        self.rng = np.random.RandomState(seed)

        # Pre-generate and cache all tensors once in memory for ultra-fast training
        self.cached_tensors: List[torch.Tensor] = []
        self.cached_labels: List[int] = []

        n_recs = len(record_paths)
        logger.info("Pre-generating %d cached samples across %d layouts...", samples_per_layout * len(layouts), len(layouts))

        for l_idx, layout_name in enumerate(layouts):
            for k in range(samples_per_layout):
                rec_idx = (l_idx * samples_per_layout + k) % n_recs
                sample_seed = seed + l_idx * 1000 + k
                rec_path = self.record_paths[rec_idx]

                try:
                    record = wfdb.rdrecord(str(rec_path))
                    sig = record.p_signal.T.astype(np.float32)
                except Exception:
                    sig = np.random.randn(12, 1000).astype(np.float32) * 0.5

                deg = None
                if self.augment:
                    rng = np.random.RandomState(sample_seed)
                    deg = DegradationConfig(
                        enabled=True,
                        rotation_deg=float(rng.uniform(-2.5, 2.5)),
                        perspective_factor=float(rng.uniform(0.0, 0.02)),
                        gaussian_noise_sigma=float(rng.uniform(0.0, 8.0)),
                        blur_kernel_size=int(rng.choice([0, 0, 3])),
                        shadow_gradient=bool(rng.rand() > 0.6),
                        grid_color_style=str(rng.choice(["standard", "faded", "dark"])),
                        clinical_header=bool(rng.rand() > 0.4),
                        vertical_dividers=bool(rng.rand() > 0.4),
                        footer_annotation=bool(rng.rand() > 0.4),
                    )

                if layout_name == "single_lead":
                    h, w = 288, 432
                    img = np.full((h, w, 3), 255, dtype=np.uint8)
                    lead_idx = self.rng.randint(0, 12)
                    sig_slice = sig[lead_idx, :300]
                    pts_y = np.clip(h // 2 - sig_slice * 40.0, 10, h - 10).astype(np.int32)
                    pts_x = np.linspace(20, w - 20, len(sig_slice)).astype(np.int32)
                    pts = np.column_stack([pts_x, pts_y])
                    cv2.polylines(img, [pts], isClosed=False, color=(30, 30, 30), thickness=2, lineType=cv2.LINE_AA)
                else:
                    synth_res = self.generator.generate(
                        signal=sig,
                        sampling_rate_hz=100.0,
                        degradation=deg,
                        seed=sample_seed,
                        layout_type=layout_name,
                    )
                    img = synth_res.image

                # Localize active canvas to train on the true waveform region
                svc_temp = LayoutDetectorService()
                y1, y2, x1, x2, _ = svc_temp.detect_active_canvas(img)
                canvas_crop = img[y1:y2, x1:x2]

                gray = cv2.cvtColor(canvas_crop, cv2.COLOR_BGR2GRAY) if len(canvas_crop.shape) == 3 else canvas_crop
                inverted = 255 - gray
                resized = cv2.resize(inverted, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
                tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0

                self.cached_tensors.append(tensor)
                self.cached_labels.append(l_idx)

    def __len__(self) -> int:
        return len(self.cached_tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.cached_tensors[idx], self.cached_labels[idx]


def train_layout_classifier():
    parser = argparse.ArgumentParser(description="Train ECG Sheet Layout Classifier")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--samples-per-layout", type=int, default=80, help="Train samples per layout class")
    parser.add_argument("--val-samples-per-layout", type=int, default=20, help="Val samples per layout class")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Load PTB-XL records split strictly by fold
    db_csv = Path("data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl_database.csv")
    base_dir = db_csv.parent

    df = pd.read_csv(db_csv)
    train_df = df[df["strat_fold"].between(1, 8)]
    val_df = df[df["strat_fold"] == 9]
    test_df = df[df["strat_fold"] == 10]

    logger.info("Folds 1-8 (Train): %d records", len(train_df))
    logger.info("Fold 9 (Val): %d records", len(val_df))
    logger.info("Fold 10 (Quarantined Test): %d records (Strictly Preserved)", len(test_df))

    train_paths = [base_dir / p for p in train_df["filename_lr"].head(200)]
    val_paths = [base_dir / p for p in val_df["filename_lr"].head(50)]

    train_ds = SyntheticLayoutDataset(train_paths, samples_per_layout=args.samples_per_layout, augment=True, seed=42)
    val_ds = SyntheticLayoutDataset(val_paths, samples_per_layout=args.val_samples_per_layout, augment=False, seed=999)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    logger.info("Built dataset: Train=%d samples, Val=%d samples", len(train_ds), len(val_ds))

    # 2. Build Model
    model = ECGLayoutCNN(num_classes=len(LAYOUT_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    checkpoint_path = Path("models/ecg_layout_classifier.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_log = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            total_train += len(targets)

        train_loss /= total_train
        train_acc = train_correct / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        class_correct = {cls: 0 for cls in LAYOUT_CLASSES}
        class_total = {cls: 0 for cls in LAYOUT_CLASSES}

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * len(targets)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                total_val += len(targets)

                for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                    cls_name = LAYOUT_CLASSES[t]
                    class_total[cls_name] += 1
                    if t == p:
                        class_correct[cls_name] += 1

        val_loss /= total_val
        val_acc = val_correct / total_val

        per_class_acc = {
            cls: (class_correct[cls] / class_total[cls]) if class_total[cls] > 0 else 0.0
            for cls in LAYOUT_CLASSES
        }

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%%",
            epoch, args.epochs, train_loss, train_acc * 100.0, val_loss, val_acc * 100.0
        )

        metrics_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "per_class_acc": per_class_acc,
        })

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": LAYOUT_CLASSES,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
            }, checkpoint_path)

    duration = time.time() - start_time
    logger.info("Training completed in %.2f s. Best Val Acc: %.2f%%", duration, best_val_acc * 100.0)

    # Save metrics
    metrics_path = Path("models/layout_classifier_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model_architecture": "ECGLayoutCNN (Lightweight 2D ConvNet)",
            "classes": LAYOUT_CLASSES,
            "total_training_duration_s": round(duration, 2),
            "best_val_acc": round(best_val_acc, 4),
            "epochs": metrics_log,
        }, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    train_layout_classifier()
