"""ECG DigiAI — Robust 12-Lead Digitizer Training & Controlled Evaluation Experiment.

Conducts the rigorous controlled training experiment:
1. Evaluates Existing Checkpoint (models/ecg_digitizer_unet_all12lead.pt) on realistic validation data.
2. Trains ECGDigitizerUNet with realistic hard-negative augmentations:
   - In-crop printed lead text labels (Hard Negatives, mask=0)
   - In-crop 1 mV calibration pulses (Hard Negatives, mask=0)
   - Multi-color & variable grid intensities (pink, faded tan, monochrome gray, cyan, no-grid)
   - Thin (1px) and thick (3-4px) stroke widths
   - Physiological baseline shifts and amplitude scaling
   - Optical blur, noise, shadow gradients, JPEG compression
3. Evaluates both models on identical Fold 9 validation data across:
   - All 12 standard leads (I..V6)
   - 5 degradation categories (Clean, Faded Grid, Text Overlap, Thick Stroke, Thin Stroke)
4. Saves new checkpoint as models/ecg_digitizer_unet_v3_robust.pt (preserving existing checkpoints).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wfdb

from ecg_digitization.data.realistic_crops import (
    PairedLeadCrop,
    RealisticLeadCropGenerator,
    STANDARD_12_LEADS,
)
from ecg_digitization.evaluation.metrics import compute_lead_metrics
from ecg_digitization.models.digitizer_unet import (
    DigitizerReconstructionLoss,
    DigitizerUNetOutput,
    ECGDigitizerUNet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainRobustDigitizer")


class PrecachedRealisticDataset(Dataset):
    """Pre-generates realistic crops in memory once for fast GPU/MPS training."""

    def __init__(
        self,
        record_paths: List[Path],
        num_records: int = 100,
        leads_per_record: int = 12,
        is_validation: bool = False,
        seed: int = 42,
    ) -> None:
        self.crops: List[PairedLeadCrop] = []
        generator = RealisticLeadCropGenerator()
        rng = np.random.RandomState(seed)

        selected_paths = record_paths[:num_records]
        logger.info(
            "Building %s dataset from %d PTB-XL records (%d leads/record)...",
            "Validation" if is_validation else "Training",
            len(selected_paths),
            leads_per_record,
        )

        degradations = ["clean", "text_overlap", "faded_grid", "thick_stroke", "thin_stroke"]

        for rec_idx, rec_path in enumerate(selected_paths):
            try:
                rec = wfdb.rdrecord(str(rec_path))
                sig = rec.p_signal.T.astype(np.float32)  # [12, 1000]
            except Exception:
                continue

            for lead_idx in range(min(12, leads_per_record)):
                lead_name = STANDARD_12_LEADS[lead_idx]
                col_idx = lead_idx // 3
                start = col_idx * 250
                seg = sig[lead_idx, start : start + 250]

                # Select adjoining lead for bleed
                adj_lead_idx = (lead_idx + 1) % 12
                adj_seg = sig[adj_lead_idx, start : start + 250]

                deg_mode = degradations[(rec_idx + lead_idx) % len(degradations)] if is_validation else "random"
                sample_seed = seed + rec_idx * 100 + lead_idx

                crop = generator.generate_crop(
                    lead_signal=seg,
                    lead_name=lead_name,
                    seed=sample_seed,
                    is_validation=is_validation,
                    degradation_mode=deg_mode,
                    adjoining_lead_signal=adj_seg,
                )
                self.crops.append(crop)

        logger.info("Successfully precached %d realistic crops.", len(self.crops))

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> PairedLeadCrop:
        return self.crops[idx]


def collate_crops(batch: List[PairedLeadCrop]):
    images = torch.stack([b.image_tensor for b in batch])
    masks = torch.stack([b.target_mask for b in batch])
    signals = torch.stack([b.target_signal for b in batch])
    coords = torch.stack([b.target_coords for b in batch])
    valid_cols = torch.stack([b.valid_columns for b in batch])
    baselines = torch.tensor([b.baseline_y for b in batch], dtype=torch.float32)
    lead_names = [b.lead_name for b in batch]
    deg_types = [b.degradation_type for b in batch]
    return images, masks, signals, coords, valid_cols, baselines, lead_names, deg_types


def evaluate_model_on_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, any]:
    """Runs rigorous quantitative evaluation on realistic validation crops."""
    model.eval()

    total_samples = 0
    total_dice = 0.0
    total_iou = 0.0
    total_pixel_mae = 0.0
    total_pearson = 0.0
    total_rmse_mv = 0.0

    per_lead_metrics = {lead: {"pearson": [], "rmse": [], "mae_px": []} for lead in STANDARD_12_LEADS}
    per_deg_metrics = {}

    with torch.no_grad():
        for images, masks, signals, coords, valid_cols, baselines, lead_names, deg_types in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            signals = signals.to(device)
            coords = coords.to(device)
            valid_cols = valid_cols.to(device)
            baselines = baselines.to(device)

            output: DigitizerUNetOutput = model(images, baseline_y=baselines)

            # 1. Segmentation Dice & IoU
            probs = output.segmentation_probs  # [B, 1, H, W]
            pred_bin = (probs > 0.5).float()
            target_f = masks.float()

            intersection = torch.sum(pred_bin * target_f, dim=(1, 2, 3))
            union = torch.sum((pred_bin + target_f) > 0, dim=(1, 2, 3)).float()
            cardinality = torch.sum(pred_bin + target_f, dim=(1, 2, 3))

            dice_batch = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
            iou_batch = (intersection + 1e-6) / (union + 1e-6)

            # 2. Pixel Coordinate Error & Signal Metrics
            pred_coords = output.pixel_coordinates  # [B, W]
            cal_mv = output.calibrated_voltage      # [B, W]

            batch_sz = len(images)
            for i in range(batch_sz):
                v_cols = valid_cols[i] > 0.5
                if torch.sum(v_cols) < 10:
                    continue

                p_c = pred_coords[i, v_cols].cpu().numpy()
                t_c = coords[i, v_cols].cpu().numpy()
                mae_px = float(np.mean(np.abs(p_c - t_c)))

                p_sig = cal_mv[i, v_cols].cpu().numpy()
                t_sig = signals[i, v_cols].cpu().numpy()

                m = compute_lead_metrics(ground_truth=t_sig, reconstructed=p_sig, lead_name=lead_names[i])
                r_val = float(m.pearson_r)
                rmse_val = float(m.rmse_mv)

                d_val = float(dice_batch[i].item())
                iou_val = float(iou_batch[i].item())

                total_dice += d_val
                total_iou += iou_val
                total_pixel_mae += mae_px
                total_pearson += r_val
                total_rmse_mv += rmse_val
                total_samples += 1

                # Lead breakdown
                l_name = lead_names[i]
                if l_name in per_lead_metrics:
                    per_lead_metrics[l_name]["pearson"].append(r_val)
                    per_lead_metrics[l_name]["rmse"].append(rmse_val)
                    per_lead_metrics[l_name]["mae_px"].append(mae_px)

                # Degradation breakdown
                d_type = deg_types[i]
                if d_type not in per_deg_metrics:
                    per_deg_metrics[d_type] = {"pearson": [], "rmse": [], "mae_px": [], "dice": []}
                per_deg_metrics[d_type]["pearson"].append(r_val)
                per_deg_metrics[d_type]["rmse"].append(rmse_val)
                per_deg_metrics[d_type]["mae_px"].append(mae_px)
                per_deg_metrics[d_type]["dice"].append(d_val)

    if total_samples == 0:
        return {"samples": 0}

    lead_summary = {}
    for l_name, vals in per_lead_metrics.items():
        if vals["pearson"]:
            lead_summary[l_name] = {
                "mean_pearson_r": round(float(np.mean(vals["pearson"])), 4),
                "mean_rmse_mv": round(float(np.mean(vals["rmse"])), 4),
                "mean_mae_px": round(float(np.mean(vals["mae_px"])), 2),
                "count": len(vals["pearson"]),
            }

    deg_summary = {}
    for d_type, vals in per_deg_metrics.items():
        if vals["pearson"]:
            deg_summary[d_type] = {
                "mean_pearson_r": round(float(np.mean(vals["pearson"])), 4),
                "mean_rmse_mv": round(float(np.mean(vals["rmse"])), 4),
                "mean_mae_px": round(float(np.mean(vals["mae_px"])), 2),
                "mean_dice": round(float(np.mean(vals["dice"])), 4),
                "count": len(vals["pearson"]),
            }

    return {
        "samples": total_samples,
        "mean_dice": round(total_dice / total_samples, 4),
        "mean_iou": round(total_iou / total_samples, 4),
        "mean_pixel_mae": round(total_pixel_mae / total_samples, 2),
        "mean_pearson_r": round(total_pearson / total_samples, 4),
        "mean_rmse_mv": round(total_rmse_mv / total_samples, 4),
        "per_lead": lead_summary,
        "per_degradation": deg_summary,
    }


def run_controlled_training_experiment():
    parser = argparse.ArgumentParser(description="Train Robust ECG Digitizer Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--train-records", type=int, default=120, help="Number of training PTB-XL records")
    parser.add_argument("--val-records", type=int, default=40, help="Number of validation PTB-XL records")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Compute device: %s", device)

    # 1. Strict Fold Separation
    db_csv = Path("data/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl_database.csv")
    base_dir = db_csv.parent
    df = pd.read_csv(db_csv)

    train_df = df[df["strat_fold"].between(1, 8)]
    val_df = df[df["strat_fold"] == 9]
    test_df = df[df["strat_fold"] == 10]

    logger.info("PTB-XL Patient Split: Train(Folds 1-8)=%d, Val(Fold 9)=%d, Test(Fold 10)=%d [Quarantined]",
                len(train_df), len(val_df), len(test_df))

    train_paths = [base_dir / p for p in train_df["filename_lr"].head(args.train_records)]
    val_paths = [base_dir / p for p in val_df["filename_lr"].head(args.val_records)]

    train_ds = PrecachedRealisticDataset(train_paths, num_records=args.train_records, is_validation=False, seed=42)
    val_ds = PrecachedRealisticDataset(val_paths, num_records=args.val_records, is_validation=True, seed=999)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_crops)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_crops)

    # 2. Phase A: Evaluate Existing Baseline Checkpoint
    baseline_ckpt_path = Path("models/ecg_digitizer_unet_all12lead.pt")
    baseline_eval = {}
    if baseline_ckpt_path.is_file():
        logger.info("Evaluating Existing Baseline Checkpoint: %s", baseline_ckpt_path)
        baseline_model = ECGDigitizerUNet(in_channels=3, base_channels=32, temperature=1.0)
        ckpt = torch.load(baseline_ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        baseline_model.load_state_dict(state)
        baseline_model.to(device)
        baseline_eval = evaluate_model_on_dataset(baseline_model, val_loader, device)
        logger.info(
            "BASELINE METRICS on Realistic Val: Dice=%.4f, Pixel MAE=%.2f px, Pearson r=%.4f, RMSE=%.4f mV",
            baseline_eval.get("mean_dice", 0.0),
            baseline_eval.get("mean_pixel_mae", 0.0),
            baseline_eval.get("mean_pearson_r", 0.0),
            baseline_eval.get("mean_rmse_mv", 0.0),
        )

    # 3. Phase B: Train Retrained Robust Model with Sharpened Soft-Argmax
    logger.info("Initializing Retrained Robust ECGDigitizerUNet (T=0.5)...")
    model = ECGDigitizerUNet(in_channels=3, base_channels=32, temperature=0.5).to(device)
    loss_fn = DigitizerReconstructionLoss(
        bce_weight=1.0,
        dice_weight=1.5,
        waveform_l1_weight=2.0,
        waveform_corr_weight=1.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_r = -1.0
    robust_ckpt_path = Path("models/ecg_digitizer_unet_v3_robust.pt")
    robust_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_logs = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for images, masks, signals, coords, valid_cols, baselines, _, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            signals = signals.to(device)
            valid_cols = valid_cols.to(device)
            baselines = baselines.to(device)

            optimizer.zero_grad()
            output = model(images, baseline_y=baselines)
            loss, _ = loss_fn(output, masks, target_signal=signals, confidence_mask=valid_cols)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(1, n_batches)

        # Validation evaluation
        val_metrics = evaluate_model_on_dataset(model, val_loader, device)
        val_r = val_metrics.get("mean_pearson_r", 0.0)
        val_mae = val_metrics.get("mean_pixel_mae", 0.0)
        val_dice = val_metrics.get("mean_dice", 0.0)
        val_rmse = val_metrics.get("mean_rmse_mv", 0.0)

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Dice: %.4f | Val MAE: %.2f px | Val Pearson r: %.4f | Val RMSE: %.4f mV",
            epoch, args.epochs, train_loss, val_dice, val_mae, val_r, val_rmse
        )

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_dice": val_dice,
            "val_pixel_mae": val_mae,
            "val_pearson_r": val_r,
            "val_rmse_mv": val_rmse,
        })

        if val_r >= best_val_r:
            best_val_r = val_r
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_pearson_r": best_val_r,
                "temperature": 0.5,
                "base_channels": 32,
            }, robust_ckpt_path)

    duration = time.time() - t_start
    logger.info("Training completed in %.2f s. Best Val Pearson r: %.4f. Saved to %s",
                duration, best_val_r, robust_ckpt_path)

    # 4. Final Verification of Best Model
    best_ckpt = torch.load(robust_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    final_robust_eval = evaluate_model_on_dataset(model, val_loader, device)

    # 5. Export Comparative Report
    metrics_file = Path("models/digitizer_v3_robust_metrics.json")
    summary = {
        "experiment_title": "Controlled Experiment: Baseline vs. Retrained Robust ECGDigitizerUNet",
        "training_duration_s": round(duration, 2),
        "compute_device": str(device),
        "dataset_split": {
            "train_folds": "1-8",
            "val_fold": "9",
            "test_fold": "10 (quarantined)",
            "train_crops_count": len(train_ds),
            "val_crops_count": len(val_ds),
        },
        "baseline_checkpoint": {
            "path": str(baseline_ckpt_path),
            "metrics": baseline_eval,
        },
        "retrained_robust_model": {
            "path": str(robust_ckpt_path),
            "metrics": final_robust_eval,
        },
        "epochs": epoch_logs,
    }

    with open(metrics_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved complete comparative metrics to %s", metrics_file)


if __name__ == "__main__":
    run_controlled_training_experiment()
