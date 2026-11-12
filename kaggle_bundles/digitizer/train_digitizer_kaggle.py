"""ECG DigiAI — Autonomous Kaggle GPU Training Script for Image Digitization.
Trains ECGDigitizerUNet (2D U-Net + Soft-Argmax) on paired synthetic ECG crops from PTB-XL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# 1. In-Memory Synthetic ECG Crop Generator
def render_lead_crop(lead_signal, crop_h=128, crop_w=256, pixels_per_mv=59.055, seed=None):
    rng = np.random.RandomState(seed)
    canvas = np.full((crop_h, crop_w, 3), 252, dtype=np.uint8)

    # Draw grid lines (BGR pink grid: Blue=215, Green=215, Red=255 minor; Blue=140, Green=140, Red=235 major)
    px_1mm = 150 / 25.4  # ~5.9 px
    px_5mm = 5.0 * px_1mm
    for x in np.arange(0, crop_w, px_1mm):
        cv2.line(canvas, (int(round(x)), 0), (int(round(x)), crop_h), (215, 215, 255), 1)
    for y in np.arange(0, crop_h, px_1mm):
        cv2.line(canvas, (0, int(round(y))), (crop_w, int(round(y))), (215, 215, 255), 1)
    for x in np.arange(0, crop_w, px_5mm):
        cv2.line(canvas, (int(round(x)), 0), (int(round(x)), crop_h), (140, 140, 235), 1)
    for y in np.arange(0, crop_h, px_5mm):
        cv2.line(canvas, (0, int(round(y))), (crop_w, int(round(y))), (140, 140, 235), 1)

    baseline_y = float(crop_h / 2.0)
    trace_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

    # Resample lead signal to crop_w
    num_pts = len(lead_signal)
    gt_resampled = np.interp(np.linspace(0, num_pts - 1, crop_w), np.arange(num_pts), lead_signal)
    y_coords = baseline_y - (gt_resampled * pixels_per_mv)

    pts = np.column_stack([np.arange(crop_w), y_coords]).astype(np.int32)
    cv2.polylines(canvas, [pts], isClosed=False, color=(20, 20, 30), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(trace_mask, [pts], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_8)

    # Optional augmentation: scanner sensor noise
    if rng.rand() > 0.5:
        noise = rng.normal(0, 3.0, canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    mask_tensor = (torch.from_numpy(trace_mask).unsqueeze(0).float() > 127).float()
    sig_tensor = torch.from_numpy(gt_resampled).float()

    return img_tensor, mask_tensor, sig_tensor, baseline_y


class PairedCropDataset(Dataset):
    def __init__(self, signals, crops_per_record=12, seed=42):
        self.samples = []
        for i, sig in enumerate(signals):
            for crop_idx in range(crops_per_record):
                # Balanced sampling across all 12 standard leads
                lead_i = crop_idx % 12
                col_i = lead_i // 3
                start = col_i * 250
                seg = sig[lead_i, start:start+250]
                self.samples.append((seg, seed + i * 100 + crop_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg, s = self.samples[idx]
        return render_lead_crop(seg, crop_h=128, crop_w=256, seed=s)


# 2. Architecture: ConvBlock2D and ECGDigitizerUNet
class ConvBlock2D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c)) if in_c != out_c else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + res)


class ECGDigitizerUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, temperature=1.0, default_ppmv=59.055):
        super().__init__()
        self.in_channels = in_channels
        self.temperature = float(temperature)
        self.default_ppmv = float(default_ppmv)

        c = base_channels
        self.enc1 = ConvBlock2D(in_channels, c)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock2D(c, c * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = ConvBlock2D(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = ConvBlock2D(c * 4, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.dec3 = ConvBlock2D(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.dec2 = ConvBlock2D(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.dec1 = ConvBlock2D(c * 2, c)

        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x, baseline_y=64.0):
        b, c, h, w = x.shape
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        seg_logits = self.head(d1)
        seg_probs = torch.sigmoid(seg_logits)

        # Differentiable Soft-Argmax
        logits_map = seg_logits.squeeze(1)
        weights_y = F.softmax(logits_map / max(self.temperature, 1e-4), dim=1)
        y_grid = torch.arange(h, dtype=torch.float32, device=x.device).view(1, h, 1)
        pixel_coords = torch.sum(y_grid * weights_y, dim=1)

        base_y = float(baseline_y) if isinstance(baseline_y, (int, float)) else baseline_y.view(-1, 1)
        calibrated_mv = (base_y - pixel_coords) / self.default_ppmv

        return seg_logits, seg_probs, pixel_coords, calibrated_mv


# 3. Composite Reconstruction Loss
class CompositeReconstructionLoss(nn.Module):
    def __init__(self, bce_w=1.0, dice_w=1.0, l1_w=2.0, corr_w=1.0):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.l1_w = l1_w
        self.corr_w = corr_w
        self.eps = 1e-6

    def forward(self, logits, probs, pred_sig, target_mask, target_sig):
        bce = F.binary_cross_entropy_with_logits(logits, target_mask)
        inter = torch.sum(probs * target_mask, dim=(1, 2, 3))
        card = torch.sum(probs + target_mask, dim=(1, 2, 3))
        dice = torch.mean(1.0 - (2.0 * inter + self.eps) / (card + self.eps))

        l1 = F.smooth_l1_loss(pred_sig, target_sig)

        p_cent = pred_sig - torch.mean(pred_sig, dim=1, keepdim=True)
        t_cent = target_sig - torch.mean(target_sig, dim=1, keepdim=True)
        cov = torch.sum(p_cent * t_cent, dim=1)
        p_std = torch.sqrt(torch.sum(p_cent ** 2, dim=1) + self.eps)
        t_std = torch.sqrt(torch.sum(t_cent ** 2, dim=1) + self.eps)
        r = cov / (p_std * t_std)
        corr_loss = torch.mean(1.0 - r)

        total = self.bce_w * bce + self.dice_w * dice + self.l1_w * l1 + self.corr_w * corr_loss
        return total, bce.item(), dice.item(), l1.item(), torch.mean(r).item()


def load_record_signals(df_subset, dataset_root, max_records=2500):
    try:
        import wfdb
    except ImportError:
        os.system("pip install -q wfdb")
        import wfdb

    signals = []
    valid_ids = []
    count = 0
    for ecg_id, row in df_subset.iterrows():
        if count >= max_records:
            break
        fn = str(row["filename_lr"])
        full_path = dataset_root / fn
        try:
            rec = wfdb.rdrecord(str(full_path))
            sig = rec.p_signal.T  # (12, 1000)
            if sig.shape == (12, 1000) and np.isfinite(sig).all():
                signals.append(sig.astype(np.float32))
                valid_ids.append(ecg_id)
                count += 1
        except Exception:
            continue
    return np.array(signals), valid_ids


def main():
    print("=== Starting ECG DigiAI Digitizer U-Net GPU Training on Kaggle ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # Locate PTB-XL Dataset on Kaggle
    input_dir = Path("/kaggle/input")
    csv_paths = list(input_dir.glob("**/ptbxl_database.csv"))
    if not csv_paths:
        raise FileNotFoundError("Could not locate ptbxl_database.csv in /kaggle/input.")

    db_path = csv_paths[0]
    dataset_root = db_path.parent
    print(f"Found PTB-XL database at: {db_path}")

    df = pd.read_csv(db_path, index_col="ecg_id")
    print(f"Total PTB-XL records: {len(df)}")

    # Stratified Split (Folds 1-8 Train, 9 Val, 10 Test)
    train_mask = df["strat_fold"].isin(range(1, 9))
    val_mask = df["strat_fold"] == 9
    test_mask = df["strat_fold"] == 10

    print("Loading training and validation signals from PTB-XL...")
    train_sigs, train_ids = load_record_signals(df[train_mask], dataset_root, max_records=2500)
    val_sigs, val_ids = load_record_signals(df[val_mask], dataset_root, max_records=500)
    print(f"Loaded train: {len(train_sigs)} unique records, val: {len(val_sigs)} unique records")

    train_dataset = PairedCropDataset(train_sigs, crops_per_record=12, seed=42)
    val_dataset = PairedCropDataset(val_sigs, crops_per_record=12, seed=123)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Total training crops: {len(train_dataset)}, Validation crops: {len(val_dataset)}")

    model = ECGDigitizerUNet(in_channels=3, base_channels=32, temperature=1.0).to(device)
    criterion = CompositeReconstructionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    best_val_r = -1.0
    metrics_history = []

    for epoch in range(1, 5 + 1):
        model.train()
        train_loss, train_r = 0.0, 0.0
        for imgs, masks, sigs, bases in train_loader:
            imgs, masks, sigs = imgs.to(device), masks.to(device), sigs.to(device)
            bases = bases.float().to(device)

            optimizer.zero_grad()
            logits, probs, coords, pred_mv = model(imgs, baseline_y=bases)
            loss, bce_v, dice_v, l1_v, r_v = criterion(logits, probs, pred_mv, masks, sigs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(imgs)
            train_r += r_v * len(imgs)

        train_loss /= len(train_dataset)
        train_r /= len(train_dataset)

        model.eval()
        val_loss, val_r, val_rmse = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, masks, sigs, bases in val_loader:
                imgs, masks, sigs = imgs.to(device), masks.to(device), sigs.to(device)
                bases = bases.float().to(device)

                logits, probs, coords, pred_mv = model(imgs, baseline_y=bases)
                v_loss, _, _, l1_val, r_val = criterion(logits, probs, pred_mv, masks, sigs)

                val_loss += v_loss.item() * len(imgs)
                val_r += r_val * len(imgs)
                val_rmse += torch.sqrt(torch.mean((pred_mv - sigs) ** 2, dim=1)).sum().item()

        val_loss /= len(val_dataset)
        val_r /= len(val_dataset)
        val_rmse /= len(val_dataset)

        print(f"Epoch {epoch}/5 - Train Loss: {train_loss:.4f} (r={train_r:.4f}) | Val Loss: {val_loss:.4f} (r={val_r:.4f}, RMSE={val_rmse:.4f} mV)")
        metrics_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_pearson_r": round(train_r, 4),
            "val_loss": round(val_loss, 4),
            "val_pearson_r": round(val_r, 4),
            "val_rmse_mv": round(val_rmse, 4),
        })

        if val_r > best_val_r:
            best_val_r = val_r
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": {"in_channels": 3, "base_channels": 32, "temperature": 1.0, "default_pixels_per_mv": 59.055},
                "epoch": epoch,
                "best_val_pearson_r": best_val_r,
                "val_rmse_mv": val_rmse,
                "supported_leads": ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6", "II_RHYTHM"],
            }
            torch.save(ckpt, "ecg_digitizer_unet_all12lead.pt")
            torch.save(ckpt, "ecg_digitizer_unet.pt")
            print(f"--> Saved best digitizer checkpoint (Pearson r={best_val_r:.4f}) to ecg_digitizer_unet_all12lead.pt")

    metrics_dict = {
        "best_val_pearson_r": round(best_val_r, 4),
        "val_rmse_mv": round(val_rmse, 4),
        "supported_leads": ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6", "II_RHYTHM"],
        "epochs": metrics_history,
    }
    with open("digitizer_all12lead_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    with open("digitizer_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print("=== Digitizer Training Complete. Artifacts saved to working directory. ===")


if __name__ == "__main__":
    main()
