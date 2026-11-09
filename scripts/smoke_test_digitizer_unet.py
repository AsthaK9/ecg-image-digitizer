"""CPU Smoke Test for ECGDigitizerUNet.

Validates:
1. Small synthetic batch data generation (4 lead crops).
2. Model forward pass, loss calculation, backward gradient flow, and parameter updates.
3. Loss decrease across 3 optimization steps.
4. Waveform metrics: MAE, RMSE, Pearson r, SNR.
5. Checkpoint saving and reload verification.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from ecg_digitization.data.types import ECGRecordMetadata, ECGSignal
from ecg_digitization.evaluation.metrics import compute_lead_metrics
from ecg_digitization.models.digitizer_unet import (
    DigitizerReconstructionLoss,
    ECGDigitizerUNet,
)
from ecg_digitization.synthetic.generator import (
    DegradationConfig,
    SyntheticECGGenerator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmokeTest")


def generate_synthetic_crops(num_samples: int = 4, crop_h: int = 128, crop_w: int = 256):
    """Generates synthetic lead crops with exact paired masks and signals."""
    generator = SyntheticECGGenerator()
    crops = []
    masks = []
    signals = []
    baselines = []

    leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fs = 100.0
    t = np.linspace(0, 10.0, 1000, endpoint=False)

    for i in range(num_samples):
        # Deterministic synthetic 12-lead signal
        raw_vals = np.zeros((12, 1000), dtype=np.float32)
        for lead_i in range(12):
            raw_vals[lead_i] = (
                0.25 * np.sin(2 * np.pi * 1.2 * t + (lead_i * 0.3) + i)
                + 0.9 * np.exp(-((t % 0.8 - 0.2) ** 2) / 0.001)
            )

        meta = ECGRecordMetadata(
            record_id=f"smoke_{i:04d}",
            source_dataset="ptbxl_sim",
            sampling_rate_hz=fs,
            num_leads=12,
            lead_names=leads,
            signal_length=1000,
        )
        sig = ECGSignal(values=raw_vals, metadata=meta)
        result = generator.generate(sig, seed=100 + i)

        # Extract Lead II crop
        lead_name = "II"
        x1, y1, x2, y2 = result.lead_bounding_boxes[lead_name]
        crop_img = result.clean_image[y1:y2, x1:x2]
        crop_mask = result.trace_mask[y1:y2, x1:x2]

        # Resize to fixed standard training dimensions [crop_h, crop_w]
        import cv2
        resized_img = cv2.resize(crop_img, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(crop_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        # Convert to float tensors
        # Image: [3, H, W] in [0, 1]
        img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
        # Mask: [1, H, W] in {0, 1}
        mask_tensor = (torch.from_numpy(resized_mask).unsqueeze(0).float() > 127).float()

        # Resample ground truth lead signal to match crop_w points
        gt_lead_sig = result.exact_lead_signals[lead_name]
        sig_resampled = np.interp(
            np.linspace(0, len(gt_lead_sig) - 1, crop_w),
            np.arange(len(gt_lead_sig)),
            gt_lead_sig,
        )
        sig_tensor = torch.from_numpy(sig_resampled).float()

        crops.append(img_tensor)
        masks.append(mask_tensor)
        signals.append(sig_tensor)
        baselines.append(float(crop_h / 2.0))

    return (
        torch.stack(crops, dim=0),      # [B, 3, H, W]
        torch.stack(masks, dim=0),      # [B, 1, H, W]
        torch.stack(signals, dim=0),    # [B, W]
        torch.tensor(baselines).float() # [B]
    )


def run_smoke_test():
    """Runs CPU training smoke test and metrics verification."""
    logger.info("Starting CPU Smoke Test for ECGDigitizerUNet...")

    # 1. Generate synthetic mini-batch
    batch_x, batch_mask, batch_sig, batch_base = generate_synthetic_crops(num_samples=4)
    logger.info("Generated synthetic batch: x=%s, mask=%s, signal=%s", batch_x.shape, batch_mask.shape, batch_sig.shape)

    # 2. Instantiate Model
    model = ECGDigitizerUNet(
        in_channels=3,
        base_channels=16,
        temperature=1.0,
        default_pixels_per_mv=59.055,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = DigitizerReconstructionLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        waveform_l1_weight=2.0,
        waveform_corr_weight=1.0,
    )

    # 3. Training Loop (3 steps)
    losses = []
    model.train()
    for step in range(3):
        optimizer.zero_grad()
        out = model(batch_x, baseline_y=batch_base)
        loss, metrics = criterion(out, batch_mask, target_signal=batch_sig)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        logger.info(
            "Step %d: Loss=%.4f (BCE=%.4f, Dice=%.4f, L1=%.4f, Corr=%.4f, Pearson r=%.4f)",
            step + 1,
            loss.item(),
            metrics.get("loss_bce", 0.0),
            metrics.get("loss_dice", 0.0),
            metrics.get("loss_signal_l1", 0.0),
            metrics.get("loss_signal_corr", 0.0),
            metrics.get("mean_pearson_r", 0.0),
        )

    # Check that loss is decreasing or stable
    assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    logger.info("Loss successfully decreased from %.4f to %.4f", losses[0], losses[-1])

    # 4. Waveform Reconstruction Metrics Evaluation
    model.eval()
    with torch.no_grad():
        eval_out = model(batch_x, baseline_y=batch_base)
        pred_sig = eval_out.calibrated_voltage.cpu().numpy()
        true_sig = batch_sig.cpu().numpy()

        all_r, all_rmse, all_snr = [], [], []
        for i in range(len(pred_sig)):
            m = compute_lead_metrics(ground_truth=true_sig[i], reconstructed=pred_sig[i], lead_name=f"crop_{i}")
            all_r.append(m.pearson_r)
            all_rmse.append(m.rmse_mv)
            all_snr.append(m.snr_db)

        logger.info(
            "Smoke Evaluation: Mean Pearson r=%.4f, Mean RMSE=%.4f mV, Mean SNR=%.2f dB",
            float(np.mean(all_r)),
            float(np.mean(all_rmse)),
            float(np.mean(all_snr)),
        )

    # 5. Checkpoint Saving and Reloading
    ckpt_dir = Path("models")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "digitizer_unet_smoke_test.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "in_channels": 3,
            "base_channels": 16,
            "temperature": 1.0,
            "default_pixels_per_mv": 59.055,
        },
        "metrics": {
            "final_loss": losses[-1],
            "mean_pearson_r": float(np.mean(all_r)),
        },
    }
    torch.save(checkpoint, ckpt_path)
    logger.info("Saved smoke test checkpoint to %s", ckpt_path)

    # Reload and test identical prediction
    loaded_model = ECGDigitizerUNet(**checkpoint["config"])
    loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    loaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
    loaded_model.eval()

    with torch.no_grad():
        reloaded_out = loaded_model(batch_x, baseline_y=batch_base)
        max_diff = torch.max(torch.abs(reloaded_out.calibrated_voltage - eval_out.calibrated_voltage)).item()
        assert max_diff < 1e-6, f"Checkpoint reload mismatch: max diff = {max_diff}"
        logger.info("Checkpoint reload verified: max difference between original and reloaded = %.2e", max_diff)

    logger.info("=== CPU SMOKE TEST PASSED SUCCESSFULLY ===")
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_pearson_r": float(np.mean(all_r)),
        "mean_rmse_mv": float(np.mean(all_rmse)),
        "mean_snr_db": float(np.mean(all_snr)),
        "checkpoint_path": str(ckpt_path),
    }


if __name__ == "__main__":
    run_smoke_test()
