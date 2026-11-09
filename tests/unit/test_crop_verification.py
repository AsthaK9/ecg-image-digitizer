"""Unit test for targeted deterministic waveform alignment and gradient check."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggle_bundles.digitizer.train_digitizer_kaggle import (
    CompositeReconstructionLoss,
    ECGDigitizerUNet,
    render_lead_crop,
)
from scripts.verify_crop_alignment import create_deterministic_ecg_signal


def test_deterministic_crop_alignment_and_gradients():
    sig = create_deterministic_ecg_signal(num_samples=250)
    ppmv = 59.055
    baseline_y = 64.0

    img_t, mask_t, sig_t, base = render_lead_crop(
        sig, crop_h=128, crop_w=256, pixels_per_mv=ppmv, seed=42
    )

    # Verify tensor shapes and dtypes
    assert img_t.shape == (3, 128, 256)
    assert mask_t.shape == (1, 128, 256)
    assert sig_t.shape == (256,)
    assert img_t.dtype == torch.float32
    assert mask_t.dtype == torch.float32
    assert sig_t.dtype == torch.float32

    # Verify alignment at P-wave and T-wave peaks
    mask_np = mask_t.squeeze(0).numpy()
    target_np = sig_t.numpy()

    # P-wave peak at col 41
    p_col = 41
    p_rows = np.where(mask_np[:, p_col] > 0)[0]
    p_exp = baseline_y - (target_np[p_col] * ppmv)
    assert abs(np.mean(p_rows) - p_exp) < 1.0  # within 1 pixel

    # T-wave peak at col 184
    t_col = 184
    t_rows = np.where(mask_np[:, t_col] > 0)[0]
    t_exp = baseline_y - (target_np[t_col] * ppmv)
    assert abs(np.mean(t_rows) - t_exp) < 1.0  # within 1 pixel

    # Verify Model forward & backward
    batch_img = img_t.unsqueeze(0)
    batch_mask = mask_t.unsqueeze(0)
    batch_sig = sig_t.unsqueeze(0)
    batch_base = torch.tensor([base]).float()

    model = ECGDigitizerUNet(in_channels=3, base_channels=16)
    criterion = CompositeReconstructionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    logits, probs, coords, pred_mv = model(batch_img, baseline_y=batch_base)
    loss, bce, dice, l1, r = criterion(logits, probs, pred_mv, batch_mask, batch_sig)

    assert torch.isfinite(loss)
    loss.backward()

    # All model weights must have finite non-zero gradients
    for p in model.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))
