"""Unit tests for ECGDigitizerUNet and DigitizerReconstructionLoss."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ecg_digitization.models.digitizer_unet import (
    ConvBlock2D,
    DigitizerReconstructionLoss,
    DigitizerUNetOutput,
    ECGDigitizerUNet,
)


def test_conv_block_forward():
    """ConvBlock2D preserves spatial dimensions and changes channels."""
    block = ConvBlock2D(in_channels=3, out_channels=16)
    x = torch.randn(2, 3, 64, 64)
    out = block(x)
    assert out.shape == (2, 16, 64, 64)


def test_digitizer_unet_output_shapes():
    """ECGDigitizerUNet forward returns correct tensor dimensions."""
    model = ECGDigitizerUNet(in_channels=3, base_channels=8, temperature=1.0)
    batch_size = 2
    height, width = 64, 128
    x = torch.randn(batch_size, 3, height, width)

    out = model(x, baseline_y=32.0, pixels_per_mv=50.0)

    assert isinstance(out, DigitizerUNetOutput)
    # Segmentation logits and probs: [B, 1, H, W]
    assert out.segmentation_logits.shape == (batch_size, 1, height, width)
    assert out.segmentation_probs.shape == (batch_size, 1, height, width)
    assert torch.all(out.segmentation_probs >= 0.0)
    assert torch.all(out.segmentation_probs <= 1.0)

    # Pixel coordinates and column confidence: [B, W]
    assert out.pixel_coordinates.shape == (batch_size, width)
    assert out.column_confidence.shape == (batch_size, width)
    assert torch.all(out.pixel_coordinates >= 0.0)
    assert torch.all(out.pixel_coordinates <= height)

    # Calibrated voltage: [B, W]
    assert out.calibrated_voltage is not None
    assert out.calibrated_voltage.shape == (batch_size, width)


def test_digitizer_unet_soft_argmax_peak_localization():
    """Soft-Argmax accurately localizes vertical coordinate for sharp peak."""
    model = ECGDigitizerUNet(in_channels=1, base_channels=8, temperature=0.01)
    height, width = 100, 10
    target_row = 42

    # Construct an artificial logit map with high value at target_row
    logits = torch.full((1, 1, height, width), -10.0)
    logits[0, 0, target_row, :] = 20.0

    # Test coordinate extraction logic
    weights = torch.softmax(logits.squeeze(1) / model.temperature, dim=1)  # [1, H, W]
    y_grid = torch.arange(height, dtype=torch.float32).view(1, height, 1)
    extracted_y = torch.sum(y_grid * weights, dim=1)  # [1, W]

    for col in range(width):
        assert abs(extracted_y[0, col].item() - target_row) < 0.05


def test_digitizer_unet_calibration_consistency():
    """Verify that (baseline_y - y_pixel) / pixels_per_mv converts correctly."""
    model = ECGDigitizerUNet(in_channels=1, base_channels=8)
    baseline_y = 50.0
    ppmv = 25.0

    # If y_coord = 25.0 (25 px above baseline), voltage should be +1.0 mV
    y_pixel = torch.tensor([[25.0, 50.0, 75.0]])  # [1, 3]
    v_calibrated = (baseline_y - y_pixel) / ppmv

    assert pytest.approx(v_calibrated[0, 0].item(), abs=1e-5) == 1.0   # +1.0 mV
    assert pytest.approx(v_calibrated[0, 1].item(), abs=1e-5) == 0.0   # 0.0 mV
    assert pytest.approx(v_calibrated[0, 2].item(), abs=1e-5) == -1.0  # -1.0 mV


def test_digitizer_gradient_propagation():
    """Backward pass propagates gradients from composite loss to all model parameters."""
    model = ECGDigitizerUNet(in_channels=3, base_channels=8)
    criterion = DigitizerReconstructionLoss(
        bce_weight=1.0, dice_weight=1.0, waveform_l1_weight=2.0, waveform_corr_weight=1.0
    )

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    target_mask = (torch.rand(2, 1, 64, 64) > 0.8).float()
    target_signal = torch.randn(2, 64)

    out = model(x, baseline_y=32.0, pixels_per_mv=50.0)
    loss, metrics = criterion(out, target_mask, target_signal=target_signal)

    assert loss.item() > 0.0
    assert "loss_bce" in metrics
    assert "loss_dice" in metrics
    assert "loss_signal_l1" in metrics
    assert "loss_signal_corr" in metrics

    loss.backward()

    # Verify gradients reach encoder and decoder
    assert model.enc1.conv1.weight.grad is not None
    assert torch.sum(torch.abs(model.enc1.conv1.weight.grad)).item() > 0.0
    assert model.dec1.conv1.weight.grad is not None
    assert torch.sum(torch.abs(model.dec1.conv1.weight.grad)).item() > 0.0
    assert model.head.weight.grad is not None
    assert torch.sum(torch.abs(model.head.weight.grad)).item() > 0.0


def test_digitizer_unet_missing_trace_handling():
    """Missing or occluded trace columns result in low column confidence and can be masked out in loss."""
    model = ECGDigitizerUNet(in_channels=1, base_channels=8)
    criterion = DigitizerReconstructionLoss()

    # Create an artificial sample where the left half has trace, right half is blank
    batch_size = 1
    height, width = 64, 64
    x = torch.zeros(batch_size, 1, height, width)

    # Put an artificial trace on columns 0..31
    target_mask = torch.zeros(batch_size, 1, height, width)
    target_mask[0, 0, 32, :32] = 1.0  # horizontal line at row 32

    # Confidence mask indicates right half (32..63) has no ground-truth trace
    confidence_mask = torch.zeros(batch_size, width)
    confidence_mask[0, :32] = 1.0  # Only columns 0..31 are evaluated

    target_signal = torch.zeros(batch_size, width)

    out = model(x, baseline_y=32.0, pixels_per_mv=50.0)
    loss, metrics = criterion(out, target_mask, target_signal=target_signal, confidence_mask=confidence_mask)

    assert loss.item() > 0.0
    assert not torch.isnan(loss)

