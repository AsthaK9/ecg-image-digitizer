"""ECG Digitizer U-Net with Differentiable Soft-Argmax Waveform Extraction.

Implements:
1. 2D ResNet-U-Net encoder-decoder for pixel-level ECG trace segmentation.
2. Differentiable Soft-Argmax column-wise coordinate extraction.
3. Column confidence tracking and missing-trace detection.
4. Differentiable pixel-to-voltage conversion.
5. End-to-end multi-task loss (Dice + BCE + SmoothL1 + Pearson Correlation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DigitizerUNetOutput:
    """Structured output from ECGDigitizerUNet forward pass."""

    segmentation_logits: torch.Tensor       # [B, 1, H, W]
    segmentation_probs: torch.Tensor        # [B, 1, H, W] in [0, 1]
    pixel_coordinates: torch.Tensor         # [B, W] extracted continuous y-coordinates (0 at top)
    column_confidence: torch.Tensor         # [B, W] in [0, 1] column-wise peak trace confidence
    calibrated_voltage: Optional[torch.Tensor] = None  # [B, W] in mV (if calibration parameters provided)


class ConvBlock2D(nn.Module):
    """Two convolutional layers with BatchNorm, ReLU, and optional residual connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + res)


class ECGDigitizerUNet(nn.Module):
    """2D U-Net with differentiable Soft-Argmax for image-to-signal digitisation.

    Args:
        in_channels: Number of input image channels (e.g. 1 for grayscale, 3 for RGB).
        base_channels: Number of feature channels in the first encoder layer.
        temperature: Softmax temperature parameter for vertical coordinate extraction.
                     Lower values sharpen the peak; higher values smooth the distribution.
        default_pixels_per_mv: Default physical calibration gain (10 mm/mV * 150 DPI / 25.4 = ~59.06 px/mV).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        temperature: float = 1.0,
        default_pixels_per_mv: float = 59.055,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.temperature = float(temperature)
        self.default_pixels_per_mv = float(default_pixels_per_mv)

        # Encoder (Downsampling)
        c = base_channels
        self.enc1 = ConvBlock2D(in_channels, c)               # -> [B, c, H, W]
        self.pool1 = nn.MaxPool2d(2, 2)                      # -> [B, c, H/2, W/2]

        self.enc2 = ConvBlock2D(c, c * 2)                     # -> [B, 2c, H/2, W/2]
        self.pool2 = nn.MaxPool2d(2, 2)                      # -> [B, 2c, H/4, W/4]

        self.enc3 = ConvBlock2D(c * 2, c * 4)                 # -> [B, 4c, H/4, W/4]
        self.pool3 = nn.MaxPool2d(2, 2)                      # -> [B, 4c, H/8, W/8]

        # Bottleneck
        self.bottleneck = ConvBlock2D(c * 4, c * 8)           # -> [B, 8c, H/8, W/8]

        # Decoder (Upsampling with skip connections)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock2D(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2D(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock2D(c * 2, c)

        # Final 1x1 conv to produce binary trace segmentation logits
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        baseline_y: Optional[Union[float, torch.Tensor]] = None,
        pixels_per_mv: Optional[float] = None,
    ) -> DigitizerUNetOutput:
        """Forward pass.

        Args:
            x: Input image tensor of shape [B, C, H, W].
            baseline_y: Optional 0 mV baseline pixel coordinate in y (scalar or [B]).
            pixels_per_mv: Optional scale for physical voltage conversion.

        Returns:
            DigitizerUNetOutput containing segmentation logits, probabilities,
            continuous pixel coordinates, column confidence, and calibrated voltages.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")

        b, c, h, w = x.shape

        # 1. U-Net Backbone
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # [B, 1, H, W]
        seg_logits = self.head(d1)
        seg_probs = torch.sigmoid(seg_logits)

        # 2. Differentiable Soft-Argmax Coordinate Extraction
        # Squeeze channel dim: [B, H, W]
        prob_map = seg_probs.squeeze(1)
        logits_map = seg_logits.squeeze(1)

        # Softmax along vertical axis (dim=1)
        weights_y = F.softmax(logits_map / max(self.temperature, 1e-4), dim=1)  # [B, H, W]

        # Grid of vertical pixel indices [0, 1, ..., H-1]
        y_grid = torch.arange(h, dtype=torch.float32, device=x.device).view(1, h, 1)  # [1, H, 1]

        # Spatial expected value: y_hat(x) = sum_y( y * P(y | x) )
        pixel_coords = torch.sum(y_grid * weights_y, dim=1)  # [B, W]

        # 3. Column-wise Confidence (Peak probability in each column)
        col_confidence, _ = torch.max(prob_map, dim=1)  # [B, W]

        # 4. Calibrated Physical Voltage (mV)
        # In image coordinates, y increases downwards.
        # Positive physiological voltage goes upwards -> V = (y_base - y_coords) / pixels_per_mv
        ppmv = pixels_per_mv if pixels_per_mv is not None else self.default_pixels_per_mv

        if baseline_y is None:
            # Default baseline to vertical midpoint H / 2
            base_y = float(h / 2.0)
        elif isinstance(baseline_y, (int, float)):
            base_y = float(baseline_y)
        else:
            base_y = baseline_y.view(-1, 1)

        calibrated_mv = (base_y - pixel_coords) / ppmv

        return DigitizerUNetOutput(
            segmentation_logits=seg_logits,
            segmentation_probs=seg_probs,
            pixel_coordinates=pixel_coords,
            column_confidence=col_confidence,
            calibrated_voltage=calibrated_mv,
        )


class DigitizerReconstructionLoss(nn.Module):
    """End-to-end composite loss for joint segmentation and waveform reconstruction.

    Combines:
    - Binary Cross Entropy (BCE) on segmentation mask
    - Dice Loss on segmentation mask
    - Smooth L1 Loss on calibrated waveform voltage
    - Pearson Morphological Correlation Loss (1 - r)
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        waveform_l1_weight: float = 2.0,
        waveform_corr_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.l1_weight = waveform_l1_weight
        self.corr_weight = waveform_corr_weight
        self.eps = eps

    def forward(
        self,
        output: DigitizerUNetOutput,
        target_mask: torch.Tensor,
        target_signal: Optional[torch.Tensor] = None,
        confidence_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Calculates multi-task composite loss.

        Args:
            output: DigitizerUNetOutput from model.
            target_mask: Ground-truth binary trace mask [B, 1, H, W] in {0, 1}.
            target_signal: Ground-truth voltage signal [B, W] in mV.
            confidence_mask: Optional binary indicator [B, W] of valid columns.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        device = target_mask.device

        # 1. Segmentation BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(output.segmentation_logits, target_mask.float())

        # 2. Dice Loss
        probs = output.segmentation_probs
        target_f = target_mask.float()
        intersection = torch.sum(probs * target_f, dim=(1, 2, 3))
        cardinality = torch.sum(probs + target_f, dim=(1, 2, 3))
        dice_loss = torch.mean(1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps))

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        metrics = {
            "loss_bce": bce_loss.item(),
            "loss_dice": dice_loss.item(),
        }

        # 3. Waveform Signal Losses (if target signal is provided)
        if target_signal is not None and output.calibrated_voltage is not None:
            pred_sig = output.calibrated_voltage
            tgt_sig = target_signal.float()

            if confidence_mask is not None:
                mask = confidence_mask.float()
                diff = F.smooth_l1_loss(pred_sig * mask, tgt_sig * mask, reduction="sum")
                l1_loss = diff / (mask.sum() + self.eps)
            else:
                l1_loss = F.smooth_l1_loss(pred_sig, tgt_sig)

            # Differentiable Pearson Correlation loss: 1 - r
            pred_centered = pred_sig - torch.mean(pred_sig, dim=1, keepdim=True)
            tgt_centered = tgt_sig - torch.mean(tgt_sig, dim=1, keepdim=True)

            covariance = torch.sum(pred_centered * tgt_centered, dim=1)
            pred_std = torch.sqrt(torch.sum(pred_centered ** 2, dim=1) + self.eps)
            tgt_std = torch.sqrt(torch.sum(tgt_centered ** 2, dim=1) + self.eps)

            pearson_r = covariance / (pred_std * tgt_std)
            corr_loss = torch.mean(1.0 - pearson_r)

            total_loss = total_loss + self.l1_weight * l1_loss + self.corr_weight * corr_loss
            metrics["loss_signal_l1"] = l1_loss.item()
            metrics["loss_signal_corr"] = corr_loss.item()
            metrics["mean_pearson_r"] = torch.mean(pearson_r).item()

        metrics["loss_total"] = total_loss.item()
        return total_loss, metrics
