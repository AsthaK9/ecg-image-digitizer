"""1D Convolutional Neural Network / ResNet Baseline for Multi-Lead ECG Classification.

Processes standard 12-lead digital ECG waveforms (shape: [batch, 12, signal_length])
and produces multi-label logits for target cardiac pathologies.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """1D Residual block with two convolutional layers and skip connection."""

    def __init__(self, channels: int, kernel_size: int = 7, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.relu(out)


class ECGResNet1D(nn.Module):
    """Multi-lead 1D ResNet for ECG classification."""

    TARGET_CLASSES: List[str] = [
        "AFIB",   # Atrial Fibrillation
        "SBRAD",  # Bradycardia
        "STACH",  # Tachycardia
        "STE_",   # ST Elevation
        "STD_",   # ST Depression
        "BBB",    # Bundle Branch Block
        "MI",     # Myocardial Infarction
    ]

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: Optional[int] = None,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes or len(self.TARGET_CLASSES)

        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(
            ResBlock1D(base_channels, kernel_size=7),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=1),
            nn.MaxPool1d(2),
        )

        self.stage2 = nn.Sequential(
            ResBlock1D(base_channels * 2, kernel_size=7),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=1),
            nn.MaxPool1d(2),
        )

        self.stage3 = nn.Sequential(
            ResBlock1D(base_channels * 4, kernel_size=5),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 4, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, 12, signal_length]

        Returns:
            Logits of shape [batch_size, num_classes]
        """
        feats = self.initial_conv(x)
        feats = self.stage1(feats)
        feats = self.stage2(feats)
        feats = self.stage3(feats)
        logits = self.classifier(feats)
        return logits
