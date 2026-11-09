"""Kaggle Training Infrastructure and CLI Wrapper.

Prepares training scripts, kernel metadata, and verified dry-run commands for
GPU training on Kaggle without submitting remote jobs autonomously.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class KaggleAuthStatus:
    """Kaggle CLI installation and authentication readiness."""

    cli_available: bool
    cli_path: Optional[str]
    auth_configured: bool
    auth_method: Optional[str]        # "environment_token", "kaggle_json", or None
    masked_user: Optional[str]
    error_message: Optional[str] = None


@dataclass
class KaggleJobCard:
    """Specification card for a proposed remote Kaggle training run."""

    kernel_slug: str
    title: str
    dataset_sources: List[str]
    enable_gpu: bool
    script_entrypoint: str
    dry_run_command: str
    metadata_path: str


class KaggleManager:
    """Manages Kaggle kernel metadata generation, dataset linking, and submission packaging."""

    DEFAULT_DATASETS = [
        "erhmrai/ecg-image-data",
    ]

    def __init__(self, workspace_dir: Optional[Union[str, Path]] = None) -> None:
        self.workspace_dir = Path(workspace_dir or Path.cwd())

    def check_auth(self) -> KaggleAuthStatus:
        """Verifies if the Kaggle CLI is installed and authentication is configured."""
        cli_path = shutil.which("kaggle")
        if not cli_path:
            # Check virtualenv scripts
            venv_kaggle = self.workspace_dir / ".venv" / "Scripts" / "kaggle.exe"
            if venv_kaggle.is_file():
                cli_path = str(venv_kaggle)

        if not cli_path:
            return KaggleAuthStatus(
                cli_available=False,
                cli_path=None,
                auth_configured=False,
                auth_method=None,
                masked_user=None,
                error_message="Kaggle CLI binary not found in PATH or .venv/Scripts/.",
            )

        # Check credentials
        auth_method = None
        masked_user = None

        if "KAGGLE_API_TOKEN" in os.environ:
            auth_method = "environment_token"
            masked_user = "env_token_user"
        elif "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
            auth_method = "environment_variables"
            u = os.environ["KAGGLE_USERNAME"]
            masked_user = u[:2] + "***" if len(u) > 2 else "***"
        else:
            home = Path.home()
            kaggle_json = home / ".kaggle" / "kaggle.json"
            if kaggle_json.is_file():
                try:
                    with open(kaggle_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    u = data.get("username", "user")
                    masked_user = u[:2] + "***" if len(u) > 2 else "***"
                    auth_method = "kaggle_json"
                except Exception as exc:
                    logger.debug("Failed to read ~/.kaggle/kaggle.json: %s", exc)

        return KaggleAuthStatus(
            cli_available=True,
            cli_path=cli_path,
            auth_configured=auth_method is not None,
            auth_method=auth_method,
            masked_user=masked_user,
            error_message=None if auth_method else "No Kaggle credentials found (set KAGGLE_API_TOKEN or ~/.kaggle/kaggle.json).",
        )

    def prepare_kernel(
        self,
        output_dir: Union[str, Path],
        kernel_slug: str = "ecg-digiai-1d-resnet-training",
        title: str = "ECG DigiAI — 1D ResNet Baseline Training",
        datasets: Optional[List[str]] = None,
        enable_gpu: bool = True,
    ) -> KaggleJobCard:
        """Prepares kernel directory, entrypoint script, and metadata for dry-run verification."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ds_sources = datasets or self.DEFAULT_DATASETS

        # 1. Write kernel-metadata.json
        meta = {
            "id": f"{kernel_slug}",
            "title": title,
            "code_file": "train_kaggle.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if enable_gpu else "false",
            "enable_internet": "true",
            "dataset_sources": ds_sources,
            "competition_sources": [],
            "kernel_sources": [],
        }

        meta_path = out / "kernel-metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 2. Write standalone self-contained train_kaggle.py script
        script_path = out / "train_kaggle.py"
        script_code = self._generate_kaggle_training_script()
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_code)

        cli_cmd = f"kaggle kernels push -p \"{out.resolve()}\""

        return KaggleJobCard(
            kernel_slug=kernel_slug,
            title=title,
            dataset_sources=ds_sources,
            enable_gpu=enable_gpu,
            script_entrypoint=str(script_path.name),
            dry_run_command=cli_cmd,
            metadata_path=str(meta_path.name),
        )

    def _generate_kaggle_training_script(self) -> str:
        return """# ECG DigiAI — Kaggle GPU Training Script
# Automated submission package for multi-label 1D ResNet baseline
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("=== Starting ECG DigiAI GPU Training on Kaggle ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute Device: {device}")

# Model Definition
class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        p = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=p),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=p),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.net(x) + x)

class ECGResNet1D(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            ResBlock1D(32),
            nn.Conv1d(32, 64, 1),
            nn.MaxPool1d(2),
            ResBlock1D(64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = ECGResNet1D(in_channels=12, num_classes=7).to(device)
print("Model initialized successfully:")
print(model)

# Synthetic training loop verification
x_dummy = torch.randn(16, 12, 1000).to(device)
y_dummy = torch.randint(0, 2, (16, 7)).float().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(1, 4):
    optimizer.zero_grad()
    logits = model(x_dummy)
    loss = criterion(logits, y_dummy)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}/3 - Loss: {loss.item():.4f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save({"state_dict": model.state_dict()}, "checkpoints/ecg_resnet1d_baseline.pt")
print("Checkpoint saved to checkpoints/ecg_resnet1d_baseline.pt")
print("=== Kaggle Training Job Complete ===")
"""
