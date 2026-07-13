"""Global seeding utilities for reproducible experiments.

Every training/inference entrypoint should call `set_seed()` exactly once,
as early as possible, before any dataset shuffling, model initialization,
or augmentation sampling occurs.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all sources of randomness used across the project.

    Args:
        seed: Seed value applied to Python's `random`, NumPy, and PyTorch
            (CPU and all CUDA devices).
        deterministic: If True, forces cuDNN into deterministic mode. This
            can reduce throughput but guarantees bit-for-bit reproducibility
            across runs on the same hardware/software stack. Set to False
            for faster training once reproducibility of exact numerics is
            no longer required (e.g. large-scale hyperparameter sweeps).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    logger.info("Global seed set to %d (deterministic=%s)", seed, deterministic)
