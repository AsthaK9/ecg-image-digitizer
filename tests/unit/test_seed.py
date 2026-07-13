from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from ecg_digitization.utils.seed import set_seed

pytestmark = pytest.mark.unit


def test_set_seed_reproducible_python_random() -> None:
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_reproducible_numpy() -> None:
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_seed_reproducible_torch() -> None:
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_set_seed_deterministic_flag_sets_cudnn_flags() -> None:
    set_seed(42, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    set_seed(42, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
